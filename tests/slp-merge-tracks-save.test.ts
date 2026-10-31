/**
 * A track-only re-run ("update_tracks") gives existing detections new tracks. Those
 * must survive a save: the tables store a track as its position in `labels.tracks`,
 * so a detection holding a Track missing from that list would save as no track.
 */
import { describe, it, expect } from "./bun-test";
import { Labels } from "../src/model/labels.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { PredictedInstance, Track } from "../src/model/instance.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";
import { readSlp } from "../src/codecs/slp/read.js";
import {
  buildExpectedSidecars,
  checkInPlaceWritable,
  openSlpWriter,
  saveSlpToBytes,
} from "../src/codecs/slp/write.js";

const skeleton = new Skeleton({ nodes: ["a", "b"], edges: [["a", "b"]] });

function detection(x: number, track: Track | null): PredictedInstance {
  const inst = PredictedInstance.fromArray(
    [
      [x, x],
      [x + 1, x + 1],
    ],
    skeleton,
  );
  inst.track = track;
  return inst;
}

/** Two detections on each of `n` frames, at x = 10 and x = 100. */
function project(names: string[], n = 3): Labels {
  const video = new Video({ filename: "v.mp4" });
  const tracks = names.map((name) => new Track(name));
  const labeledFrames = [...Array(n).keys()].map(
    (f) =>
      new LabeledFrame({
        video,
        frameIdx: f,
        instances: [detection(10, tracks[0]), detection(100, tracks[1])],
      }),
  );
  return new Labels({
    labeledFrames,
    videos: [video],
    skeletons: [skeleton],
    tracks,
  });
}

async function reopen(labels: Labels): Promise<Labels> {
  const bytes = await saveSlpToBytes(labels);
  return readSlp(bytes.buffer as ArrayBuffer, { openVideos: false });
}

const trackNames = (labels: Labels) =>
  labels.labeledFrames.map((lf) => lf.instances.map((i) => i.track?.name));

describe("A track-only merge, saved", () => {
  const merge = (a: Labels, b: Labels) =>
    a.merge(b, { track: "name", video: "basename", frame: "update_tracks" });

  it("new names reach the saved file", async () => {
    const labels = project(["old_0", "old_1"]);
    await merge(labels, project(["new_0", "new_1"]));
    expect(trackNames(await reopen(labels))).toEqual([
      ["new_0", "new_1"],
      ["new_0", "new_1"],
      ["new_0", "new_1"],
    ]);
  });

  it("the same names, the other way round, reach the saved file", async () => {
    const labels = project(["track_0", "track_1"]);
    await merge(labels, project(["track_1", "track_0"]));
    const known = new Set(labels.tracks);
    for (const lf of labels.labeledFrames) {
      for (const inst of lf.instances)
        expect(known.has(inst.track!)).toBe(true);
    }
    expect(labels.tracks.map((t) => t.name)).toEqual(["track_0", "track_1"]);
    const back = await reopen(labels);
    expect(back.tracks.map((t) => t.name)).toEqual(["track_0", "track_1"]);
    expect(trackNames(back)[0]).toEqual(["track_1", "track_0"]);
  });

  it("a detection the re-run did not match keeps its track", async () => {
    const labels = project(["track_0", "track_1"]);
    const other = project(["track_1", "track_0"]);
    other.labeledFrames[1].instances.splice(0, 1);
    await merge(labels, other);
    const back = await reopen(labels);
    expect(trackNames(back)[1]).toEqual(["track_0", "track_0"]);
  });
});

describe("Saving a track only its detections know about", () => {
  it("keeps it", async () => {
    const labels = project(["track_0", "track_1"]);
    labels.labeledFrames[1].instances[1].track = new Track("added");
    const back = await reopen(labels);
    expect(back.tracks.map((t) => t.name)).toEqual([
      "track_0",
      "track_1",
      "added",
    ]);
    expect(trackNames(back)[1]).toEqual(["track_0", "added"]);
  });

  it("stops the in-place save, which cannot rewrite the track list", () => {
    const labels = project(["track_0", "track_1"]);
    const onDisk = buildExpectedSidecars(labels);
    labels.labeledFrames[1].instances[1].track = new Track("added");
    const expected = buildExpectedSidecars(labels);
    expect(labels.tracks.map((t) => t.name)).toContain("added");
    const gate = checkInPlaceWritable({}, {}, { onDisk, expected });
    expect(gate.ok).toBe(false);
  });

  it("is an error on the stream writer, whose track list is already written", async () => {
    const labels = project(["track_0", "track_1"]);
    const writer = await openSlpWriter({
      videos: labels.videos,
      skeletons: labels.skeletons,
      tracks: labels.tracks,
    });
    labels.labeledFrames[0].instances[0].track = new Track("added");
    expect(() => writer.appendFrames(labels.labeledFrames)).toThrow("added");
    writer.dispose();
  });
});
