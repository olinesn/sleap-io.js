/**
 * A save must keep every frame on its video.
 *
 * `/frames/video` in a 2.x file isn't simply the video's list index. A reader takes
 * the stored ids as list indices only when they are exactly `0..n-1`; otherwise it
 * goes by each video's `videoN` number in `videos_json` ({@link buildVideoIdMap}).
 * Every writer has to store what reads back: {@link frameVideoIds}.
 */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { File as H5File, ready } from "h5wasm/node";
import { describe, expect, it } from "./bun-test";
import { StreamingH5Writer } from "../src/codecs/slp/h5-streaming.js";
import { H5_WORKER_CODE } from "../src/codecs/slp/h5-worker.js";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import {
  buildLabelTableRows,
  buildLabelTableUpdate,
  openSlpWriter,
  saveSlpMergedFromStores,
  saveSlpStructureToBytes,
  saveSlpToBytes,
  writeLabelTablesInPlace,
} from "../src/codecs/slp/write.js";
import { Instance } from "../src/model/instance.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { Labels } from "../src/model/labels.js";
import { Skeleton } from "../src/model/skeleton.js";
import {
  buildVideoIdMap,
  frameVideoIds,
  parseVideoIdFromDataset,
  videoIdsReadBack,
  type VideoNumbers,
} from "../src/model/video-id-map.js";
import { Video } from "../src/model/video.js";

const fixturePath = (name: string): string =>
  path.join(fileURLToPath(new URL("./data/slp", import.meta.url)), name);
const fixture = (name: string): ArrayBuffer =>
  new Uint8Array(readFileSync(fixturePath(name))).buffer as ArrayBuffer;
const toBuffer = (bytes: Uint8Array): ArrayBuffer =>
  bytes.slice().buffer as ArrayBuffer;
const closed = { openVideos: false };

/** Embedded videos numbered [2, 0, 1] with real images; frames in the first two. */
const EMBEDDED = "out_of_order_embedded.pkg.slp";
const EMBEDDED_WHERE = [
  [0, 5, false],
  [1, 7, false],
  [1, 9, true],
  [0, 11, false],
];
/** The same numbering; 3.0 and 2.x twins of one project. */
const HANDBUILT_2X = "handbuilt_30_2x.slp";
const HANDBUILT_30 = "handbuilt_30.slp";
const HANDBUILT_WHERE = [
  [0, 5, false],
  [1, 7, false],
  [1, 9, true],
  [0, 11, false],
  [0, 13, false],
];

/** Each frame as `[video's list index, frame index, negative]`. */
const where = (labels: Labels) =>
  labels.labeledFrames.map((f) => [
    labels.videos.indexOf(f.video),
    f.frameIdx,
    f.isNegative,
  ]);

const reload = (bytes: Uint8Array) => readSlp(toBuffer(bytes), closed);

async function tmpH5Path(): Promise<string> {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    // already exists
  }
  return `/tmp/video_ids_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
}

/** The ids a saved file stores, and the dataset of each of its videos. */
async function stored(bytes: Uint8Array): Promise<{
  frames: number[];
  negative: number[][];
  datasets: Array<string | null>;
}> {
  const module = await ready;
  const p = await tmpH5Path();
  module.FS.writeFile(p, bytes);
  const f: any = new H5File(p, "r");
  try {
    const table = f.get("frames");
    const [rows, cols] = table.shape as number[];
    const values = table.value as ArrayLike<number>;
    const negative = f.get("negative_frames");
    return {
      frames: Array.from({ length: rows }, (_, i) =>
        Number(values[i * cols + 1]),
      ),
      // A compound table in the full save, a flat one from the stream writer.
      negative: !negative
        ? []
        : negative.shape.length === 2
          ? Array.from({ length: negative.shape[0] }, (_, i) => [
              Number(negative.value[i * 2]),
              Number(negative.value[i * 2 + 1]),
            ])
          : negative.value.map((row: unknown[]) => row.map(Number)),
      datasets: (f.get("videos_json").value as string[]).map(
        (entry) => JSON.parse(entry).backend?.dataset ?? null,
      ),
    };
  } finally {
    f.close();
    module.FS.unlink(p);
  }
}

const SK = new Skeleton({ nodes: ["a"] });

/**
 * Labels on closed videos holding these numbers, with one frame per entry of
 * `framesOn` (a video's list index). The frame index is the entry's position.
 */
function labelsOn(numbers: VideoNumbers, framesOn: number[]): Labels {
  const videos = numbers.map(
    (number, i) =>
      new Video({
        filename: `clip${i}.mp4`,
        backendMetadata:
          number == null ? {} : { dataset: `video${number}/video` },
      }),
  );
  return new Labels({
    labeledFrames: framesOn.map(
      (index, frameIdx) =>
        new LabeledFrame({
          video: videos[index],
          frameIdx,
          instances: [Instance.fromArray([[frameIdx, frameIdx]], SK)],
        }),
    ),
    videos,
    skeletons: [SK],
  });
}

function product<T>(values: T[], length: number): T[][] {
  return length === 0
    ? [[]]
    : product(values, length - 1).flatMap((rest) =>
        values.map((value) => [...rest, value]),
      );
}

describe("frameVideoIds", () => {
  it("keeps list indices when every video has frames, whatever the numbers", () => {
    // Python's _written_video_id_map stores the numbers here. They are a shuffle of
    // 0..2, so readers would take them as list indices and move every frame.
    const numbers = [2, 0, 1];
    expect(
      videoIdsReadBack(
        [
          [2, 0],
          [0, 1],
          [1, 2],
        ],
        numbers,
      ),
    ).toBe(false);
    expect(frameVideoIds(new Set([0, 1, 2]), numbers)).toEqual([0, 1, 2]);
    expect(frameVideoIds(new Set([0, 1, 2]), [0, 15, 29])).toEqual([0, 1, 2]);
    expect(frameVideoIds(new Set([0, 1]), [0, 0])).toEqual([0, 1]);
  });

  it("keeps list indices when no number is in their way", () => {
    expect(frameVideoIds(new Set([0, 1]), [15, 16, 17])).toEqual([0, 1, 2]);
    expect(frameVideoIds(new Set([0]), [null, null])).toEqual([0, 1]);
    expect(frameVideoIds(new Set([0, 2]), [0, 1, 2])).toEqual([0, 1, 2]);
    expect(frameVideoIds(new Set(), [2, 0, 1])).toEqual([0, 1, 2]);
  });

  it("uses a video's number when its list index would read as another video", () => {
    expect(frameVideoIds(new Set([0, 1]), [2, 0, 1])).toEqual([2, 0, 1]);
    expect(frameVideoIds(new Set([1]), [1, 5])).toEqual([1, 5]);
    // Video 0's list index is free, and a video without a number has only that one.
    expect(frameVideoIds(new Set([0, 1]), [1, 5, 9])).toEqual([0, 5, 9]);
    expect(frameVideoIds(new Set([1, 2]), [1, 5, null])).toEqual([1, 5, 2]);
    // Video 0's number is taken by video 1, but its list index is free.
    expect(frameVideoIds(new Set([0, 1]), [5, 5, 1])).toEqual([0, 5, 1]);
  });

  it("refuses when two videos with frames would share an id", () => {
    // Video 1 has no number, so it can only be id 1, which means video 0 to a reader.
    expect(() => frameVideoIds(new Set([1]), [1, null])).toThrow(
      /frames of video 1 would load on video 0.*Nothing was written/,
    );
    // Only the last video holding a number can be reached through it, and video 0's
    // list index is that same number.
    expect(() => frameVideoIds(new Set([0, 1]), [0, 0, 0])).toThrow(
      /frames of video 0 would load on video 2/,
    );
  });

  it("is the reader's inverse for every small case, and refuses only when nothing works", () => {
    const videosOf = (numbers: VideoNumbers) =>
      numbers.map(
        (number) =>
          ({
            backendMetadata:
              number == null ? {} : { dataset: `video${number}/video` },
          }) as unknown as Video,
      );
    const readsBack = (ids: number[], used: number[], videos: Video[]) => {
      const map = buildVideoIdMap({ video: ids }, videos);
      return ids.every((id, k) => (map.get(id) ?? id) === used[k]);
    };
    let refused = 0;
    for (const n of [1, 2, 3]) {
      for (const numbers of product<number | null>([null, 0, 1, 2, 3], n)) {
        const videos = videosOf(numbers);
        for (let mask = 1; mask < 2 ** n; mask += 1) {
          const used = numbers.map((_, i) => i).filter((i) => mask & (1 << i));
          let ids: number[] | null = null;
          try {
            ids = frameVideoIds(new Set(used), numbers);
          } catch {
            refused += 1;
          }
          if (ids) {
            const written = used.map((i) => (ids as number[])[i]);
            expect(readsBack(written, used, videos)).toBe(true);
            if (readsBack(used, used, videos)) {
              expect(ids).toEqual(numbers.map((_, i) => i));
            }
          } else {
            const candidates = [0, 1, 2, 3, 4, 5];
            expect(
              product(candidates, used.length).some((written) =>
                readsBack(written, used, videos),
              ),
            ).toBe(false);
          }
        }
      }
    }
    expect(refused).toBeGreaterThan(0);
  });
});

describe("parseVideoIdFromDataset", () => {
  it("reads the number of a videoN group, as Python", () => {
    expect(parseVideoIdFromDataset("video15/video")).toBe(15);
    expect(parseVideoIdFromDataset("video0/video")).toBe(0);
    expect(parseVideoIdFromDataset("video3")).toBe(3);
  });

  it("finds no number in anything else", () => {
    // `Number("")` is 0, so "video" used to read as video0.
    for (const dataset of [
      "",
      "video",
      "video/video",
      "videos/1",
      "box",
      "video1a/video",
      "video-1/video",
      "video1.5/video",
      "video1e2/video",
      "video 1/video",
      "myvideo1/video",
    ]) {
      expect(parseVideoIdFromDataset(dataset)).toBeNull();
    }
  });
});

describe("saving keeps frames on their videos", () => {
  it("eager, closed videos: stores the numbers beside the kept videos_json", async () => {
    // The original bug: list indices [0, 1, 1, 0, 0] came back at videos 1 and 2.
    const labels = await readSlp(fixture(HANDBUILT_2X), closed);
    const bytes = await saveSlpToBytes(labels);
    expect(where(await reload(bytes))).toEqual(HANDBUILT_WHERE);
    expect(await stored(bytes)).toEqual({
      frames: [2, 0, 0, 2, 2],
      negative: [[0, 9]],
      datasets: ["video2/video", "video0/video", "video1/video"],
    });
  });

  it("eager, closed videos, from a 3.0 file", async () => {
    const labels = await readSlp(fixture(HANDBUILT_30), closed);
    const bytes = await saveSlpToBytes(labels);
    expect(where(await reload(bytes))).toEqual(HANDBUILT_WHERE);
    expect((await stored(bytes)).frames).toEqual([2, 0, 0, 2, 2]);
  });

  it("lazy store from a 3.0 file, which holds list indices", async () => {
    const lazy = await readSlpLazy(fixture(HANDBUILT_30), closed);
    expect(lazy._lazyDataStore?.framesData.video).toEqual(
      expect.objectContaining({ 0: 0, 1: 1, 2: 1, 3: 0, 4: 0 }),
    );
    const bytes = await saveSlpToBytes(lazy);
    expect(lazy.isLazy).toBe(true);
    expect(where(await reload(bytes))).toEqual(HANDBUILT_WHERE);
    expect((await stored(bytes)).frames).toEqual([2, 0, 0, 2, 2]);
    expect((await stored(bytes)).negative).toEqual([[0, 9]]);
  });

  it("lazy store from a 2.x file copies its ids as they are", async () => {
    const lazy = await readSlpLazy(fixture(HANDBUILT_2X), closed);
    const bytes = await saveSlpToBytes(lazy);
    expect(where(await reload(bytes))).toEqual(HANDBUILT_WHERE);
    expect((await stored(bytes)).frames).toEqual([2, 0, 0, 2, 2]);
    expect((await stored(bytes)).negative).toEqual([[0, 9]]);
  });

  it("lazy store keeps ids that read back, even where list indices would do", async () => {
    // A lazy save copies the tables, so a file with no problem saves byte-identically.
    const lazy = await readSlpLazy(
      fixture("sparse_video_negative.pkg.slp"),
      closed,
    );
    const bytes = await saveSlpToBytes(lazy);
    expect(await stored(bytes)).toEqual({
      frames: [2, 2],
      negative: [[2, 1]],
      datasets: ["video2/video"],
    });
    expect(where(await reload(bytes))).toEqual([
      [0, 0, false],
      [0, 1, true],
    ]);
  });

  it("lazy and eager saves of a file with one video numbered 2 both read back", async () => {
    for (const file of [
      "py_written_30_sparse.slp",
      "py_written_30_sparse_legacy.slp",
    ]) {
      const want = where(await readSlp(fixture(file), closed));
      for (const read of [readSlp, readSlpLazy]) {
        const bytes = await saveSlpToBytes(await read(fixture(file), closed));
        expect(where(await reload(bytes))).toEqual(want);
      }
    }
  });

  it("every video has frames, numbers a shuffle: list indices (Python's trap)", async () => {
    const bytes = await saveSlpToBytes(labelsOn([2, 0, 1], [0, 1, 2]));
    expect((await stored(bytes)).frames).toEqual([0, 1, 2]);
    expect(where(await reload(bytes))).toEqual([
      [0, 0, false],
      [1, 1, false],
      [2, 2, false],
    ]);
  });

  it("every video has frames, sparse numbers: list indices, unchanged", async () => {
    const bytes = await saveSlpToBytes(labelsOn([0, 15, 29], [0, 1, 2, 2]));
    expect((await stored(bytes)).frames).toEqual([0, 1, 2, 2]);
    expect(where(await reload(bytes)).map(([v]) => v)).toEqual([0, 1, 2, 2]);
  });

  it("some videos have frames, no number in the way: list indices, unchanged", async () => {
    for (const numbers of [
      [15, 16, 17],
      [null, null, null],
      [0, 1, 2],
    ]) {
      const bytes = await saveSlpToBytes(labelsOn(numbers, [0, 1]));
      expect((await stored(bytes)).frames).toEqual([0, 1]);
      expect(where(await reload(bytes)).map(([v]) => v)).toEqual([0, 1]);
    }
  });

  it("an external video named like a group isn't a number", async () => {
    const labels = labelsOn([null, null], [0]);
    for (const video of labels.videos) video.backendMetadata.dataset = "video";
    const bytes = await saveSlpToBytes(labels);
    expect((await stored(bytes)).frames).toEqual([0]);
    expect(where(await reload(bytes)).map(([v]) => v)).toEqual([0]);
  });

  it("refuses a file no reader could read back, and writes nothing", async () => {
    await expect(saveSlpToBytes(labelsOn([1, null], [1]))).rejects.toThrow(
      /frames of video 1 would load on video 0/,
    );
    await expect(saveSlpToBytes(labelsOn([0, 0, 0], [0, 1]))).rejects.toThrow(
      /frames of video 0 would load on video 2/,
    );
    // Giving the last video frames makes the ids exactly 0..n-1: list indices again.
    const bytes = await saveSlpToBytes(labelsOn([0, 0, 0], [0, 1, 2]));
    expect(where(await reload(bytes)).map(([v]) => v)).toEqual([0, 1, 2]);
  });

  it("buildLabelTableRows holds list indices unless told the numbers", () => {
    const labels = labelsOn([2, 0, 1], [0, 1]);
    const video = (rows: number[][]) => rows.map((row) => row[1]);
    expect(video(buildLabelTableRows(labels).frames)).toEqual([0, 1]);
    expect(video(buildLabelTableRows(labels, [2, 0, 1]).frames)).toEqual([
      2, 0,
    ]);
  });
});

describe("saving with open videos", () => {
  /** The stored image of each frame, per video (padding off, since widths change). */
  async function images(labels: Labels) {
    return Promise.all(
      labels.videos.map(async (video) => {
        await video.backend?.ensureLoaded?.();
        const out: Record<number, number[]> = {};
        for (const frame of video.embeddedFrameIndices ?? []) {
          const blob = Array.from((await video.getFrameBuffer(frame)) ?? []);
          while (blob.at(-1) === 0) blob.pop();
          out[frame] = blob;
        }
        return out;
      }),
    );
  }

  function withSavedFile<T>(
    bytes: Uint8Array,
    use: (file: string) => Promise<T>,
  ): Promise<T> {
    const dir = mkdtempSync(path.join(os.tmpdir(), "video-ids-"));
    const file = path.join(dir, "saved.pkg.slp");
    writeFileSync(file, bytes);
    return use(file).finally(() => rmSync(dir, { recursive: true }));
  }

  it.each([
    ["eager", readSlp],
    ["lazy", readSlpLazy],
  ] as const)("%s: embedded videos are rewritten under video{i}, so list indices stay right", async (_, read) => {
    const labels = await read(fixturePath(EMBEDDED), { openVideos: true });
    const want = await images(await readSlp(fixturePath(EMBEDDED)));
    expect(Object.keys(want[2])).toEqual(["3"]);

    const bytes = await saveSlpToBytes(labels);
    expect(await stored(bytes)).toEqual({
      frames: [0, 1, 1, 0],
      negative: [[1, 9]],
      datasets: ["video0/video", "video1/video", "video2/video"],
    });
    expect(where(await reload(bytes))).toEqual(EMBEDDED_WHERE);
    await withSavedFile(bytes, async (file) => {
      const again = await readSlp(file, { openVideos: true });
      expect(where(again)).toEqual(EMBEDDED_WHERE);
      expect(await images(again)).toEqual(want);
    });
  });

  it("saving a save changes nothing about where frames are", async () => {
    const first = await saveSlpToBytes(
      await readSlp(fixturePath(EMBEDDED), { openVideos: true }),
    );
    await withSavedFile(first, async (file) => {
      const second = await saveSlpToBytes(
        await readSlp(file, { openVideos: true }),
      );
      expect((await stored(second)).frames).toEqual([0, 1, 1, 0]);
      expect(where(await reload(second))).toEqual(EMBEDDED_WHERE);
    });
  });

  it("the structure-only half of the streaming save stores the same ids", async () => {
    const labels = await readSlp(fixturePath(EMBEDDED), { openVideos: true });
    expect(await stored(await saveSlpStructureToBytes(labels))).toEqual({
      frames: [0, 1, 1, 0],
      negative: [[1, 9]],
      datasets: ["video0/video", "video1/video", "video2/video"],
    });
  });

  it('embed: "source" (eager) writes the source videos, under list indices', async () => {
    const labels = await readSlp(fixturePath(EMBEDDED), { openVideos: true });
    const bytes = await saveSlpToBytes(labels, { embed: "source" });
    const saved = await stored(bytes);
    expect(saved.frames).toEqual([0, 1, 1, 0]);
    expect(saved.datasets).toEqual([null, null, null]);
    const back = await reload(bytes);
    expect(back.videos.map((v) => v.filename)).toEqual([
      "clip_a.mp4",
      "clip_b.mp4",
      "clip_c.mp4",
    ]);
    expect(where(back).map(([v, f]) => [v, f])).toEqual(
      EMBEDDED_WHERE.map(([v, f]) => [v, f]),
    );
  });

  it('embed: "source" (lazy) rewrites ids that no longer mean anything', async () => {
    const lazy = await readSlpLazy(fixturePath(EMBEDDED), { openVideos: true });
    const bytes = await saveSlpToBytes(lazy, { embed: "source" });
    expect(lazy.isLazy).toBe(true);
    expect(await stored(bytes)).toEqual({
      frames: [0, 1, 1, 0],
      negative: [[1, 9]],
      datasets: [null, null, null],
    });
    expect(where(await reload(bytes))).toEqual(EMBEDDED_WHERE);
  });

  it('embed: "source" (lazy) doesn\'t lose the frames of a sparse file', async () => {
    // The only video was video2. Without the number id 2 is past the list, and every
    // frame used to be dropped on reload.
    const lazy = await readSlpLazy(
      fixturePath("sparse_video_negative.pkg.slp"),
      {
        openVideos: true,
      },
    );
    const bytes = await saveSlpToBytes(lazy, { embed: "source" });
    expect(where(await reload(bytes))).toEqual([
      [0, 0, false],
      [0, 1, true],
    ]);
  });
});

describe("saving in place", () => {
  /** A file holding `bytes` opened for writing, closed by `done` into new bytes. */
  async function openForWriting(bytes: Uint8Array) {
    const module = await ready;
    const p = await tmpH5Path();
    module.FS.writeFile(p, bytes);
    const file: any = new H5File(p, "a");
    return {
      file,
      done: (): Uint8Array => {
        file.close();
        const out = module.FS.readFile(p);
        module.FS.unlink(p);
        return out;
      },
    };
  }

  /** A JS-written file (so h5wasm can patch it) that keeps the numbers [2, 0, 1]. */
  const savedWithNumbers = async () =>
    saveSlpToBytes(await readSlp(fixture(EMBEDDED), closed));

  it("goes by the numbers in the file", async () => {
    const original = await savedWithNumbers();
    const labels = await reload(original);
    labels.labeledFrames[0].instances[0].points[0].xy = [100, 200];
    labels.labeledFrames[1].isNegative = true;

    const { file, done } = await openForWriting(original);
    writeLabelTablesInPlace(file, buildLabelTableUpdate(labels));
    const bytes = done();

    const saved = await stored(bytes);
    expect(saved.frames).toEqual([2, 0, 0, 2]);
    expect(saved.negative).toEqual([
      [0, 7],
      [0, 9],
    ]);
    const back = await reload(bytes);
    expect(where(back)).toEqual([
      [0, 5, false],
      [1, 7, true],
      [1, 9, true],
      [0, 11, false],
    ]);
    expect(back.labeledFrames[0].instances[0].points[0].xy).toEqual([100, 200]);
  });

  it("goes by the file, not the videos in memory, after a full re-save renumbered it", async () => {
    // Open videos: this save rewrites them as video0, video1, video2 on disk, while
    // the Video objects in memory still say video2, video0, video1.
    const labels = await readSlp(fixturePath(EMBEDDED), { openVideos: true });
    const resaved = await saveSlpToBytes(labels);
    expect(labels.videos[0].backend?.dataset).toBe("video2/video");

    const { file, done } = await openForWriting(resaved);
    writeLabelTablesInPlace(file, buildLabelTableUpdate(labels));
    const bytes = done();
    expect((await stored(bytes)).frames).toEqual([0, 1, 1, 0]);
    expect(where(await reload(bytes))).toEqual(EMBEDDED_WHERE);
  });

  it("through the streaming writer's worker, too", async () => {
    const original = await savedWithNumbers();
    const labels = await reload(original);
    labels.labeledFrames[0].instances[0].points[0].xy = [100, 200];

    const { file, done } = await openForWriting(original);
    const worker = new Function(
      "self",
      `${H5_WORKER_CODE}
      return {
        setDest(f) { currentDestFile = f; },
        getDestVideosJson,
        updateLabelsInPlace,
      };`,
    )({ postMessage() {} });
    worker.setDest(file);
    const writer: any = Object.create(StreamingH5Writer.prototype);
    writer.send = async (type: string, payload: unknown) =>
      type === "getDestVideosJson"
        ? { success: true, entries: worker.getDestVideosJson() }
        : worker.updateLabelsInPlace(payload);

    const update = buildLabelTableUpdate(labels);
    const result = await writer.updateLabelsInPlace(update);
    expect(result.success).toBe(true);
    const bytes = done();

    expect((await stored(bytes)).frames).toEqual([2, 0, 0, 2]);
    expect((await stored(bytes)).negative).toEqual([[0, 9]]);
    const back = await reload(bytes);
    expect(where(back)).toEqual(EMBEDDED_WHERE);
    expect(back.labeledFrames[0].instances[0].points[0].xy).toEqual([100, 200]);
    // The caller's update is left in list indices, for its own checks.
    expect(update.frames.rows.map((row) => row[1])).toEqual([0, 1, 1, 0]);
  });

  it("refuses before writing anything if nothing would read back", async () => {
    const original = await saveSlpToBytes(labelsOn([1, null], [0]));
    const labels = await reload(original);
    labels.labeledFrames[0].video = labels.videos[1];

    const { file, done } = await openForWriting(original);
    expect(() =>
      writeLabelTablesInPlace(file, buildLabelTableUpdate(labels)),
    ).toThrow(/frames of video 1 would load on video 0/);
    expect((await stored(done())).frames).toEqual([0]);
  });
});

describe("the stream writer", () => {
  const streamSave = async (labels: Labels): Promise<Uint8Array> => {
    const writer = await openSlpWriter({
      skeletons: labels.skeletons,
      videos: labels.videos,
      tracks: labels.tracks,
    });
    writer.appendStore(labels._lazyDataStore!);
    return writer.close();
  };

  it("appendStore: frames of a 2.x store, in windows, end up under the numbers", async () => {
    const lazy = await readSlpLazy(fixture(HANDBUILT_2X), closed);
    const writer = await openSlpWriter({
      skeletons: lazy.skeletons,
      videos: lazy.videos,
      tracks: lazy.tracks,
    });
    writer.appendStore(lazy._lazyDataStore!, { windowFrames: 2 });
    const bytes = writer.close();
    expect((await stored(bytes)).frames).toEqual([2, 0, 0, 2, 2]);
    expect((await stored(bytes)).negative).toEqual([[0, 9]]);
    expect(where(await reload(bytes))).toEqual(HANDBUILT_WHERE);
  });

  it("appendStore: frames of a 3.0 store", async () => {
    const lazy = await readSlpLazy(fixture(HANDBUILT_30), closed);
    expect(where(await reload(await streamSave(lazy)))).toEqual(
      HANDBUILT_WHERE,
    );
  });

  it("appendFrames: frames already built", async () => {
    const labels = labelsOn([2, 0, 1], [0, 1]);
    labels.labeledFrames[1].isNegative = true;
    const writer = await openSlpWriter({
      skeletons: labels.skeletons,
      videos: labels.videos,
    });
    writer.appendFrames(labels.labeledFrames);
    const bytes = writer.close();
    expect((await stored(bytes)).frames).toEqual([2, 0]);
    expect((await stored(bytes)).negative).toEqual([[0, 1]]);
    expect(where(await reload(bytes))).toEqual([
      [0, 0, false],
      [1, 1, true],
    ]);
  });

  it("leaves list indices alone when they read back", async () => {
    const labels = labelsOn([15, 16, 17], [0, 1]);
    const writer = await openSlpWriter({
      skeletons: labels.skeletons,
      videos: labels.videos,
    });
    writer.appendFrames(labels.labeledFrames);
    expect((await stored(writer.close())).frames).toEqual([0, 1]);
  });

  it("merging stores that share a number works while every video has frames", async () => {
    const stores = [
      (await readSlpLazy(fixture("sparse_video_negative.pkg.slp"), closed))
        ._lazyDataStore!,
      (await readSlpLazy(fixture("sparse_video_negative.pkg.slp"), closed))
        ._lazyDataStore!,
    ];
    const bytes = await saveSlpMergedFromStores(stores);
    expect((await stored(bytes)).frames).toEqual([0, 0, 1, 1]);
    expect(where(await reload(bytes))).toEqual([
      [0, 0, false],
      [0, 1, true],
      [1, 0, false],
      [1, 1, true],
    ]);
  });

  it("refuses at close if nothing would read back, and releases the file", async () => {
    const labels = labelsOn([0, 0, 0], [0, 1]);
    const writer = await openSlpWriter({
      skeletons: labels.skeletons,
      videos: labels.videos,
    });
    writer.appendFrames(labels.labeledFrames);
    expect(() => writer.close()).toThrow(
      /frames of video 0 would load on video 2/,
    );
    expect(() => writer.close()).toThrow(/already closed/);
  });
});
