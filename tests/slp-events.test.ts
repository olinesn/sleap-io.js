/**
 * Frame-spanning events (SLP 2.6): the `/event_types` + `/events` layout shared with
 * Python sleap-io (`sleap_io/io/slp.py` `read_events` / `write_events`).
 *
 * Covers the model and the `Labels` catalog rules (ported from Python), every read
 * path (eager, lazy, streaming) and write path (eager, lazy, streaming writer, merged
 * stores, embed: "source"), both directions with Python, and the in-place gate.
 */
import { describe, it, expect, setDefaultTimeout } from "./bun-test";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import {
  readEventSignaturesStreaming,
  readFromStreamingFile,
} from "../src/codecs/slp/read-streaming.js";
import type { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import {
  buildEventSignatures,
  buildExpectedSidecars,
  buildLabelTableUpdate,
  checkInPlaceWritable,
  saveSlpToBytes,
} from "../src/io/main.js";
import {
  openSlpWriter,
  saveSlpMergedFromStores,
} from "../src/codecs/slp/write.js";
import {
  type Event,
  EventType,
  PredictedEvent,
  UserEvent,
  cloneEvent,
} from "../src/model/event.js";
import { Identity } from "../src/model/identity.js";
import { Instance, Track } from "../src/model/instance.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { Labels } from "../src/model/labels.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";
import { ready, File as H5File } from "h5wasm/node";
import { execFileSync } from "node:child_process";
import { readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// The Python-direction test shells out to `uv run --with sleap-io`.
setDefaultTimeout(120_000);

const SK = new Skeleton({ nodes: ["A", "B"], edges: [] });

/** A project with one of everything an event can hold. */
function makeLabels(): Labels {
  const v0 = new Video({ filename: "v0.mp4" });
  const v1 = new Video({ filename: "v1.mp4" });
  const t1 = new Track("mouse1");
  const t2 = new Track("mouse2");
  const animal = new Identity({ name: "animal_A", color: "#112233" });
  const attack = new EventType({
    name: "attack",
    description: "One mouse attacks another.",
    metadata: { color: "#e6194b", social: "true" },
  });
  const rear = new EventType({ name: "rear" });
  const unused = new EventType({ name: "never used" });
  const inst = Instance.fromArray(
    [
      [1, 2],
      [3, 4],
    ],
    SK,
  );
  inst.track = t1;
  return new Labels({
    labeledFrames: [
      new LabeledFrame({ video: v0, frameIdx: 0, instances: [inst] }),
    ],
    videos: [v0, v1],
    skeletons: [SK],
    tracks: [t1, t2],
    identities: [animal],
    eventTypes: [attack, rear, unused],
    events: [
      new UserEvent({
        type: attack,
        video: v0,
        startFrame: 100,
        endFrame: 140,
        subject: t1,
        target: t2,
        name: "first attack",
        source: "manual",
        metadata: { note: "ünïcode ok" },
      }),
      new UserEvent({ type: rear, video: v1, startFrame: 50, subject: animal }),
      new PredictedEvent({
        type: attack,
        video: v0,
        startFrame: 200,
        endFrame: 204,
        subject: t2,
        scores: [0.5, 0.75, 1, 0.75, 0.5],
        score: 0.74,
        source: "classifier",
      }),
      new PredictedEvent({ type: rear, video: v0, startFrame: 7, endFrame: 9 }),
    ],
  });
}

/** What an event says, in plain values, so two files can be compared. */
function describeEvents(labels: Labels): unknown[] {
  const who = (p: Event["subject"]) =>
    p === null ? null : [p instanceof Track ? "track" : "identity", p.name];
  return labels.events.map((e) => ({
    type: e.type.name,
    video: labels.videos.indexOf(e.video as Video),
    start: e.startFrame,
    end: e.endFrame,
    subject: who(e.subject),
    target: who(e.target),
    predicted: e.isPredicted,
    score: e instanceof PredictedEvent ? e.score : null,
    scores:
      e instanceof PredictedEvent && e.scores ? Array.from(e.scores) : null,
    name: e.name,
    source: e.source,
    metadata: e.metadata,
  }));
}

function describeTypes(labels: Labels): unknown[] {
  return labels.eventTypes.map((t) => ({
    name: t.name,
    description: t.description,
    metadata: t.metadata,
  }));
}

/** Back a StreamingH5File with a real h5wasm/node file opened from `bytes`. */
async function makeStreamingFake(
  bytes: Uint8Array,
): Promise<{ fake: StreamingH5File; close: () => void }> {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    /* exists */
  }
  const p = `/tmp/events_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(p, bytes);
  const h5 = new H5File(p, "r");
  const get = (dp: string) => h5.get(dp) as any;
  const fake = {
    keys: () => h5.keys() as string[],
    getKeys: async (dp: string) => {
      const g = get(dp);
      return typeof g?.keys === "function" ? (g.keys() as string[]) : [];
    },
    getAttrs: async (dp: string) => {
      const a = (get(dp)?.attrs as Record<string, any>) ?? {};
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(a)) out[k] = v?.value ?? v;
      return out;
    },
    getDatasetMeta: async (dp: string) => {
      const d = get(dp);
      return {
        shape: (d?.shape as number[]) ?? [],
        dtype: (d?.dtype as string) ?? "",
      };
    },
    getDatasetValue: async (dp: string) => {
      const d = get(dp);
      return {
        value: d?.value,
        shape: (d?.shape as number[]) ?? [],
        dtype: (d?.dtype as string) ?? "",
      };
    },
  } as unknown as StreamingH5File;
  return {
    fake,
    close: () => {
      h5.close();
      try {
        module.FS.unlink(p);
      } catch {
        /* ignore */
      }
    },
  };
}

/** The top-level group names in a saved file, and its format id. */
async function fileLayout(bytes: Uint8Array) {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    /* exists */
  }
  const p = `/tmp/events_layout_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(p, bytes);
  const h5 = new H5File(p, "r");
  try {
    return {
      keys: h5.keys() as string[],
      formatId: Number((h5.get("metadata") as any).attrs.format_id.value),
    };
  } finally {
    h5.close();
    module.FS.unlink(p);
  }
}

const PY_FIXTURE = join(
  fileURLToPath(new URL("./data", import.meta.url)),
  "slp",
  "py_written_26_events.slp",
);

describe("Event model", () => {
  const video = new Video({ filename: "v.mp4" });

  it("covers an inclusive span, and defaults to one frame", () => {
    const bout = new UserEvent({
      type: "attack",
      video,
      startFrame: 100,
      endFrame: 140,
    });
    expect(bout.type).toBeInstanceOf(EventType);
    expect(bout.type.name).toBe("attack");
    expect(bout.nFrames).toBe(41);
    expect(bout.contains(100)).toBe(true);
    expect(bout.contains(140)).toBe(true);
    expect(bout.contains(141)).toBe(false);
    const point = new UserEvent({ type: "light", video, startFrame: 200 });
    expect(point.isInstantaneous).toBe(true);
    expect(point.endFrame).toBe(200);
    expect(point.isPredicted).toBe(false);
    expect(point.isDirected).toBe(false);
  });

  it("refuses an end before the start, and scores of the wrong length", () => {
    expect(
      () => new UserEvent({ type: "a", video, startFrame: 5, endFrame: 4 }),
    ).toThrow(/endFrame >= startFrame/);
    expect(
      () =>
        new PredictedEvent({
          type: "a",
          video,
          startFrame: 0,
          endFrame: 4,
          scores: [1, 2],
        }),
    ).toThrow(/length nFrames/);
  });

  it("overlaps only within one video", () => {
    const other = new Video({ filename: "w.mp4" });
    const a = new UserEvent({ type: "a", video, startFrame: 0, endFrame: 10 });
    const b = new UserEvent({ type: "b", video, startFrame: 8, endFrame: 20 });
    const c = new UserEvent({
      type: "b",
      video: other,
      startFrame: 8,
      endFrame: 20,
    });
    expect(a.overlaps(b)).toBe(true);
    expect(a.overlaps(c)).toBe(false);
  });
});

describe("Labels events", () => {
  it("collects the catalog by name, and what events reference", () => {
    const v = new Video({ filename: "v.mp4" });
    const t = new Track("t");
    const id = new Identity({ name: "i" });
    const a1 = new UserEvent({
      type: "a",
      video: v,
      startFrame: 0,
      subject: t,
    });
    const a2 = new UserEvent({
      type: "a",
      video: v,
      startFrame: 1,
      subject: id,
    });
    const labels = new Labels({ events: [a1, a2] });
    expect(labels.eventTypes.map((e) => e.name)).toEqual(["a"]);
    expect(a2.type).toBe(a1.type);
    expect(labels.videos).toEqual([v]);
    expect(labels.tracks).toEqual([t]);
    expect(labels.identities).toEqual([id]);
  });

  it("answers queries by video, frame, type, subject and kind", () => {
    const labels = makeLabels();
    const [v0, v1] = labels.videos;
    expect(labels.getEvents()).toHaveLength(4);
    expect(labels.getEvents({ video: v1 })).toHaveLength(1);
    expect(labels.getEvents({ type: "attack" })).toHaveLength(2);
    expect(labels.getEvents({ type: labels.eventTypes[1] })).toHaveLength(2);
    expect(labels.getEvents({ predicted: true })).toHaveLength(2);
    expect(labels.getEvents({ subject: labels.tracks[0] })).toHaveLength(1);
    expect(labels.eventsAt(v0, 120).map((e) => e.type.name)).toEqual([
      "attack",
    ]);
    expect(labels.eventsAt(v0, 141)).toHaveLength(0);
  });

  it("copy() keeps events, on the copied objects", () => {
    const labels = makeLabels();
    const copy = labels.copy();
    expect(describeEvents(copy)).toEqual(describeEvents(labels));
    expect(describeTypes(copy)).toEqual(describeTypes(labels));
    expect(copy.events[0]).not.toBe(labels.events[0]);
    expect(copy.events[0].video).toBe(copy.videos[0]);
    expect(copy.events[0].subject).toBe(copy.tracks[0]);
    expect(copy.events[1].subject).toBe(copy.identities[0]);
    expect(copy.events[0].type).toBe(copy.eventTypes[0]);
  });

  it("replaceVideos and removeVideos carry events", () => {
    const labels = makeLabels();
    const [v0, v1] = labels.videos;
    const replacement = new Video({ filename: "other.mp4" });
    labels.replaceVideos({ videoMap: new Map([[v1, replacement]]) });
    expect(labels.events[1].video).toBe(replacement);
    labels.removeVideos([replacement]);
    expect(labels.events).toHaveLength(3);
    expect(labels.events.every((e) => e.video === v0)).toBe(true);
  });

  it("merge() brings events across by name, once", async () => {
    const labels = makeLabels();
    const other = makeLabels();
    const added = new UserEvent({
      type: "groom",
      video: other.videos[0],
      startFrame: 500,
      endFrame: 510,
      subject: other.tracks[1],
    });
    other.events.push(added);
    await labels.merge(other, { video: "basename", track: "name" });
    expect(labels.eventTypes.map((t) => t.name)).toEqual([
      "attack",
      "rear",
      "never used",
      "groom",
    ]);
    expect(labels.events).toHaveLength(5);
    const merged = labels.events[4];
    expect(merged.video).toBe(labels.videos[0]);
    expect(merged.subject).toBe(labels.tracks[1]);
    // Merging the same thing again adds nothing.
    await labels.merge(other, { video: "basename", track: "name" });
    expect(labels.events).toHaveLength(5);
    expect(labels.identities.map((i) => i.name)).toEqual(["animal_A"]);
  });
});

describe("SLP events round trip", () => {
  it("writes nothing and keeps the format id when there are none", async () => {
    const labels = makeLabels();
    labels.events = [];
    labels.eventTypes = [];
    const { keys, formatId } = await fileLayout(await saveSlpToBytes(labels));
    expect(keys.includes("events")).toBe(false);
    expect(keys.includes("event_types")).toBe(false);
    expect(formatId).toBeLessThan(2.6);
  });

  it("a catalog with no events is still written", async () => {
    const labels = makeLabels();
    labels.events = [];
    const bytes = await saveSlpToBytes(labels);
    const { keys, formatId } = await fileLayout(bytes);
    expect(keys.includes("event_types")).toBe(true);
    expect(keys.includes("events")).toBe(false);
    expect(formatId).toBe(2.6);
    const loaded = await readSlp(new Uint8Array(bytes).buffer as ArrayBuffer, {
      openVideos: false,
    });
    expect(describeTypes(loaded)).toEqual(describeTypes(labels));
  });

  it("eager read of an eager write is exact", async () => {
    const labels = makeLabels();
    const bytes = await saveSlpToBytes(labels);
    const { formatId } = await fileLayout(bytes);
    expect(formatId).toBe(2.6);
    const loaded = await readSlp(new Uint8Array(bytes).buffer as ArrayBuffer, {
      openVideos: false,
    });
    expect(describeTypes(loaded)).toEqual(describeTypes(labels));
    expect(describeEvents(loaded)).toEqual(describeEvents(labels));
    // References resolve to the file's own catalogs, not copies.
    expect(loaded.events[0].subject).toBe(loaded.tracks[0]);
    expect(loaded.events[1].subject).toBe(loaded.identities[0]);
    expect(loaded.events[0].type).toBe(loaded.eventTypes[0]);
    expect(loaded.events[0].video).toBe(loaded.videos[0]);
  });

  it("lazy read, then lazy write, keeps them", async () => {
    const labels = makeLabels();
    const path = join(tmpdir(), `events_${Date.now()}.slp`);
    writeFileSync(path, await saveSlpToBytes(labels));
    try {
      const lazy = await readSlpLazy(path, { openVideos: false });
      expect(lazy.isLazy).toBe(true);
      expect(describeEvents(lazy)).toEqual(describeEvents(labels));
      const again = await readSlp(
        new Uint8Array(await saveSlpToBytes(lazy)).buffer as ArrayBuffer,
        { openVideos: false },
      );
      expect(describeTypes(again)).toEqual(describeTypes(labels));
      expect(describeEvents(again)).toEqual(describeEvents(labels));
    } finally {
      unlinkSync(path);
    }
  });

  it("the streaming reader reads the same", async () => {
    const labels = makeLabels();
    const f = await makeStreamingFake(await saveSlpToBytes(labels));
    try {
      const loaded = await readFromStreamingFile(
        f.fake,
        "t.slp",
        "t.slp",
        false,
      );
      expect(describeTypes(loaded)).toEqual(describeTypes(labels));
      expect(describeEvents(loaded)).toEqual(describeEvents(labels));
    } finally {
      f.close();
    }
  });

  it("the streaming writer's header carries them", async () => {
    const labels = makeLabels();
    const writer = await openSlpWriter({
      skeletons: labels.skeletons,
      videos: labels.videos,
      tracks: labels.tracks,
      identities: labels.identities,
      eventTypes: labels.eventTypes,
      events: labels.events,
    });
    const bytes = writer.close();
    const loaded = await readSlp(new Uint8Array(bytes).buffer as ArrayBuffer, {
      openVideos: false,
    });
    expect(describeTypes(loaded)).toEqual(describeTypes(labels));
    expect(describeEvents(loaded)).toEqual(describeEvents(labels));
  });

  it("merged stores carry events passed in the options", async () => {
    const labels = makeLabels();
    const path = join(tmpdir(), `events_merge_${Date.now()}.slp`);
    writeFileSync(path, await saveSlpToBytes(labels));
    try {
      const lazy = await readSlpLazy(path, { openVideos: false });
      const store = lazy._lazyDataStore;
      if (!store) throw new Error("no lazy store");
      const bytes = await saveSlpMergedFromStores([store], {
        identities: lazy.identities,
        eventTypes: lazy.eventTypes,
        events: lazy.events,
      });
      const loaded = await readSlp(
        new Uint8Array(bytes).buffer as ArrayBuffer,
        {
          openVideos: false,
        },
      );
      expect(describeEvents(loaded)).toEqual(describeEvents(labels));
    } finally {
      unlinkSync(path);
    }
  });

  it('embed: "source" moves events onto the restored videos', async () => {
    const labels = makeLabels();
    const source = new Video({ filename: "original.mp4" });
    labels.videos[0].sourceVideo = source;
    const bytes = await saveSlpToBytes(labels, { embed: "source" });
    const loaded = await readSlp(new Uint8Array(bytes).buffer as ArrayBuffer, {
      openVideos: false,
    });
    expect(loaded.videos).toHaveLength(2);
    expect(describeEvents(loaded)).toEqual(describeEvents(labels));
    // The source labels keep their own video objects.
    expect(labels.events[0].video).toBe(labels.videos[0]);
  });

  it("an event appended after construction is saved, with what it references", async () => {
    const labels = makeLabels();
    const late = new Video({ filename: "late.mp4" });
    const t = new Track("late track");
    labels.events.push(
      new UserEvent({
        type: "late type",
        video: late,
        startFrame: 3,
        subject: t,
      }),
    );
    const loaded = await readSlp(
      new Uint8Array(await saveSlpToBytes(labels)).buffer as ArrayBuffer,
      { openVideos: false },
    );
    expect(loaded.eventTypes.map((e) => e.name)).toContain("late type");
    const ev = loaded.events[loaded.events.length - 1];
    expect((ev.video as Video).filename).toBe("late.mp4");
    expect(ev.subject?.name).toBe("late track");
  });

  it("cloneEvent copies a prediction's scores and metadata", () => {
    const labels = makeLabels();
    const pred = labels.events[2] as PredictedEvent;
    const clone = cloneEvent(pred) as PredictedEvent;
    expect(clone).toBeInstanceOf(PredictedEvent);
    expect(clone.scores).not.toBe(pred.scores);
    expect(Array.from(clone.scores as Float32Array)).toEqual(
      Array.from(pred.scores as Float32Array),
    );
    expect(clone.score).toBe(pred.score);
  });
});

describe("Python interop", () => {
  const expectPythonFile = (loaded: Labels) => {
    expect(loaded.videos.map((v) => v.filename)).toEqual([
      "clip0.mp4",
      "clip1.mp4",
    ]);
    expect(describeTypes(loaded)).toEqual([
      {
        name: "attack",
        description: "One mouse attacks another.",
        metadata: { color: "#e6194b", social: "true" },
      },
      { name: "rear", description: "", metadata: {} },
      {
        name: "never used",
        description: "in the catalog, no events",
        metadata: {},
      },
    ]);
    const evs = describeEvents(loaded) as any[];
    expect(evs).toHaveLength(5);
    expect(evs[0]).toEqual({
      type: "attack",
      video: 0,
      start: 100,
      end: 140,
      subject: ["track", "mouse1"],
      target: ["track", "mouse2"],
      predicted: false,
      score: null,
      scores: null,
      name: "first attack",
      source: "manual",
      metadata: { note: "ünïcode ok" },
    });
    expect(evs[1].video).toBe(1);
    expect(evs[1].subject).toEqual(["identity", "animal_A"]);
    expect(evs[1].end).toBe(50);
    expect(evs[2].predicted).toBe(true);
    expect(evs[2].score).toBeCloseTo(0.74, 6);
    expect(evs[2].scores).toHaveLength(5);
    expect(evs[2].scores[2]).toBeCloseTo(0.9, 5);
    expect(evs[3].predicted).toBe(true);
    expect(evs[3].score).toBe(null);
    expect(evs[3].scores).toBe(null);
    expect(evs[3].subject).toBe(null);
    expect(evs[4].predicted).toBe(false);
  };

  it("reads a Python-written file: eager", async () => {
    const loaded = await readSlp(
      new Uint8Array(readFileSync(PY_FIXTURE)).buffer as ArrayBuffer,
      { openVideos: false },
    );
    expectPythonFile(loaded);
  });

  it("reads a Python-written file: lazy", async () => {
    expectPythonFile(await readSlpLazy(PY_FIXTURE, { openVideos: false }));
  });

  it("reads a Python-written file: streaming", async () => {
    const f = await makeStreamingFake(new Uint8Array(readFileSync(PY_FIXTURE)));
    try {
      expectPythonFile(
        await readFromStreamingFile(f.fake, "t.slp", "t.slp", false),
      );
    } finally {
      f.close();
    }
  });

  it("keeps Python's events through a JS load and save", async () => {
    const loaded = await readSlp(
      new Uint8Array(readFileSync(PY_FIXTURE)).buffer as ArrayBuffer,
      { openVideos: false },
    );
    const again = await readSlp(
      new Uint8Array(await saveSlpToBytes(loaded)).buffer as ArrayBuffer,
      { openVideos: false },
    );
    expectPythonFile(again);
  });

  it("Python reads a JS-written file exactly", async () => {
    const labels = makeLabels();
    const path = join(tmpdir(), `events_py_${Date.now()}.slp`);
    writeFileSync(path, await saveSlpToBytes(labels));
    const script = `
import json, sys, sleap_io as sio
labels = sio.load_slp(sys.argv[1], open_videos=False)
def who(p):
    if p is None:
        return None
    return [type(p).__name__.lower(), p.name]
print(json.dumps({
    "types": [{"name": t.name, "description": t.description, "metadata": t.metadata} for t in labels.event_types],
    "events": [{
        "type": e.type.name,
        "video": labels.videos.index(e.video),
        "start": e.start_frame,
        "end": e.end_frame,
        "subject": who(e.subject),
        "target": who(e.target),
        "predicted": e.is_predicted,
        "score": getattr(e, "score", None),
        "scores": None if getattr(e, "scores", None) is None else [float(x) for x in e.scores],
        "name": e.name,
        "source": e.source,
        "metadata": e.metadata,
    } for e in labels.events],
}))
`;
    try {
      const out = execFileSync(
        "uv",
        ["run", "--with", "sleap-io>=0.9.5", "python", "-c", script, path],
        { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] },
      );
      const py = JSON.parse(out.trim().split("\n").pop() as string);
      expect(py.types).toEqual(describeTypes(labels));
      expect(py.events).toEqual(describeEvents(labels));
    } finally {
      unlinkSync(path);
    }
  });
});

describe("in-place save gate", () => {
  /** Tables that pass the table checks, so only the sidecars decide. */
  const tables = (labels: Labels) => {
    const update = buildLabelTableUpdate(labels);
    return { update, onDisk: {} };
  };

  it("passes when the events on disk match", async () => {
    const labels = makeLabels();
    const f = await makeStreamingFake(await saveSlpToBytes(labels));
    try {
      const events = await readEventSignaturesStreaming(f.fake);
      expect(events).toEqual(buildEventSignatures(labels));
      const expected = buildExpectedSidecars(labels);
      const { update, onDisk } = tables(labels);
      const res = checkInPlaceWritable(update, onDisk, {
        onDisk: { ...expected, events },
        expected,
      });
      expect(res.ok).toBe(true);
    } finally {
      f.close();
    }
  });

  it("refuses when the events or the catalog changed", async () => {
    const labels = makeLabels();
    const f = await makeStreamingFake(await saveSlpToBytes(labels));
    try {
      const events = await readEventSignaturesStreaming(f.fake);
      const edits: Array<(l: Labels) => void> = [
        (l) => {
          l.events[0].endFrame += 1;
        },
        (l) => {
          l.events.pop();
        },
        (l) => {
          l.events[1].metadata.note = "edited";
        },
        (l) => {
          (l.events[2] as PredictedEvent).score = 0.5;
        },
        (l) => {
          l.eventTypes[0].metadata.color = "#000000";
        },
        (l) => {
          l.eventTypes.push(new EventType({ name: "new" }));
        },
      ];
      for (const edit of edits) {
        const edited = labels.copy();
        edit(edited);
        const expected = buildExpectedSidecars(edited);
        const { update, onDisk } = tables(edited);
        const res = checkInPlaceWritable(update, onDisk, {
          onDisk: { ...expected, events },
          expected,
        });
        expect(res.ok).toBe(false);
        if (!res.ok) expect(res.reason).toMatch(/events/);
      }
    } finally {
      f.close();
    }
  });

  it("refuses when the caller did not probe a project that has events", () => {
    const labels = makeLabels();
    const expected = buildExpectedSidecars(labels);
    const { update, onDisk } = tables(labels);
    const res = checkInPlaceWritable(update, onDisk, {
      onDisk: { ...expected, events: undefined },
      expected,
    });
    expect(res.ok).toBe(false);
  });

  it("agrees between Python's file and the same events once read", async () => {
    const bytes = new Uint8Array(readFileSync(PY_FIXTURE));
    const loaded = await readSlp(new Uint8Array(bytes).buffer as ArrayBuffer, {
      openVideos: false,
    });
    const f = await makeStreamingFake(bytes);
    try {
      expect(await readEventSignaturesStreaming(f.fake)).toEqual(
        buildEventSignatures(loaded),
      );
    } finally {
      f.close();
    }
  });
});
