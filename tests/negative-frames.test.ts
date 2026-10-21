import { describe, it, expect } from "./bun-test";
import {
  LabeledFrame,
  Labels,
  Video,
  Skeleton,
  Instance,
  UserCentroid,
} from "../src/index.js";
import { saveSlpToBytes } from "../src/codecs/slp/write.js";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { readFromStreamingFile } from "../src/codecs/slp/read-streaming.js";
import type { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import { ready, File as H5File } from "h5wasm/node";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import path from "node:path";

const slpPath = (name: string) =>
  path.join(fileURLToPath(new URL("./data/slp", import.meta.url)), name);

/**
 * Python-written package whose only embedded video lives under `/video2` (a
 * sparse group id, as after removing a video), with frame 1 recorded in
 * `/negative_frames` as `(2, 1)` — Python's id, not the video's position 0.
 * See `gen_sparse_video_negative_fixture.py`, which checks Python reads it so.
 */
const SPARSE = slpPath("sparse_video_negative.pkg.slp");

const negativeIdxs = (labels: Labels) =>
  labels.labeledFrames.filter((f) => f.isNegative).map((f) => f.frameIdx);

/** Fake `StreamingH5File` over `h5wasm/node`: the Worker-gated streaming reader
 * can't run under bun, so drive its orchestrator directly (as in
 * streaming-lazy.test.ts). Compound tables come from a real lazy store; every
 * other dataset — `/negative_frames` included — straight from h5wasm. */
async function fakeStreamingFile(file: string) {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    // already exists
  }
  const p = `/tmp/neg_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(p, readFileSync(file));
  const h5 = new H5File(p, "r");
  const store = (await readSlpLazy(file, { openVideos: false }))
    ._lazyDataStore!;
  const columns: Record<string, Record<string, unknown[]>> = {
    frames: store.framesData,
    instances: store.instancesData,
    points: store.pointsData,
    pred_points: store.predPointsData,
  };
  const get = (dp: string) => h5.get(dp) as unknown as Record<string, unknown>;
  const unwrap = (attrs: Record<string, unknown>) =>
    Object.fromEntries(
      Object.entries(attrs ?? {}).map(([k, v]) => [
        k,
        v && typeof v === "object" && "value" in (v as object)
          ? (v as { value: unknown }).value
          : v,
      ]),
    );
  const fake = {
    keys: () => h5.keys() as string[],
    getKeys: () => h5.keys() as string[],
    getAttrs: async (dp: string) =>
      unwrap((get(dp)?.attrs as Record<string, unknown>) ?? {}),
    getDatasetMeta: async (dp: string) => {
      const d = get(dp);
      return { shape: (d?.shape as number[]) ?? [], dtype: d?.dtype ?? "" };
    },
    getDatasetValue: async (dp: string) => {
      if (dp in columns) return { value: columns[dp], shape: [], dtype: "" };
      const d = get(dp);
      return {
        value: d?.value,
        shape: (d?.shape as number[]) ?? [],
        dtype: d?.dtype ?? "",
      };
    },
  } as unknown as StreamingH5File;
  const read = (lazy: boolean) =>
    readFromStreamingFile(
      fake,
      "test.slp",
      "test.slp",
      false,
      undefined,
      false,
      lazy,
    );
  return { read, close: () => h5.close() };
}

describe("Negative Frames", () => {
  it("LabeledFrame has isNegative property defaulting to false", () => {
    const video = new Video({ filename: "test.mp4" });
    const frame = new LabeledFrame({ video, frameIdx: 0 });
    expect(frame.isNegative).toBe(false);
  });

  it("LabeledFrame can be created with isNegative=true", () => {
    const video = new Video({ filename: "test.mp4" });
    const frame = new LabeledFrame({ video, frameIdx: 0, isNegative: true });
    expect(frame.isNegative).toBe(true);
  });

  it("Labels.negativeFrames filters correctly", () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const frame1 = new LabeledFrame({ video, frameIdx: 0, isNegative: false });
    const frame2 = new LabeledFrame({ video, frameIdx: 1, isNegative: true });
    const frame3 = new LabeledFrame({ video, frameIdx: 2, isNegative: true });
    const labels = new Labels({
      labeledFrames: [frame1, frame2, frame3],
      videos: [video],
      skeletons: [skeleton],
    });
    expect(labels.negativeFrames.length).toBe(2);
    expect(labels.negativeFrames[0].frameIdx).toBe(1);
    expect(labels.negativeFrames[1].frameIdx).toBe(2);
  });

  it("round-trips negative frames through write and read", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const inst = new Instance({
      points: { A: [10, 20], B: [30, 40] },
      skeleton,
    });
    const frame1 = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const frame2 = new LabeledFrame({ video, frameIdx: 5, isNegative: true });
    const frame3 = new LabeledFrame({
      video,
      frameIdx: 10,
      instances: [inst],
      isNegative: true,
    });
    const labels = new Labels({
      labeledFrames: [frame1, frame2, frame3],
      videos: [video],
      skeletons: [skeleton],
    });

    const bytes = await saveSlpToBytes(labels);
    const loaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });

    expect(loaded.labeledFrames.length).toBe(3);
    expect(loaded.labeledFrames[0].isNegative).toBe(false);
    expect(loaded.labeledFrames[1].isNegative).toBe(true);
    expect(loaded.labeledFrames[2].isNegative).toBe(true);
    expect(loaded.negativeFrames.length).toBe(2);
  });

  it("preserves negative frames through lazy load cycle", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const inst = new Instance({
      points: { A: [10, 20], B: [30, 40] },
      skeleton,
    });
    const frame1 = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const frame2 = new LabeledFrame({ video, frameIdx: 5, isNegative: true });
    const frame3 = new LabeledFrame({
      video,
      frameIdx: 10,
      instances: [inst],
      isNegative: true,
    });
    const labels = new Labels({
      labeledFrames: [frame1, frame2, frame3],
      videos: [video],
      skeletons: [skeleton],
    });

    const bytes = await saveSlpToBytes(labels);
    const loaded = await readSlpLazy(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });

    expect(loaded.isLazy).toBe(true);
    loaded.materialize();
    expect(loaded.isLazy).toBe(false);

    expect(loaded.labeledFrames.length).toBe(3);
    expect(loaded.labeledFrames[0].isNegative).toBe(false);
    expect(loaded.labeledFrames[1].isNegative).toBe(true);
    expect(loaded.labeledFrames[2].isNegative).toBe(true);
    expect(loaded.negativeFrames.length).toBe(2);
  });

  it("numpy() output spans full video length, not just to last labeled frame", () => {
    const video = new Video({ filename: "test.mp4" });
    // Set video shape: 100 frames, 384x384, 1 channel
    video.shape = [100, 384, 384, 1];
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const inst = new Instance({
      points: { A: [10, 20], B: [30, 40] },
      skeleton,
    });
    // Only label frame 5 — video has 100 frames
    const frame = new LabeledFrame({ video, frameIdx: 5, instances: [inst] });
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const arr = labels.numpy();
    // Should span all 100 frames (indices 0..99), not just 0..5
    expect(arr.length).toBe(100);
    // Frame 5 should have data
    expect(arr[5][0][0][0]).toBe(10);
    expect(arr[5][0][0][1]).toBe(20);
    // Other frames should be NaN-filled
    expect(Number.isNaN(arr[0][0][0][0])).toBe(true);
    expect(Number.isNaN(arr[99][0][0][0])).toBe(true);
  });

  it("Labels.userLabeledFrames matches sleap-io's user_labeled_frames", () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const user = new Instance({ points: { A: [1, 2], B: [3, 4] }, skeleton });
    const labels = new Labels({
      labeledFrames: [
        new LabeledFrame({ video, frameIdx: 0, instances: [user] }),
        new LabeledFrame({ video, frameIdx: 1, isNegative: true }),
        new LabeledFrame({
          video,
          frameIdx: 2,
          centroids: [new UserCentroid({ x: 5, y: 5 })],
        }),
        new LabeledFrame({ video, frameIdx: 3 }), // empty: not user-labeled
      ],
      videos: [video],
      skeletons: [skeleton],
    });
    expect(labels.userLabeledFrames.map((f) => f.frameIdx)).toEqual([0, 1, 2]);
  });
});

describe("Negative frames with sparse video ids (Python-written)", () => {
  it("the eager reader maps /negative_frames' video id like /frames'", async () => {
    const labels = await readSlp(SPARSE, { openVideos: false });
    expect(negativeIdxs(labels)).toEqual([1]);
  });

  it("the lazy reader does too", async () => {
    const labels = await readSlpLazy(SPARSE, { openVideos: false });
    labels.materialize();
    expect(negativeIdxs(labels)).toEqual([1]);
  });

  it("the streaming reader reads /negative_frames, eager and lazy", async () => {
    const { read, close } = await fakeStreamingFile(SPARSE);
    try {
      expect(negativeIdxs(await read(false))).toEqual([1]);
      const lazy = await read(true);
      lazy.materialize();
      expect(negativeIdxs(lazy)).toEqual([1]);
    } finally {
      close();
    }
  });

  it("keeps the flag through a save and reload", async () => {
    const labels = await readSlp(SPARSE, { openVideos: false });
    const again = await readSlp(
      new Uint8Array(await saveSlpToBytes(labels)).buffer,
      { openVideos: false },
    );
    expect(negativeIdxs(again)).toEqual([1]);
  });
});

describe("embed: 'user' includes every user-labeled frame (sleap-io parity)", () => {
  /** A continuous (mp4-style) in-memory video, so saving takes the new-embed
   * path that selects frames by embed mode (a re-saved pkg copies its stored
   * frames instead). */
  function memoryVideo(nFrames = 5): Video {
    const backend = {
      filename: "external.mp4",
      shape: [nFrames, 4, 4, 3],
      async getFrame(idx: number) {
        if (idx < 0 || idx >= nFrames) return null;
        return {
          data: new Uint8ClampedArray(4 * 4 * 4).fill(200),
          width: 4,
          height: 4,
        };
      },
    };
    return new Video({ filename: "external.mp4", backend: backend as never });
  }

  async function embeddedFrames(embed: string): Promise<number[] | null> {
    const video = memoryVideo();
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const user = new Instance({ points: { A: [1, 2], B: [3, 4] }, skeleton });
    const labels = new Labels({
      labeledFrames: [
        new LabeledFrame({ video, frameIdx: 0, instances: [user] }),
        new LabeledFrame({ video, frameIdx: 1, isNegative: true }),
        new LabeledFrame({
          video,
          frameIdx: 2,
          centroids: [new UserCentroid({ x: 1, y: 1 })],
        }),
        new LabeledFrame({ video, frameIdx: 3 }), // empty: never embedded by "user"
      ],
      videos: [video],
      skeletons: [skeleton],
    });
    const bytes = await saveSlpToBytes(labels, { embed });
    const reloaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: true,
    });
    return reloaded.videos[0].embeddedFrameIndices;
  }

  it("'user' embeds instance, negative and centroid-only frames, like sleap-io", async () => {
    expect(await embeddedFrames("user")).toEqual([0, 1, 2]);
  });

  it("'user+suggestions' does too", async () => {
    expect(await embeddedFrames("user+suggestions")).toEqual([0, 1, 2]);
  });
});
