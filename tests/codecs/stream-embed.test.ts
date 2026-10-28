/**
 * Streaming NEW-embed: `appendEncodedVideos` (main thread) + the worker's
 * encode/append helpers (`EMBED_ENCODE_WORKER_CODE`), which embed a continuous
 * video's frames into a pkg.slp written straight to disk instead of building
 * the file in memory.
 *
 * The worker helpers are evaluated here directly and driven against a real
 * h5wasm file on Node, through a sink that serializes calls the way the worker
 * queues its messages. The result is reloaded with the normal reader.
 */
import { describe, it, expect } from "../bun-test";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import os from "node:os";
import path from "node:path";
import { readPng } from "../png-read";
import { ready, File as H5File } from "h5wasm/node";
import { readSlp } from "../../src/codecs/slp/read.js";
import { loadSlp } from "../../src/index.js";
import {
  buildSerializableEmbedPlan,
  saveSlpStructureToBytes,
  saveSlpToBytes,
  ExportCancelled,
  type SerializableEncodeEntry,
} from "../../src/codecs/slp/write.js";
import {
  appendEncodedVideos,
  toEncodeFramePayload,
  type EncodeFramePayload,
  type EncodedVideoSink,
} from "../../src/codecs/slp/stream-embed.js";
import { EMBED_ENCODE_WORKER_CODE } from "../../src/codecs/slp/embed-encode-worker.js";
import { H5_WORKER_CODE } from "../../src/codecs/slp/h5-worker.js";
import { subFilterScanlines } from "../../src/video/png-encode.js";
import { GrayscaleVideoBackend } from "../../src/video/grayscale-backend.js";
import { Labels } from "../../src/model/labels.js";
import { Video } from "../../src/model/video.js";
import { Skeleton } from "../../src/model/skeleton.js";
import { LabeledFrame } from "../../src/model/labeled-frame.js";
import { Instance } from "../../src/model/instance.js";
import type { VideoBackend, VideoFrame } from "../../src/video/backend.js";

type Helpers = {
  subFilterScanlinesEnc: (
    pixels: Uint8Array,
    w: number,
    h: number,
    channels: number,
  ) => Uint8Array;
  encodePngEnc: (
    pixels: Uint8Array,
    w: number,
    h: number,
    channels: number,
  ) => Promise<Uint8Array>;
  encodeFrameEnc: (
    frame: EncodeFramePayload,
    collapseToGray: boolean,
  ) => Promise<Uint8Array>;
  beginEncodedVideoEnc: (
    file: unknown,
    entry: SerializableEncodeEntry,
  ) => {
    collapseToGray?: boolean;
  };
  appendEncodedBytesEnc: (
    state: unknown,
    idx: number,
    bytes: Uint8Array,
  ) => void;
  endEncodedVideoEnc: (
    file: unknown,
    state: unknown,
  ) => { framesWritten: number };
};

/** Evaluate the worker helpers with what `H5_WORKER_CODE` supplies them. */
function loadHelpers({
  compressionStream = (globalThis as { CompressionStream?: unknown })
    .CompressionStream,
}: {
  compressionStream?: unknown;
} = {}): Helpers {
  // Same as h5-worker.ts's writeSourceVideoJsonWorker (for small metadata).
  const writeSourceVideoJsonWorker = (
    file: {
      create_group: (p: string) => void;
      get: (p: string) => { create_attribute: (...a: unknown[]) => void };
    },
    group: string,
    dict: unknown,
  ) => {
    file.create_group(`${group}/source_video`);
    file
      .get(`${group}/source_video`)
      .create_attribute("json", JSON.stringify(dict));
  };
  const make = new Function(
    "writeSourceVideoJsonWorker",
    "CompressionStream",
    `${EMBED_ENCODE_WORKER_CODE}\nreturn { encodePngEnc, encodeFrameEnc, subFilterScanlinesEnc, beginEncodedVideoEnc, appendEncodedBytesEnc, endEncodedVideoEnc };`,
  );
  return make(writeSourceVideoJsonWorker, compressionStream) as Helpers;
}

/** A sink over an open h5wasm file, serialized like the worker's queue. */
function fileSink(file: unknown, helpers: Helpers) {
  let state: { collapseToGray?: boolean } | null = null;
  let chain: Promise<unknown> = Promise.resolve();
  const queue = <T>(fn: () => Promise<T> | T): Promise<T> => {
    const next = chain.then(fn);
    chain = next.catch(() => {});
    return next;
  };
  const calls = { collapseToGray: [] as boolean[], frames: [] as number[] };
  const sink: EncodedVideoSink = {
    beginEncodedVideo: (entry, collapseToGray) =>
      queue(() => {
        calls.collapseToGray.push(collapseToGray);
        state = helpers.beginEncodedVideoEnc(file, entry);
        state.collapseToGray = collapseToGray;
      }),
    appendEncodedFrame: (frameIdx, frame) =>
      queue(async () => {
        calls.frames.push(frameIdx);
        const bytes = await helpers.encodeFrameEnc(
          frame,
          !!state!.collapseToGray,
        );
        helpers.appendEncodedBytesEnc(state, frameIdx, bytes);
      }),
    endEncodedVideo: () =>
      queue(() => {
        const r = helpers.endEncodedVideoEnc(file, state);
        state = null;
        return r;
      }),
  };
  return { sink, calls };
}

const W = 6;
const H = 4;
/** RGBA pixel value of frame `idx` at byte `i`: distinct per frame/channel. */
const px = (idx: number, i: number) => (idx * 37 + i * 3) % 256;

/** A continuous-video backend returning RGBA `ImageData`-like frames. */
function rgbaBackend(name: string, nFrames = 20) {
  const reads: number[] = [];
  const backend = {
    filename: name,
    shape: [nFrames, H, W, 3],
    fps: 30,
    async getFrame(idx: number): Promise<VideoFrame | null> {
      reads.push(idx);
      if (idx < 0 || idx >= nFrames) return null;
      const data = new Uint8ClampedArray(W * H * 4);
      for (let i = 0; i < data.length; i++) data[i] = px(idx, i);
      return { data, width: W, height: H } as unknown as VideoFrame;
    },
    close() {},
  } as unknown as VideoBackend;
  return { backend, reads };
}

function labelsOn(videos: Video[], userFrames: number[]): Labels {
  const skeleton = new Skeleton({ nodes: ["A"] });
  return new Labels({
    labeledFrames: userFrames.map(
      (frameIdx) =>
        new LabeledFrame({
          video: videos[0],
          frameIdx,
          instances: [new Instance({ points: { A: [1, 2] }, skeleton })],
        }),
    ),
    videos,
    skeletons: [skeleton],
  });
}

function tmpFile(tag: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `sleap-stream-${tag}-`));
  return path.join(dir, "out.pkg.slp");
}

/** Structure write + streamed frames, the way the app's desktop export does. */
async function streamSave(
  labels: Labels,
  embed: "user" | "user+suggestions" | "all",
  tag: string,
  opts: Parameters<typeof appendEncodedVideos>[3] = {},
) {
  const file = tmpFile(tag);
  fs.writeFileSync(file, await saveSlpStructureToBytes(labels, { embed }));
  const plan = await buildSerializableEmbedPlan(labels, embed, "", {
    allowEncode: true,
  });
  await ready;
  const h5 = new H5File(file, "a");
  const { sink, calls } = fileSink(h5, loadHelpers());
  try {
    await appendEncodedVideos(sink, labels, plan.encodeEntries, opts);
  } finally {
    h5.close();
  }
  return { file, plan, calls };
}

describe("appendEncodedVideos — streaming new-embed to disk", () => {
  it("embeds the same frames as the in-memory save, as RGB PNGs of the source pixels", async () => {
    const { backend } = rgbaBackend("clip.mp4");
    const video = new Video({ filename: "clip.mp4", backend });
    const labels = labelsOn([video], [5, 0, 3]);

    const { file } = await streamSave(labels, "user", "rgb");
    const streamed = await readSlp(file, { openVideos: true });

    const memFile = tmpFile("mem");
    fs.writeFileSync(memFile, await saveSlpToBytes(labels, { embed: "user" }));
    const inMemory = await readSlp(memFile, { openVideos: true });

    expect(streamed.videos[0].embeddedFrameIndices).toEqual([0, 3, 5]);
    expect(streamed.videos[0].embeddedFrameIndices).toEqual(
      inMemory.videos[0].embeddedFrameIndices,
    );
    expect(streamed.labeledFrames.map((f) => f.frameIdx).sort()).toEqual([
      0, 3, 5,
    ]);
    for (const idx of [0, 3, 5]) {
      const png = readPng((await streamed.videos[0].getFrameBuffer(idx))!);
      expect(png.colorType).toBe(2);
      const want = new Uint8Array(W * H * 3);
      for (let p = 0; p < W * H; p++) {
        for (let c = 0; c < 3; c++) want[p * 3 + c] = px(idx, p * 4 + c);
      }
      expect([...png.pixels]).toEqual([...want]);
    }
  });

  it("writes the same datasets, dtypes and attributes as the in-memory save (Python's layout)", async () => {
    // Frames of different PNG sizes, so the streamed width has to grow.
    const a = new Video({
      filename: "a.mp4",
      backend: rgbaBackend("a.mp4").backend,
    });
    const b = new Video({
      filename: "b.mp4",
      backend: rgbaBackend("b.mp4").backend,
    });
    const labels = labelsOn([a, b], [0, 9, 4]);
    const { file } = await streamSave(labels, "user", "layout");
    const memFile = tmpFile("layout-mem");
    fs.writeFileSync(memFile, await saveSlpToBytes(labels, { embed: "user" }));

    await ready;
    const dump = (p: string) => {
      const f = new H5File(p, "r");
      try {
        return ["video0", "video1"].map((g) => {
          const ds = f.get(`${g}/video`) as any;
          const fns = f.get(`${g}/frame_numbers`) as any;
          const rows =
            ds.shape.length === 2
              ? Array.from({ length: ds.shape[0] }, (_, r) => {
                  const row = new Uint8Array(ds.slice([[r, r + 1], []]).buffer);
                  let end = row.length;
                  while (end > 0 && row[end - 1] === 0) end--;
                  // Same pixels; the PNG bytes may differ (the worker's zlib
                  // can't use OpenCV's level/strategy like the main thread's).
                  return [...readPng(row.subarray(0, end)).pixels];
                })
              : [];
          return {
            dtype: ds.dtype,
            rank: ds.shape.length,
            rows,
            attrs: Object.fromEntries(
              Object.entries(ds.attrs).map(([k, v]: [string, any]) => [
                k,
                [v.dtype, String(v.value)],
              ]),
            ),
            frameNumbers: [
              fns.dtype,
              Array.from(fns.value as ArrayLike<bigint | number>, Number),
            ],
            sizes: f.get(`${g}/frame_sizes`) ? "present" : "absent",
            sourceVideo: (f.get(`${g}/source_video`) as any).attrs.json.dtype,
          };
        });
      } finally {
        f.close();
      }
    };
    const streamed = dump(file);
    const inMemory = dump(memFile);
    expect(streamed).toEqual(inMemory);
    expect(streamed[0].dtype).toBe("<b");

    expect(streamed[0].frameNumbers).toEqual(["<q", [0, 4, 9]]);
    expect(streamed[0].sizes).toBe("absent");
    expect(Object.keys(streamed[0].attrs).sort()).toEqual([
      "channel_order",
      "channels",
      "format",
      "fps",
      "frames",
      "height",
      "width",
    ]);
    // An empty embedded video: Python's 1-D empty dataset, float64 frame numbers.
    expect(streamed[1].rank).toBe(1);
    expect(streamed[1].frameNumbers).toEqual(["<d", []]);
  });

  it("reads a grayscale video's inner decoder and collapses to one channel in the worker", async () => {
    const { backend: inner, reads } = rgbaBackend("gray.mp4");
    const video = new Video({
      filename: "gray.mp4",
      backend: GrayscaleVideoBackend.wrap({ inner, grayscale: true }),
    });
    const labels = labelsOn([video], [2, 7]);

    const { file, calls } = await streamSave(labels, "user", "gray");
    expect(calls.collapseToGray).toEqual([true]);
    // Only the inner decoder was read, once per frame.
    expect(reads).toEqual([2, 7]);

    const r = await readSlp(file, { openVideos: true });
    for (const idx of [2, 7]) {
      const png = readPng((await r.videos[0].getFrameBuffer(idx))!);
      expect(png.colorType).toBe(0);
      const want = Uint8Array.from({ length: W * H }, (_, p) => px(idx, p * 4));
      expect([...png.pixels]).toEqual([...want]);
    }
  });

  it("writes an empty embedded video for a video with nothing to embed, as the in-memory save", async () => {
    const a = new Video({
      filename: "a.mp4",
      backend: rgbaBackend("a.mp4").backend,
    });
    const b = new Video({
      filename: "b.mp4",
      backend: rgbaBackend("b.mp4").backend,
    });
    const labels = labelsOn([a, b], [1]);

    const { file, plan } = await streamSave(labels, "user", "empty");
    expect(plan.encodeEntries.map((e) => e.frameNumbers)).toEqual([[1], []]);
    expect(plan.encodeEntries[1].videoAttrs).toEqual({
      frames: 20,
      height: H,
      width: W,
      channels: 3,
      fps: 30,
    });

    const r = await readSlp(file, { openVideos: true });
    expect(r.videos.map((v) => v.hasEmbeddedImages)).toEqual([true, true]);
    expect(r.videos[1].embeddedFrameIndices ?? []).toEqual([]);
  });

  it("reports progress per frame and cancels with ExportCancelled when it returns false", async () => {
    const video = new Video({
      filename: "c.mp4",
      backend: rgbaBackend("c.mp4").backend,
    });
    const labels = labelsOn([video], [0, 1, 2, 3, 4, 5]);
    const seen: Array<[number, number, string | undefined]> = [];
    await streamSave(labels, "user", "progress", {
      onProgress: (c, t, m) => {
        seen.push([c, t, m]);
      },
    });
    expect(seen.map(([c]) => c)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(seen.every(([, t, m]) => t === 6 && m === "Embedding frames")).toBe(
      true,
    );

    await expect(
      streamSave(labels, "user", "cancel", { onProgress: (c) => c < 2 }),
    ).rejects.toBeInstanceOf(ExportCancelled);
  });

  it("fails on a frame that can't be read instead of writing a package without it", async () => {
    const backend = {
      ...rgbaBackend("bad.mp4").backend,
      async getFrame(idx: number) {
        return idx === 4 ? null : rgbaBackend("bad.mp4").backend.getFrame(idx);
      },
    } as unknown as VideoBackend;
    const labels = labelsOn(
      [new Video({ filename: "bad.mp4", backend })],
      [3, 4],
    );
    await expect(streamSave(labels, "user", "unreadable")).rejects.toThrow(
      'Couldn\'t read frame 4 from "bad.mp4"',
    );
  });
});

describe("streaming new-embed plan", () => {
  it("plans without reading a single frame (structure write + plan)", async () => {
    const { backend, reads } = rgbaBackend("lazy.mp4");
    const labels = labelsOn(
      [new Video({ filename: "lazy.mp4", backend })],
      [0, 9],
    );
    await saveSlpStructureToBytes(labels, { embed: "user" });
    const plan = await buildSerializableEmbedPlan(labels, "user", "", {
      allowEncode: true,
    });
    expect(reads).toEqual([]);
    expect(plan.entries).toEqual([]);
    expect(
      plan.encodeEntries.map((e) => [e.videoIndex, e.frameNumbers, e.format]),
    ).toEqual([[0, [0, 9], "png"]]);
  });

  it("leaves frames past the video's end out, as reading would", async () => {
    const { backend } = rgbaBackend("short.mp4", 5);
    const labels = labelsOn(
      [new Video({ filename: "short.mp4", backend })],
      [2, 8],
    );
    const plan = await buildSerializableEmbedPlan(labels, "user", "", {
      allowEncode: true,
    });
    expect(plan.encodeEntries[0].frameNumbers).toEqual([2]);
  });

  it("still refuses a new embed without allowEncode (raw-copy callers)", async () => {
    const labels = labelsOn(
      [new Video({ filename: "x.mp4", backend: rgbaBackend("x.mp4").backend })],
      [0],
    );
    await expect(
      buildSerializableEmbedPlan(labels, "user", ""),
    ).rejects.toThrow("new-embed");
  });
});

describe("worker frame encoding", () => {
  it("inlines into the h5 worker as valid JavaScript, with its message handlers", () => {
    expect(H5_WORKER_CODE).toContain("function encodeFrameEnc(");
    for (const type of [
      "beginEncodedVideo",
      "appendEncodedFrame",
      "endEncodedVideo",
    ]) {
      expect(H5_WORKER_CODE).toContain(`case '${type}':`);
    }
    expect(() => new Function(H5_WORKER_CODE)).not.toThrow();
  });

  it("writes a valid PNG with stored deflate blocks when CompressionStream is missing", async () => {
    const helpers = loadHelpers({ compressionStream: undefined });
    // > 65535 bytes, so the stored stream spans several blocks.
    const w = 300;
    const h = 250;
    const pixels = Uint8Array.from({ length: w * h }, (_, i) => (i * 13) % 256);
    const png = readPng(await helpers.encodePngEnc(pixels, w, h, 1));
    expect(png.colorType).toBe(0);
    expect([...png.pixels]).toEqual([...pixels]);
  });

  it("Sub-filters every row exactly as the main-thread encoder (OpenCV's choice)", () => {
    const helpers = loadHelpers();
    for (const channels of [1, 3]) {
      const w = 37;
      const h = 23;
      const px = Uint8Array.from(
        { length: w * h * channels },
        (_, i) => (i * 2654435761) >>> 24,
      );
      const main = subFilterScanlines(px, w, h, channels);
      expect([...helpers.subFilterScanlinesEnc(px, w, h, channels)]).toEqual([
        ...main,
      ]);
      const filters = Array.from(
        { length: h },
        (_, y) => main[y * (w * channels + 1)],
      );
      expect(new Set(filters)).toEqual(new Set([1]));
    }
  });

  it("stores already-encoded PNG bytes verbatim", async () => {
    const helpers = loadHelpers();
    const png = await helpers.encodePngEnc(
      Uint8Array.from([1, 2, 3, 4]),
      2,
      2,
      1,
    );
    const payload = await toEncodeFramePayload(png as unknown as VideoFrame);
    expect(payload.kind).toBe("png");
    expect([...(await helpers.encodeFrameEnc(payload, false))]).toEqual([
      ...png,
    ]);
  });

  it("keeps a RawFrame's own channel count (only RGBA ImageData drops its alpha)", async () => {
    const helpers = loadHelpers();
    const data = Uint8Array.from({ length: 2 * 2 * 4 }, (_, i) => i * 9);
    const payload = await toEncodeFramePayload({
      data,
      width: 2,
      height: 2,
      channels: 4,
    } as unknown as VideoFrame);
    const png = readPng(await helpers.encodeFrameEnc(payload, false));
    expect(png.channels).toBe(4);
    expect([...png.pixels]).toEqual([...data]);
  });

  it("hands the worker a copy, so the backend's frame stays intact", async () => {
    const data = new Uint8ClampedArray(2 * 2 * 4).fill(7);
    const payload = await toEncodeFramePayload({
      data,
      width: 2,
      height: 2,
    } as unknown as VideoFrame);
    if (payload.kind !== "pixels") throw new Error("expected pixels");
    payload.data.fill(0);
    expect(data[0]).toBe(7);
    expect(payload.rgba).toBe(true);
  });
});

describe("worker raw copy (re-saving a pkg.slp) — Python's layout", () => {
  it("copies stored images into the same datasets and attributes as the in-memory save", async () => {
    const fixture = fileURLToPath(
      new URL("../data/slp/minimal_instance.pkg.slp", import.meta.url),
    );
    const labels = await loadSlp(fixture, { openVideos: true });

    // In memory: a plain re-save preserves (raw-copies) the stored images.
    const memFile = tmpFile("raw-mem");
    fs.writeFileSync(memFile, await saveSlpToBytes(labels));

    // Worker: structure, then the real worker code's appendEmbeddedVideos.
    const out = tmpFile("raw-worker");
    fs.writeFileSync(out, await saveSlpStructureToBytes(labels));
    const plan = await buildSerializableEmbedPlan(labels, false, fixture);
    expect(plan.entries.length).toBe(1);
    await ready;
    const src = new H5File(fixture, "r");
    const dest = new H5File(out, "a");
    try {
      const run = new Function(
        "self",
        `${H5_WORKER_CODE}\nreturn function(s, d, e) { currentSourceFile = s; currentDestFile = d; return appendEmbeddedVideos(e); };`,
      )({ postMessage() {} });
      const res = run(src, dest, plan.entries);
      expect(res.perVideo).toEqual([
        { videoIndex: 0, framesWritten: plan.entries[0].frameNumbers.length },
      ]);
    } finally {
      dest.close();
      src.close();
    }

    const dump = (p: string) => {
      const f = new H5File(p, "r");
      try {
        const ds = f.get("video0/video") as any;
        const fns = f.get("video0/frame_numbers") as any;
        return {
          dtype: ds.dtype,
          shape: ds.shape,
          rows: Array.from({ length: ds.shape[0] }, (_, r) => [
            ...new Uint8Array(ds.slice([[r, r + 1], []]).buffer),
          ]),
          attrs: Object.fromEntries(
            Object.entries(ds.attrs).map(([k, v]: [string, any]) => [
              k,
              [v.dtype, String(v.value)],
            ]),
          ),
          frameNumbers: [
            fns.dtype,
            Array.from(fns.value as ArrayLike<bigint | number>, Number),
          ],
          sizes: f.get("video0/frame_sizes") ? "present" : "absent",
        };
      } finally {
        f.close();
      }
    };
    const worker = dump(out);
    expect(worker).toEqual(dump(memFile));
    expect(worker.dtype).toBe("<b");
    expect(worker.shape.length).toBe(2);
    expect(worker.sizes).toBe("absent");

    const reloaded = await readSlp(out, { openVideos: true });
    const frame = await reloaded.videos[0].getFrameBuffer(
      reloaded.videos[0].embeddedFrameIndices![0],
    );
    expect([...frame!]).toEqual([
      ...(await labels.videos[0].getFrameBuffer(
        labels.videos[0].embeddedFrameIndices![0],
      ))!,
    ]);
  });
});

describe("appendEncodedVideos frameReader (frames from outside the backend, e.g. ffmpeg)", () => {
  it("stores the reader's PNG bytes as-is, never decodes the backend, and closes the reader", async () => {
    const { backend, reads } = rgbaBackend("ff.mp4");
    const labels = labelsOn(
      [new Video({ filename: "ff.mp4", backend })],
      [2, 5],
    );
    const helpers = loadHelpers();
    const pngFor = async (f: number) =>
      helpers.encodePngEnc(
        Uint8Array.from({ length: 4 }, () => f),
        2,
        2,
        1,
      );
    const asked: number[] = [];
    let announced: readonly number[] = [];
    let closed = 0;
    const file = tmpFile("reader");
    fs.writeFileSync(
      file,
      await saveSlpStructureToBytes(labels, { embed: "user" }),
    );
    const plan = await buildSerializableEmbedPlan(labels, "user", "", {
      allowEncode: true,
    });
    await ready;
    const h5 = new H5File(file, "a");
    const { sink } = fileSink(h5, helpers);
    try {
      await appendEncodedVideos(sink, labels, plan.encodeEntries, {
        frameReader: async (_video, frameNumbers) => {
          announced = frameNumbers;
          return {
            read: async (f) => {
              asked.push(f);
              return (await pngFor(f)) as never;
            },
            close: () => void closed++,
          };
        },
      });
    } finally {
      h5.close();
    }
    expect(announced).toEqual([2, 5]);
    expect(asked).toEqual([2, 5]);
    expect(reads).toEqual([]);
    expect(closed).toBe(1);
    const r = await readSlp(file, { openVideos: true });
    expect([...(await r.videos[0].getFrameBuffer(5))!]).toEqual([
      ...(await pngFor(5)),
    ]);
  });

  it("closes the reader when the export is cancelled or a frame is missing", async () => {
    const labels = labelsOn(
      [new Video({ filename: "x.mp4", backend: rgbaBackend("x.mp4").backend })],
      [0, 1, 2],
    );
    const plan = await buildSerializableEmbedPlan(labels, "user", "", {
      allowEncode: true,
    });
    const noopSink = {
      beginEncodedVideo: async () => {},
      appendEncodedFrame: async () => {},
      endEncodedVideo: async () => ({ framesWritten: 3 }),
    };
    let closed = 0;
    const reader = (missing: number) => async () => ({
      read: async (f: number) =>
        f === missing
          ? null
          : (Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0]) as never),
      close: () => void closed++,
    });
    await expect(
      appendEncodedVideos(noopSink, labels, plan.encodeEntries, {
        frameReader: reader(-1),
        onProgress: () => false,
      }),
    ).rejects.toBeInstanceOf(ExportCancelled);
    await expect(
      appendEncodedVideos(noopSink, labels, plan.encodeEntries, {
        frameReader: reader(1),
      }),
    ).rejects.toThrow("Couldn't read frame 1");
    expect(closed).toBe(2);
  });
});
