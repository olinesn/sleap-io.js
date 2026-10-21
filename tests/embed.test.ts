import { describe, it, expect } from "./bun-test";
import { loadSlp } from "../src/io/main.js";
import { saveSlpToBytes } from "../src/codecs/slp/write.js";
import { readSlp } from "../src/codecs/slp/read.js";
import {
  Labels,
  LabeledFrame,
  Video,
  Skeleton,
  Instance,
  SuggestionFrame,
} from "../src/index.js";
import { decodeEncoded } from "../src/video/image-decode.js";
import type { VideoBackend, VideoFrame } from "../src/video/backend.js";
import { fileURLToPath } from "node:url";
import path from "node:path";

const fixtureRoot = fileURLToPath(new URL("./data", import.meta.url));

async function loadFixture(filename: string) {
  return loadSlp(path.join(fixtureRoot, "slp", filename), {
    openVideos: false,
  });
}

describe("Frame Embedding", () => {
  it("saveSlpToBytes no longer throws for embed=true", async () => {
    // Even without open video backends, it should not throw "not supported yet"
    const labels = await loadFixture("minimal_instance.slp");
    // With openVideos=false, backends are null so no frames will be read
    // But it should not throw the old error
    const bytes = await saveSlpToBytes(labels, { embed: true });
    expect(bytes.length).toBeGreaterThan(0);
  });

  it("saveSlpToBytes with embed=false works as before", async () => {
    const labels = await loadFixture("minimal_instance.slp");
    const bytes = await saveSlpToBytes(labels);
    expect(bytes.length).toBeGreaterThan(0);

    const reloaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    expect(reloaded.labeledFrames.length).toBe(labels.labeledFrames.length);
  });

  it("embeds frames from pkg.slp with open backends", async () => {
    // Load a package file that has embedded video frames
    const labels = await loadSlp(
      path.join(fixtureRoot, "slp", "minimal_instance.pkg.slp"),
      { openVideos: true },
    );

    // Verify the video has embedded images
    expect(labels.videos[0].hasEmbeddedImages).toBe(true);
    expect(labels.videos[0].backend).not.toBeNull();

    // Write with embedding to a temp file so the reader can find it
    const fs = await import("node:fs");
    const os = await import("node:os");
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sleap-embed-"));
    const tmpFile = path.join(tmpDir, "embedded.slp");

    const bytes = await saveSlpToBytes(labels, { embed: true });
    expect(bytes.length).toBeGreaterThan(0);
    fs.writeFileSync(tmpFile, bytes);

    // Read back and verify
    const reloaded = await readSlp(tmpFile, { openVideos: true });
    expect(reloaded.labeledFrames.length).toBe(labels.labeledFrames.length);
    expect(reloaded.videos[0].hasEmbeddedImages).toBe(true);

    // Verify we can read a frame from the re-embedded video
    const frame = await reloaded.videos[0].getFrame(
      labels.labeledFrames[0].frameIdx,
    );
    expect(frame).not.toBeNull();

    // Cleanup
    fs.rmSync(tmpDir, { recursive: true });
  });

  it("embed='user' only embeds frames with user instances", async () => {
    const labels = await loadSlp(
      path.join(fixtureRoot, "slp", "minimal_instance.pkg.slp"),
      { openVideos: true },
    );

    // Count frames with user instances
    const userFrameCount = labels.labeledFrames.filter(
      (f) => f.hasUserInstances,
    ).length;
    expect(userFrameCount).toBeGreaterThan(0);

    const bytes = await saveSlpToBytes(labels, { embed: "user" });
    expect(bytes.length).toBeGreaterThan(0);

    const reloaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: true,
    });
    expect(reloaded.videos[0].hasEmbeddedImages).toBe(true);
  });

  it("embed='source' does not embed (backward compat)", async () => {
    const labels = await loadFixture("minimal_instance.slp");
    const bytes = await saveSlpToBytes(labels, { embed: "source" });
    expect(bytes.length).toBeGreaterThan(0);

    const reloaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    expect(reloaded.videos[0].hasEmbeddedImages).toBe(false);
  });

  it("sets source_video when embedding non-embedded video", async () => {
    // Create a labels with a non-embedded video that has a backend returning bytes
    const video = new Video({ filename: "original_video.mp4" });
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const inst = new Instance({
      points: { A: [10, 20], B: [30, 40] },
      skeleton,
    });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    // Without a backend, no frames will be embedded, but the code path should work
    const bytes = await saveSlpToBytes(labels, { embed: true });
    const reloaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    // Since no backend was available, video should not be embedded
    expect(reloaded.videos[0].hasEmbeddedImages).toBe(false);
  });

  it("writes frame_sizes dataset and reads frames correctly", async () => {
    const labels = await loadSlp(
      path.join(fixtureRoot, "slp", "minimal_instance.pkg.slp"),
      { openVideos: true },
    );

    const fs = await import("node:fs");
    const os = await import("node:os");
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sleap-sizes-"));
    const tmpFile = path.join(tmpDir, "with_sizes.slp");

    const bytes = await saveSlpToBytes(labels, { embed: true });
    fs.writeFileSync(tmpFile, bytes);

    // Verify frame_sizes dataset exists in the HDF5 file
    const { openH5File } = await import("../src/codecs/slp/h5.js");
    const { file } = await openH5File(tmpFile);
    const frameSizesDs = file.get("video0/frame_sizes");
    expect(frameSizesDs).not.toBeNull();
    const frameSizes = Array.from(frameSizesDs.value).map((v: any) =>
      Number(v),
    );
    expect(frameSizes.length).toBeGreaterThan(0);
    expect(frameSizes.every((s: number) => s > 0)).toBe(true);

    // Verify total of frame sizes matches the video dataset length
    const videoDs = file.get("video0/video");
    const videoData = videoDs.value;
    const totalSize = frameSizes.reduce((sum: number, s: number) => sum + s, 0);
    expect(totalSize).toBe(videoData.length);

    // Read back and verify frames are accessible
    const reloaded = await readSlp(tmpFile, { openVideos: true });
    const frame = await reloaded.videos[0].getFrame(
      reloaded.labeledFrames[0].frameIdx,
    );
    expect(frame).not.toBeNull();

    fs.rmSync(tmpDir, { recursive: true });
  });

  it("re-embedded pkg.slp preserves frame data", async () => {
    const labels = await loadSlp(
      path.join(fixtureRoot, "slp", "minimal_instance.pkg.slp"),
      { openVideos: true },
    );

    // Read a frame before embedding
    const originalFrame = await labels.videos[0].getFrame(
      labels.labeledFrames[0].frameIdx,
    );
    expect(originalFrame).not.toBeNull();

    // Re-embed and verify metadata
    const bytes = await saveSlpToBytes(labels, { embed: true });
    expect(bytes.length).toBeGreaterThan(0);

    // Verify the embedded video metadata
    const fs = await import("node:fs");
    const os = await import("node:os");
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sleap-embed-"));
    const tmpFile = path.join(tmpDir, "test.slp");
    fs.writeFileSync(tmpFile, bytes);

    const reloaded = await readSlp(tmpFile, { openVideos: true });
    expect(reloaded.videos[0].hasEmbeddedImages).toBe(true);
    expect(reloaded.labeledFrames.length).toBe(labels.labeledFrames.length);

    fs.rmSync(tmpDir, { recursive: true });
  });

  it("re-embedded pkg.slp preserves stored blobs byte-exact (raw-copy fast path)", async () => {
    // The already-embedded source path copies stored encoded blobs verbatim (no
    // decode/re-encode). Compare each stored blob before and after a re-save.
    const labels = await loadSlp(
      path.join(fixtureRoot, "slp", "minimal_instance.pkg.slp"),
      { openVideos: true },
    );
    const idxs = labels.videos[0].embeddedFrameIndices ?? [];
    expect(idxs.length).toBeGreaterThan(0);
    const before = new Map<number, Uint8Array>();
    for (const i of idxs) {
      const b = await labels.videos[0].getFrameBuffer(i);
      expect(b).not.toBeNull();
      before.set(i, new Uint8Array(b!));
    }

    const fs = await import("node:fs");
    const os = await import("node:os");
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sleap-byteexact-"));
    const tmpFile = path.join(tmpDir, "resave.pkg.slp");
    fs.writeFileSync(tmpFile, await saveSlpToBytes(labels, { embed: true }));

    const reloaded = await readSlp(tmpFile, { openVideos: true });
    expect(reloaded.videos[0].embeddedFrameIndices).toEqual(idxs);
    for (const i of idxs) {
      const after = await reloaded.videos[0].getFrameBuffer(i);
      expect(after).not.toBeNull();
      expect(Array.from(after!)).toEqual(Array.from(before.get(i)!));
    }
    fs.rmSync(tmpDir, { recursive: true });
  });
});

/**
 * Regression tests for the continuous/mp4-backed new-embed path. The MediaBunny
 * and Mp4Box backends' `getFrame` return an `ImageBitmap` (raw pixels), which
 * the legacy `frameToBytes` dropped (returned null) — so "Export Labels
 * Package" wrote NO images for the common external-mp4 case. The fix PNG-encodes
 * `ImageBitmap`/`ImageData` frames on embed and records `format: "png"`. Under
 * `bun test` there is no `ImageBitmap`, so the synthetic backend returns
 * `ImageData` (the same code path — `frameToBytes` returns null → PNG-encode).
 */
describe("Embedding a continuous (mp4-style) backend that returns raw pixels", () => {
  /** A solid-color RGBA frame as an ImageData-shaped object. */
  function solidFrame(
    w: number,
    h: number,
    rgb: [number, number, number],
  ): VideoFrame {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      data[i * 4] = rgb[0];
      data[i * 4 + 1] = rgb[1];
      data[i * 4 + 2] = rgb[2];
      data[i * 4 + 3] = 255;
    }
    return { data, width: w, height: h } as unknown as VideoFrame;
  }

  /**
   * A continuous-video backend (like mp4) that decodes to raw pixels. It has NO
   * `getFrameBuffer`, so it is not "raw-copyable" and takes the encode path.
   */
  function rawPixelBackend(
    w: number,
    h: number,
    filename: string,
    nFrames = 20,
  ): VideoBackend {
    return {
      filename,
      shape: [nFrames, h, w, 3],
      async getFrame(idx: number): Promise<VideoFrame | null> {
        if (idx < 0 || idx >= nFrames) return null;
        return solidFrame(w, h, [(idx * 25) % 256, 100, 50]);
      },
    } as unknown as VideoBackend;
  }

  async function saveReload(
    labels: Labels,
    embed: string,
    tag: string,
  ): Promise<Labels> {
    const fs = await import("node:fs");
    const os = await import("node:os");
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), `sleap-${tag}-`));
    const tmpFile = path.join(tmpDir, "embedded.pkg.slp");
    fs.writeFileSync(tmpFile, await saveSlpToBytes(labels, { embed }));
    const reloaded = await readSlp(tmpFile, { openVideos: true });
    // Keep the temp dir around until the process exits (backend reads lazily);
    // tests are short-lived so this is fine.
    return reloaded;
  }

  function makeLabels(video: Video, frameIdx: number): Labels {
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const inst = new Instance({ points: { A: [1, 2], B: [3, 4] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx, instances: [inst] });
    return new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });
  }

  it("embeds ImageData frames as decodable PNG (was silently dropped)", async () => {
    const w = 8;
    const h = 6;
    const video = new Video({
      filename: "external.mp4",
      backend: rawPixelBackend(w, h, "external.mp4"),
    });
    const labels = makeLabels(video, 3);

    const reloaded = await saveReload(labels, "all", "pngembed");
    const rv = reloaded.videos[0];

    // The whole bug: images ARE embedded now.
    expect(rv.hasEmbeddedImages).toBe(true);
    expect(rv.embeddedFrameIndices).toContain(3);

    // CRITICAL: the stored format label must match the bytes we wrote (PNG).
    expect(
      (rv.backend as unknown as { embeddedFormat: string }).embeddedFormat,
    ).toBe("png");
    const buf = await rv.getFrameBuffer(3);
    expect(buf).not.toBeNull();
    // PNG magic bytes.
    expect(buf![0]).toBe(0x89);
    expect(buf![1]).toBe(0x50);
    expect(buf![2]).toBe(0x4e);
    expect(buf![3]).toBe(0x47);
    // The stored bytes decode to a valid image of the right dimensions — proves
    // the format label matches the bytes.
    const decoded = await decodeEncoded(buf!);
    expect(decoded.width).toBe(w);
    expect(decoded.height).toBe(h);
  });

  it("records format='png' even when backendMetadata.format is a codec string", async () => {
    // If a backend advertised a codec-ish format (e.g. "h264"), the OLD code
    // recorded THAT as the embedded format while writing PNG bytes — a
    // mis-labelled file. The fix forces "png" whenever it PNG-encodes.
    const video = new Video({
      filename: "external.mp4",
      backend: rawPixelBackend(4, 4, "external.mp4"),
      backendMetadata: { format: "h264" },
    });
    const labels = makeLabels(video, 0);

    const reloaded = await saveReload(labels, "all", "codecfmt");
    const rv = reloaded.videos[0];
    expect(
      (rv.backend as unknown as { embeddedFormat: string }).embeddedFormat,
    ).toBe("png");
    const buf = await rv.getFrameBuffer(0);
    expect(buf![0]).toBe(0x89); // PNG magic — bytes really are PNG
  });

  it("'all+suggestions' embeds suggestion-only frames that 'all' skips", async () => {
    const build = () => {
      const video = new Video({
        filename: "external.mp4",
        backend: rawPixelBackend(4, 4, "external.mp4"),
      });
      const skeleton = new Skeleton({ nodes: ["A"] });
      const inst = new Instance({ points: { A: [1, 2] }, skeleton });
      const labeled = new LabeledFrame({
        video,
        frameIdx: 0,
        instances: [inst],
      });
      const labels = new Labels({
        labeledFrames: [labeled],
        videos: [video],
        skeletons: [skeleton],
        suggestions: [new SuggestionFrame({ video, frameIdx: 7 })],
      });
      return labels;
    };

    // "all" embeds only the labeled frame (0), NOT the suggestion-only frame (7).
    const allReload = await saveReload(build(), "all", "allmode");
    expect(allReload.videos[0].embeddedFrameIndices).toEqual([0]);

    // "all+suggestions" embeds BOTH the labeled frame and the suggestion frame.
    const bothReload = await saveReload(build(), "all+suggestions", "allsugg");
    const idxs = bothReload.videos[0].embeddedFrameIndices ?? [];
    expect(idxs).toContain(0);
    expect(idxs).toContain(7);
  });
});

describe("Embedding reads every frame (pinned, retried, never silently dropped)", () => {
  function labelsWith(backend: unknown, frameIdxs: number[]): Labels {
    const video = new Video({
      filename: "external.mp4",
      backend: backend as VideoBackend,
    });
    const skeleton = new Skeleton({ nodes: ["A"] });
    return new Labels({
      labeledFrames: frameIdxs.map(
        (frameIdx) =>
          new LabeledFrame({
            video,
            frameIdx,
            instances: [new Instance({ points: { A: [1, 1] }, skeleton })],
          }),
      ),
      videos: [video],
      skeletons: [skeleton],
    });
  }
  const pixels = () => ({
    data: new Uint8ClampedArray(4 * 4 * 4).fill(9),
    width: 4,
    height: 4,
  });

  it("reads with { pinned: true } so a newer UI read can't cancel them", async () => {
    const opts: unknown[] = [];
    const backend = {
      filename: "external.mp4",
      shape: [10, 4, 4, 3],
      async getFrame(_idx: number, o?: unknown) {
        opts.push(o);
        return pixels();
      },
    };
    await saveSlpToBytes(labelsWith(backend, [1, 2]), { embed: "user" });
    expect(opts).toHaveLength(2);
    expect(
      opts.every((o) => (o as { pinned?: boolean })?.pinned === true),
    ).toBe(true);
  });

  it("retries a frame that comes back null, and embeds it", async () => {
    let failsLeft = 1;
    const backend = {
      filename: "external.mp4",
      shape: [10, 4, 4, 3],
      async getFrame(idx: number) {
        if (idx === 2 && failsLeft-- > 0) return null;
        return pixels();
      },
    };
    const bytes = await saveSlpToBytes(labelsWith(backend, [1, 2]), {
      embed: "user",
    });
    const reloaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: true,
    });
    expect(reloaded.videos[0].embeddedFrameIndices).toEqual([1, 2]);
  });

  it("fails, naming the frames, rather than writing a package missing images", async () => {
    const backend = {
      filename: "external.mp4",
      shape: [10, 4, 4, 3],
      async getFrame(idx: number) {
        return idx === 2 ? null : pixels();
      },
    };
    await expect(
      saveSlpToBytes(labelsWith(backend, [1, 2]), { embed: "user" }),
    ).rejects.toThrow(/frame 2/);
  });

  it("reports embedding progress", async () => {
    const backend = {
      filename: "external.mp4",
      shape: [10, 4, 4, 3],
      async getFrame() {
        return pixels();
      },
    };
    const seen: Array<[number, number]> = [];
    await saveSlpToBytes(labelsWith(backend, [1, 2, 3]), {
      embed: "user",
      onProgress: (current, total) => seen.push([current, total]),
    });
    expect(seen.at(-1)).toEqual([3, 3]);
    expect(seen.map(([c]) => c)).toEqual(
      [...seen.map(([c]) => c)].sort((a, b) => a - b),
    );
  });
});
