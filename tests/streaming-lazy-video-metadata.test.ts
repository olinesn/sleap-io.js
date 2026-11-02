/**
 * lazyVideoMetadata option for the streaming SLP reader
 * (`readSlpStreaming({ lazyVideoMetadata: true })`).
 *
 * When set, embedded videos are built from `videos_json` ALONE and every
 * per-video HDF5 read (dataset lookup, `getAttrs`, `frame_numbers`,
 * `frame_sizes`, `source_video`) is deferred to a self-initializing backend
 * that reads them on the first `getFrame`/`ensureLoaded`. That is what lets a
 * many-video `pkg.slp` open fast over high-latency storage: the ~N serial
 * per-video reads that dominate open time are skipped for videos never viewed.
 *
 * `readSlpStreaming` is Worker-gated, so we drive the exported orchestrator
 * `readFromStreamingFile` with a fake `StreamingH5File` (backed by h5wasm/node)
 * that COUNTS reads per path — so we can assert the per-video reads do NOT
 * happen until the backend's `ensureLoaded()` runs.
 */
import { describe, it, expect } from "./bun-test";
import { readSlpLazy } from "../src/codecs/slp/read.js";
import { readFromStreamingFile } from "../src/codecs/slp/read-streaming.js";
import { saveSlpToBytes } from "../src/codecs/slp/write.js";
import type { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import { StreamingHdf5VideoBackend } from "../src/video/streaming-hdf5-video.js";
import { ready, File as H5File } from "h5wasm/node";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const fixtureRoot = fileURLToPath(new URL("./data", import.meta.url));
const slpPath = (name: string) => path.join(fixtureRoot, "slp", name);

// minimal_instance embeds 1 frame at source index 0 (H=384,W=384,C=1) with NO
// `frames` attr — so the true source count is max(frame_numbers)+1 = 1.
const FIXTURE = "minimal_instance.pkg.slp";
const EXPECTED_SHAPE = [1, 384, 384, 1];

/** Unwrap h5wasm/node attr entries (`{ value }`) to the worker's bare form. */
function unwrapAttrs(attrs: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(attrs ?? {})) {
    out[k] =
      v && typeof v === "object" && "value" in (v as Record<string, unknown>)
        ? (v as { value: unknown }).value
        : v;
  }
  return out;
}

type Counts = {
  attrs: Map<string, number>;
  keys: Map<string, number>;
  values: Map<string, number>;
};

/** A fake StreamingH5File that records per-path read counts. */
async function makeCountingFake(opts?: {
  framesAttrOverride?: number;
  frameNumbersOverride?: number[];
}): Promise<{
  fake: StreamingH5File;
  counts: Counts;
  close: () => void;
}> {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    /* already exists */
  }
  const p = `/tmp/lvm_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(p, readFileSync(slpPath(FIXTURE)));
  const h5 = new H5File(p, "r");

  const store = (await readSlpLazy(slpPath(FIXTURE), { openVideos: false }))
    ._lazyDataStore;
  if (!store) throw new Error("readSlpLazy did not produce a lazy store");
  const columns: Record<string, Record<string, unknown[]>> = {
    frames: store.framesData,
    instances: store.instancesData,
    points: store.pointsData,
    pred_points: store.predPointsData,
  };

  const get = (dp: string) => h5.get(dp) as unknown as Record<string, unknown>;
  const counts: Counts = {
    attrs: new Map(),
    keys: new Map(),
    values: new Map(),
  };
  const bump = (m: Map<string, number>, k: string) =>
    m.set(k, (m.get(k) ?? 0) + 1);

  const fake = {
    keys: () => h5.keys() as string[],
    getKeys: async (dp: string) => {
      bump(counts.keys, dp);
      const g = get(dp);
      if (g && typeof (g as { keys?: unknown }).keys === "function") {
        return (g as unknown as { keys: () => string[] }).keys();
      }
      return h5.keys() as string[];
    },
    getAttrs: async (dp: string) => {
      bump(counts.attrs, dp);
      const a = unwrapAttrs((get(dp)?.attrs as Record<string, unknown>) ?? {});
      if (opts?.framesAttrOverride !== undefined && dp.endsWith("/video")) {
        a.frames = opts.framesAttrOverride;
      }
      return a;
    },
    getDatasetMeta: async (dp: string) => {
      const d = get(dp);
      return {
        shape: (d?.shape as number[]) ?? [],
        dtype: (d?.dtype as string) ?? "",
      };
    },
    getDatasetValue: async (dp: string) => {
      bump(counts.values, dp);
      if (dp in columns) return { value: columns[dp], shape: [], dtype: "" };
      if (opts?.frameNumbersOverride && dp.endsWith("/frame_numbers")) {
        return { value: opts.frameNumbersOverride, shape: [], dtype: "" };
      }
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
    counts,
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

// (file, url, filenameHint, openVideos, onProgress, rawSessions, lazy, lazyVideoMetadata)
function read(
  fake: StreamingH5File,
  { openVideos = false, lazyVideoMetadata = false } = {},
) {
  return readFromStreamingFile(
    fake,
    "test.slp",
    "test.slp",
    openVideos,
    undefined,
    false,
    false,
    lazyVideoMetadata,
  );
}

// Minimal shape of the deferred backend surface we assert against.
type DeferredBackend = {
  isLoaded: boolean;
  ensureLoaded: () => Promise<void>;
  shape?: number[];
};

describe("readSlpStreaming lazyVideoMetadata option", () => {
  it("defers ALL per-video reads until the backend is opened", async () => {
    const { fake, counts, close } = await makeCountingFake();
    try {
      const labels = await read(fake, { lazyVideoMetadata: true });
      expect(labels.videos.length).toBe(1);

      const backend = labels.videos[0].backend as unknown as DeferredBackend;
      expect(backend).toBeTruthy();
      expect(backend.isLoaded).toBe(false);

      // The master list (videos_json) is read, but NOT any per-video metadata.
      expect(counts.values.has("videos_json")).toBe(true);
      expect(counts.attrs.get("video0/video")).toBeUndefined();
      expect(counts.values.has("video0/frame_numbers")).toBe(false);

      // Opening the backend performs the deferred per-video reads and resolves
      // the true shape (H/W/C from attrs; count from max(frame_numbers)+1).
      await backend.ensureLoaded();
      expect(backend.isLoaded).toBe(true);
      expect(counts.attrs.get("video0/video")).toBeGreaterThanOrEqual(1);
      expect(counts.values.has("video0/frame_numbers")).toBe(true);
      expect(backend.shape).toEqual(EXPECTED_SHAPE);
    } finally {
      close();
    }
  });

  it("never shrinks the frame axis below max(frame_numbers)+1 (bogus `frames` attr)", async () => {
    // Real pkg.slp files carry a `frames` attr that badly under-reports (e.g.
    // 18 while frame_numbers span ~47k). Simulate a sparse video whose labels
    // sit at source index 100, with a too-small `frames`=2: the resolved count
    // must be the frame-number span (101), never the bogus attr.
    const { fake, close } = await makeCountingFake({
      framesAttrOverride: 2,
      frameNumbersOverride: [0, 100],
    });
    try {
      const labels = await read(fake, { lazyVideoMetadata: true });
      const backend = labels.videos[0].backend as unknown as DeferredBackend;
      await backend.ensureLoaded();
      expect(backend.shape?.[0]).toBe(101); // max(2, 100+1)
    } finally {
      close();
    }
  });

  describe("when a per-video read fails", () => {
    /** Fail this request of the fake until `heal` is called. */
    function failing(fake: StreamingH5File, method: string, path: string) {
      const real = (fake as any)[method];
      let broken = true;
      (fake as any)[method] = async (p: string, ...rest: unknown[]) => {
        if (broken && p === path) throw new Error("could not read");
        return real(p, ...rest);
      };
      return () => {
        broken = false;
      };
    }

    for (const [method, path] of [
      ["getDatasetValue", "video0/frame_numbers"],
      ["getKeys", "video0"],
      ["getAttrs", "video0/video"],
    ]) {
      it(`${method} ${path}: not loaded, and the next call reads again`, async () => {
        const { fake, close } = await makeCountingFake();
        try {
          const labels = await read(fake, { lazyVideoMetadata: true });
          const backend = labels.videos[0]
            .backend as unknown as DeferredBackend;
          const heal = failing(fake, method, path);
          await expect(backend.ensureLoaded()).rejects.toThrow(
            "could not read",
          );
          expect(backend.isLoaded).toBe(false);

          heal();
          await backend.ensureLoaded();
          expect(backend.isLoaded).toBe(true);
          expect(backend.shape).toEqual(EXPECTED_SHAPE);
        } finally {
          close();
        }
      });
    }

    it("a frame reads as nothing, not as the frame at that index", async () => {
      const { fake, close } = await makeCountingFake({
        frameNumbersOverride: [100, 200],
      });
      try {
        const labels = await read(fake, { lazyVideoMetadata: true });
        const backend = labels.videos[0].backend as any;
        failing(fake, "getDatasetValue", "video0/frame_numbers");
        expect(await backend.getFrameBuffer(0)).toBeNull();
        expect(await backend.getFrame(0)).toBeNull();
      } finally {
        close();
      }
    });

    it("a save fails, instead of leaving the video's images out", async () => {
      const { fake, close } = await makeCountingFake();
      try {
        const labels = await read(fake, { lazyVideoMetadata: true });
        const backend = labels.videos[0].backend as any;
        // What a save does first, to find which frames it has to copy.
        failing(fake, "getDatasetValue", "video0/frame_numbers");
        await expect(saveSlpToBytes(labels)).rejects.toThrow("could not read");
        expect(backend.isLoaded).toBe(false);
      } finally {
        close();
      }
    });

    it("a video with no group in the file isn't a failure", async () => {
      const { fake, close } = await makeCountingFake();
      try {
        (fake as any).keys = () => [];
        const backend = new StreamingHdf5VideoBackend({
          filename: "test.slp",
          h5file: fake,
          datasetPath: "video0/video",
          shape: [5, 10, 20, 1],
          deferred: true,
        });
        await backend.ensureLoaded();
        expect(backend.isLoaded).toBe(true);
        expect(backend.frameNumbers).toEqual([]);
        expect(backend.shape).toEqual([5, 10, 20, 1]);
      } finally {
        close();
      }
    });
  });

  it("eager path (openVideos) still reads per-video metadata at load", async () => {
    const { fake, counts, close } = await makeCountingFake();
    try {
      const labels = await read(fake, { openVideos: true });
      expect(labels.videos.length).toBe(1);
      // Eager opens backends up front — the per-video reads happen during load.
      expect(counts.values.has("video0/frame_numbers")).toBe(true);
    } finally {
      close();
    }
  });
});
