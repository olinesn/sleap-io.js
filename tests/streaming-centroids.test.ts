/**
 * Regression: the STREAMING SLP reader (browser/Tauri WebView path) must read
 * the `/centroids` group. It previously read none — the eager/lazy readers had
 * `readCentroids` but `read-streaming.ts` did not — so a loaded project silently
 * dropped its centroid annotations in the browser. Drives the exported
 * `readFromStreamingFile` with a fake `StreamingH5File` (no Worker), same
 * pattern as streaming-lazy.test.ts.
 */
import { describe, it, expect } from "./bun-test";
import { readSlpLazy } from "../src/codecs/slp/read.js";
import { readFromStreamingFile } from "../src/codecs/slp/read-streaming.js";
import type { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import {
  Labels,
  LabeledFrame,
  Instance,
  Skeleton,
  Video,
  UserCentroid,
  PredictedCentroid,
  PredictedCentroid as _PC,
  saveSlpToBytes,
} from "../src/index.js";
import { ready, File as H5File } from "h5wasm/node";
import { writeFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

void _PC;

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

/** A project: pose instance + a UserCentroid linked to it (frame 0), and a
 * standalone PredictedCentroid on frame 1 (centroid-only frame). */
function centroidLabels(): Labels {
  const pose = new Skeleton({ nodes: ["nose", "tail"], name: "rodent" });
  const video = new Video({ filename: "v.mp4" });
  const inst = new Instance({
    skeleton: pose,
    points: [
      { xy: [10, 10], visible: true, complete: true },
      { xy: [30, 30], visible: true, complete: true },
    ],
  });
  const uc = new UserCentroid({ x: 12, y: 12, instance: inst, name: "cm" });
  const pc = new PredictedCentroid({ x: 99, y: 88, score: 0.77 });
  return new Labels({
    skeletons: [pose],
    videos: [video],
    labeledFrames: [
      new LabeledFrame({
        video,
        frameIdx: 0,
        instances: [inst],
        centroids: [uc],
      }),
      new LabeledFrame({ video, frameIdx: 1, instances: [], centroids: [pc] }),
    ],
  });
}

/** Build a fake StreamingH5File over a temp .slp (compound pose tables from a
 * real lazy store, other datasets from h5wasm/node). */
async function makeFakeStreamingFile(bytes: Uint8Array): Promise<{
  fake: StreamingH5File;
  close: () => void;
}> {
  const module = await ready;
  const disk = join(
    tmpdir(),
    `stream_centroids_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`,
  );
  writeFileSync(disk, bytes);
  try {
    module.FS.mkdir("/tmp");
  } catch {
    /* exists */
  }
  const mem = `/tmp/stream_centroids_${Date.now()}.slp`;
  module.FS.writeFile(mem, bytes);
  const h5 = new H5File(mem, "r");

  const store = (await readSlpLazy(disk, { openVideos: false }))._lazyDataStore;
  if (!store) throw new Error("readSlpLazy produced no lazy store");
  const columns: Record<string, Record<string, unknown[]>> = {
    frames: store.framesData,
    instances: store.instancesData,
    points: store.pointsData,
    pred_points: store.predPointsData,
  };
  const get = (dp: string) => h5.get(dp) as unknown as Record<string, unknown>;

  const fake = {
    keys: () => h5.keys() as string[],
    getKeys: async (dp: string) => {
      const item = get(dp) as unknown as { keys?: () => string[] } | null;
      return item?.keys ? item.keys() : [];
    },
    getAttrs: async (dp: string) =>
      unwrapAttrs((get(dp)?.attrs as Record<string, unknown>) ?? {}),
    getDatasetMeta: async (dp: string) => {
      const d = get(dp);
      return {
        shape: (d?.shape as number[]) ?? [],
        dtype: (d?.dtype as string) ?? "",
      };
    },
    getDatasetValue: async (dp: string) => {
      if (dp in columns) return { value: columns[dp], shape: [], dtype: "" };
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
        module.FS.unlink(mem);
      } catch {
        /* ignore */
      }
      try {
        unlinkSync(disk);
      } catch {
        /* ignore */
      }
    },
  };
}

const readStreaming = (fake: StreamingH5File, lazy: boolean) =>
  readFromStreamingFile(
    fake,
    "test.slp",
    "test.slp",
    false,
    undefined,
    false,
    lazy,
  );

describe("streaming reader — centroids", () => {
  it("eager streaming reads /centroids (was silently dropped)", async () => {
    const bytes = await saveSlpToBytes(centroidLabels());
    const { fake, close } = await makeFakeStreamingFile(bytes);
    try {
      const labels = await readStreaming(fake, false);
      const cents = labels.labeledFrames.flatMap((lf) => lf.centroids);
      expect(cents.length).toBe(2);
      const user = cents.find((c) => !c.isPredicted)!;
      const pred = cents.find((c) => c.isPredicted)!;
      expect(user.xy).toEqual([12, 12]);
      expect(user.name).toBe("cm");
      // The centroid→instance link resolves to a live pose instance.
      expect(user.instance).not.toBeNull();
      expect((pred as PredictedCentroid).score).toBeCloseTo(0.77, 5);
    } finally {
      close();
    }
  });

  it("lazy streaming materializes /centroids per frame", async () => {
    const bytes = await saveSlpToBytes(centroidLabels());
    const { fake, close } = await makeFakeStreamingFile(bytes);
    try {
      const labels = await readStreaming(fake, true);
      let n = 0;
      for (let i = 0; i < labels.length; i++) {
        const frame = labels._lazyFrameList!.at(i);
        n += frame?.centroids.length ?? 0;
      }
      expect(n).toBe(2);
    } finally {
      close();
    }
  });
});
