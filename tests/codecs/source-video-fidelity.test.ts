/**
 * `source_video` lineage fidelity across a save/reload (issue #160).
 *
 * A video's `source_video` must round-trip with its recorded shape (and any
 * deeper chain), not filename-only, so `_getEffectiveShape` can resolve an
 * embedded subset's full frame extent after a real reload — the prerequisite
 * for the embedded-subset -> restore-original matching workflow. These tests
 * exercise the shared reconstruction helper, the eager reader against real
 * Python-written `.pkg.slp` fixtures (source in the HDF5 group and/or nested in
 * videos_json), the streaming group reader, and a full JS -> JS round-trip.
 */
import { describe, it, expect } from "../bun-test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import { loadSlp } from "../../src/io/main.js";
import { saveSlpToBytes } from "../../src/codecs/slp/write.js";
import {
  readSlp,
  readSourceVideoGroupJson,
  readVideoCrops,
} from "../../src/codecs/slp/read.js";
import { Video } from "../../src/model/video.js";
import { Instance } from "../../src/model/instance.js";
import { Skeleton } from "../../src/model/skeleton.js";
import { LabeledFrame } from "../../src/model/labeled-frame.js";
import { Labels } from "../../src/model/labels.js";
import {
  _getEffectiveShape,
  shapesCompatible,
} from "../../src/model/matching.js";
import { buildSourceVideoFromDict } from "../../src/codecs/slp/source-video.js";
import { readSourceVideoGroupJsonStreaming } from "../../src/codecs/slp/read-streaming.js";
import { datasetValueToString } from "../../src/codecs/slp/parsers.js";

const fixtureRoot = fileURLToPath(new URL("../data", import.meta.url));
const slp = (name: string) => path.join(fixtureRoot, "slp", name);

describe("buildSourceVideoFromDict", () => {
  it("recovers shape and a multi-level chain, backend left unopened", () => {
    const dict = {
      filename: "subset.pkg.slp",
      backend: { shape: [27, 384, 384, 1], filename: "subset.pkg.slp" },
      source_video: {
        filename: "original.mp4",
        backend: {
          type: "MediaVideo",
          shape: [80, 384, 384, 1],
          filename: "original.mp4",
        },
      },
    };
    const v = buildSourceVideoFromDict(dict);
    expect(v.backend).toBeNull(); // metadata-only; never opens the lineage file
    expect(v.shape).toEqual([27, 384, 384, 1]);
    expect(v.sourceVideo).not.toBeNull();
    expect(v.sourceVideo?.shape).toEqual([80, 384, 384, 1]);
    // The video's OWN shape is the subset extent...
    expect(v.shape).toEqual([27, 384, 384, 1]);
    // ...but _getEffectiveShape walks source-first, resolving to the source's
    // full extent (the point of the source chain for matching).
    expect(_getEffectiveShape(v)).toEqual([80, 384, 384, 1]);
  });

  it("resolves a '.' self-reference to the labels path", () => {
    const v = buildSourceVideoFromDict(
      { backend: { filename: ".", dataset: "video0/video" } },
      "/data/project.pkg.slp",
    );
    expect(v.filename).toBe("/data/project.pkg.slp");
    expect(v.hasEmbeddedImages).toBe(true);
  });

  it("falls back to the top-level filename when backend.filename is absent", () => {
    const v = buildSourceVideoFromDict({ filename: "bare.mp4", backend: {} });
    expect(v.filename).toBe("bare.mp4");
    expect(v.shape).toBeNull();
  });
});

describe("datasetValueToString (scalar |S<n> json dataset decode, #214)", () => {
  const enc = (s: string) => new TextEncoder().encode(s);
  const json = '{"backend":{"type":"ImageVideo"}}';

  it("decodes a plain string (h5wasm scalar |S<n>)", () => {
    expect(datasetValueToString(json)).toBe(json);
  });

  it("decodes a Uint8Array (raw bytes)", () => {
    expect(datasetValueToString(enc(json))).toBe(json);
  });

  it("decodes an ArrayBuffer", () => {
    expect(datasetValueToString(enc(json).buffer)).toBe(json);
  });

  it("decodes a length-1 array of string (jsfive scalar |S<n>)", () => {
    expect(datasetValueToString([json])).toBe(json);
  });

  it("decodes a length-1 array whose element is a Uint8Array", () => {
    // The pre-#214 `String(v[0])` path yielded "[object Uint8Array]" here.
    expect(datasetValueToString([enc(json)])).toBe(json);
  });

  it("decodes a { buffer } typed-array wrapper (streaming transport)", () => {
    expect(datasetValueToString({ buffer: enc(json).buffer })).toBe(json);
  });

  it("unwraps a { value } object", () => {
    expect(datasetValueToString({ value: json })).toBe(json);
  });

  it("trims trailing NUL padding from fixed-length storage", () => {
    expect(datasetValueToString(`${json}\0\0`)).toBe(json);
    expect(datasetValueToString(enc(`${json}\0`))).toBe(json);
  });

  it("returns undefined for empty / absent values (never throws)", () => {
    expect(datasetValueToString(undefined)).toBeUndefined();
    expect(datasetValueToString(null)).toBeUndefined();
    expect(datasetValueToString("")).toBeUndefined();
    expect(datasetValueToString("   ")).toBeUndefined();
    expect(datasetValueToString(new Uint8Array(0))).toBeUndefined();
    expect(datasetValueToString([])).toBeUndefined();
    expect(datasetValueToString(42)).toBeUndefined();
  });
});

describe("readSourceVideoGroupJson (sync reader, scalar dataset + crash-safety, #214)", () => {
  const blob =
    '{"filename":"o.mp4","backend":{"shape":[80,384,384,1],"filename":"o.mp4"}}';
  const enc = (s: string) => new TextEncoder().encode(s);

  // Minimal h5wasm-file stand-in: file.get(group) yields { attrs }, and
  // file.get(group/json) yields a dataset whose `.value` getter returns a
  // representation (or throws). These branches are unreachable through a real
  // fixture because h5wasm always decodes the scalar |S<n> to a plain string.
  const svFile = (opts: {
    datasetValue?: unknown;
    datasetThrows?: boolean;
    hasDataset?: boolean;
    attrJson?: string;
  }) => ({
    get(path: string) {
      if (path === "video0/source_video") {
        return {
          attrs: opts.attrJson !== undefined ? { json: opts.attrJson } : {},
        };
      }
      if (path === "video0/source_video/json") {
        const exists =
          opts.hasDataset ??
          (opts.datasetThrows === true || opts.datasetValue !== undefined);
        if (!exists) return null;
        return {
          get value() {
            if (opts.datasetThrows) throw new Error("h5wasm .value boom");
            return opts.datasetValue;
          },
        };
      }
      return null;
    },
  });

  it("decodes a Uint8Array scalar dataset value", () => {
    const d = readSourceVideoGroupJson(
      svFile({ datasetValue: enc(blob) }),
      "video0",
    );
    expect((d?.backend as any).shape).toEqual([80, 384, 384, 1]);
  });

  it("decodes a length-1 array whose element is a Uint8Array (pre-fix String(v[0]) mis-decode)", () => {
    // Pre-fix `String(v[0])` produced "[object Uint8Array]" -> JSON.parse fails -> null.
    const d = readSourceVideoGroupJson(
      svFile({ datasetValue: [enc(blob)] }),
      "video0",
    );
    expect((d?.backend as any).shape).toEqual([80, 384, 384, 1]);
  });

  it("does not abort the open when .value throws; falls back to the attribute", () => {
    // Pre-fix: the unguarded `.value` access threw out of readVideos, failing
    // the whole file open (the issue's "cannot open" symptom).
    let d: Record<string, unknown> | null = null;
    expect(() => {
      d = readSourceVideoGroupJson(
        svFile({ datasetThrows: true, attrJson: blob }),
        "video0",
      );
    }).not.toThrow();
    expect((d as any)?.backend?.shape).toEqual([80, 384, 384, 1]);
  });

  it("returns null (no throw) when .value throws and no attribute exists", () => {
    let d: Record<string, unknown> | null | undefined;
    expect(() => {
      d = readSourceVideoGroupJson(svFile({ datasetThrows: true }), "video0");
    }).not.toThrow();
    expect(d).toBeNull();
  });
});

describe("readVideoCrops (sync reader crash-safety, #214)", () => {
  const cropsJson = '[{"video":0,"crop":[1,2,3,4],"fill":0}]';
  const cropsFile = (opts: {
    value?: unknown;
    throws?: boolean;
    hasKey?: boolean;
  }) => ({
    keys: () => (opts.hasKey === false ? [] : ["video_crops"]),
    get(path: string) {
      if (path === "video_crops") {
        return {
          get value() {
            if (opts.throws) throw new Error("h5wasm .value boom");
            return opts.value;
          },
        };
      }
      return null;
    },
  });

  it("decodes a Uint8Array /video_crops value", () => {
    const m = readVideoCrops(
      cropsFile({ value: new TextEncoder().encode(cropsJson) }),
    );
    expect(m.get(0)?.crop).toEqual([1, 2, 3, 4]);
  });

  it("does not throw when .value throws (returns empty map)", () => {
    // Pre-fix `let raw = ds.value` was unguarded -> the throw propagated.
    let m: Map<number, unknown> | undefined;
    expect(() => {
      m = readVideoCrops(cropsFile({ throws: true }));
    }).not.toThrow();
    expect(m?.size).toBe(0);
  });
});

describe("readSourceVideoGroupJsonStreaming", () => {
  const blob =
    '{"filename":"o.mp4","backend":{"shape":[80,384,384,1],"filename":"o.mp4"}}';

  /** A file with `video0/source_video`, whose members and attributes are these. */
  const fileWith = (
    members: string[],
    file: {
      getAttrs?: (path: string) => Promise<Record<string, unknown>>;
      getDatasetValue?: () => Promise<{
        value: unknown;
        shape: number[];
        dtype: string;
      }>;
    },
  ) => ({
    keys: () => ["video0"],
    getKeys: async (p: string) =>
      p === "video0" ? ["video", "source_video"] : members,
    getAttrs: file.getAttrs ?? (async () => ({})),
    getDatasetValue:
      file.getDatasetValue ??
      (async () => {
        throw new Error("no dataset");
      }),
  });

  it("reads the json attribute (normal case)", async () => {
    const file = fileWith([], {
      getAttrs: async (p: string) =>
        p === "video0/source_video" ? { json: blob } : {},
    });
    const dict = await readSourceVideoGroupJsonStreaming(file, "video0");
    expect(dict).not.toBeNull();
    expect((dict?.backend as any).shape).toEqual([80, 384, 384, 1]);
  });

  it("prefers the json dataset when present (oversized metadata)", async () => {
    const file = fileWith(["json"], {
      getDatasetValue: async () => ({ value: blob, shape: [1], dtype: "S" }),
    });
    const dict = await readSourceVideoGroupJsonStreaming(file, "video0");
    expect((dict?.backend as any).shape).toEqual([80, 384, 384, 1]);
  });

  it("decodes a scalar |S<n> json dataset returned as a plain string (#214)", async () => {
    // What the h5wasm worker returns for Python's np.bytes_ scalar spill.
    const file = fileWith(["json"], {
      getDatasetValue: async () => ({ value: blob, shape: [], dtype: "A73" }),
    });
    const dict = await readSourceVideoGroupJsonStreaming(file, "video0");
    expect((dict?.backend as any).shape).toEqual([80, 384, 384, 1]);
  });

  it("decodes a scalar json dataset returned as raw bytes (#214)", async () => {
    const file = fileWith(["json"], {
      getDatasetValue: async () => ({
        value: new TextEncoder().encode(blob),
        shape: [],
        dtype: "B",
      }),
    });
    const dict = await readSourceVideoGroupJsonStreaming(file, "video0");
    expect((dict?.backend as any).shape).toEqual([80, 384, 384, 1]);
  });

  it("falls back to the attribute when the dataset read throws (#214)", async () => {
    // A throw from getDatasetValue must not lose the lineage when the attribute
    // is still available (parity with the sync reader's scoped guard).
    const file = fileWith(["json"], {
      getAttrs: async (p: string) =>
        p === "video0/source_video" ? { json: blob } : {},
      getDatasetValue: async () => {
        throw new Error("boom");
      },
    });
    const dict = await readSourceVideoGroupJsonStreaming(file, "video0");
    expect((dict?.backend as any).shape).toEqual([80, 384, 384, 1]);
  });

  it("gives the error when the dataset read throws and there is no attribute", async () => {
    // The lineage is in that dataset, so a null here would lose it at the next save.
    const file = fileWith(["json"], {
      getDatasetValue: async () => {
        throw new Error("boom");
      },
    });
    await expect(
      readSourceVideoGroupJsonStreaming(file, "video0"),
    ).rejects.toThrow("boom");
  });

  it("gives the error when the group's attributes can't be read", async () => {
    const file = fileWith([], {
      getAttrs: async () => {
        throw new Error("boom");
      },
    });
    await expect(
      readSourceVideoGroupJsonStreaming(file, "video0"),
    ).rejects.toThrow("boom");
  });

  it("returns null when the group is absent", async () => {
    const throwing = async () => {
      throw new Error("no such path");
    };
    expect(
      await readSourceVideoGroupJsonStreaming(
        {
          keys: () => ["video0"],
          getKeys: async () => ["video"],
          getAttrs: throwing,
          getDatasetValue: throwing,
        },
        "video0",
      ),
    ).toBeNull();
    expect(
      await readSourceVideoGroupJsonStreaming(
        {
          keys: () => [],
          getKeys: throwing,
          getAttrs: throwing,
          getDatasetValue: throwing,
        },
        "video0",
      ),
    ).toBeNull();
  });
});

describe("eager reader recovers source_video lineage from fixtures", () => {
  it("cropped_format_2_3.pkg.slp: source shape + chain from the nested videos_json dict", async () => {
    const labels = await readSlp(slp("cropped_format_2_3.pkg.slp"), {
      openVideos: false,
    });
    const v = labels.videos[0];
    // The video itself is a 192x256 crop of a 384x384 source.
    expect(v.shape).toEqual([1, 192, 256, 1]);
    expect(v.sourceVideo).not.toBeNull();
    expect(v.sourceVideo?.shape).toEqual([1, 384, 384, 1]);
    // Two-level chain: crop -> embedded subset -> original mp4.
    expect(v.sourceVideo?.sourceVideo).not.toBeNull();
    // Effective shape resolves to the uncropped source, not the cropped facade.
    expect(_getEffectiveShape(v)).toEqual([1, 384, 384, 1]);
  });

  it("minimal_instance.pkg.slp: source recovered from the {group}/source_video HDF5 group", async () => {
    const labels = await readSlp(slp("minimal_instance.pkg.slp"), {
      openVideos: false,
    });
    const v = labels.videos[0];
    // videos_json for this older file carries NO source_video; it lives only in
    // the HDF5 group. Before the fix, sourceVideo was dropped entirely.
    expect(v.sourceVideo).not.toBeNull();
    expect(String(v.sourceVideo?.filename)).toContain(".mp4");
  });

  it("spilled_source_video.pkg.slp: source recovered from a scalar |S<n> json DATASET (#214)", async () => {
    // Python spills oversized (>64 KB) source_video metadata to a scalar
    // fixed-length-string DATASET (np.bytes_) instead of the @json attribute.
    // The whole open must not throw, and the ImageVideo lineage (thousands of
    // frame filenames + backend) must be recovered from that dataset — the
    // attribute is absent in this fixture. See gen_spilled_source_video_fixture.py.
    const labels = await readSlp(slp("spilled_source_video.pkg.slp"), {
      openVideos: false,
    });
    const v = labels.videos[0];
    expect(v.sourceVideo).not.toBeNull();
    const src = v.sourceVideo!;
    const meta = src.backendMetadata as Record<string, unknown>;
    expect(meta.type).toBe("ImageVideo");
    expect(meta.shape).toEqual([2200, 384, 384, 1]);
    // The image-sequence filename list survives in full.
    const fn = src.filename as unknown;
    expect(Array.isArray(fn)).toBe(true);
    expect((fn as string[]).length).toBe(2200);
    expect((fn as string[])[0]).toBe("raw_images_top/frame_00000.jpg");
  });
});

describe("#160 end-to-end: embedded subset -> restore-original matching survives a reload", () => {
  it("recovers the source's full shape and pairs the subset with the restored original", async () => {
    // An embedded pkg whose frames we can re-embed.
    const labels = await loadSlp(slp("minimal_instance.pkg.slp"), {
      openVideos: true,
    });
    const subset = labels.videos[0];

    // Model the workflow: this embedded video is a SUBSET of a larger (80-frame)
    // original. Attach the original as its source (full extent in metadata).
    subset.sourceVideo = new Video({
      filename: "the_original.mp4",
      backendMetadata: { shape: [80, 384, 384, 1] },
    });
    expect(_getEffectiveShape(subset)).toEqual([80, 384, 384, 1]);

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sleap-160-"));
    try {
      const tmpFile = path.join(tmpDir, "subset.pkg.slp");
      fs.writeFileSync(tmpFile, await saveSlpToBytes(labels, { embed: true }));

      const reloaded = await readSlp(tmpFile, { openVideos: false });
      const rv = reloaded.videos[0];
      expect(rv.hasEmbeddedImages).toBe(true);
      expect(rv.sourceVideo).not.toBeNull();
      // The source's full frame extent survives the round-trip...
      expect(rv.sourceVideo?.shape).toEqual([80, 384, 384, 1]);
      expect(_getEffectiveShape(rv)).toEqual([80, 384, 384, 1]);

      // ...so the restored original (80 frames) is judged compatible with the
      // subset instead of being rejected on the subset's own frame count.
      const restoredOriginal = new Video({
        filename: "the_original.mp4",
        backendMetadata: { shape: [80, 384, 384, 1] },
      });
      expect(shapesCompatible(rv, restoredOriginal)).toBe(true);

      // The authoritative HDF5 group (what Python reads for embedded videos) is
      // written, not only the videos_json nested dict.
      const { openH5File } = await import("../../src/codecs/slp/h5.js");
      const { file, close } = await openH5File(tmpFile);
      try {
        expect(file.get("video0/source_video")).not.toBeNull();
      } finally {
        close();
      }
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});

describe("non-embedded videos_json source_video round-trip", () => {
  it("serializes and recovers a source with shape via the nested videos_json dict", async () => {
    const source = new Video({
      filename: "original.mp4",
      backendMetadata: { shape: [80, 384, 384, 1] },
    });
    const video = new Video({ filename: "derived.mp4", sourceVideo: source });
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const inst = new Instance({ points: { A: [1, 2], B: [3, 4] }, skeleton });
    const labels = new Labels({
      labeledFrames: [
        new LabeledFrame({ video, frameIdx: 0, instances: [inst] }),
      ],
      videos: [video],
      skeletons: [skeleton],
    });

    const bytes = await saveSlpToBytes(labels); // no embedding
    const reloaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    const rv = reloaded.videos[0];
    expect(rv.hasEmbeddedImages).toBe(false);
    expect(rv.sourceVideo).not.toBeNull();
    expect(rv.sourceVideo?.shape).toEqual([80, 384, 384, 1]);
    expect(_getEffectiveShape(rv)).toEqual([80, 384, 384, 1]);
  });
});
