/**
 * SLP format 3.0, where every table is a group of columns, read through every path.
 *
 * The main check is that one project saved in both layouts reads the same. The
 * `py_written_30_*` pairs come from Python's own writer, with `legacy_format=True` for
 * the 2.x file (`tests/data/slp/gen_format_30_twins_fixture.py`), which also shows that
 * re-saving that way works for anything that can't read 3.0. The `handbuilt_30*` files
 * (`gen_format_30_handbuilt_fixture.py`) follow Python's layout by hand, to hold what its
 * writer doesn't produce: embedded videos numbered out of list order, unknown columns.
 *
 * `readSlpStreaming` needs a browser Worker, so the streaming tests run the real worker
 * code and the real `StreamingH5File` over h5wasm/node, with only the message
 * channel replaced (see `worker-file.ts`). The same tests hold for the 2.x layouts
 * where the streaming reader and the eager one must agree on any format.
 */
import { afterEach, describe, expect, it, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { ready, File as H5File } from "h5wasm/node";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import {
  readFromStreamingFile,
  readPixelDataStreaming,
} from "../src/codecs/slp/read-streaming.js";
import type { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import { H5_WORKER_CODE } from "../src/codecs/slp/h5-worker.js";
import { loadSlp } from "../src/io/main.js";
import { loadSlpMetadata, validateSlpBuffer } from "../src/lite.js";
import { UnsupportedSlpFormatError, type Labels } from "../src/index.js";
import {
  checkInPlaceWritable,
  onDiskTableFromMeta,
  saveSlpMergedFromStores,
  saveSlpToBytes,
  writeLabelTablesInPlace,
} from "../src/codecs/slp/write.js";
import { checkSlpFormat } from "../src/codecs/slp/parsers.js";
import {
  checkDatasetFilters,
  numberColumn,
} from "../src/codecs/slp/columnar.js";
import {
  buildLabelImages,
  type LabelImagePixels,
  type Rows,
} from "../src/codecs/slp/annotations.js";
import { encodeWkb } from "../src/model/roi.js";
import { readRecording, tmpPath, workerFile } from "./worker-file";

const fixturePath = (name: string): string =>
  path.join(fileURLToPath(new URL("./data/slp", import.meta.url)), name);
const fixture = (name: string): ArrayBuffer =>
  new Uint8Array(readFileSync(fixturePath(name))).buffer as ArrayBuffer;

const COLUMNAR = "handbuilt_30.slp";
const LEGACY = "handbuilt_30_2x.slp";
const PYTHON = "py_written_30_columnar.slp";
const PYTHON_TWIN = "py_written_30_twin.slp";

/** One project saved in both layouts: the 3.0 file, then the 2.x file. */
const TWINS: Array<[string, string]> = [
  [PYTHON_TWIN, "py_written_30_twin_legacy.slp"],
  ["py_written_30_sparse.slp", "py_written_30_sparse_legacy.slp"],
  ["py_written_30_embedded.pkg.slp", "py_written_30_embedded_legacy.pkg.slp"],
  [COLUMNAR, LEGACY],
];

afterEach(() => {
  vi.restoreAllMocks();
});

/** Silence (and record) the warnings about what a save won't keep. */
const quietWarnings = () =>
  vi.spyOn(console, "warn").mockImplementation(() => {});

async function readStreaming(
  bytes: ArrayBuffer,
  lazy = false,
): Promise<Labels> {
  const { file, close } = await workerFile(bytes);
  try {
    return await readFromStreamingFile(
      file,
      "https://x.test/mice.slp?a=1",
      undefined,
      false,
      undefined,
      false,
      lazy,
    );
  } finally {
    close();
  }
}

/** Every way to read a whole project, as (bytes) -> Labels. */
const READERS: Array<[string, (bytes: ArrayBuffer) => Promise<Labels>]> = [
  ["readSlp", (b) => readSlp(b, { openVideos: false })],
  [
    "readSlpLazy",
    async (b) => {
      const labels = await readSlpLazy(b, { openVideos: false });
      labels.materialize();
      return labels;
    },
  ],
  ["loadSlp", (b) => loadSlp(b, { openVideos: false })],
  ["readFromStreamingFile", (b) => readStreaming(b)],
  [
    "readFromStreamingFile lazy",
    async (b) => {
      const labels = await readStreaming(b, true);
      labels.materialize();
      return labels;
    },
  ],
];

/** 2.x stores these columns as float32, so a value may differ past that precision. */
const f4 = (v: number | null | undefined): number | null =>
  v === null || v === undefined ? null : Math.fround(v);

/** Everything the frame tables hold, in a form `toEqual` can compare. */
function poseSnapshot(labels: Labels) {
  return {
    frames: Array.from(labels.labeledFrames).map((frame) => ({
      video: labels.videos.indexOf(frame.video),
      frameIdx: frame.frameIdx,
      isNegative: !!frame.isNegative,
      instances: frame.instances.map((inst: any) => ({
        type: inst.constructor.name,
        score: f4(inst.score),
        trackingScore: f4(inst.trackingScore),
        track: inst.track?.name ?? null,
        fromPredicted: inst.fromPredicted
          ? frame.instances.indexOf(inst.fromPredicted)
          : null,
        identity: inst.identity?.name ?? null,
        identityScore: f4(inst.identityScore),
        category: inst.category ?? null,
        categoryScore: f4(inst.categoryScore),
        identityEmbedding: inst.identityEmbedding
          ? Array.from(inst.identityEmbedding.vector)
          : null,
        categoryEmbedding: inst.categoryEmbedding
          ? Array.from(inst.categoryEmbedding.vector)
          : null,
        points: inst.points.map((p: any) => [
          p.xy[0],
          p.xy[1],
          p.visible,
          p.complete,
          p.score ?? null,
        ]),
      })),
    })),
    videos: labels.videos.map(
      (v) => (v.backendMetadata as { dataset?: string }).dataset,
    ),
    tracks: labels.tracks.map((t) => t.name),
    suggestions: labels.suggestions.map((s) => [
      labels.videos.indexOf(s.video),
      s.frameIdx,
    ]),
    identities: labels.identities.map((i) => [i.name, i.color ?? null]),
    provenance: labels.provenance,
    skeleton: labels.skeletons.map((s) => s.nodeNames),
    sessions: labels.sessions.map((session) => ({
      cameras: session.cameraGroup.cameras.map((c) => c.name),
      videos: labels.videos.map((v) => session.getCamera(v)?.name ?? null),
      groups: [...session.frameGroups.values()].map((fg: any) => ({
        frameIdx: fg.frameIdx,
        metadata: fg.metadata,
        instanceGroups: fg.instanceGroups.map((ig: any) => ({
          score: ig.score ?? null,
          identity: ig.identity?.name ?? null,
          metadata: ig.metadata,
          members: [...(ig._instanceRefsByCamera ?? new Map()).entries()].map(
            ([camera, ref]: any) => [camera.name, ref],
          ),
          instance3d: ig.instance3d
            ? {
                type: ig.instance3d.constructor.name,
                score: ig.instance3d.score ?? null,
                points: ig.instance3d.points,
                pointScores: ig.instance3d.pointScores ?? null,
              }
            : null,
        })),
      })),
    })),
  };
}

/** Everything the annotation tables hold, and where each annotation ended up. */
function annotationSnapshot(labels: Labels) {
  const rowOf = new Map<unknown, number>(
    labels.labeledFrames
      .flatMap((f) => f.instances)
      .map((inst, i) => [inst, i]),
  );
  const link = (ann: any) =>
    ann.instance
      ? (rowOf.get(ann.instance) ?? "other")
      : (ann._instanceIdx ?? -1);
  const vector = (embedding: any) =>
    embedding ? Array.from(embedding.vector) : null;
  const perDetection = (d: any) => [
    d.identity?.name ?? null,
    f4(d.identityScore),
    vector(d.identityEmbedding),
    f4(d.categoryScore),
    vector(d.categoryEmbedding),
  ];
  return {
    placement: labels.labeledFrames.map((f) => [
      labels.videos.indexOf(f.video),
      f.frameIdx,
      f.rois.length,
      f.bboxes.length,
      f.masks.length,
      f.centroids.length,
      f.labelImages.length,
    ]),
    rois: labels.rois.map((r: any) => [
      r.constructor.name,
      r.name,
      r.category,
      r.source,
      f4(r.score),
      f4(r.trackingScore),
      r.track?.name ?? null,
      r.video ? labels.videos.indexOf(r.video) : null,
      r.geometry,
      perDetection(r),
      link(r),
    ]),
    bboxes: labels.bboxes.map((b: any) => [
      b.constructor.name,
      b.x1,
      b.y1,
      b.x2,
      b.y2,
      b.angle,
      b.name,
      b.category,
      b.source,
      f4(b.score),
      f4(b.trackingScore),
      b.track?.name ?? null,
      perDetection(b),
      link(b),
    ]),
    masks: labels.masks.map((m: any) => [
      m.constructor.name,
      m.name,
      m.category,
      m.source,
      f4(m.score),
      f4(m.trackingScore),
      m.track?.name ?? null,
      m.height,
      m.width,
      Array.from(m.rleCounts),
      m.scale,
      m.offset,
      m.fromPredicted?.name ?? null,
      m.scoreMap ? Array.from(m.scoreMap) : null,
      perDetection(m),
      link(m),
    ]),
    centroids: labels.centroids.map((c: any) => [
      c.constructor.name,
      c.x,
      c.y,
      c.z,
      c.name,
      c.category,
      c.source,
      f4(c.score),
      f4(c.trackingScore),
      c.track?.name ?? null,
      perDetection(c),
      link(c),
    ]),
    labelImages: labels.labelImages.map((li: any) => [
      li.constructor.name,
      li.source,
      f4(li.score),
      li.height,
      li.width,
      li.scale,
      li.offset,
      Array.from(li.data),
      li.scoreMap ? Array.from(li.scoreMap) : null,
      [...li.objects.entries()].map(([id, o]: any) => [
        id,
        o.name,
        o.category,
        f4(o.score),
        f4(o.trackingScore),
        o.track?.name ?? null,
        o.instance ? (rowOf.get(o.instance) ?? "other") : -1,
      ]),
    ]),
  };
}

for (const [columnarFile, legacyFile] of TWINS) {
  describe(`${columnarFile} and ${legacyFile} read the same`, () => {
    for (const [name, read] of READERS) {
      it(name, async () => {
        quietWarnings();
        const columnar = await read(fixture(columnarFile));
        const legacy = await read(fixture(legacyFile));
        expect(columnar.labeledFrames.length).toBeGreaterThan(0);
        expect(poseSnapshot(columnar)).toEqual(poseSnapshot(legacy));
        expect(annotationSnapshot(columnar)).toEqual(
          annotationSnapshot(legacy),
        );
      });
    }

    it("loadSlpMetadata", async () => {
      quietWarnings();
      const columnar = await loadSlpMetadata(fixture(columnarFile));
      const legacy = await loadSlpMetadata(fixture(legacyFile));
      expect(columnar.formatId).toBe(3);
      expect(legacy.formatId).toBeLessThan(3);
      expect(columnar.counts.labeledFrames).toBeGreaterThan(0);
      expect(columnar.counts).toEqual(legacy.counts);
      expect(columnar.version).toBe(legacy.version);
      expect(columnar.skeletons[0].nodeNames).toEqual(
        legacy.skeletons[0].nodeNames,
      );
      expect(columnar.tracks.map((t) => t.name)).toEqual(
        legacy.tracks.map((t) => t.name),
      );
      expect(columnar.suggestions).toEqual(legacy.suggestions);
      expect(columnar.provenance).toEqual(legacy.provenance);
      expect(columnar.hasEmbeddedImages).toBe(legacy.hasEmbeddedImages);
      expect(columnar.sessions.map((s) => s.cameras)).toEqual(
        legacy.sessions.map((s) => s.cameras),
      );
      expect(columnar.sessions.map((s) => s.videosByCamera)).toEqual(
        legacy.sessions.map((s) => s.videosByCamera),
      );
    });
  });
}

describe("other layouts read the same as the 2.x file", () => {
  it("gzip-compressed columns read the same as plain ones", async () => {
    quietWarnings();
    const gzip = "handbuilt_30_gzip.slp";
    const frames = (labels: Labels) => poseSnapshot(labels).frames;
    const expected = frames(
      await readSlp(fixture(LEGACY), { openVideos: false }),
    );
    expect(frames(await readSlp(fixture(gzip), { openVideos: false }))).toEqual(
      expected,
    );
    expect(frames(await readStreaming(fixture(gzip)))).toEqual(expected);
    const lite = await loadSlpMetadata(fixture(gzip));
    expect(lite.counts).toEqual({
      labeledFrames: 5,
      instances: 4,
      points: 6,
      predictedPoints: 6,
    });
    expect(lite.sessions[0].frameGroups?.length).toBe(2);
  });

  it("each table goes by its own layout, whatever format_id says", async () => {
    quietWarnings();
    const mixed = "handbuilt_30_mixed.slp";
    expect(
      poseSnapshot(await readSlp(fixture(mixed), { openVideos: false })),
    ).toEqual(
      poseSnapshot(await readSlp(fixture(LEGACY), { openVideos: false })),
    );
    expect(poseSnapshot(await readStreaming(fixture(mixed)))).toEqual(
      poseSnapshot(await readStreaming(fixture(LEGACY))),
    );
  });
});

// The same project in both, so the same values must come out.
for (const file of [PYTHON_TWIN, COLUMNAR]) {
  describe(`what ${file} holds`, () => {
    it("poses, links and flags come through with their values", async () => {
      quietWarnings();
      const labels = await readSlp(fixture(file), { openVideos: false });
      expect(labels.labeledFrames.map((f) => f.frameIdx)).toEqual([
        5, 7, 9, 11, 13,
      ]);
      const [p0, u0] = labels.labeledFrames[0].instances as any[];
      expect(p0.constructor.name).toBe("PredictedInstance");
      // Float64 survives: 2.x would round this to the nearest float32.
      expect(p0.score).toBe(0.1);
      expect(p0.trackingScore).toBe(0.75);
      expect(p0.track.name).toBe("mouse1");
      expect(p0.points.map((p: any) => p.score)).toEqual([0.25, 0.5, 0.75]);
      expect(u0.constructor.name).toBe("Instance");
      expect(u0.fromPredicted).toBe(p0);
      // `verified` is what the rest of the library calls `complete`. A hidden point
      // keeps its coordinates; an absent one has none.
      expect(u0.points[0]).toMatchObject({ visible: true, complete: true });
      expect(u0.points[1]).toMatchObject({
        xy: [3.5, 4.5],
        visible: false,
        complete: true,
      });
      expect(u0.points[2].xy.every(Number.isNaN)).toBe(true);
      expect(p0.identity.name).toBe("animal_A");
      expect(p0.identityScore).toBe(0.5);
      expect(u0.category).toBe("rodent");
    });

    it("session tables give 3D points, scores and member references", async () => {
      quietWarnings();
      const labels = await readSlp(fixture(file), { openVideos: false });
      const groups = [...labels.sessions[0].frameGroups.values()] as any[];
      expect(groups.map((g) => g.frameIdx)).toEqual([5, 7]);
      expect(groups[0].metadata).toEqual({ fg: 1 });
      const [first, second] = groups[0].instanceGroups;
      expect(first.identity.name).toBe("animal_A");
      expect(first.instance3d.points[0]).toEqual([1.5, 2.5, 3.5]);
      expect(first._instanceRefsByCamera.size).toBe(2);
      expect(second.instance3d).toBeUndefined();
      expect(second.metadata).toEqual({ ig: "x" });
      const predicted = groups[1].instanceGroups[0].instance3d;
      expect(predicted.constructor.name).toBe("PredictedInstance3D");
      expect(predicted.pointScores.slice(0, 2)).toEqual([0.5, 0.25]);
      // Also through the lite reader, which has no frames to resolve them against.
      const lite = await loadSlpMetadata(fixture(file));
      expect(lite.sessions[0].frameGroups?.map((g) => g.frameIdx)).toEqual([
        5, 7,
      ]);
    });

    for (const [name, read] of READERS) {
      it(`${name} takes annotations from their 3.0 columns and buffers`, async () => {
        quietWarnings();
        const labels = await read(fixture(file));
        const frame = labels.labeledFrames[0];
        const roi = frame.rois[0] as any;
        expect([roi.name, roi.category, roi.source]).toEqual([
          "roi-a",
          "cat-a",
          "src-a",
        ]);
        expect(roi.instance).toBe(frame.instances[1]);
        expect(labels.rois.map((r) => r.name)).toContain("static");
        expect(frame.bboxes[0].x2).toBe(5);
        expect(frame.bboxes[0].instance).toBe(frame.instances[0]);
        const predicted = frame.masks[0] as any;
        const user = frame.masks[1] as any;
        expect(Array.from(predicted.rleCounts)).toEqual([1, 2, 3]);
        expect(Array.from(predicted.scoreMap)).toEqual([0, 1, 2, 3, 4, 5]);
        expect(user.fromPredicted).toBe(predicted);
        expect(user.scale).toEqual([2, 0.5]);
        const image = frame.labelImages[0] as any;
        expect(Array.from(image.data)).toEqual([0, 1, 1, 2]);
        expect(image.objects.get(1).name).toBe("o1");
        expect(image.objects.get(1).instance).toBe(frame.instances[0]);
        const second = labels.labeledFrames[1].labelImages[0] as any;
        expect(Array.from(second.data)).toEqual([2, 2, 0, 1]);
        expect(Array.from(second.scoreMap)).toEqual([0, 1, 2, 3]);
      });

      it(`${name} attaches per-detection links to annotations, not just instances`, async () => {
        quietWarnings();
        const labels = await read(fixture(file));
        expect(labels.labeledFrames[1].centroids[0].identity?.name).toBe(
          "animal_A",
        );
        expect(labels.labeledFrames[0].masks[0].identity?.name).toBe(
          "animal_B",
        );
        expect(labels.labeledFrames[0].bboxes[0].identity?.name).toBe(
          "animal_A",
        );
        expect(labels.labeledFrames[0].masks[1].category).toBe("rodent");
      });
    }
  });
}

describe("what a 3.0 file holds", () => {
  it("/frames/video is the list index, not the videoN number", async () => {
    quietWarnings();
    // videos are [video2, video0, video1], and only the first two have frames, so
    // the 2.x guess at sparse ids would map list index 0 to video0.
    for (const read of [
      (b: ArrayBuffer) => readSlp(b, { openVideos: false }),
      (b: ArrayBuffer) => readStreaming(b),
    ]) {
      const labels = await read(fixture(COLUMNAR));
      const datasets = labels.labeledFrames.map(
        (f) => (f.video.backendMetadata as { dataset: string }).dataset,
      );
      expect(datasets).toEqual([
        "video2/video",
        "video0/video",
        "video0/video",
        "video2/video",
        "video2/video",
      ]);
    }
    const lazy = await readSlpLazy(fixture(COLUMNAR), { openVideos: false });
    const frame = lazy.frameAt?.(0) ?? lazy.labeledFrames[0];
    expect((frame.video.backendMetadata as { dataset: string }).dataset).toBe(
      "video2/video",
    );
  });

  it("is_negative marks only the negative frame", async () => {
    quietWarnings();
    for (const [, read] of READERS) {
      const labels = await read(fixture(COLUMNAR));
      expect(
        labels.labeledFrames.filter((f) => f.isNegative).map((f) => f.frameIdx),
      ).toEqual([9]);
    }
  });

  it("the lazy store holds the columns the 2.x tables have, as numbers", async () => {
    quietWarnings();
    const labels = await readSlpLazy(fixture(COLUMNAR), { openVideos: false });
    const store = labels._lazyDataStore!;
    expect(Array.from(store.framesData.frame_id)).toEqual([0, 1, 2, 3, 4]);
    expect(Array.from(store.instancesData.instance_id)).toEqual([0, 1, 2, 3]);
    for (const table of [store.framesData, store.instancesData]) {
      for (const column of Object.values(table)) {
        expect(typeof (column as unknown[])[0]).toBe("number");
      }
    }
    expect(Object.keys(store.pointsData).sort()).toEqual([
      "complete",
      "visible",
      "x",
      "y",
    ]);
    expect(Object.keys(store.predPointsData).sort()).toEqual([
      "complete",
      "score",
      "visible",
      "x",
      "y",
    ]);
    expect(store.negativeFrames).toEqual(new Set(["1:9"]));
    expect(store.copy().materializeFrame(2)?.isNegative).toBe(true);
    expect(store.toNumpy({ video: labels.videos[0] })[5][0][0]).toEqual([
      1.5, 2.5,
    ]);
  });

  it("Python stores a video by list index in 3.0 and by videoN number in 2.x", async () => {
    const module = await ready;
    const videoColumn = async (name: string) => {
      const p = await tmpPath();
      module.FS.writeFile(p, new Uint8Array(fixture(name)));
      const f: any = new H5File(p, "r");
      const frames = f.get("frames");
      const values =
        frames.shape === undefined
          ? Array.from(frames.get("video").value as ArrayLike<bigint>, Number)
          : frames.value.map((row: unknown[]) => Number(row[1]));
      f.close();
      module.FS.unlink(p);
      return values;
    };
    expect(await videoColumn("py_written_30_sparse.slp")).toEqual([0, 0]);
    expect(await videoColumn("py_written_30_sparse_legacy.slp")).toEqual([
      2, 2,
    ]);
    for (const name of [
      "py_written_30_sparse.slp",
      "py_written_30_sparse_legacy.slp",
    ]) {
      const labels = await readSlp(fixture(name), { openVideos: false });
      expect(labels.labeledFrames.map((f) => f.video)).toEqual([
        labels.videos[0],
        labels.videos[0],
      ]);
      expect(labels.labeledFrames.map((f) => f.isNegative)).toEqual([
        false,
        true,
      ]);
    }
  });

  it("embedded images read the same from both layouts", async () => {
    const frameBytes = async (name: string) => {
      const labels = await readSlp(fixturePath(name), { openVideos: true });
      const bytes = await labels.videos[0].getFrameBuffer(0);
      labels.videos[0].close?.();
      return Array.from(new Uint8Array(bytes as ArrayBufferLike));
    };
    const columnar = await frameBytes("py_written_30_embedded.pkg.slp");
    expect(columnar.length).toBeGreaterThan(100);
    expect(columnar).toEqual(
      await frameBytes("py_written_30_embedded_legacy.pkg.slp"),
    );
  });

  it("Python's float64 scores survive in 3.0, and round to float32 in 2.x", async () => {
    quietWarnings();
    const score = async (name: string) => {
      const labels = await readSlp(fixture(name), { openVideos: false });
      return (labels.labeledFrames[0].instances[0] as any).score as number;
    };
    expect(await score(PYTHON_TWIN)).toBe(0.1);
    expect(await score("py_written_30_twin_legacy.slp")).toBe(Math.fround(0.1));
  });

  it("the warning names what Python's own file holds that a save won't keep", async () => {
    const warn = quietWarnings();
    await readSlp(fixture(PYTHON_TWIN), {
      openVideos: false,
      h5: { filenameHint: "mice.slp" },
    });
    const lines = warn.mock.calls.map((c) => String(c[0]));
    expect(lines.some((l) => l.includes("/points has 1 unlabeled"))).toBe(true);
    for (const table of ["/instances", "/frames", "/bboxes"]) {
      expect(lines.some((l) => l.includes(`${table} holds metadata,`))).toBe(
        true,
      );
    }
    expect(lines.length).toBe(4);
  });

  it("the 2.x file has none of what the 3.0 one warns about", async () => {
    const warn = quietWarnings();
    await readSlp(fixture("py_written_30_twin_legacy.slp"), {
      openVideos: false,
    });
    expect(warn).not.toHaveBeenCalled();
  });

  it("the file Python really writes reads in every reader, with no warnings", async () => {
    const warn = quietWarnings();
    for (const [, read] of READERS) {
      const labels = await read(fixture(PYTHON));
      expect(labels.labeledFrames.length).toBe(1);
      const [instance] = labels.labeledFrames[0].instances;
      expect(instance.points.map((p) => p.xy)).toEqual([
        [1, 2],
        [3, 4],
      ]);
      expect(instance.track?.name).toBe("mouse1");
      expect(labels.labeledFrames[0].isNegative).toBe(false);
      expect(labels.identities.map((i) => i.name)).toEqual(["animal_A"]);
    }
    const lite = await loadSlpMetadata(fixture(PYTHON));
    expect(lite.formatId).toBe(3);
    expect(lite.counts).toEqual({
      labeledFrames: 1,
      instances: 1,
      points: 2,
      predictedPoints: 0,
    });
    expect(validateSlpBuffer(fixture(PYTHON))).toBe(true);
    expect(warn).not.toHaveBeenCalled();
  });
});

/** Tools to change a fixture, or write a small file, with h5wasm. */
async function editFile(
  name: string,
  edit: (f: any) => void,
): Promise<ArrayBuffer> {
  const module = await ready;
  const p = await tmpPath();
  module.FS.writeFile(p, new Uint8Array(fixture(name)));
  const f = new H5File(p, "a");
  edit(f);
  f.close();
  const out = module.FS.readFile(p);
  module.FS.unlink(p);
  return new Uint8Array(out).buffer as ArrayBuffer;
}

/** A minimal 3.0 file from columns given as `{ "points/x": Float64Array, ... }`. */
async function smallFile(
  columns: Record<string, ArrayLike<number> | BigInt64Array>,
): Promise<ArrayBuffer> {
  const module = await ready;
  const p = await tmpPath();
  module.FS.writeFile(p, new Uint8Array(0));
  const f: any = new H5File(p, "w");
  const meta = f.create_group("metadata");
  meta.create_attribute("format_id", 3.0);
  meta.create_attribute(
    "json",
    JSON.stringify({
      version: "2.0.0",
      skeletons: [{ graph: { name: "s" }, links: [], nodes: [{ id: 0 }] }],
      nodes: [{ name: "a" }],
    }),
  );
  f.create_dataset({
    name: "videos_json",
    data: [
      JSON.stringify({ filename: "a.mp4", backend: { filename: "a.mp4" } }),
    ],
  });
  for (const [column, data] of Object.entries(columns)) {
    const [group, name] = column.split("/");
    if (!f.get(group)) f.create_group(group);
    f.get(group).create_dataset({ name, data });
  }
  f.close();
  const out = module.FS.readFile(p);
  module.FS.unlink(p);
  return new Uint8Array(out).buffer as ArrayBuffer;
}

const ONE_FRAME = {
  "frames/video": new Int32Array([0]),
  "frames/frame_idx": new Int32Array([3]),
  "frames/instance_id_start": new Int32Array([0]),
  "frames/instance_id_end": new Int32Array([1]),
};
const ONE_INSTANCE = {
  "instances/instance_type": new Uint8Array([0]),
  "instances/frame_id": new Int32Array([0]),
  "instances/skeleton": new Int32Array([0]),
  "instances/point_id_start": new Int32Array([0]),
  "instances/point_id_end": new Int32Array([1]),
};
const ONE_POINT = {
  "points/x": new Float64Array([1.5]),
  "points/y": new Float64Array([2.5]),
  "points/visible": new Uint8Array([1]),
};
const NO_PRED_POINTS = {
  "pred_points/x": new Float64Array(0),
  "pred_points/y": new Float64Array(0),
};

describe("the reading rules for 3.0 tables", () => {
  const SMALL_READERS: Array<
    [string, (bytes: ArrayBuffer) => Promise<Labels>]
  > = [
    ["readSlp", (b) => readSlp(b, { openVideos: false })],
    ["readFromStreamingFile", (b) => readStreaming(b)],
  ];

  for (const [name, read] of SMALL_READERS) {
    it(`${name} fills in optional columns and takes any numeric dtype`, async () => {
      const labels = await read(
        await smallFile({
          ...ONE_FRAME,
          ...ONE_INSTANCE,
          ...ONE_POINT,
          ...NO_PRED_POINTS,
        }),
      );
      const [frame] = labels.labeledFrames;
      expect(frame.frameIdx).toBe(3);
      expect(frame.isNegative).toBe(false);
      const [instance] = frame.instances;
      expect(instance.track).toBeNull();
      expect(instance.fromPredicted).toBeNull();
      expect(instance.points[0]).toMatchObject({
        xy: [1.5, 2.5],
        visible: true,
        complete: false,
      });
    });

    it(`${name} reads a missing visible as hidden, and any non-zero flag as set`, async () => {
      const labels = await read(
        await smallFile({
          ...ONE_FRAME,
          ...ONE_INSTANCE,
          "points/x": new Float64Array([1.5]),
          "points/y": new Float64Array([2.5]),
          "points/verified": new Int32Array([7]),
          "frames/is_negative": new Int32Array([2]),
          ...NO_PRED_POINTS,
        }),
      );
      const [frame] = labels.labeledFrames;
      expect(frame.isNegative).toBe(true);
      expect(frame.instances[0].points[0]).toMatchObject({
        visible: false,
        complete: true,
      });
    });

    it(`${name} refuses a table missing a required column`, async () => {
      const { "points/x": _x, ...noX } = {
        ...ONE_FRAME,
        ...ONE_INSTANCE,
        ...ONE_POINT,
        ...NO_PRED_POINTS,
      };
      await expect(read(await smallFile(noX))).rejects.toThrow(
        /\/points is missing column\(s\) x/,
      );
      const { "instances/skeleton": _s, ...noSkeleton } = {
        ...ONE_FRAME,
        ...ONE_INSTANCE,
        ...ONE_POINT,
        ...NO_PRED_POINTS,
      };
      await expect(read(await smallFile(noSkeleton))).rejects.toThrow(
        /\/instances is missing column\(s\) skeleton/,
      );
    });

    it(`${name} refuses columns of different lengths`, async () => {
      await expect(
        read(
          await smallFile({
            ...ONE_FRAME,
            ...ONE_INSTANCE,
            ...ONE_POINT,
            ...NO_PRED_POINTS,
            "points/y": new Float64Array([2.5, 3.5]),
          }),
        ),
      ).rejects.toThrow(/\/points columns have different lengths/);
    });

    it(`${name} refuses ranges that don't add up, naming the table`, async () => {
      const base = {
        ...ONE_FRAME,
        ...ONE_INSTANCE,
        ...ONE_POINT,
        ...NO_PRED_POINTS,
      };
      const cases: Array<[Record<string, Int32Array>, RegExp]> = [
        [
          { "frames/instance_id_end": new Int32Array([2]) },
          /\/frames instance ranges end at 2, not at the instance count \(1\)/,
        ],
        [
          { "instances/frame_id": new Int32Array([4]) },
          /\/instances frame_id disagrees with the \/frames ranges/,
        ],
        [
          { "instances/point_id_end": new Int32Array([2]) },
          /\/instances point ranges fall outside \/points \(1 rows\)/,
        ],
        [
          { "instances/from_predicted": new Int32Array([5]) },
          /\/instances from_predicted is not an instance row/,
        ],
        [
          { "frames/video": new Int32Array([1]) },
          /\/frames video is not one of the 1 videos/,
        ],
      ];
      for (const [change, message] of cases) {
        await expect(
          read(await smallFile({ ...base, ...change })),
        ).rejects.toThrow(message);
      }
    });

    it(`${name} skips instances of an unknown type, keeping links by row`, async () => {
      const labels = await read(
        await smallFile({
          ...ONE_FRAME,
          "frames/instance_id_end": new Int32Array([2]),
          "instances/instance_type": new Uint8Array([9, 0]),
          "instances/frame_id": new Int32Array([0, 0]),
          "instances/skeleton": new Int32Array([0, 0]),
          "instances/point_id_start": new Int32Array([0, 0]),
          "instances/point_id_end": new Int32Array([0, 1]),
          ...ONE_POINT,
          ...NO_PRED_POINTS,
        }),
      );
      expect(labels.labeledFrames[0].instances.length).toBe(1);
    });
  }

  it("an empty project is valid", async () => {
    const bytes = await smallFile({
      "frames/video": new Int32Array(0),
      "frames/frame_idx": new Int32Array(0),
      "frames/instance_id_start": new Int32Array(0),
      "frames/instance_id_end": new Int32Array(0),
      "instances/instance_type": new Uint8Array(0),
      "instances/frame_id": new Int32Array(0),
      "instances/skeleton": new Int32Array(0),
      "instances/point_id_start": new Int32Array(0),
      "instances/point_id_end": new Int32Array(0),
      "points/x": new Float64Array(0),
      "points/y": new Float64Array(0),
      ...NO_PRED_POINTS,
    });
    expect((await readSlp(bytes, { openVideos: false })).labeledFrames).toEqual(
      [],
    );
    expect((await readStreaming(bytes)).labeledFrames).toEqual([]);
    expect((await loadSlpMetadata(bytes)).counts.labeledFrames).toBe(0);
  });

  it("an unlabeled point becomes a hidden point with no location", async () => {
    const warn = quietWarnings();
    const labels = await readSlp(
      await smallFile({
        ...ONE_FRAME,
        ...ONE_INSTANCE,
        ...ONE_POINT,
        "points/visible": new Uint8Array([0]),
        "points/labeled": new Uint8Array([0]),
        ...NO_PRED_POINTS,
      }),
      { openVideos: false },
    );
    const [point] = labels.labeledFrames[0].instances[0].points;
    expect(point.visible).toBe(false);
    expect(point.xy.every(Number.isNaN)).toBe(true);
    expect(
      warn.mock.calls.some((c) => String(c[0]).includes("1 unlabeled point")),
    ).toBe(true);
  });

  it("a visible point counts as labeled, as Python stores it", async () => {
    const warn = quietWarnings();
    const labels = await readSlp(
      await smallFile({
        ...ONE_FRAME,
        ...ONE_INSTANCE,
        ...ONE_POINT,
        "points/labeled": new Uint8Array([0]),
        ...NO_PRED_POINTS,
      }),
      { openVideos: false },
    );
    expect(labels.labeledFrames[0].instances[0].points[0].xy).toEqual([
      1.5, 2.5,
    ]);
    expect(warn).not.toHaveBeenCalled();
  });

  it("64-bit integer columns become numbers without one BigInt per row", () => {
    expect(
      Array.from(
        numberColumn(BigInt64Array.from([0n, -1n, 2n ** 40n, -(2n ** 40n)])),
      ),
    ).toEqual([0, -1, 2 ** 40, -(2 ** 40)]);
    expect(
      Array.from(numberColumn(BigUint64Array.from([0n, 2n ** 40n + 5n]))),
    ).toEqual([0, 2 ** 40 + 5]);
    const floats = new Float32Array([1.5]);
    expect(numberColumn(floats)).toBe(floats);
  });
});

describe("what 3.0 readers still refuse", () => {
  const READERS: Array<[string, (b: ArrayBuffer) => Promise<unknown>]> = [
    ["readSlp", (b) => readSlp(b, { openVideos: false })],
    ["readSlpLazy", (b) => readSlpLazy(b, { openVideos: false })],
    ["readFromStreamingFile", (b) => readStreaming(b)],
    ["loadSlpMetadata", (b) => loadSlpMetadata(b)],
  ];
  const withRequiredFeatures = (features: string | string[]) =>
    editFile(PYTHON, (f) => {
      f.get("metadata").create_attribute("required_features", features);
    });

  for (const [name, read] of READERS) {
    it(`${name} refuses a feature it doesn't understand`, async () => {
      for (const [features, listed] of [
        [["zeta", "alpha"], "alpha, zeta"],
        ["zeta", "zeta"],
      ] as const) {
        let caught: unknown;
        try {
          await read(await withRequiredFeatures(features as string[]));
        } catch (e) {
          caught = e;
        }
        expect(caught).toBeInstanceOf(UnsupportedSlpFormatError);
        expect((caught as UnsupportedSlpFormatError).formatId).toBe(3);
        expect((caught as Error).message).toContain(
          `requires SLP features this version of sleap-io.js doesn't support (${listed})`,
        );
      }
    });
  }

  it("validateSlpBuffer refuses it too", async () => {
    const bytes = await withRequiredFeatures(["zeta"]);
    expect(() => validateSlpBuffer(bytes)).toThrow(UnsupportedSlpFormatError);
  });

  it("no required features is no refusal", () => {
    for (const required_features of [undefined, [], { value: [] }]) {
      expect(checkSlpFormat({ format_id: 3, required_features })).toBe(3);
    }
  });

  it("the check reads the attribute in every form a reader returns it", () => {
    for (const required_features of [
      "x",
      ["x"],
      { value: "x" },
      { value: ["x"] },
      new TextEncoder().encode("x"),
      [new TextEncoder().encode("x")],
    ]) {
      let caught: UnsupportedSlpFormatError | undefined;
      try {
        checkSlpFormat({ format_id: 3, required_features });
      } catch (e) {
        caught = e as UnsupportedSlpFormatError;
      }
      expect(caught?.unsupportedFeatures).toEqual(["x"]);
    }
  });

  const LZF = "handbuilt_30_lzf.slp";
  for (const [name, read] of READERS) {
    it(`${name} refuses columns it can't decompress, instead of reading garbage`, async () => {
      quietWarnings();
      await expect(read(fixture(LZF))).rejects.toThrow(/lzf|32000/);
    });
  }

  it("a 2.x table that can't be decompressed is refused too", () => {
    expect(() =>
      checkDatasetFilters(
        {
          filters: [
            { id: 2, name: "shuffle" },
            { id: 32000, name: "lzf" },
          ],
        },
        "points",
      ),
    ).toThrow(/\/points is compressed with "lzf"/);
    expect(() =>
      checkDatasetFilters(
        { filters: [{ id: 2 }, { id: 1, name: "deflate" }] },
        "points",
      ),
    ).not.toThrow();
  });

  it("the message says what to do about lzf", async () => {
    quietWarnings();
    for (const read of [
      (b: ArrayBuffer) => readSlp(b, { openVideos: false }),
      (b: ArrayBuffer) => readStreaming(b),
    ]) {
      await expect(read(fixture(LZF))).rejects.toThrow(
        /\/frames\/\w+ is compressed with "lzf", which sleap-io\.js cannot read\. Re-save/,
      );
    }
  });
});

describe("the warning about what a save won't keep", () => {
  it("names unknown columns and record metadata, table by table", async () => {
    const warn = quietWarnings();
    await readSlp(fixture(COLUMNAR), {
      openVideos: false,
      h5: { filenameHint: "mice.slp" },
    });
    const lines = warn.mock.calls.map((c) => String(c[0]));
    const about = (text: string) => lines.filter((l) => l.includes(text));
    expect(about("mice.slp: /points holds brightness,").length).toBe(1);
    expect(about("mice.slp: /instances holds future, metadata,").length).toBe(
      1,
    );
    expect(about("mice.slp: /frames holds metadata,").length).toBe(1);
    expect(about("mice.slp: /bboxes holds metadata,").length).toBe(1);
    expect(about("will not keep them").length).toBe(4);
  });

  it("the streaming reader gives the same warnings", async () => {
    const warn = quietWarnings();
    await readStreaming(fixture(COLUMNAR));
    const lines = warn.mock.calls.map((c) => String(c[0]));
    // The name, without the URL's query.
    expect(lines.every((l) => !l.includes("a=1"))).toBe(true);
    warn.mockClear();
    await readSlp(fixture(COLUMNAR), {
      openVideos: false,
      h5: { filenameHint: "mice.slp" },
    });
    expect(lines.sort()).toEqual(
      warn.mock.calls.map((c) => String(c[0])).sort(),
    );
  });

  for (const [name, read] of READERS) {
    it(`${name}: buffers and sub-tables a reader handles itself are not called unknown`, async () => {
      const warn = quietWarnings();
      await read(fixture(COLUMNAR));
      const lines = warn.mock.calls.map((c) => String(c[0]));
      for (const table of ["/rois", "/masks", "/label_images"]) {
        expect(lines.some((l) => l.includes(`${table} holds`))).toBe(false);
      }
    });
  }
});

/** The five-item per-detection part of an {@link annotationSnapshot} row. */
const isLinks = (value: unknown): boolean =>
  Array.isArray(value) &&
  value.length === 5 &&
  (value[0] === null || typeof value[0] === "string") &&
  (value[2] === null || Array.isArray(value[2]));

describe("annotations through the streaming reader", () => {
  /** A file as `saveSlpToBytes` writes it: flat tables, strings as JSON bytes. */
  const jsWritten = async (): Promise<ArrayBuffer> => {
    const labels = await readSlp(fixture("handbuilt_30_gzip.slp"), {
      openVideos: false,
    });
    return new Uint8Array(await saveSlpToBytes(labels)).buffer as ArrayBuffer;
  };
  const LAYOUTS: Array<[string, () => Promise<ArrayBuffer>]> = [
    ...TWINS.flat().map((name): [string, () => Promise<ArrayBuffer>] => [
      name,
      async () => fixture(name),
    ]),
    ["compressed columns", async () => fixture("handbuilt_30_gzip.slp")],
    ["3.0 and 2.x tables mixed", async () => fixture("handbuilt_30_mixed.slp")],
    ["what this library saves", jsWritten],
    [
      "py_written_annotations.slp (real Python, every kind)",
      async () => fixture("py_written_annotations.slp"),
    ],
    [
      "js_flat_annotations.slp (flat tables, strings as JSON)",
      async () => fixture("js_flat_annotations.slp"),
    ],
  ];

  for (const [layout, bytes] of LAYOUTS) {
    it(`reads the same as the eager reader: ${layout}`, async () => {
      quietWarnings();
      const eager = await readSlp(await bytes(), { openVideos: false });
      const streamed = await readStreaming(await bytes());
      expect(eager.labeledFrames.length).toBeGreaterThan(0);
      expect(poseSnapshot(streamed)).toEqual(poseSnapshot(eager));
      expect(annotationSnapshot(streamed)).toEqual(annotationSnapshot(eager));
    });

    it(`hands the lazy store the same as readSlpLazy: ${layout}`, async () => {
      quietWarnings();
      const read = await readSlpLazy(await bytes(), { openVideos: false });
      const streamed = await readStreaming(await bytes(), true);
      expect(streamed.isLazy).toBe(true);
      const store = streamed._lazyDataStore!;
      const expected = read._lazyDataStore!;
      for (const map of [
        "_centroidByFrame",
        "_bboxByFrame",
        "_maskByFrame",
        "_roiByFrame",
        "_labelImageByFrame",
      ] as const) {
        expect([...store[map].keys()]).toEqual([...expected[map].keys()]);
      }
      expect(store._undistributedRois.length).toBe(
        expected._undistributedRois.length,
      );
      expect(store._instanceIdentityLinks).toEqual(
        expected._instanceIdentityLinks,
      );
      expect(store._instanceCategoryLinks).toEqual(
        expected._instanceCategoryLinks,
      );
      expect(store.categoryCatalog).toEqual(expected.categoryCatalog);
      // Reached without materializing a frame.
      expect(streamed.isLazy).toBe(true);
      expect(streamed.rois.length).toBe(read.rois.length);
      expect(streamed.masks.length).toBe(read.masks.length);
      expect(streamed.bboxes.length).toBe(read.bboxes.length);
      expect(streamed.labelImages.length).toBe(read.labelImages.length);
      expect(streamed.centroids.length).toBe(read.centroids.length);
      read.materialize();
      streamed.materialize();
      expect(annotationSnapshot(streamed)).toEqual(annotationSnapshot(read));
    });
  }

  const ANNOTATION_PATH =
    /^(rois|roi_|masks|mask_|bboxes|bbox_|label_image|centroid)/;

  it("asks for nothing about a table the file doesn't have", async () => {
    const { requests } = await readRecording(fixture(PYTHON));
    expect(requests.filter(([, path]) => ANNOTATION_PATH.test(path))).toEqual(
      [],
    );
  });

  it("on a 2.x file of poses alone, annotations add no request at all", async () => {
    const { requests } = await readRecording(fixture("minimal_instance.slp"));
    expect(requests.every(([, path]) => !ANNOTATION_PATH.test(path))).toBe(
      true,
    );
    expect(requests.some(([, path]) => path.startsWith("embeddings"))).toBe(
      false,
    );
  });

  it("reads a table's buffer once, and not along with its rows", async () => {
    quietWarnings();
    const { requests } = await readRecording(fixture(COLUMNAR));
    const reads = (path: string) =>
      requests.filter(([type, p]) => type === "getDatasetValue" && p === path)
        .length;
    for (const path of [
      "rois",
      "rois/wkb",
      "masks",
      "masks/rle",
      "masks/score_maps",
      "masks/score_maps/data",
      "label_images",
      "label_images/data",
      "label_images/objects",
      "label_images/score_maps",
      "label_images/score_maps/data",
      "bboxes",
    ]) {
      expect([path, reads(path)]).toEqual([path, 1]);
    }
  });

  it("doesn't read the buffer of a table with no rows", async () => {
    const bytes = await smallFile({
      ...ONE_FRAME,
      ...ONE_INSTANCE,
      ...ONE_POINT,
      ...NO_PRED_POINTS,
      "rois/video": new Int32Array(0),
      "rois/frame_idx": new Int32Array(0),
      "rois/track": new Int32Array(0),
      "rois/wkb_start": new Int32Array(0),
      "rois/wkb_end": new Int32Array(0),
      "rois/wkb": new Uint8Array(0),
    });
    const { labels, requests } = await readRecording(bytes);
    expect(labels.rois).toEqual([]);
    expect(requests.some(([, path]) => path === "rois/wkb")).toBe(false);
  });

  for (const [layout, name, path] of [
    ["3.0 rows", COLUMNAR, "rois"],
    ["3.0 buffer", COLUMNAR, "masks/rle"],
    ["3.0 sub-table", COLUMNAR, "label_images/objects"],
    ["3.0 column", COLUMNAR, "centroids/x"],
    ["2.x rows", LEGACY, "masks"],
    ["2.x buffer", LEGACY, "roi_wkb"],
    ["2.x strings", LEGACY, "mask_names"],
    ["2.x sub-table", LEGACY, "label_image_objects"],
    ["links", LEGACY, "categories/links"],
    ["embedding", LEGACY, "embeddings/category_vectors"],
  ] as const) {
    it(`a failed read is an error, not a missing table: ${layout}`, async () => {
      quietWarnings();
      await expect(
        readRecording(fixture(name), (type, p) =>
          type === "getDatasetValue" && p === path
            ? new Error(`could not read ${path}`)
            : undefined,
        ),
      ).rejects.toThrow(`could not read ${path}`);
    });
  }

  it("makes a frame for an annotation that isn't on one in /frames", async () => {
    const onlyBox = {
      ...ONE_FRAME,
      ...ONE_INSTANCE,
      ...ONE_POINT,
      ...NO_PRED_POINTS,
      "bboxes/x1": new Float64Array([1, 7]),
      "bboxes/y1": new Float64Array([2, 8]),
      "bboxes/x2": new Float64Array([3, 9]),
      "bboxes/y2": new Float64Array([4, 10]),
      "bboxes/video": new Int32Array([0, 0]),
      "bboxes/frame_idx": new Int32Array([3, 50]),
      "bboxes/track": new Int32Array([-1, -1]),
      "bboxes/instance": new Int32Array([0, -1]),
      "bboxes/is_predicted": new Uint8Array([0, 0]),
    };
    const bytes = await smallFile(onlyBox);
    const eager = await readSlp(bytes, { openVideos: false });
    const streamed = await readStreaming(bytes);
    expect(streamed.labeledFrames.map((f) => f.frameIdx)).toEqual([3, 50]);
    expect(streamed.labeledFrames[0].bboxes[0].instance).toBe(
      streamed.labeledFrames[0].instances[0],
    );
    expect(annotationSnapshot(streamed)).toEqual(annotationSnapshot(eager));

    const lazy = await readStreaming(bytes, true);
    const lazyEager = await readSlpLazy(bytes, { openVideos: false });
    expect(lazy._lazyFrameList!.length).toBe(2);
    expect(lazy._lazyFrameList!._supplementary.length).toBe(1);
    lazy.materialize();
    lazyEager.materialize();
    expect(annotationSnapshot(lazy)).toEqual(annotationSnapshot(lazyEager));
    expect(lazy.labeledFrames[1].bboxes[0].x1).toBe(7);
    expect(lazy.labeledFrames[0].bboxes[0].instance).toBe(
      lazy.labeledFrames[0].instances[0],
    );
  });

  it("links a static ROI to its instance", async () => {
    const wkb = encodeWkb({
      type: "Polygon",
      coordinates: [
        [
          [0, 0],
          [2, 0],
          [2, 3],
          [0, 0],
        ],
      ],
    });
    const bytes = await smallFile({
      ...ONE_FRAME,
      ...ONE_INSTANCE,
      ...ONE_POINT,
      ...NO_PRED_POINTS,
      "rois/video": new Int32Array([-1]),
      "rois/frame_idx": new Int32Array([-1]),
      "rois/track": new Int32Array([-1]),
      "rois/instance": new Int32Array([0]),
      "rois/wkb_start": new Int32Array([0]),
      "rois/wkb_end": new Int32Array([wkb.length]),
      "rois/wkb": wkb,
    });
    for (const labels of [
      await readSlp(bytes, { openVideos: false }),
      await readStreaming(bytes),
    ]) {
      const [roi] = labels.rois as any[];
      expect(labels.labeledFrames[0].rois.length).toBe(0);
      expect(roi.instance).toBe(labels.labeledFrames[0].instances[0]);
    }
  });

  describe("label image pixels too large for one read", () => {
    /** Read a dataset's pixels through a real worker, recording each read's slice. */
    async function readPixels(
      bytes: ArrayBuffer,
      path: string,
      blockBytes?: number,
    ): Promise<{ pixels: LabelImagePixels; slices: unknown[] }> {
      const { file, close } = await workerFile(bytes);
      const slices: unknown[] = [];
      const send = (file as any).send.bind(file);
      (file as any).send = (type: string, payload: any) => {
        if (type === "getDatasetValue") {
          slices.push(payload.slice?.[0] ?? "whole");
        }
        return send(type, payload);
      };
      try {
        const pixels = await readPixelDataStreaming(file, path, blockBytes);
        return { pixels, slices };
      } finally {
        close();
      }
    }
    const flat = ({ value, framesPerBlock }: LabelImagePixels): number[] =>
      framesPerBlock
        ? (value as Int32Array[]).flatMap((block) => Array.from(block))
        : Array.from(value as Int32Array);

    for (const [name, path] of [
      [COLUMNAR, "label_images/data"],
      [LEGACY, "label_image_data"],
      [PYTHON_TWIN, "label_images/data"],
      ["py_written_30_twin_legacy.slp", "label_image_data"],
    ]) {
      it(`${name}: blocks hold the same pixels as one read`, async () => {
        const whole = await readPixels(fixture(name), path);
        expect(whole.slices).toEqual(["whole"]);
        expect(whole.pixels.framesPerBlock).toBeUndefined();
        const [nFrames, h, w] = whole.pixels.shape;
        const blocks = await readPixels(fixture(name), path, h * w * 4);
        expect(blocks.slices).toEqual(
          Array.from({ length: nFrames }, (_, i) => [i, i + 1]),
        );
        expect(blocks.pixels.framesPerBlock).toBe(1);
        expect(blocks.pixels.shape).toEqual(whole.pixels.shape);
        expect(flat(blocks.pixels)).toEqual(flat(whole.pixels));
      });
    }

    it("a block holds as many frames as fit, in the file's integer type", async () => {
      const pixels = Array.from({ length: 30 }, (_, i) => i);
      const bytes = await editFile(PYTHON, (f) => {
        f.create_dataset({
          name: "pixels",
          data: new Uint16Array(pixels),
          shape: [5, 3, 2],
          dtype: "<H",
        });
      });
      // 12 bytes a frame, so 2 frames fit in 30 bytes, and the last block is short.
      const read = await readPixels(bytes, "pixels", 30);
      expect(read.slices).toEqual([
        [0, 2],
        [2, 4],
        [4, 5],
      ]);
      expect(read.pixels.framesPerBlock).toBe(2);
      expect((read.pixels.value as unknown[])[0]).toBeInstanceOf(Uint16Array);
      expect(flat(read.pixels)).toEqual(pixels);
    });

    it("a frame larger than a block is read alone", async () => {
      const read = await readPixels(fixture(COLUMNAR), "label_images/data", 1);
      expect(read.pixels.framesPerBlock).toBe(1);
      expect(read.slices.length).toBe(read.pixels.shape[0]);
    });

    it("a blob is one read, however large", async () => {
      const read = await readPixels(fixture(COLUMNAR), "rois/wkb", 1);
      expect(read.slices).toEqual(["whole"]);
    });

    it("block reads refuse a filter this build can't decompress", async () => {
      // The worker reports the filter, as whole reads go by it too.
      const { file, close } = await workerFile(fixture("handbuilt_30_lzf.slp"));
      const meta = await file.getDatasetMeta("frames/video");
      close();
      expect(() => checkDatasetFilters(meta, "frames/video")).toThrow(/"lzf"/);
      const reads: unknown[] = [];
      const threeD = {
        getDatasetMeta: async () => ({ ...meta, shape: [2, 2, 2] }),
        getDatasetValue: async (...args: unknown[]) => reads.push(args),
      } as unknown as StreamingH5File;
      await expect(readPixelDataStreaming(threeD, "pixels", 1)).rejects.toThrow(
        /\/pixels is compressed with "lzf"/,
      );
      expect(reads).toEqual([]);
    });

    const OBJECTLESS = {
      video: [0, 0, 0],
      frame_idx: [1, 2, 3],
      height: [1, 1, 1],
      width: [2, 2, 2],
      n_objects: [0, 0, 0],
      objects_start: [0, 0, 0],
    };
    const build = (data: LabelImagePixels, rows: Rows = OBJECTLESS) =>
      buildLabelImages(
        {
          rows,
          columnar: false,
          sources: [],
          data,
          objects: null,
          scoreMaps: null,
        },
        [],
        [],
        { labelsPath: "mice.slp", nInstances: 0 },
      ).map(([image]) => Array.from(image.data));

    it("frames come from the right block", () => {
      const expected = [
        [1, 2],
        [3, 4],
        [5, 6],
      ];
      const shape = [3, 1, 2];
      expect(
        build({
          value: [new Int32Array([1, 2, 3, 4]), new Int32Array([5, 6])],
          shape,
          framesPerBlock: 2,
        }),
      ).toEqual(expected);
      expect(
        build({
          value: [new Uint16Array([1, 2, 3, 4]), new Uint16Array([5, 6])],
          shape,
          framesPerBlock: 2,
        }),
      ).toEqual(expected);
      expect(
        build({ value: new Int32Array([1, 2, 3, 4, 5, 6]), shape }),
      ).toEqual(expected);
    });

    it("more rows than frames is refused, not read as blank images", () => {
      expect(() =>
        build({
          value: [new Int32Array([1, 2, 3, 4])],
          shape: [2, 1, 2],
          framesPerBlock: 2,
        }),
      ).toThrow(/\/label_images has 3 rows but 2 frames of data/);
    });
  });

  describe("older layouts", () => {
    const attrJson = (ds: any, name: string, values: string[]) =>
      ds.create_attribute(name, JSON.stringify(values));
    const flatTable = (
      f: any,
      name: string,
      fields: string[],
      rows: number[][],
    ) => {
      f.create_dataset({
        name,
        data: new Float64Array(rows.flat()),
        shape: [rows.length, fields.length],
        dtype: "<f8",
      });
      f.get(name).create_attribute("field_names", JSON.stringify(fields));
      return f.get(name);
    };
    const polygon = (x: number): Uint8Array =>
      encodeWkb({
        type: "Polygon",
        coordinates: [
          [
            [x, 0],
            [x + 2, 0],
            [x + 2, 3],
            [x, 3],
            [x, 0],
          ],
        ],
      });
    const onBase = (edit: (f: any) => void) => editFile(PYTHON, edit);

    /** Both readers agree, and the result is what `check` expects. */
    async function expectBoth(
      bytes: ArrayBuffer,
      check: (labels: Labels) => void,
    ): Promise<void> {
      const eager = await readSlp(bytes, { openVideos: false });
      const streamed = await readStreaming(bytes);
      expect(annotationSnapshot(streamed)).toEqual(annotationSnapshot(eager));
      check(eager);
      check(streamed);
      const lazy = await readStreaming(bytes, true);
      lazy.materialize();
      check(lazy);
    }

    it("a flat /centroids table with sibling string datasets", async () => {
      const bytes = await onBase((f) => {
        flatTable(
          f,
          "centroids",
          ["x", "y", "video", "frame_idx", "track", "instance", "is_predicted"],
          [[4, 5, 0, 0, 0, 0, 1]],
        );
        f.get("centroids").create_attribute("categories", '["from attr"]');
        f.create_dataset({ name: "centroid_names", data: ["c-name"] });
        f.create_dataset({ name: "centroid_sources", data: ["c-source"] });
      });
      await expectBoth(bytes, (labels) => {
        const [c] = labels.centroids as any[];
        expect([c.x, c.y, c.name, c.source]).toEqual([
          4,
          5,
          "c-name",
          "c-source",
        ]);
        expect(c.category).toBe("from attr");
        expect(c.isPredicted).toBe(true);
        expect(c.instance).toBe(labels.labeledFrames[0].instances[0]);
      });
    });

    it("a center-based /bboxes table with strings in attributes", async () => {
      const bytes = await onBase((f) => {
        const table = flatTable(
          f,
          "bboxes",
          [
            "x_center",
            "y_center",
            "width",
            "height",
            "angle",
            "video",
            "frame_idx",
            "track",
            "score",
            "instance",
          ],
          [[10, 20, 4, 6, 0.5, 0, 0, -1, 0.75, 0]],
        );
        attrJson(table, "categories", ["b-cat"]);
        attrJson(table, "names", ["b-name"]);
        attrJson(table, "sources", ["b-source"]);
      });
      await expectBoth(bytes, (labels) => {
        const [b] = labels.bboxes as any[];
        expect([b.x1, b.y1, b.x2, b.y2, b.angle]).toEqual([8, 17, 12, 23, 0.5]);
        expect([b.category, b.name, b.source]).toEqual([
          "b-cat",
          "b-name",
          "b-source",
        ]);
        expect(b.constructor.name).toBe("PredictedBoundingBox");
        expect(b.score).toBe(0.75);
      });
    });

    it("box-shaped ROIs become boxes when the file has no /bboxes", async () => {
      const box = polygon(1);
      const shape = polygon(10);
      const bytes = await onBase((f) => {
        const table = flatTable(
          f,
          "rois",
          [
            "annotation_type",
            "video",
            "frame_idx",
            "track",
            "score",
            "wkb_start",
            "wkb_end",
            "instance",
          ],
          [
            [1, 0, 0, -1, Number.NaN, 0, box.length, 0],
            [0, 0, 0, -1, Number.NaN, box.length, box.length + shape.length, 0],
          ],
        );
        attrJson(table, "categories", ["box-cat", "roi-cat"]);
        attrJson(table, "names", ["box-name", "roi-name"]);
        attrJson(table, "sources", ["box-src", "roi-src"]);
        const wkb = new Uint8Array(box.length + shape.length);
        wkb.set(box);
        wkb.set(shape, box.length);
        f.create_dataset({
          name: "roi_wkb",
          data: wkb,
          shape: [wkb.length],
          dtype: "<B",
        });
      });
      await expectBoth(bytes, (labels) => {
        expect(labels.rois.map((r: any) => r.name)).toEqual(["roi-name"]);
        const [b] = labels.bboxes as any[];
        expect([b.x1, b.y1, b.x2, b.y2, b.name]).toEqual([
          1,
          0,
          3,
          3,
          "box-name",
        ]);
        const instance = labels.labeledFrames[0].instances[0];
        expect(b.instance).toBe(instance);
        expect((labels.rois[0] as any).instance).toBe(instance);
      });
    });

    it("3-D pixel data in another integer type", async () => {
      const bytes = await onBase((f) => {
        flatTable(
          f,
          "label_images",
          [
            "video",
            "frame_idx",
            "height",
            "width",
            "n_objects",
            "objects_start",
          ],
          [
            [0, 0, 1, 2, 0, 0],
            [0, 8, 1, 2, 0, 0],
          ],
        );
        f.create_dataset({
          name: "label_image_data",
          data: new Uint16Array([1, 2, 3, 4]),
          shape: [2, 1, 2],
          dtype: "<H",
        });
      });
      await expectBoth(bytes, (labels) => {
        expect(labels.labelImages.map((li) => Array.from(li.data))).toEqual([
          [1, 2],
          [3, 4],
        ]);
        expect(labels.labelImages[1].data).toBeInstanceOf(Int32Array);
      });
    });

    it("a flat table whose field names don't match its columns", async () => {
      const bytes = await onBase((f) => {
        f.create_dataset({
          name: "bboxes",
          data: new Float64Array([1, 2, 3, 4, 0, 0, 0, -1, 99]),
          shape: [1, 9],
          dtype: "<f8",
        });
        // One column has no name, and one name (score) has no column.
        f.get("bboxes").create_attribute(
          "field_names",
          JSON.stringify([
            "x1",
            "y1",
            "x2",
            "y2",
            "angle",
            "video",
            "frame_idx",
            "track",
          ]),
        );
      });
      await expectBoth(bytes, (labels) => {
        const [b] = labels.bboxes as any[];
        expect([b.x1, b.y1, b.x2, b.y2]).toEqual([1, 2, 3, 4]);
        expect(b.constructor.name).toBe("UserBoundingBox");
      });
    });

    it("a 2.x table with no buffer reads as none, as it always has", async () => {
      const bytes = await onBase((f) => {
        flatTable(
          f,
          "rois",
          [
            "annotation_type",
            "video",
            "frame_idx",
            "track",
            "wkb_start",
            "wkb_end",
          ],
          [[0, 0, 0, -1, 0, 1]],
        );
        flatTable(
          f,
          "masks",
          [
            "height",
            "width",
            "video",
            "frame_idx",
            "track",
            "rle_start",
            "rle_end",
          ],
          [[1, 1, 0, 0, -1, 0, 4]],
        );
        flatTable(
          f,
          "label_images",
          [
            "video",
            "frame_idx",
            "height",
            "width",
            "n_objects",
            "objects_start",
          ],
          [[0, 0, 1, 1, 0, 0]],
        );
      });
      await expectBoth(bytes, (labels) => {
        expect(labels.rois.length).toBe(0);
        expect(labels.masks.length).toBe(0);
        expect(labels.labelImages.length).toBe(0);
      });
    });
  });

  describe("what both readers refuse", () => {
    const BASE = {
      ...ONE_FRAME,
      ...ONE_INSTANCE,
      ...ONE_POINT,
      ...NO_PRED_POINTS,
    };
    const ROI = {
      "rois/video": new Int32Array([0]),
      "rois/frame_idx": new Int32Array([3]),
      "rois/track": new Int32Array([-1]),
      "rois/wkb_start": new Int32Array([0]),
      "rois/wkb_end": new Int32Array([1]),
      "rois/wkb": new Uint8Array(1),
    };
    const MASK = {
      "masks/height": new Int32Array([1]),
      "masks/width": new Int32Array([1]),
      "masks/video": new Int32Array([0]),
      "masks/frame_idx": new Int32Array([3]),
      "masks/track": new Int32Array([-1]),
      "masks/rle_start": new Int32Array([0]),
      "masks/rle_end": new Int32Array([4]),
      "masks/rle": new Uint8Array(4),
    };
    const IMAGE = {
      "label_images/video": new Int32Array([0]),
      "label_images/frame_idx": new Int32Array([3]),
      "label_images/height": new Int32Array([1]),
      "label_images/width": new Int32Array([1]),
      "label_images/n_objects": new Int32Array([0]),
      "label_images/objects_start": new Int32Array([0]),
      "label_images/data_start": new Int32Array([0]),
      "label_images/data_end": new Int32Array([1]),
    };
    const cases: Array<[string, Record<string, ArrayLike<number>>, RegExp]> = [
      [
        "an ROI on a video that doesn't exist",
        { ...ROI, "rois/video": new Int32Array([4]) },
        /\/rois video is not one of the 1 videos/,
      ],
      [
        "an ROI past its buffer",
        { ...ROI, "rois/wkb_end": new Int32Array([9]) },
        /\/rois wkb ranges fall outside the buffer \(1\)/,
      ],
      [
        "an ROI missing a column",
        (({ "rois/track": _t, ...rest }) => rest)(ROI),
        /\/rois is missing column\(s\) track/,
      ],
      [
        "a mask linked to a mask that doesn't exist",
        { ...MASK, "masks/from_predicted": new Int32Array([3]) },
        /\/masks from_predicted is not a mask row/,
      ],
      [
        "a mask past its buffer",
        { ...MASK, "masks/rle_end": new Int32Array([8]) },
        /\/masks rle ranges fall outside the buffer \(4\)/,
      ],
      [
        "label images with no pixel data",
        IMAGE,
        /\/label_images has no pixel data/,
      ],
      [
        "label images past their pixel data",
        {
          ...IMAGE,
          "label_images/data": new Uint8Array(1),
          "label_images/data_end": new Int32Array([5]),
        },
        /\/label_images data ranges fall outside the buffer \(1\)/,
      ],
      [
        "a link to an instance that doesn't exist",
        {
          "bboxes/x1": new Float64Array([1]),
          "bboxes/y1": new Float64Array([1]),
          "bboxes/x2": new Float64Array([2]),
          "bboxes/y2": new Float64Array([2]),
          "bboxes/video": new Int32Array([0]),
          "bboxes/frame_idx": new Int32Array([3]),
          "bboxes/track": new Int32Array([-1]),
          "bboxes/instance": new Int32Array([0]),
          "bboxes/is_predicted": new Uint8Array([0]),
          ...ROI,
          "rois/instance": new Int32Array([9]),
        },
        /\/rois instance is not one of the 1 instances/,
      ],
    ];
    for (const [what, columns, message] of cases) {
      it(what, async () => {
        const bytes = await smallFile({ ...BASE, ...columns });
        await expect(readSlp(bytes, { openVideos: false })).rejects.toThrow(
          message,
        );
        await expect(readStreaming(bytes)).rejects.toThrow(message);
        await expect(readStreaming(bytes, true)).rejects.toThrow(message);
      });
    }
  });

  it("reports a stage for reading annotations", async () => {
    quietWarnings();
    const { file, close } = await workerFile(fixture(COLUMNAR));
    const calls: Array<[number, number, string | undefined]> = [];
    try {
      await readFromStreamingFile(
        file,
        "mice.slp",
        undefined,
        false,
        (c, t, m) => calls.push([c, t, m]),
      );
    } finally {
      close();
    }
    const total = calls[0][1];
    expect(calls.map(([c]) => c)).toEqual(
      Array.from({ length: total + 1 }, (_, i) => i),
    );
    expect(calls.map(([, , m]) => m)).toContain("Reading annotations");
    expect(calls[calls.length - 1]).toEqual([total, total, "Finalizing"]);
  });
});

describe("provenance", () => {
  it("comes from /provenance_json when the metadata attribute dropped it", async () => {
    quietWarnings();
    const bytes = await editFile(COLUMNAR, (f) => {
      const meta = f.get("metadata");
      const json = JSON.parse(meta.attrs.json.value);
      delete json.provenance;
      meta.delete_attribute("json");
      meta.create_attribute("json", JSON.stringify(json));
    });
    const expected = { sleap_version: "stand-in" };
    expect((await readSlp(bytes, { openVideos: false })).provenance).toEqual(
      expected,
    );
    expect(
      (await readSlpLazy(bytes, { openVideos: false })).provenance,
    ).toEqual(expected);
    expect((await readStreaming(bytes)).provenance).toEqual(expected);
    expect((await loadSlpMetadata(bytes)).provenance).toEqual(expected);
  });
});

describe("loading a 3.0 file and saving it", () => {
  /**
   * The embedded videos of `handbuilt_30.slp` are numbered out of list order, with
   * frames in only some of them. A save writes `/frames/video` as the list index beside
   * those numbers, and a 2.x reader then guesses wrong, whichever format was loaded (it
   * happens to `handbuilt_30_2x.slp` too). So these tests use videos that aren't embedded.
   */
  const SAVED = [PYTHON_TWIN, "handbuilt_30_gzip.slp"];

  /** Save and reload, returning the reloaded labels and the file's top-level facts. */
  async function saveAndInspect(labels: Labels) {
    const bytes = new Uint8Array(await saveSlpToBytes(labels)).buffer;
    const reloaded = await readSlp(bytes as ArrayBuffer, { openVideos: false });
    const module = await ready;
    const p = await tmpPath();
    module.FS.writeFile(p, new Uint8Array(bytes));
    const f: any = new H5File(p, "r");
    const facts = {
      formatId: f.get("metadata").attrs.format_id.value as number,
      keys: f.keys() as string[],
      framesIsDataset: f.get("frames").shape !== undefined,
      pointFields: f
        .get("points")
        .metadata.compound_type.members.map((m: { name: string }) => m.name),
    };
    f.close();
    module.FS.unlink(p);
    return { reloaded, facts };
  }

  for (const file of SAVED) {
    for (const [name, read] of READERS) {
      it(`${file}: ${name} then a save keeps the poses, in the 2.x layout`, async () => {
        quietWarnings();
        const loaded = await read(fixture(file));
        const before = poseSnapshot(loaded);
        const { reloaded, facts } = await saveAndInspect(loaded);
        expect(facts.formatId).toBeLessThan(3);
        expect(facts.framesIsDataset).toBe(true);
        expect(facts.keys).toContain("negative_frames");
        const after = poseSnapshot(reloaded);
        expect(after.frames).toEqual(before.frames);
        expect(after.tracks).toEqual(before.tracks);
        expect(after.skeleton).toEqual(before.skeleton);
        expect(after.suggestions).toEqual(before.suggestions);
        expect(after.sessions).toEqual(before.sessions);
        expect(after.provenance).toEqual(before.provenance);
      });

      it(`${file}: ${name} then a save keeps every annotation`, async () => {
        quietWarnings();
        const loaded = await read(fixture(file));
        for (const kind of [
          loaded.rois,
          loaded.masks,
          loaded.bboxes,
          loaded.labelImages,
          loaded.centroids,
        ]) {
          expect(kind.length).toBeGreaterThan(0);
        }
        const before = annotationSnapshot(loaded);
        const { reloaded } = await saveAndInspect(loaded);
        expect(annotationSnapshot(reloaded)).toEqual(before);
      });
    }

    it(`${file}: a streaming lazy load saved without materializing keeps every annotation`, async () => {
      quietWarnings();
      const loaded = await readStreaming(fixture(file), true);
      const { reloaded } = await saveAndInspect(loaded);
      loaded.materialize();
      // A lazy save has never kept per-detection identities or categories, on any format.
      const noLinks = (labels: Labels) => {
        const { rois, bboxes, masks, centroids, labelImages, placement } =
          annotationSnapshot(labels);
        const drop = (row: unknown[]) => row.filter((v) => !isLinks(v));
        return {
          placement,
          labelImages,
          rois: rois.map(drop),
          bboxes: bboxes.map(drop),
          masks: masks.map(drop),
          centroids: centroids.map(drop),
        };
      };
      expect(noLinks(reloaded)).toEqual(noLinks(loaded));
    });

    it(`${file}: a lazy load saved without materializing keeps the poses`, async () => {
      quietWarnings();
      const loaded = await readSlpLazy(fixture(file), { openVideos: false });
      const { reloaded, facts } = await saveAndInspect(loaded);
      expect(facts.formatId).toBeLessThan(3);
      loaded.materialize();
      // A lazy save has never kept per-detection identities or categories, on any format.
      const poses = (labels: Labels) =>
        poseSnapshot(labels).frames.map((f) => [
          f.video,
          f.frameIdx,
          f.isNegative,
          f.instances.map((i) => [i.type, i.score, i.track, i.points]),
        ]);
      expect(poses(reloaded)).toEqual(poses(loaded));
    });

    it(`${file}: the merged streaming writer takes a 3.0 lazy store`, async () => {
      quietWarnings();
      const loaded = await readSlpLazy(fixture(file), { openVideos: false });
      const bytes = await saveSlpMergedFromStores([loaded._lazyDataStore!]);
      const reloaded = await readSlp(
        new Uint8Array(bytes as Uint8Array).buffer as ArrayBuffer,
        { openVideos: false },
      );
      loaded.materialize();
      const frames = (labels: Labels) =>
        poseSnapshot(labels).frames.map((f) => [
          f.video,
          f.frameIdx,
          f.isNegative,
          f.instances.map((i) => i.points),
        ]);
      expect(frames(reloaded)).toEqual(frames(loaded));
    });
  }

  it("what a save loses: unknown columns, record metadata, the labeled flag", async () => {
    quietWarnings();
    const loaded = await readSlp(fixture(COLUMNAR), { openVideos: false });
    const { reloaded, facts } = await saveAndInspect(loaded);
    expect(facts.pointFields).toEqual(["x", "y", "visible", "complete"]);
    // An unlabeled point comes back as an absent one, like any hidden point with no
    // coordinates: visible false, no location.
    const unlabeled = reloaded.labeledFrames
      .flatMap((f) => f.instances)
      .flatMap((i) => i.points)
      .filter((p) => !p.visible && Number.isNaN(p.xy[0]));
    expect(unlabeled.length).toBeGreaterThanOrEqual(2);
    const module = await ready;
    const p = await tmpPath();
    module.FS.writeFile(
      p,
      new Uint8Array(new Uint8Array(await saveSlpToBytes(loaded)).buffer),
    );
    const f: any = new H5File(p, "r");
    expect(f.get("instances").metadata.compound_type).toBeDefined();
    expect(f.get("metadata").attrs.required_features).toBeUndefined();
    f.close();
    module.FS.unlink(p);
  });

  it("the in-place save gate refuses a 3.0 file", () => {
    // What `StreamingH5File.getDatasetMeta` reports for a group of columns.
    const table = onDiskTableFromMeta({ shape: undefined } as never);
    expect(table?.layout).toBe("columnar");
    const sidecars = {
      tracksJson: null,
      suggestionsJson: null,
      metadataJson: null,
      videos: null,
    };
    const gate = checkInPlaceWritable(
      {
        frames: { fields: [], rows: [] },
        instances: { fields: [], rows: [] },
        points: { fields: [], rows: [] },
        predPoints: { fields: [], rows: [] },
      },
      { frames: table },
      { onDisk: sidecars, expected: sidecars },
    );
    expect(gate).toEqual({
      ok: false,
      reason: expect.stringContaining("group of columns"),
    });
    expect(onDiskTableFromMeta(undefined)).toBeUndefined();
    expect(onDiskTableFromMeta(null)).toBeUndefined();
  });

  it("the in-place writer leaves a 3.0 file alone if the gate was skipped", async () => {
    const module = await ready;
    const p = await tmpPath();
    module.FS.writeFile(p, new Uint8Array(fixture(PYTHON)));
    const f: any = new H5File(p, "a");
    const empty = { fields: ["x"], rows: [[1]] };
    expect(() =>
      writeLabelTablesInPlace(f, {
        frames: empty,
        instances: empty,
        points: empty,
        predPoints: empty,
      }),
    ).toThrow(/group of columns \(SLP format 3\.0\)/);
    f.close();

    const self: any = { postMessage() {} };
    const write = new Function(
      "self",
      `${H5_WORKER_CODE}\nreturn writeOneTableInPlaceWorker;`,
    )(self);
    const again: any = new H5File(p, "a");
    expect(() =>
      write(again, "frames", ["x"], new Float64Array([1]), 1, 1, "<i8"),
    ).toThrow(/group of columns \(SLP format 3\.0\)/);
    again.close();
    module.FS.unlink(p);
  });
});
