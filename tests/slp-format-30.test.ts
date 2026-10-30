/**
 * SLP format 3.0, where every table is a group of columns, read through every path.
 *
 * `handbuilt_30.slp` and `handbuilt_30_2x.slp` hold one project in the two layouts (see
 * `tests/data/slp/gen_format_30_handbuilt_fixture.py`), so the main check is that they
 * read the same. `py_written_30_columnar.slp` is a real file from Python's writer.
 *
 * `readSlpStreaming` needs a browser Worker, so the streaming tests run the real worker
 * code and the real `StreamingH5File` over h5wasm/node, with only the message
 * channel replaced (see {@link workerFile}).
 */
import { afterEach, describe, expect, it, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { ready, File as H5File } from "h5wasm/node";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { readFromStreamingFile } from "../src/codecs/slp/read-streaming.js";
import { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import { H5_WORKER_CODE } from "../src/codecs/slp/h5-worker.js";
import { getH5EmscriptenModule } from "../src/codecs/slp/h5.js";
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

const fixture = (name: string): ArrayBuffer =>
  new Uint8Array(
    readFileSync(
      path.join(fileURLToPath(new URL("./data/slp", import.meta.url)), name),
    ),
  ).buffer as ArrayBuffer;

const COLUMNAR = "handbuilt_30.slp";
const LEGACY = "handbuilt_30_2x.slp";
const PYTHON = "py_written_30_columnar.slp";

afterEach(() => {
  vi.restoreAllMocks();
});

/** Silence (and record) the warnings about what a save won't keep. */
const quietWarnings = () =>
  vi.spyOn(console, "warn").mockImplementation(() => {});

async function tmpPath(): Promise<string> {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    /* already exists */
  }
  return `/tmp/slp30_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
}

/**
 * A real `StreamingH5File` talking to the real worker code, over h5wasm/node. What a
 * browser adds is the `postMessage` boundary and the byte source.
 */
async function workerFile(
  bytes: ArrayBuffer,
): Promise<{ file: StreamingH5File; close: () => void }> {
  const module = await ready;
  const p = await tmpPath();
  module.FS.writeFile(p, new Uint8Array(bytes));
  const h5 = new H5File(p, "r");
  const self: any = {
    postMessage: (msg: unknown) => file.handleMessage({ data: msg }),
  };
  const install = new Function(
    "self",
    `${H5_WORKER_CODE}\nreturn function(f, m) { currentFile = f; h5wasmModule = m; };`,
  )(self);
  install(h5, { Module: await getH5EmscriptenModule() });
  const file: any = Object.create(StreamingH5File.prototype);
  Object.assign(file, {
    worker: {
      postMessage: (message: unknown) => self.onmessage({ data: message }),
    },
    messageId: 0,
    pendingMessages: new Map(),
    _keys: h5.keys(),
    _isOpen: true,
  });
  return {
    file: file as StreamingH5File,
    close: () => {
      h5.close();
      module.FS.unlink(p);
    },
  };
}

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

/**
 * Every way to read a whole project, as (bytes) -> Labels. The streaming reader only
 * reads pose tables, identities, centroids and sessions, on any format.
 */
const FULL_READERS: Array<[string, (bytes: ArrayBuffer) => Promise<Labels>]> = [
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
];
const STREAMING_READERS: Array<
  [string, (bytes: ArrayBuffer) => Promise<Labels>]
> = [
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
    centroids: labels.labeledFrames.flatMap((f) =>
      f.centroids.map((c: any) => [
        c.constructor.name,
        c.x,
        c.y,
        c.z,
        c.name,
        c.source,
        f4(c.score),
        f4(c.trackingScore),
        c.track?.name ?? null,
        c.identity?.name ?? null,
        c.instance ? labels.labeledFrames.indexOf(f) : null,
      ]),
    ),
  };
}

/** What only the eager and lazy readers read: ROIs, boxes, masks, label images. */
function annotationSnapshot(labels: Labels) {
  return {
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
      r.identity?.name ?? null,
      r.instance ? true : (r._instanceIdx ?? null),
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
      b.identity?.name ?? null,
      b.instance ? true : (b._instanceIdx ?? null),
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
      m.identity?.name ?? null,
      m.instance ? true : (m._instanceIdx ?? null),
    ]),
    labelImages: labels.labelImages.map((li: any) => [
      li.constructor.name,
      li.source,
      f4(li.score),
      li.height,
      li.width,
      Array.from(li.data),
      li.scoreMap ? Array.from(li.scoreMap) : null,
      [...li.objects.entries()].map(([id, o]: any) => [
        id,
        o.name,
        o.category,
        f4(o.score),
        f4(o.trackingScore),
        o.track?.name ?? null,
        o.instance ? true : (o._instanceIdx ?? null),
      ]),
    ]),
  };
}

describe("a 3.0 file and its 2.x twin read the same", () => {
  for (const [name, read] of FULL_READERS) {
    it(name, async () => {
      quietWarnings();
      const columnar = await read(fixture(COLUMNAR));
      const legacy = await read(fixture(LEGACY));
      expect(poseSnapshot(columnar)).toEqual(poseSnapshot(legacy));
      expect(annotationSnapshot(columnar)).toEqual(annotationSnapshot(legacy));
    });
  }

  for (const [name, read] of STREAMING_READERS) {
    it(name, async () => {
      quietWarnings();
      const columnar = await read(fixture(COLUMNAR));
      const legacy = await read(fixture(LEGACY));
      expect(poseSnapshot(columnar)).toEqual(poseSnapshot(legacy));
    });
  }

  it("the streaming reader reads the same poses as the eager one", async () => {
    quietWarnings();
    const eager = poseSnapshot(
      await readSlp(fixture(COLUMNAR), { openVideos: false }),
    );
    const streamed = poseSnapshot(await readStreaming(fixture(COLUMNAR)));
    // On any format, it links identities to instances only (not to centroids).
    expect({ ...streamed, centroids: [] }).toEqual({ ...eager, centroids: [] });
    expect(streamed.centroids.length).toBe(eager.centroids.length);
  });

  it("loadSlpMetadata", async () => {
    quietWarnings();
    const columnar = await loadSlpMetadata(fixture(COLUMNAR));
    const legacy = await loadSlpMetadata(fixture(LEGACY));
    expect(columnar.formatId).toBe(3);
    expect(legacy.formatId).toBeCloseTo(2.8, 5);
    expect(columnar.counts).toEqual({
      labeledFrames: 5,
      instances: 4,
      points: 6,
      predictedPoints: 6,
    });
    expect(columnar.counts).toEqual(legacy.counts);
    expect(columnar.version).toBe(legacy.version);
    expect(columnar.skeletons[0].nodeNames).toEqual(
      legacy.skeletons[0].nodeNames,
    );
    expect(columnar.tracks.map((t) => t.name)).toEqual(
      legacy.tracks.map((t) => t.name),
    );
    expect(columnar.videos).toEqual(legacy.videos);
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
    expect(lite.counts.points).toBe(6);
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

describe("what a 3.0 file holds", () => {
  it("poses, links and flags come through with their values", async () => {
    quietWarnings();
    const labels = await readSlp(fixture(COLUMNAR), { openVideos: false });
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
    for (const [, read] of [...FULL_READERS, ...STREAMING_READERS]) {
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

  it("session tables give 3D points, scores and member references", async () => {
    quietWarnings();
    const labels = await readSlp(fixture(COLUMNAR), { openVideos: false });
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
    const lite = await loadSlpMetadata(fixture(COLUMNAR));
    expect(lite.sessions[0].frameGroups?.map((g) => g.frameIdx)).toEqual([
      5, 7,
    ]);
  });

  it("annotations come from their 3.0 columns and buffers", async () => {
    quietWarnings();
    const labels = await readSlp(fixture(COLUMNAR), { openVideos: false });
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

  it("per-detection links reach annotations, not just instances", async () => {
    quietWarnings();
    const labels = await readSlp(fixture(COLUMNAR), { openVideos: false });
    expect(labels.labeledFrames[1].centroids[0].identity?.name).toBe(
      "animal_A",
    );
    expect(labels.labeledFrames[0].masks[0].identity?.name).toBe("animal_B");
    expect(labels.labeledFrames[0].bboxes[0].identity?.name).toBe("animal_A");
    expect(labels.labeledFrames[0].masks[1].category).toBe("rodent");
  });

  it("the file Python really writes reads in every reader, with no warnings", async () => {
    const warn = quietWarnings();
    for (const [, read] of [...FULL_READERS, ...STREAMING_READERS]) {
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

  it("the streaming reader gives it too", async () => {
    const warn = quietWarnings();
    await readStreaming(fixture(COLUMNAR));
    const lines = warn.mock.calls.map((c) => String(c[0]));
    expect(
      lines.some((l) => l.includes("/instances holds future, metadata,")),
    ).toBe(true);
    // The name, without the URL's query.
    expect(lines.every((l) => !l.includes("a=1"))).toBe(true);
  });

  it("buffers and sub-tables a reader handles itself are not called unknown", async () => {
    const warn = quietWarnings();
    await readSlp(fixture(COLUMNAR), { openVideos: false });
    const lines = warn.mock.calls.map((c) => String(c[0]));
    for (const table of ["/rois", "/masks", "/label_images"]) {
      expect(lines.some((l) => l.includes(`${table} holds`))).toBe(false);
    }
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
  const SAVED = "handbuilt_30_gzip.slp";

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

  for (const [name, read] of [...FULL_READERS, ...STREAMING_READERS]) {
    it(`${name} then a save keeps the poses, in the 2.x layout`, async () => {
      quietWarnings();
      const loaded = await read(fixture(SAVED));
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
  }

  it("a lazy load saved without materializing keeps the poses", async () => {
    quietWarnings();
    const loaded = await readSlpLazy(fixture(SAVED), { openVideos: false });
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

  it("the merged streaming writer takes a 3.0 lazy store", async () => {
    quietWarnings();
    const loaded = await readSlpLazy(fixture(SAVED), { openVideos: false });
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
