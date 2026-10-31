/**
 * Python sleap-io opens `.slp` files this library saved with ROIs, masks, label
 * images and bounding boxes in them, and reads the right values. Runs the real
 * sleap-io on real saved files, one process for every file (`beforeAll` would need
 * its own timeout, so the first test that asks starts it).
 *
 * The Python version is pinned, to 0.9.5. `release/0.10`'s reader takes the same
 * tables (`_TABLE_SPECS`, its `legacy` dtypes), which was checked by reading it, not
 * by running it. To try another version on a saved file, see the top of
 * tests/python-read-annotations.py.
 */
import { afterAll, describe, it, expect, setDefaultTimeout } from "./bun-test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { ready, File as H5File } from "h5wasm/node";
import { openSlpWriter, saveSlpToBytes } from "../src/codecs/slp/write.js";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { UserBoundingBox } from "../src/model/bbox.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { UserLabelImage } from "../src/model/label-image.js";
import { Labels } from "../src/model/labels.js";
import type { PredictedSegmentationMask } from "../src/model/mask.js";
import { type Geometry, UserROI } from "../src/model/roi.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";
import {
  ANNOTATION_KINDS,
  type AnnotationKind,
  FRAME_PAST_2_24,
  UNICODE_NAME,
  annotatedLabels,
} from "./annotation-labels.js";

setDefaultTimeout(120_000);

afterAll(async () => {
  if (savedFiles)
    rmSync((await savedFiles).dir, { recursive: true, force: true });
});

const PYTHON_SLEAP_IO = "sleap-io==0.9.5";
const READER = fileURLToPath(
  new URL("./python-read-annotations.py", import.meta.url),
);
/**
 * Written by this library before annotation tables changed (fork/main 63695ff, with
 * `annotatedLabels(undefined, 11)`): flat float32 tables and strings as JSON bytes.
 * That writer is gone, so this file can't be regenerated. Don't.
 */
const OLD_FILE = fileURLToPath(
  new URL("./data/slp/js_flat_annotations.slp", import.meta.url),
);
const FAR_FRAME_IN_OLD_FILE = 11;
/** Written by real Python: see tests/data/slp/gen_annotations_fixture.py. */
const PYTHON_FILE = fileURLToPath(
  new URL("./data/slp/py_written_annotations.slp", import.meta.url),
);

/** Every annotation, the way tests/python-read-annotations.py prints it. */
type Read = Record<string, any>;

/** What the project in annotation-labels.ts holds, written out by hand. */
function expectedAnnotations(far: number = FRAME_PAST_2_24): Read {
  const noIdentity = {
    identity: null,
    identity_score: null,
    category_score: null,
    identity_embedding: null,
    category_embedding: null,
  };
  const noLinks = {
    ...noIdentity,
    name: "",
    category: "",
    source: "",
    track: null,
    tracking_score: null,
    instance: null,
  };
  return {
    n_instances: 3,
    rois: [
      {
        ...noIdentity,
        video: 0,
        frame_idx: 3,
        type: "UserROI",
        wkt: "POLYGON ((0 0, 4 0, 4 3, 0 0))",
        name: UNICODE_NAME,
        category: "arena",
        source: "manual",
        track: "track a",
        tracking_score: 0.5,
        instance: 0,
        score: null,
      },
      {
        ...noLinks,
        video: 1,
        frame_idx: 7,
        type: "PredictedROI",
        wkt: "POINT (1.5 2.5)",
        instance: 2,
        score: 0.75,
      },
      {
        ...noLinks,
        video: 1,
        frame_idx: far,
        type: "UserROI",
        wkt: "POINT (8 9)",
        name: "far",
        score: null,
      },
    ],
    static_rois: [
      {
        ...noLinks,
        video: 1,
        type: "UserROI",
        wkt: "LINESTRING (0 1, 2 3)",
        name: "static",
        score: null,
      },
    ],
    masks: [
      {
        ...noIdentity,
        video: 0,
        frame_idx: 3,
        type: "PredictedSegmentationMask",
        name: UNICODE_NAME,
        category: "cell",
        source: "model",
        track: "track b",
        tracking_score: 0.25,
        instance: 1,
        rle: [1, 2, 3],
        height: 2,
        width: 3,
        scale: [2, 4],
        offset: [10, 20],
        score: 0.75,
        score_map: [
          [0, 0.25, 0.5],
          [0.75, 1, 0.5],
        ],
        score_map_scale: [0.5, 0.25],
        score_map_offset: [1, 2],
        from_predicted: null,
      },
      {
        ...noLinks,
        video: 0,
        frame_idx: 3,
        type: "UserSegmentationMask",
        instance: 0,
        rle: [6],
        height: 2,
        width: 3,
        scale: [1, 1],
        offset: [0, 0],
        score: null,
        score_map: null,
        score_map_scale: [],
        score_map_offset: [],
        from_predicted: 0,
      },
      {
        ...noLinks,
        video: 1,
        frame_idx: far,
        type: "PredictedSegmentationMask",
        rle: [0, 4],
        height: 2,
        width: 2,
        scale: [1, 1],
        offset: [0, 0],
        score: 0.5,
        score_map: null,
        score_map_scale: [1, 1],
        score_map_offset: [0, 0],
        from_predicted: null,
      },
    ],
    label_images: [
      {
        video: 0,
        frame_idx: 3,
        type: "PredictedLabelImage",
        data: [
          [0, 1, 1],
          [2, 2, 0],
        ],
        height: 2,
        width: 3,
        source: "cellpose",
        objects: {
          "1": {
            name: UNICODE_NAME,
            category: "neuron",
            track: "track a",
            score: 0.5,
            tracking_score: 0.25,
            instance: 0,
          },
          "2": {
            name: "",
            category: "",
            track: null,
            score: null,
            tracking_score: null,
            instance: 1,
          },
        },
        scale: [2, 4],
        offset: [10, 20],
        score: 0.75,
        score_map: [
          [0, 0.25, 0.5],
          [0.75, 1, 0.5],
        ],
        score_map_scale: [0.5, 0.25],
        score_map_offset: [1, 2],
      },
      {
        video: 1,
        frame_idx: far,
        type: "UserLabelImage",
        data: [
          [0, 0],
          [0, 0],
        ],
        height: 2,
        width: 2,
        source: "",
        objects: {},
        scale: [1, 1],
        offset: [0, 0],
        score: null,
        score_map: null,
        score_map_scale: [],
        score_map_offset: [],
      },
    ],
    bboxes: [
      {
        ...noIdentity,
        video: 0,
        frame_idx: 3,
        type: "UserBoundingBox",
        xyxy: [1.5, 2.5, 3.25, 4.75],
        angle: 0.5,
        name: UNICODE_NAME,
        category: "mouse",
        source: "manual",
        track: "track a",
        tracking_score: 0.5,
        instance: 0,
        score: null,
      },
      {
        ...noLinks,
        video: 1,
        frame_idx: far,
        type: "PredictedBoundingBox",
        xyxy: [10, 20, 30, 40],
        angle: 0,
        track: "track b",
        score: 0.75,
      },
    ],
    centroids: [
      {
        ...noLinks,
        video: 1,
        frame_idx: 7,
        type: "UserCentroid",
        xy: [1, 2],
        name: "c",
        instance: 2,
      },
    ],
  };
}

/** The expected annotations with only one kind left in. */
function expectedOnly(kind: AnnotationKind): Read {
  const key = kind === "labelImages" ? "label_images" : kind;
  const all = expectedAnnotations();
  const only: Read = { n_instances: all.n_instances };
  for (const name of [
    "rois",
    "static_rois",
    "masks",
    "label_images",
    "bboxes",
    "centroids",
  ]) {
    only[name] =
      name === key || (kind === "rois" && name === "static_rois")
        ? all[name]
        : [];
  }
  return only;
}

/** The same shape as tests/python-read-annotations.py, read through this library. */
function readInThisLibrary(labels: Labels): Read {
  labels.materialize();
  const frames = labels.labeledFrames;
  const instanceIndex = new Map<unknown, number>();
  for (const lf of frames) {
    for (const inst of lf.instances) {
      instanceIndex.set(inst, instanceIndex.size);
    }
  }
  const maskIndex = new Map<unknown, number>();
  for (const lf of frames) {
    for (const mask of lf.masks) maskIndex.set(mask, maskIndex.size);
  }
  const instance = (value: unknown): number | null =>
    value == null ? null : (instanceIndex.get(value) ?? null);
  const nullIfUnset = (value: number | null | undefined): number | null =>
    value == null || Number.isNaN(value) ? null : value;
  const common = (a: any) => ({
    name: a.name,
    category: a.category,
    source: a.source,
    track: a.track?.name ?? null,
    tracking_score: nullIfUnset(a.trackingScore),
    instance: instance(a.instance),
    identity: a.identity?.name ?? null,
    identity_score: nullIfUnset(a.identityScore),
    category_score: nullIfUnset(a.categoryScore),
    identity_embedding: a.identityEmbedding?.vector ?? null,
    category_embedding: a.categoryEmbedding?.vector ?? null,
  });
  const rows = (data: ArrayLike<number>, height: number, width: number) =>
    Array.from({ length: height }, (_, r) =>
      Array.from(data).slice(r * width, (r + 1) * width),
    );
  const scoreMapFields = (a: any) => ({
    score: nullIfUnset(a.isPredicted ? a.score : null),
    score_map: a.scoreMap ? rows(a.scoreMap, a.height, a.width) : null,
    score_map_scale: a.isPredicted ? a.scoreMapScale : [],
    score_map_offset: a.isPredicted ? a.scoreMapOffset : [],
  });
  const roi = (r: any) => ({
    ...common(r),
    type: r.constructor.name,
    wkt: wkt(r.geometry),
    score: nullIfUnset(r.isPredicted ? r.score : null),
  });

  const out: Read = {
    n_instances: instanceIndex.size,
    rois: [],
    masks: [],
    label_images: [],
    bboxes: [],
    centroids: [],
    static_rois: labels.staticRois.map((r) => ({
      ...roi(r),
      video: r.video ? labels.videos.indexOf(r.video) : null,
    })),
  };
  const byPlace = [...frames].sort(
    (a, b) =>
      labels.videos.indexOf(a.video) - labels.videos.indexOf(b.video) ||
      a.frameIdx - b.frameIdx,
  );
  for (const lf of byPlace) {
    const where = {
      video: labels.videos.indexOf(lf.video),
      frame_idx: lf.frameIdx,
    };
    for (const r of lf.rois) out.rois.push({ ...where, ...roi(r) });
    for (const m of lf.masks as any[]) {
      out.masks.push({
        ...where,
        ...common(m),
        type: m.constructor.name,
        rle: Array.from(m.rleCounts),
        height: m.height,
        width: m.width,
        scale: m.scale,
        offset: m.offset,
        ...scoreMapFields(m),
        from_predicted: m.fromPredicted
          ? (maskIndex.get(m.fromPredicted) ?? null)
          : null,
      });
    }
    for (const li of lf.labelImages as any[]) {
      const objects: Read = {};
      for (const [id, info] of li.objects as Map<number, any>) {
        objects[String(id)] = {
          name: info.name,
          category: info.category,
          track: info.track?.name ?? null,
          score: nullIfUnset(info.score),
          tracking_score: nullIfUnset(info.trackingScore),
          instance: instance(info.instance),
        };
      }
      out.label_images.push({
        ...where,
        type: li.constructor.name,
        data: rows(li.data, li.height, li.width),
        height: li.height,
        width: li.width,
        source: li.source,
        objects,
        scale: li.scale,
        offset: li.offset,
        ...scoreMapFields(li),
      });
    }
    for (const b of lf.bboxes as any[]) {
      out.bboxes.push({
        ...where,
        ...common(b),
        type: b.constructor.name,
        xyxy: [b.x1, b.y1, b.x2, b.y2],
        angle: b.angle,
        score: nullIfUnset(b.isPredicted ? b.score : null),
      });
    }
    for (const c of lf.centroids as any[]) {
      out.centroids.push({
        ...where,
        ...common(c),
        type: c.constructor.name,
        xy: [c.x, c.y],
      });
    }
  }
  return out;
}

/** The well-known text of the shapes in annotation-labels.ts, as shapely prints it. */
function wkt(geometry: Geometry): string {
  const point = (p: number[]) => p.join(" ");
  if (geometry.type === "Point")
    return `POINT (${point(geometry.coordinates)})`;
  if (geometry.type === "LineString") {
    return `LINESTRING (${geometry.coordinates.map(point).join(", ")})`;
  }
  if (geometry.type === "Polygon") {
    return `POLYGON ((${geometry.coordinates[0].map(point).join(", ")}))`;
  }
  throw new Error(`no text for ${geometry.type}`);
}

/** Open a saved file with h5wasm for a look at what's on disk. */
async function inspect<T>(
  bytes: Uint8Array,
  look: (file: any) => T,
): Promise<T> {
  const module = await ready;
  const path = `/tmp/annotations_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(path, bytes);
  const file = new H5File(path, "r");
  try {
    return look(file);
  } finally {
    file.close();
    module.FS.unlink(path);
  }
}

/** A polygon big enough to push the next ROI's WKB bytes past 2^24, at an odd offset. */
function bigPolygonLabels(): Labels {
  const skeleton = new Skeleton({ name: "s", nodes: ["a"] });
  const video = new Video({ filename: "a.mp4" });
  const ring: number[][] = [];
  for (let i = 0; i < 1_100_000; i++) ring.push([i, i % 2]);
  ring.push(ring[0]);
  return new Labels({
    skeletons: [skeleton],
    videos: [video],
    labeledFrames: [
      new LabeledFrame({
        video,
        frameIdx: 0,
        rois: [
          new UserROI({
            geometry: { type: "Polygon", coordinates: [ring] },
            name: "big",
          }),
          new UserROI({
            geometry: { type: "Point", coordinates: [5, 6] },
            name: "after",
          }),
        ],
      }),
    ],
  });
}

/** One label image with no objects at all, and a box with no strings. */
function sparseLabels(): Labels {
  const skeleton = new Skeleton({ name: "s", nodes: ["a"] });
  const video = new Video({ filename: "a.mp4" });
  return new Labels({
    skeletons: [skeleton],
    videos: [video],
    labeledFrames: [
      new LabeledFrame({
        video,
        frameIdx: 2,
        labelImages: [
          new UserLabelImage({
            data: new Int32Array([0, 0]),
            height: 1,
            width: 2,
          }),
        ],
        bboxes: [new UserBoundingBox({ x1: 0, y1: 0, x2: 1, y2: 1 })],
      }),
    ],
  });
}

/** The same project written by the two streaming write paths. */
async function streamed(source: Labels, how: "frames" | "store") {
  const writer = await openSlpWriter({
    skeletons: source.skeletons,
    videos: source.videos,
    tracks: source.tracks,
  });
  if (how === "frames") {
    writer.appendFrames(source.labeledFrames);
  } else {
    writer.appendStore(source._lazyDataStore!);
  }
  return new Uint8Array(writer.close());
}

interface PythonFile {
  eager?: Read;
  lazy?: Read;
  resaved?: string;
  error?: Record<string, string>;
  warnings: string[];
}

let savedFiles:
  | Promise<{
      dir: string;
      python: Record<string, PythonFile>;
      bytes: Record<string, Uint8Array>;
    }>
  | undefined;

/** Save every variant, then have real Python read them all, once. */
function pythonReads() {
  savedFiles ??= (async () => {
    const dir = mkdtempSync(join(tmpdir(), "sleap-io-js-annotations-"));
    const all = new Uint8Array(await saveSlpToBytes(annotatedLabels()));
    const lazy = await readSlpLazy(new Uint8Array(all).buffer, {
      openVideos: false,
    });
    const old = new Uint8Array(readFileSync(OLD_FILE));
    const fromPython = new Uint8Array(readFileSync(PYTHON_FILE));
    const bytes: Record<string, Uint8Array> = {
      all,
      lazy: new Uint8Array(await saveSlpToBytes(lazy)),
      streamedFrames: await streamed(annotatedLabels(), "frames"),
      streamedStore: await streamed(lazy, "store"),
      old,
      oldResaved: new Uint8Array(
        await saveSlpToBytes(
          await readSlp(new Uint8Array(old).buffer, { openVideos: false }),
        ),
      ),
      fromPython,
      fromPythonSavedHere: new Uint8Array(
        await saveSlpToBytes(
          await readSlp(new Uint8Array(fromPython).buffer, {
            openVideos: false,
          }),
        ),
      ),
      sparse: new Uint8Array(await saveSlpToBytes(sparseLabels())),
      big: new Uint8Array(await saveSlpToBytes(bigPolygonLabels())),
    };
    for (const kind of ANNOTATION_KINDS) {
      bytes[`only_${kind}`] = new Uint8Array(
        await saveSlpToBytes(annotatedLabels(kind)),
      );
    }
    const paths: Record<string, string> = {};
    for (const [name, data] of Object.entries(bytes)) {
      paths[name] = join(dir, `${name}.slp`);
      writeFileSync(paths[name], data);
    }
    const stdout = execFileSync(
      "uv",
      [
        "run",
        "--no-project",
        "--with",
        PYTHON_SLEAP_IO,
        "--with",
        "shapely",
        "python",
        "-I",
        READER,
        ...Object.values(paths),
      ],
      { encoding: "utf-8", timeout: 120_000, maxBuffer: 64 * 1024 * 1024 },
    );
    const output = JSON.parse(stdout) as {
      version: string;
      files: Record<string, PythonFile>;
    };
    expect(output.version).toBe("0.9.5");
    const python: Record<string, PythonFile> = {};
    for (const [name, path] of Object.entries(paths)) {
      python[name] = output.files[path];
    }
    return { dir, python, bytes };
  })();
  return savedFiles;
}

describe("Python sleap-io reads annotations saved here", () => {
  for (const kind of ANNOTATION_KINDS) {
    it(`opens a file holding only ${kind}, with the right values`, async () => {
      const { python } = await pythonReads();
      const result = python[`only_${kind}`];
      expect(result.error).toBeUndefined();
      expect(result.warnings).toEqual([]);
      expect(result.eager).toEqual(expectedOnly(kind));
      expect(result.lazy).toEqual(expectedOnly(kind));
    });
  }

  it("opens a file holding every kind, with the right values", async () => {
    const { python } = await pythonReads();
    expect(python.all.error).toBeUndefined();
    expect(python.all.warnings).toEqual([]);
    expect(python.all.eager).toEqual(expectedAnnotations());
    expect(python.all.lazy).toEqual(expectedAnnotations());
  });

  it("reads the same from a save of lazily loaded labels", async () => {
    const { python } = await pythonReads();
    expect(python.lazy.error).toBeUndefined();
    expect(python.lazy.eager).toEqual(expectedAnnotations());
  });

  it("reads the same from the streaming writer", async () => {
    const { python } = await pythonReads();
    expect(python.streamedStore.error).toBeUndefined();
    expect(python.streamedStore.eager).toEqual(expectedAnnotations());

    // Frames only: a frame has no static ROIs to hand over, and appendFrames doesn't
    // carry a label image object's link to its instance yet (for any reader).
    const frames = expectedAnnotations();
    frames.static_rois = [];
    for (const info of Object.values<Read>(frames.label_images[0].objects)) {
      info.instance = null;
    }
    expect(python.streamedFrames.error).toBeUndefined();
    expect(python.streamedFrames.eager).toEqual(frames);
  });

  it("reads a file Python wrote the same after a load and save here", async () => {
    const { python } = await pythonReads();
    expect(python.fromPython.error).toBeUndefined();
    expect(python.fromPythonSavedHere.error).toBeUndefined();
    expect(python.fromPythonSavedHere.eager).toEqual(python.fromPython.eager);

    // The comparison above is only as good as what's in the file.
    const read = python.fromPython.eager!;
    expect(read.masks[0].score_map).toHaveLength(4);
    expect(read.masks[0].score_map[0]).toHaveLength(6);
    expect(read.masks[0].score_map_scale).toEqual([2, 2]);
    expect(read.masks[0].score_map_offset).toEqual([1, 1]);
    expect(read.masks[0].identity).toBe("mouse-1");
    expect(read.masks[0].identity_embedding).toEqual([0.5, 0.25]);
    expect(read.masks[0].category_embedding).toEqual([1.5, 2.5, 3.5]);
    expect(read.masks[1].from_predicted).toBe(0);
    expect(read.label_images[0].score_map).toHaveLength(4);
    expect(read.bboxes[1]).toMatchObject({
      identity: "mouse-1",
      identity_score: 0.5,
      category: "rat",
      category_score: 0.25,
      identity_embedding: [0.75, 0.5],
    });
    expect(read.rois[1].identity).toBe("mouse-1");
    expect(read.centroids[1].identity).toBe("mouse-1");
    expect(read.static_rois[0].video).toBe(1);
  });

  it("opens a label image with no object table and a box with no strings", async () => {
    const { python } = await pythonReads();
    expect(python.sparse.error).toBeUndefined();
    const read = python.sparse.eager!;
    expect(read.label_images).toHaveLength(1);
    expect(read.label_images[0].objects).toEqual({});
    expect(read.label_images[0].data).toEqual([[0, 0]]);
    expect(read.bboxes).toHaveLength(1);
    expect(read.bboxes[0]).toMatchObject({
      name: "",
      category: "",
      source: "",
      xyxy: [0, 0, 1, 1],
      frame_idx: 2,
    });
  });

  it("finds a ROI whose bytes start past 2^24", async () => {
    const { python } = await pythonReads();
    expect(python.big.error).toBeUndefined();
    const rois = python.big.eager!.rois;
    expect(rois.map((r: Read) => r.name)).toEqual(["big", "after"]);
    expect(rois[0].wkt).toBe("Polygon: 1100001");
    expect(rois[1].wkt).toBe("POINT (5 6)");
  });

  it("can't open a file saved before this change, until it is saved again", async () => {
    const { python } = await pythonReads();
    expect(python.old.error?.eager).toContain("TypeError");
    expect(python.oldResaved.error).toBeUndefined();
    expect(python.oldResaved.eager).toMatchObject({
      rois: [
        { name: UNICODE_NAME, wkt: "POLYGON ((0 0, 4 0, 4 3, 0 0))" },
        { wkt: "POINT (1.5 2.5)" },
        { name: "far", frame_idx: FAR_FRAME_IN_OLD_FILE },
      ],
    });
  });
});

describe("this library reads the same files", () => {
  it("its own save, with the right values", async () => {
    const { bytes } = await pythonReads();
    const labels = await readSlp(new Uint8Array(bytes.all).buffer, {
      openVideos: false,
    });
    expect(readInThisLibrary(labels)).toEqual(expectedAnnotations());
  });

  it("a lazily loaded save of its own", async () => {
    const { bytes } = await pythonReads();
    const labels = await readSlpLazy(new Uint8Array(bytes.all).buffer, {
      openVideos: false,
    });
    expect(readInThisLibrary(labels)).toEqual(expectedAnnotations());
  });

  it("a file Python wrote", async () => {
    const { python } = await pythonReads();
    const labels = await readSlp(
      new Uint8Array(readFileSync(PYTHON_FILE)).buffer,
      { openVideos: false },
    );
    const read = readInThisLibrary(labels);
    const expected = python.fromPython.eager!;
    // Python gives a score map its shape; here it's flat, on its owner's size.
    for (const table of ["masks", "label_images"]) {
      for (const entry of expected[table]) {
        delete entry.score_map;
      }
      for (const entry of read[table]) {
        delete entry.score_map;
      }
    }
    expect(read).toEqual(expected);
    const mask = labels.labeledFrames[0].masks[0] as PredictedSegmentationMask;
    expect(mask.scoreMap).toHaveLength(24);
    expect(mask.scoreMapScale).toEqual([2, 2]);
    expect(mask.scoreMapOffset).toEqual([1, 1]);
    expect(mask._scoreMapSize).toEqual([4, 6]);
  });

  it("the same file saved again by Python", async () => {
    const { python } = await pythonReads();
    expect(python.all.error).toBeUndefined();
    const resaved = new Uint8Array(readFileSync(python.all.resaved!));
    const labels = await readSlp(resaved.buffer, { openVideos: false });
    expect(readInThisLibrary(labels)).toEqual(expectedAnnotations());
  });

  it("its own older files: flat tables and strings as JSON bytes", async () => {
    const old = new Uint8Array(readFileSync(OLD_FILE));
    // The fixture really is the old layout.
    await inspect(old, (file) => {
      expect(file.get("rois").shape).toEqual([4, 10]);
      expect(file.get("bboxes").shape).toEqual([2, 11]);
      expect(file.get("roi_names").metadata.type).not.toBe(3); // not a string
      expect(file.get("roi_names").attrs.json).toBeDefined();
    });

    const expected = expectedAnnotations(FAR_FRAME_IN_OLD_FILE);
    // The old writer dropped these two; there is nothing in the file to read.
    for (const entry of [expected.masks[0], expected.label_images[0]]) {
      entry.score_map_scale = [1, 1];
      entry.score_map_offset = [0, 0];
    }
    for (const load of [readSlp, readSlpLazy]) {
      const labels = await load(new Uint8Array(old).buffer, {
        openVideos: false,
      });
      expect(readInThisLibrary(labels)).toEqual(expected);
    }
  });

  it("keeps a ROI's bytes straight past 2^24", async () => {
    const { bytes } = await pythonReads();
    const labels = await readSlp(new Uint8Array(bytes.big).buffer, {
      openVideos: false,
    });
    const rois = labels.labeledFrames[0].rois;
    expect(rois[1].name).toBe("after");
    expect(rois[1].geometry).toEqual({ type: "Point", coordinates: [5, 6] });
  });
});

describe("what a save puts on disk", () => {
  /** A compound table's members as `name: <type class><bytes>`, by name. */
  const members = (ds: any): Record<string, string> =>
    Object.fromEntries(
      ds.metadata.compound_type.members.map((m: any) => [
        m.name,
        `${m.type === 0 ? (m.signed ? "i" : "u") : "f"}${m.size}`,
      ]),
    );

  it("gives each table Python's members and dtypes", async () => {
    const { bytes } = await pythonReads();
    const roiTable = {
      annotation_type: "u1",
      video: "i4",
      frame_idx: "i8",
      track: "i4",
      is_predicted: "u1",
      score: "f4",
      tracking_score: "f4",
      wkb_start: "u8",
      wkb_end: "u8",
      instance: "i4",
    };
    const maskTable = {
      height: "u4",
      width: "u4",
      annotation_type: "u1",
      video: "i4",
      frame_idx: "i8",
      track: "i4",
      instance: "i4",
      from_predicted: "i4",
      is_predicted: "u1",
      score: "f4",
      tracking_score: "f4",
      rle_start: "u8",
      rle_end: "u8",
      scale_x: "f4",
      scale_y: "f4",
      offset_x: "f4",
      offset_y: "f4",
    };
    const labelImageTable = {
      video: "i4",
      frame_idx: "i8",
      height: "u4",
      width: "u4",
      n_objects: "u4",
      objects_start: "u4",
      data_start: "u8",
      data_end: "u8",
      is_predicted: "u1",
      score: "f4",
      scale_x: "f4",
      scale_y: "f4",
      offset_x: "f4",
      offset_y: "f4",
    };
    const objectTable = {
      label_id: "i4",
      track: "i4",
      instance: "i4",
      score: "f4",
      tracking_score: "f4",
    };
    const scoreMapTable = (owner: string) => ({
      [owner]: "u4",
      data_start: "u8",
      data_end: "u8",
      height: "u4",
      width: "u4",
      scale_x: "f4",
      scale_y: "f4",
      offset_x: "f4",
      offset_y: "f4",
    });
    await inspect(bytes.all, (file) => {
      expect(members(file.get("rois"))).toEqual(roiTable);
      expect(members(file.get("masks"))).toEqual(maskTable);
      expect(members(file.get("label_images"))).toEqual(labelImageTable);
      expect(members(file.get("label_image_objects"))).toEqual(objectTable);
      expect(members(file.get("mask_score_map_index"))).toEqual(
        scoreMapTable("mask_idx"),
      );
      expect(members(file.get("label_image_score_map_index"))).toEqual(
        scoreMapTable("li_idx"),
      );
    });
  });

  it("writes every string column as UTF-8 strings, not JSON bytes", async () => {
    const { bytes } = await pythonReads();
    await inspect(bytes.all, (file) => {
      for (const path of [
        "roi_categories",
        "roi_names",
        "roi_sources",
        "mask_categories",
        "mask_names",
        "mask_sources",
        "label_image_sources",
        "label_image_obj_categories",
        "label_image_obj_names",
        "bboxes/category",
        "bboxes/name",
        "bboxes/source",
      ]) {
        const ds = file.get(path);
        expect(ds.metadata.type, path).toBe(3); // H5T_STRING
        expect(ds.metadata.vlen, path).toBe(true);
        expect(ds.metadata.cset, path).toBe(1); // UTF-8
        expect(ds.attrs.json, path).toBeUndefined();
      }
    });
  });

  it("writes /bboxes as a group of columns and no root string datasets", async () => {
    const { bytes } = await pythonReads();
    await inspect(bytes.all, (file) => {
      expect(file.get("bboxes").keys().sort()).toEqual(
        [
          "x1",
          "y1",
          "x2",
          "y2",
          "angle",
          "video",
          "frame_idx",
          "track",
          "instance",
          "is_predicted",
          "score",
          "tracking_score",
          "category",
          "name",
          "source",
        ].sort(),
      );
      for (const old of ["bbox_categories", "bbox_names", "bbox_sources"]) {
        expect(file.get(old), old).toBeFalsy();
      }
    });
  });

  it("writes none of this for a project without these annotations", async () => {
    const labels = annotatedLabels("centroids");
    await inspect(new Uint8Array(await saveSlpToBytes(labels)), (file) => {
      const names = file.keys();
      for (const name of [
        "rois",
        "roi_wkb",
        "roi_categories",
        "masks",
        "mask_rle",
        "mask_categories",
        "mask_score_map_index",
        "label_images",
        "label_image_data",
        "label_image_sources",
        "label_image_objects",
        "bboxes",
        "bbox_categories",
      ]) {
        expect(names, name).not.toContain(name);
      }
    });
  });
});
