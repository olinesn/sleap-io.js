/**
 * Tests for the lazy SLP write fast path (`writeSlpToFileLazy`).
 *
 * These exercise `saveSlpToBytes(lazy)` without explicit materialization —
 * the dispatch in `saveSlpToBytes` should detect lazy mode and route to the
 * fast path that pulls raw column data + per-frame annotation maps directly
 * from `LazyDataStore`.
 */
import { describe, it, expect } from "./bun-test";
import { Labels } from "../src/model/labels.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { Instance, PredictedInstance, Track } from "../src/model/instance.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";
import { UserCentroid, PredictedCentroid } from "../src/model/centroid.js";
import { UserBoundingBox, PredictedBoundingBox } from "../src/model/bbox.js";
import { UserROI } from "../src/model/roi.js";
import { UserSegmentationMask } from "../src/model/mask.js";
import { UserLabelImage } from "../src/model/label-image.js";
import type { LabelImageObjectInfo } from "../src/model/label-image.js";
import { Identity } from "../src/model/identity.js";
import { Embedding } from "../src/model/embedding.js";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { openH5File } from "../src/codecs/slp/h5.js";
import { saveSlpToBytes } from "../src/codecs/slp/write.js";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/** Save eager → read lazy → save lazy → read eager. The middle step is the
 *  lazy fast path; we verify that `lazy.materialize()` is never needed. */
async function lazyRoundTrip(labels: Labels): Promise<Labels> {
  const bytes1 = await saveSlpToBytes(labels);
  const lazy = await readSlpLazy(new Uint8Array(bytes1).buffer, {
    openVideos: false,
  });
  expect(lazy.isLazy).toBe(true);
  const bytes2 = await saveSlpToBytes(lazy);
  // Confirm the lazy save did not materialize as a side effect.
  expect(lazy.isLazy).toBe(true);
  return readSlp(new Uint8Array(bytes2).buffer, { openVideos: false });
}

describe("lazy SLP write fast path", () => {
  it("preserves user instances", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const track = new Track("t1");
    const inst1 = new Instance({
      points: { A: [1, 2], B: [3, 4] },
      skeleton,
      track,
    });
    const inst2 = new Instance({
      points: { A: [5, 6], B: [7, 8] },
      skeleton,
      track,
    });
    const lf0 = new LabeledFrame({ video, frameIdx: 0, instances: [inst1] });
    const lf1 = new LabeledFrame({ video, frameIdx: 1, instances: [inst2] });
    const labels = new Labels({ labeledFrames: [lf0, lf1] });

    const loaded = await lazyRoundTrip(labels);
    expect(loaded.labeledFrames).toHaveLength(2);
    expect(loaded.labeledFrames[0].instances).toHaveLength(1);
    expect(loaded.labeledFrames[1].instances).toHaveLength(1);
    expect(loaded.labeledFrames[0].instances[0].points[0].xy).toEqual([1, 2]);
    expect(loaded.labeledFrames[1].instances[0].points[1].xy).toEqual([7, 8]);
  });

  it("preserves predicted instances mixed with user instances", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const userInst = new Instance({ points: { A: [1, 2] }, skeleton });
    const predInst = PredictedInstance.fromArray([[3, 4]], skeleton, 0.85);
    const lf = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [userInst, predInst],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const loaded = await lazyRoundTrip(labels);
    expect(loaded.labeledFrames[0].instances).toHaveLength(2);
    const loadedUser = loaded.labeledFrames[0].instances.find(
      (i) => !(i instanceof PredictedInstance),
    );
    const loadedPred = loaded.labeledFrames[0].instances.find(
      (i) => i instanceof PredictedInstance,
    ) as PredictedInstance | undefined;
    expect(loadedUser).toBeDefined();
    expect(loadedPred).toBeDefined();
    expect(loadedPred!.score).toBeCloseTo(0.85);
  });

  it("preserves frame-bound centroids", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const c1 = new UserCentroid({ x: 1, y: 2, category: "cell" });
    const c2 = new PredictedCentroid({ x: 3, y: 4, score: 0.9 });
    const lf = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [inst],
      centroids: [c1, c2],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const loaded = await lazyRoundTrip(labels);
    expect(loaded.labeledFrames[0].centroids).toHaveLength(2);
    const xs = loaded.labeledFrames[0].centroids.map((c) => c.x).sort();
    expect(xs).toEqual([1, 3]);
  });

  it("preserves frame-bound bboxes", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [0, 0] }, skeleton });
    const b1 = new UserBoundingBox({
      x1: 0,
      y1: 0,
      x2: 10,
      y2: 10,
      category: "obj",
    });
    const b2 = new PredictedBoundingBox({
      x1: 5,
      y1: 5,
      x2: 15,
      y2: 15,
      score: 0.7,
    });
    const lf = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [inst],
      bboxes: [b1, b2],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const loaded = await lazyRoundTrip(labels);
    expect(loaded.labeledFrames[0].bboxes).toHaveLength(2);
    expect(
      loaded.labeledFrames[0].bboxes.find((b) => b.category === "obj"),
    ).toBeDefined();
  });

  it("preserves frame-bound masks", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [0, 0] }, skeleton });
    const mask = UserSegmentationMask.fromArray(new Uint8Array(16), 4, 4, {
      name: "m1",
      category: "cell",
    });
    const lf = new LabeledFrame({
      video,
      frameIdx: 3,
      instances: [inst],
      masks: [mask],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const loaded = await lazyRoundTrip(labels);
    expect(loaded.labeledFrames[0].masks).toHaveLength(1);
    expect(loaded.labeledFrames[0].masks[0].name).toBe("m1");
    expect(loaded.labeledFrames[0].masks[0].category).toBe("cell");
    expect(loaded.labeledFrames[0].frameIdx).toBe(3);
  });

  it("preserves frame-bound label images", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const track = new Track("t1");
    const inst = new Instance({ points: { A: [0, 0] }, skeleton });
    const li = new UserLabelImage({
      data: new Int32Array([0, 1, 0, 1]),
      height: 2,
      width: 2,
      objects: new Map<number, LabelImageObjectInfo>([
        [1, { track, category: "cell", name: "obj", instance: null }],
      ]),
    });
    const lf = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [inst],
      labelImages: [li],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const loaded = await lazyRoundTrip(labels);
    expect(loaded.labeledFrames[0].labelImages).toHaveLength(1);
    const loadedLi = loaded.labeledFrames[0].labelImages[0];
    expect(loadedLi.height).toBe(2);
    expect(loadedLi.width).toBe(2);
    expect(loadedLi.nObjects).toBe(1);
  });

  it("preserves static ROI video association (Side fix B regression)", async () => {
    // Static ROIs with a video reference must round-trip with the video
    // intact through the lazy fast path. Before Side fix B this was lost
    // because the collector emitted (-1, -1) for all undistributed entries.
    const v1 = new Video({ filename: "v1.mp4" });
    const v2 = new Video({ filename: "v2.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [0, 0] }, skeleton });
    const frameRoi = UserROI.fromBbox(0, 0, 50, 50, {
      video: v1,
      category: "frame-roi",
    });
    const staticRoi = UserROI.fromBbox(0, 0, 100, 100, {
      video: v2,
      category: "arena",
    });
    const lf = new LabeledFrame({
      video: v1,
      frameIdx: 0,
      instances: [inst],
      rois: [frameRoi],
    });
    const labels = new Labels({
      labeledFrames: [lf],
      videos: [v1, v2],
      rois: [staticRoi],
    });

    const loaded = await lazyRoundTrip(labels);

    // Frame-bound ROI preserved on its frame
    expect(loaded.labeledFrames[0].rois).toHaveLength(1);
    expect(loaded.labeledFrames[0].rois[0].category).toBe("frame-roi");

    // Static ROI preserved as static, with video association intact
    expect(loaded.staticRois).toHaveLength(1);
    expect(loaded.staticRois[0].category).toBe("arena");
    expect(loaded.staticRois[0].video).not.toBeNull();
    expect(loaded.staticRois[0].video!.filename).toBe("v2.mp4");
  });

  it("preserves negative frames", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [0, 0] }, skeleton });
    const lfNorm = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const lfNeg = new LabeledFrame({ video, frameIdx: 5, isNegative: true });
    const labels = new Labels({ labeledFrames: [lfNorm, lfNeg] });

    const loaded = await lazyRoundTrip(labels);
    const negFrames = loaded.labeledFrames.filter((f) => f.isNegative);
    expect(negFrames).toHaveLength(1);
    expect(negFrames[0].frameIdx).toBe(5);
  });

  it("preserves instance association on centroid via _instanceIdx (Side fix A)", async () => {
    // After lazy read, the centroid has _instanceIdx set but instance===null.
    // The lazy writer must persist _instanceIdx so the link survives a round
    // trip. After eager read, instance points to the materialized instance.
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const c = new UserCentroid({ x: 1, y: 2, instance: inst });
    const lf = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [inst],
      centroids: [c],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const loaded = await lazyRoundTrip(labels);
    expect(loaded.labeledFrames[0].centroids).toHaveLength(1);
    expect(loaded.labeledFrames[0].instances).toHaveLength(1);
    expect(loaded.labeledFrames[0].centroids[0].instance).toBe(
      loaded.labeledFrames[0].instances[0],
    );
  });

  it("preserves instance association on bbox via _instanceIdx (Side fix A)", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const b = new UserBoundingBox({
      x1: 0,
      y1: 0,
      x2: 10,
      y2: 10,
      instance: inst,
    });
    const lf = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [inst],
      bboxes: [b],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const loaded = await lazyRoundTrip(labels);
    expect(loaded.labeledFrames[0].bboxes[0].instance).toBe(
      loaded.labeledFrames[0].instances[0],
    );
  });

  it("preserves instance association on roi via _instanceIdx (Side fix A)", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const roi = UserROI.fromBbox(0, 0, 50, 50, { instance: inst });
    const lf = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [inst],
      rois: [roi],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const loaded = await lazyRoundTrip(labels);
    expect(loaded.labeledFrames[0].rois[0].instance).toBe(
      loaded.labeledFrames[0].instances[0],
    );
  });

  it("mixed annotation types round-trip in a single frame", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const track = new Track("t1");
    const inst = new Instance({ points: { A: [10, 20] }, skeleton, track });
    const c = new UserCentroid({ x: 1, y: 2 });
    const b = new UserBoundingBox({ x1: 0, y1: 0, x2: 10, y2: 10 });
    const m = UserSegmentationMask.fromArray(new Uint8Array(16), 4, 4);
    const li = new UserLabelImage({
      data: new Int32Array([0, 1]),
      height: 1,
      width: 2,
    });
    const roi = UserROI.fromBbox(0, 0, 50, 50);

    const lf = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [inst],
      centroids: [c],
      bboxes: [b],
      masks: [m],
      labelImages: [li],
      rois: [roi],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const loaded = await lazyRoundTrip(labels);
    const frame = loaded.labeledFrames[0];
    expect(frame.instances).toHaveLength(1);
    expect(frame.centroids).toHaveLength(1);
    expect(frame.bboxes).toHaveLength(1);
    expect(frame.masks).toHaveLength(1);
    expect(frame.labelImages).toHaveLength(1);
    expect(frame.rois).toHaveLength(1);
  });

  it("embed:'source' restores source video paths in the lazy fast path", async () => {
    // Construct a Labels where the (only) video has a sourceVideo set.
    // Save eager → read lazy → save lazy with embed:'source' → read eager.
    // The loaded video should match the source filename.
    const sourceVideo = new Video({ filename: "original.mp4" });
    const wrappedVideo = new Video({
      filename: "wrapper.mp4",
      sourceVideo,
    });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const lf = new LabeledFrame({
      video: wrappedVideo,
      frameIdx: 0,
      instances: [inst],
    });
    const labels = new Labels({ labeledFrames: [lf] });

    const bytes1 = await saveSlpToBytes(labels);
    const lazy = await readSlpLazy(new Uint8Array(bytes1).buffer, {
      openVideos: false,
    });
    expect(lazy.isLazy).toBe(true);
    // Manually attach a sourceVideo to the lazy-loaded video so the swap
    // has something to do (the read path doesn't re-create sourceVideo refs).
    lazy.videos[0].sourceVideo = new Video({ filename: "restored.mp4" });

    const bytes2 = await saveSlpToBytes(lazy, { embed: "source" });
    expect(lazy.isLazy).toBe(true); // still lazy after the source-mode save

    const loaded = await readSlp(new Uint8Array(bytes2).buffer, {
      openVideos: false,
    });
    expect(loaded.videos[0].filename).toBe("restored.mp4");
  });

  it("embed:'all' forces materialization in lazy mode", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const lf = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const labels = new Labels({ labeledFrames: [lf] });

    const bytes1 = await saveSlpToBytes(labels);
    const lazy = await readSlpLazy(new Uint8Array(bytes1).buffer, {
      openVideos: false,
    });
    expect(lazy.isLazy).toBe(true);

    // embed:"all" tries to read pixel data — must materialize first.
    // We can't actually embed pixels (no real video), so embed mode that
    // requires pixel reads will go through the materialize path. Without
    // a backing video file, the embed step itself may throw, so we just
    // verify the materialization side-effect: after the call attempt,
    // lazy.isLazy should become false (materialization happens before
    // embed processing).
    try {
      await saveSlpToBytes(lazy, { embed: "all" });
    } catch {
      // Expected — no real video to embed pixels from.
    }
    expect(lazy.isLazy).toBe(false);
  });
});

describe("lazy SLP write keeps identity and category links", () => {
  const fixture = (name: string): ArrayBuffer =>
    new Uint8Array(
      readFileSync(
        path.join(fileURLToPath(new URL("./data/slp", import.meta.url)), name),
      ),
    ).buffer as ArrayBuffer;
  const closed = { openVideos: false };
  const reload = (bytes: Uint8Array) =>
    readSlp(new Uint8Array(bytes).buffer as ArrayBuffer, closed);

  type Detection = {
    identity?: { name: string } | null;
    identityScore?: number | null;
    identityEmbedding?: { vector: number[] } | null;
    category?: string | null;
    categoryScore?: number | null;
    categoryEmbedding?: { vector: number[] } | null;
  };
  const linksOf = (d: Detection) => ({
    identity: d.identity?.name ?? null,
    identityScore: d.identityScore ?? null,
    identityEmbedding: d.identityEmbedding?.vector ?? null,
    category: d.category ?? null,
    categoryScore: d.categoryScore ?? null,
    categoryEmbedding: d.categoryEmbedding?.vector ?? null,
  });
  /** Every detection's links by kind. Instances in table order; a lazy save may
   * reorder annotations, so those are compared as a set. */
  const allLinks = (labels: Labels) => {
    const set = (list: Detection[]) =>
      list.map((d) => JSON.stringify(linksOf(d))).sort();
    return {
      instances: labels.labeledFrames.flatMap((f) =>
        (f.instances as Detection[]).map(linksOf),
      ),
      centroids: set(labels.centroids as Detection[]),
      masks: set(labels.masks as Detection[]),
      bboxes: set(labels.bboxes as Detection[]),
      rois: set(labels.rois as Detection[]),
    };
  };

  /** Read something out of a saved file, then close it. */
  async function inFile<T>(
    bytes: Uint8Array,
    read: (file: any) => T,
  ): Promise<T> {
    const { file, close } = await openH5File(
      new Uint8Array(bytes).buffer as ArrayBuffer,
    );
    try {
      return read(file);
    } finally {
      close();
    }
  }
  const formatId = (bytes: Uint8Array) =>
    inFile(bytes, (file) => {
      const attrs = file.get("metadata").attrs;
      return Number(attrs.format_id?.value ?? attrs.format_id);
    });

  it.each([
    "py_written_25_identity.slp",
    "py_written_27_category.slp",
    "py_written_30_twin.slp",
    "py_written_30_twin_legacy.slp",
    "handbuilt_30.slp",
    "handbuilt_30_2x.slp",
    "handbuilt_30_gzip.slp",
  ])("%s", async (name) => {
    const want = allLinks(await readSlp(fixture(name), closed));
    const lazy = await readSlpLazy(fixture(name), closed);
    const bytes = await saveSlpToBytes(lazy);
    expect(lazy.isLazy).toBe(true);
    expect(allLinks(await reload(bytes))).toEqual(want);
    // There was something to lose.
    const flat = JSON.stringify(want);
    expect(flat).toMatch(/"(identity|category)":"[^"]/);
  });

  it("the Python identity file: scores and embeddings, and format 2.5", async () => {
    const lazy = await readSlpLazy(
      fixture("py_written_25_identity.slp"),
      closed,
    );
    const bytes = await saveSlpToBytes(lazy);
    const all = (await reload(bytes)).labeledFrames.flatMap((f) => f.instances);
    expect(all.map((i) => i.identity?.name)).toEqual(["A", "B", "A"]);
    expect(all[0].identityScore).toBeCloseTo(0.9, 5);
    expect(all[1].identityEmbedding?.vector).toHaveLength(3);
    expect(await formatId(bytes)).toBe(2.5);
  });

  it("the Python category file: scores and embeddings, and format 2.7", async () => {
    const lazy = await readSlpLazy(
      fixture("py_written_27_category.slp"),
      closed,
    );
    const bytes = await saveSlpToBytes(lazy);
    const all = (await reload(bytes)).labeledFrames.flatMap((f) => f.instances);
    expect(all.map((i) => i.category)).toEqual(["left", "right", "left"]);
    expect(all[1].categoryScore).toBeCloseTo(0.8, 5);
    expect(all[2].categoryEmbedding?.vector).toHaveLength(2);
    expect(await formatId(bytes)).toBe(2.7);
  });

  it.each([
    "py_written_25_identity.slp",
    "py_written_27_category.slp",
  ])("%s: the same tables as an eager save", async (name) => {
    const tables = (bytes: Uint8Array) =>
      inFile(bytes, (file) =>
        [
          "identity/links",
          "embeddings/vectors",
          "embeddings/owner_type",
          "embeddings/owner_id",
          "categories/name",
          "categories/links",
          "embeddings/category_vectors",
          "embeddings/category_owner_type",
          "embeddings/category_owner_id",
        ].map((table) => {
          const value = file.get(table)?.value;
          return value == null ? null : Array.from(value as ArrayLike<unknown>);
        }),
      );
    const eager = await saveSlpToBytes(await readSlp(fixture(name), closed));
    const lazy = await saveSlpToBytes(await readSlpLazy(fixture(name), closed));
    expect(await tables(lazy)).toEqual(await tables(eager));
  });

  /** Instances with a link of one kind each, and the rest with none. */
  function someLinked(): Labels {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const animal = new Identity({ name: "animal" });
    const plain = () =>
      new Instance({ points: { A: [1, 2] }, skeleton }) as Instance & Detection;
    const instances = Array.from({ length: 7 }, plain);
    instances[1].identity = animal;
    instances[1].identityScore = 0.5;
    instances[2].identity = animal; // no score
    instances[3].identityEmbedding = new Embedding([1, 2]); // no identity
    instances[4].category = "left";
    instances[4].categoryScore = 0.25;
    instances[5].categoryEmbedding = new Embedding([3, 4, 5]); // no category
    return new Labels({
      labeledFrames: instances.map(
        (instance, frameIdx) =>
          new LabeledFrame({ video, frameIdx, instances: [instance] }),
      ),
      videos: [video],
      skeletons: [skeleton],
      identities: [animal],
    });
  }

  it("each link stays on its own instance, with the others left without", async () => {
    const want = allLinks(someLinked()).instances;
    const back = await lazyRoundTrip(someLinked());
    expect(allLinks(back).instances).toEqual(want);
    expect(want[0].identity).toBeNull();
    expect(want[2].identityScore).toBeNull();
    expect(want[3].identityEmbedding).toEqual([1, 2]);
    expect(want[6].category).toBeNull();
  });

  it("identity links follow the catalog as it stands", async () => {
    const lazy = await readSlpLazy(
      fixture("py_written_25_identity.slp"),
      closed,
    );
    lazy.identities[0].name = "renamed";
    lazy.identities.push(new Identity({ name: "new" }));
    const back = await reload(await saveSlpToBytes(lazy));
    expect(back.identities.map((i) => i.name)).toEqual(["renamed", "B", "new"]);
    expect(
      back.labeledFrames.flatMap((f) =>
        f.instances.map((i) => i.identity?.name),
      ),
    ).toEqual(["renamed", "B", "renamed"]);
  });

  it("a copy of lazy labels keeps them too, saved lazy or materialized", async () => {
    const name = "py_written_25_identity.slp";
    const want = allLinks(await readSlp(fixture(name), closed));
    const original = await readSlpLazy(fixture(name), closed);

    const copy = original.copy();
    expect(copy.isLazy).toBe(true);
    expect(allLinks(await reload(await saveSlpToBytes(copy)))).toEqual(want);

    const materialized = original.copy();
    materialized.materialize();
    expect(materialized.identities).toContain(
      materialized.labeledFrames[0].instances[0].identity as Identity,
    );
    expect(allLinks(await reload(await saveSlpToBytes(materialized)))).toEqual(
      want,
    );
  });

  it("a link to a row or catalog entry that isn't there attaches to nothing", async () => {
    const lazy = await readSlpLazy(
      fixture("py_written_25_identity.slp"),
      closed,
    );
    const store = lazy._lazyDataStore!;
    store._instanceIdentityLinks = new Map([
      [0, [99, null]], // no such identity
      [99, [0, null]], // no such instance
      [2, [1, 0.5]], // fine
    ]);
    store._instanceEmbeddings = new Map();
    const bytes = await saveSlpToBytes(lazy);
    expect(
      (await reload(bytes)).labeledFrames.flatMap((f) =>
        f.instances.map((i) => i.identity?.name ?? null),
      ),
    ).toEqual([null, null, "B"]);
    expect(
      await inFile(bytes, (file) => file.get("identity/links").shape),
    ).toEqual([1, 4]);
  });

  it("writes nothing new and stays at its format when there are no links", async () => {
    const video = new Video({ filename: "v.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const labels = new Labels({
      labeledFrames: [
        new LabeledFrame({
          video,
          frameIdx: 0,
          instances: [new Instance({ points: { A: [1, 2] }, skeleton })],
        }),
      ],
      videos: [video],
      skeletons: [skeleton],
      identities: [new Identity({ name: "unused" })],
    });
    const eager = await saveSlpToBytes(labels);
    const lazy = await saveSlpToBytes(
      await readSlpLazy(new Uint8Array(eager).buffer as ArrayBuffer, closed),
    );
    expect(await formatId(lazy)).toBe(await formatId(eager));
    expect(await formatId(lazy)).toBeLessThan(2.5);
    expect(
      await inFile(lazy, (file) => [
        file.get("identity/links"),
        file.get("embeddings"),
        file.get("categories"),
      ]),
    ).toEqual([null, null, null]);
  });
});
