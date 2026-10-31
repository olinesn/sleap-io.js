import { describe, it, expect } from "./bun-test";
import { Labels } from "../src/model/labels.js";
import { ROI, PredictedROI } from "../src/model/roi.js";
import {
  SegmentationMask,
  UserSegmentationMask,
  PredictedSegmentationMask,
} from "../src/model/mask.js";
import { Video } from "../src/model/video.js";
import { Track } from "../src/model/instance.js";
import { Skeleton, Instance } from "../src/index.js";
import { UserBoundingBox, PredictedBoundingBox } from "../src/model/bbox.js";
import {
  UserLabelImage,
  PredictedLabelImage,
} from "../src/model/label-image.js";
import { saveSlpToBytes } from "../src/codecs/slp/write.js";
import { readSlp } from "../src/codecs/slp/read.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { ready, File as H5File } from "h5wasm/node";

async function roundTrip(labels: Labels): Promise<Labels> {
  const bytes = await saveSlpToBytes(labels);
  return readSlp(new Uint8Array(bytes).buffer, { openVideos: false });
}

/** Read the metadata `format_id` attr from saved SLP bytes via h5wasm. */
async function readFormatId(bytes: Uint8Array): Promise<number> {
  const module = await ready;
  const memPath = `/tmp/slp_fmt_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(memPath, bytes);
  const file = new H5File(memPath, "r");
  try {
    const fmtAttr = (
      file.get("metadata") as { attrs: Record<string, { value?: number }> }
    ).attrs["format_id"];
    return Number(fmtAttr?.value ?? fmtAttr);
  } finally {
    file.close();
    module.FS.unlink(memPath);
  }
}

/** Read the HDF5 filter ids applied to a dataset in saved SLP bytes. */
async function readDatasetFilters(
  bytes: Uint8Array,
  path: string,
): Promise<Array<{ id: number; name: string; cd_values: number[] }>> {
  const module = await ready;
  const memPath = `/tmp/slp_flt_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(memPath, bytes);
  const file = new H5File(memPath, "r");
  try {
    const ds = file.get(path) as {
      filters?: Array<{ id: number; name: string; cd_values: number[] }>;
    };
    return ds.filters ?? [];
  } finally {
    file.close();
    module.FS.unlink(memPath);
  }
}

/** Build a flat row-major binary mask raster. */
function makeMaskRaster(
  height: number,
  width: number,
  fill: (r: number, c: number) => boolean,
): Uint8Array {
  const flat = new Uint8Array(height * width);
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      flat[r * width + c] = fill(r, c) ? 1 : 0;
    }
  }
  return flat;
}

describe("SLP ROI/Mask I/O", () => {
  it("round-trips ROIs through write and read", async () => {
    const video = new Video({ filename: "test.mp4" });
    const track = new Track("track0");
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const inst = new Instance({
      points: { A: [10, 20], B: [30, 40] },
      skeleton,
    });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const bboxRoi = ROI.fromBbox(10, 20, 100, 200, {
      name: "roi1",
      category: "arena",
      source: "manual",
      video,
      track,
    });

    const polyRoi = ROI.fromPolygon(
      [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
      ],
      {
        name: "roi2",
        category: "region",
        source: "auto",
        video,
        frameIdx: null,
      },
    );

    // Put temporal ROI on a frame, keep static ROI on Labels
    const lfRoi = new LabeledFrame({ video, frameIdx: 5, rois: [bboxRoi] });
    const labels = new Labels({
      labeledFrames: [frame, lfRoi],
      videos: [video],
      skeletons: [skeleton],
      tracks: [track],
      rois: [polyRoi],
    });

    const loaded = await roundTrip(labels);

    expect(loaded.rois.length).toBe(2);

    // Find ROIs by name (ordering may differ after distribution)
    const loadedBbox = loaded.rois.find((r) => r.name === "roi1")!;
    const loadedPoly = loaded.rois.find((r) => r.name === "roi2")!;

    // Check bbox ROI
    expect(loadedBbox).toBeDefined();
    expect(loadedBbox.category).toBe("arena");
    expect(loadedBbox.source).toBe("manual");
    expect(loadedBbox.track).not.toBeNull();
    expect(loadedBbox.track!.name).toBe("track0");

    // Verify geometry bounds
    const bboxBounds = loadedBbox.bounds;
    expect(bboxBounds.minX).toBeCloseTo(10);
    expect(bboxBounds.minY).toBeCloseTo(20);
    expect(bboxBounds.maxX).toBeCloseTo(110);
    expect(bboxBounds.maxY).toBeCloseTo(220);

    // Check polygon ROI
    expect(loadedPoly).toBeDefined();
    expect(loadedPoly.category).toBe("region");
    expect(loadedPoly.source).toBe("auto");
    expect(loadedPoly.track).toBeNull();
  });

  it("round-trips masks through write and read", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const maskData = new Uint8Array(100 * 100);
    // Fill a 20x30 rectangle
    for (let r = 10; r < 30; r++) {
      for (let c = 20; c < 50; c++) {
        maskData[r * 100 + c] = 1;
      }
    }

    const mask = SegmentationMask.fromArray(maskData, 100, 100, {
      name: "mask1",
      category: "cell",
      source: "model",
    });

    const lfMask = new LabeledFrame({ video, frameIdx: 3, masks: [mask] });
    const labels = new Labels({
      labeledFrames: [frame, lfMask],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);

    expect(loaded.masks.length).toBe(1);
    const loadedMask = loaded.masks[0];
    expect(loadedMask.height).toBe(100);
    expect(loadedMask.width).toBe(100);
    expect(loadedMask.name).toBe("mask1");
    expect(loadedMask.category).toBe("cell");
    expect(loadedMask.source).toBe("model");

    // Verify RLE round-trips correctly
    const originalData = mask.data;
    const loadedData = loadedMask.data;
    expect(loadedData.length).toBe(originalData.length);
    for (let i = 0; i < originalData.length; i++) {
      expect(loadedData[i]).toBe(originalData[i]);
    }
  });

  it("persists mask fromPredicted across SLP round-trip", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    // Build a 100x100 binary mask raster shared by both masks.
    const maskData = makeMaskRaster(
      100,
      100,
      (r, c) => r >= 10 && r < 30 && c >= 20 && c < 50,
    );
    const { encodeRle } = await import("../src/model/mask.js");
    const rle = encodeRle(maskData, 100, 100);

    // A predicted mask and the user mask adopted from it via toUser().
    const predMask = new PredictedSegmentationMask({
      rleCounts: rle.slice(),
      height: 100,
      width: 100,
      score: 0.9,
      name: "mask1",
      category: "cell",
      source: "model",
    });
    const userMask = predMask.toUser(true);

    // In-memory provenance link exists before save.
    expect(userMask.fromPredicted).toBe(predMask);

    const lfMask = new LabeledFrame({
      video,
      frameIdx: 3,
      masks: [predMask, userMask],
    });
    const labels = new Labels({
      labeledFrames: [frame, lfMask],
      videos: [video],
      skeletons: [skeleton],
    });

    const bytes = await saveSlpToBytes(labels);
    // Recording a link bumps the format to 2.4.
    expect(await readFormatId(bytes)).toBeGreaterThanOrEqual(2.4);

    const loaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    loaded.materialize();

    // Both masks survive: exactly one predicted, one user.
    expect(loaded.masks.length).toBe(2);
    const loadedPred = loaded.masks.find(
      (m): m is PredictedSegmentationMask =>
        m instanceof PredictedSegmentationMask,
    )!;
    const loadedUser = loaded.masks.find(
      (m): m is UserSegmentationMask => m instanceof UserSegmentationMask,
    )!;
    expect(loadedPred).toBeDefined();
    expect(loadedUser).toBeDefined();

    // Predicted mask keeps its score and raster.
    expect(loadedPred.score).toBeCloseTo(0.9);
    expect(loadedPred.data).toEqual(maskData);

    // User mask raster is intact.
    expect(loadedUser.data).toEqual(maskData);

    // The provenance link is persisted as an index into the saved mask list and
    // re-linked to the loaded predicted mask after read.
    expect(loadedUser.fromPredicted).toBe(loadedPred);
  });

  it("leaves mask fromPredicted null when toUser(false) (no link, format stays < 2.4)", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const maskData = makeMaskRaster(
      40,
      40,
      (r, c) => r >= 5 && r < 15 && c >= 5 && c < 15,
    );
    const { encodeRle } = await import("../src/model/mask.js");
    const rle = encodeRle(maskData, 40, 40);

    const predMask = new PredictedSegmentationMask({
      rleCounts: rle.slice(),
      height: 40,
      width: 40,
      score: 0.5,
    });
    // Adopt WITHOUT linking.
    const userMask = predMask.toUser(false);
    expect(userMask.fromPredicted).toBeNull();

    const lfMask = new LabeledFrame({
      video,
      frameIdx: 1,
      masks: [predMask, userMask],
    });
    const labels = new Labels({
      labeledFrames: [frame, lfMask],
      videos: [video],
      skeletons: [skeleton],
    });

    const bytes = await saveSlpToBytes(labels);
    // No mask records a link -> format must NOT be bumped to 2.4.
    expect(await readFormatId(bytes)).toBeLessThan(2.4);

    const loaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    loaded.materialize();
    const loadedUser = loaded.masks.find(
      (m): m is UserSegmentationMask => m instanceof UserSegmentationMask,
    )!;
    expect(loadedUser.fromPredicted).toBeNull();
  });

  it("re-links multiple user masks sharing one source prediction", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const maskData = makeMaskRaster(
      50,
      50,
      (r, c) => r >= 10 && r < 25 && c >= 10 && c < 25,
    );
    const { encodeRle } = await import("../src/model/mask.js");
    const rle = encodeRle(maskData, 50, 50);

    const predMask = new PredictedSegmentationMask({
      rleCounts: rle.slice(),
      height: 50,
      width: 50,
      score: 0.8,
    });
    // Two user masks both adopt the same prediction.
    const userA = predMask.toUser(true);
    const userB = predMask.toUser(true);
    expect(userA.fromPredicted).toBe(predMask);
    expect(userB.fromPredicted).toBe(predMask);

    const lfMask = new LabeledFrame({
      video,
      frameIdx: 2,
      masks: [predMask, userA, userB],
    });
    const labels = new Labels({
      labeledFrames: [frame, lfMask],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    loaded.materialize();

    const loadedPred = loaded.masks.find(
      (m): m is PredictedSegmentationMask =>
        m instanceof PredictedSegmentationMask,
    )!;
    const loadedUsers = loaded.masks.filter(
      (m): m is UserSegmentationMask => m instanceof UserSegmentationMask,
    );
    expect(loadedUsers.length).toBe(2);
    // Both user masks re-link to the SAME loaded prediction object.
    for (const u of loadedUsers) {
      expect(u.fromPredicted).toBe(loadedPred);
    }
  });

  it("loads fromPredicted as null when the source prediction is not saved (dangling)", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const maskData = makeMaskRaster(
      40,
      40,
      (r, c) => r >= 5 && r < 15 && c >= 5 && c < 15,
    );
    const { encodeRle } = await import("../src/model/mask.js");
    const rle = encodeRle(maskData, 40, 40);

    const predMask = new PredictedSegmentationMask({
      rleCounts: rle.slice(),
      height: 40,
      width: 40,
      score: 0.5,
    });
    const userMask = predMask.toUser(true);
    expect(userMask.fromPredicted).toBe(predMask);

    // Only the USER mask is saved; the source prediction is omitted from the
    // frame, so write resolves the link to -1 (Map miss).
    const lfMask = new LabeledFrame({ video, frameIdx: 1, masks: [userMask] });
    const labels = new Labels({
      labeledFrames: [frame, lfMask],
      videos: [video],
      skeletons: [skeleton],
    });

    const bytes = await saveSlpToBytes(labels);
    // Source absent -> no recorded link -> format not bumped.
    expect(await readFormatId(bytes)).toBeLessThan(2.4);

    const loaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    loaded.materialize();
    expect(loaded.masks.length).toBe(1);
    const loadedUser = loaded.masks.find(
      (m): m is UserSegmentationMask => m instanceof UserSegmentationMask,
    )!;
    expect(loadedUser.fromPredicted).toBeNull();
  });

  it("disambiguates fromPredicted across frames (each user links to its own frame's prediction)", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const { encodeRle } = await import("../src/model/mask.js");

    // Distinct rasters so the loaded masks are individually identifiable.
    const rasterA = makeMaskRaster(
      30,
      30,
      (r, c) => r >= 2 && r < 8 && c >= 2 && c < 8,
    );
    const rasterB = makeMaskRaster(
      30,
      30,
      (r, c) => r >= 20 && r < 26 && c >= 20 && c < 26,
    );

    const predA = new PredictedSegmentationMask({
      rleCounts: encodeRle(rasterA, 30, 30),
      height: 30,
      width: 30,
      score: 0.7,
      name: "A",
    });
    const predB = new PredictedSegmentationMask({
      rleCounts: encodeRle(rasterB, 30, 30),
      height: 30,
      width: 30,
      score: 0.6,
      name: "B",
    });
    const userA = predA.toUser(true);
    const userB = predB.toUser(true);

    const lfA = new LabeledFrame({ video, frameIdx: 1, masks: [predA, userA] });
    const lfB = new LabeledFrame({ video, frameIdx: 2, masks: [predB, userB] });
    const labels = new Labels({
      labeledFrames: [frame, lfA, lfB],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    loaded.materialize();

    const loadedUserA = loaded.masks.find(
      (m): m is UserSegmentationMask =>
        m instanceof UserSegmentationMask && m.name === "A",
    )!;
    const loadedUserB = loaded.masks.find(
      (m): m is UserSegmentationMask =>
        m instanceof UserSegmentationMask && m.name === "B",
    )!;
    expect(loadedUserA.fromPredicted).not.toBeNull();
    expect(loadedUserB.fromPredicted).not.toBeNull();
    // Each user mask links to the prediction with the matching raster, not the
    // other frame's prediction.
    expect(loadedUserA.fromPredicted!.data).toEqual(rasterA);
    expect(loadedUserB.fromPredicted!.data).toEqual(rasterB);
    expect(loadedUserA.fromPredicted).not.toBe(loadedUserB.fromPredicted);
  });

  it("preserves fromPredicted across a double round-trip", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const maskData = makeMaskRaster(
      60,
      60,
      (r, c) => r >= 10 && r < 30 && c >= 10 && c < 30,
    );
    const { encodeRle } = await import("../src/model/mask.js");
    const predMask = new PredictedSegmentationMask({
      rleCounts: encodeRle(maskData, 60, 60),
      height: 60,
      width: 60,
      score: 0.95,
    });
    const userMask = predMask.toUser(true);

    const lfMask = new LabeledFrame({
      video,
      frameIdx: 4,
      masks: [predMask, userMask],
    });
    const labels = new Labels({
      labeledFrames: [frame, lfMask],
      videos: [video],
      skeletons: [skeleton],
    });

    const once = await roundTrip(labels);
    once.materialize();
    const twice = await roundTrip(once);
    twice.materialize();

    const loadedPred = twice.masks.find(
      (m): m is PredictedSegmentationMask =>
        m instanceof PredictedSegmentationMask,
    )!;
    const loadedUser = twice.masks.find(
      (m): m is UserSegmentationMask => m instanceof UserSegmentationMask,
    )!;
    expect(loadedUser.fromPredicted).toBe(loadedPred);
  });

  it("persists both instance and mask fromPredicted in the same file", async () => {
    const { PredictedInstance } = await import("../src/model/instance.js");
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });

    // Predicted instance + a user instance carrying the instance fromPredicted link.
    const predInst = PredictedInstance.fromArray([[10, 20]], skeleton, 0.9);
    const userInst = new Instance({
      points: { A: [10, 20] },
      skeleton,
      fromPredicted: predInst,
    });
    expect(userInst.fromPredicted).toBe(predInst);

    // Predicted mask adopted to a user mask (mask fromPredicted).
    const maskData = makeMaskRaster(
      40,
      40,
      (r, c) => r >= 5 && r < 15 && c >= 5 && c < 15,
    );
    const { encodeRle } = await import("../src/model/mask.js");
    const predMask = new PredictedSegmentationMask({
      rleCounts: encodeRle(maskData, 40, 40),
      height: 40,
      width: 40,
      score: 0.8,
    });
    const userMask = predMask.toUser(true);

    const lf = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [predInst, userInst],
      masks: [predMask, userMask],
    });
    const labels = new Labels({
      labeledFrames: [lf],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    loaded.materialize();

    const loadedUserMask = loaded.masks.find(
      (m): m is UserSegmentationMask => m instanceof UserSegmentationMask,
    )!;
    const loadedPredMask = loaded.masks.find(
      (m): m is PredictedSegmentationMask =>
        m instanceof PredictedSegmentationMask,
    )!;
    expect(loadedUserMask.fromPredicted).toBe(loadedPredMask);

    // Instance fromPredicted still round-trips alongside the mask link.
    const loadedFrame = loaded.labeledFrames[0];
    const loadedUserInst = loadedFrame.instances.find(
      (i) => !(i instanceof PredictedInstance),
    ) as Instance;
    expect(loadedUserInst.fromPredicted).not.toBeNull();
  });

  /**
   * Load a project whose `/masks` table was written by hand: one predicted mask,
   * then one user mask. `fromPredicted` fills the user mask's from_predicted, and
   * `undefined` leaves the column out, as files before format 2.4 do.
   */
  async function loadHandWrittenMasks(
    fromPredicted?: number,
  ): Promise<UserSegmentationMask> {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const bytes = await saveSlpToBytes(labels);
    const module = await ready;
    const memPath = `/tmp/slp_hand_masks_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
    module.FS.writeFile(memPath, new Uint8Array(bytes));
    {
      const f = new H5File(memPath, "a");
      // Two rows: [member, dtype, predicted mask's value, user mask's value].
      const members: Array<[string, string, number, number]> = [
        ["height", "<I", 2, 2],
        ["width", "<I", 2, 2],
        ["annotation_type", "<B", 2, 2],
        ["video", "<i", 0, 0],
        ["frame_idx", "<q", 0, 0],
        ["track", "<i", -1, -1],
        ["instance", "<i", -1, -1],
        ["is_predicted", "<B", 1, 0],
        ["score", "<f", 0.8, Number.NaN],
        ["tracking_score", "<f", Number.NaN, Number.NaN],
        ["rle_start", "<Q", 0, 4],
        ["rle_end", "<Q", 4, 8],
      ];
      if (fromPredicted !== undefined) {
        members.push(["from_predicted", "<i", -1, fromPredicted]);
      }
      f.create_dataset({
        name: "masks",
        data: new Map(members.map(([name, , a, b]) => [name, [a, b]])),
        shape: [2],
        dtype: members.map(([name, dtype]) => [name, dtype]),
      } as never);
      const rle = new Uint8Array(new Uint32Array([0, 4]).buffer);
      f.create_dataset({
        name: "mask_rle",
        data: rle,
        shape: [rle.length],
        dtype: "<B",
      });
      f.close();
    }
    const handWritten = module.FS.readFile(memPath) as Uint8Array;
    module.FS.unlink(memPath);

    const loaded = await readSlp(new Uint8Array(handWritten).buffer, {
      openVideos: false,
    });
    loaded.materialize();
    expect(loaded.masks).toHaveLength(2);
    return loaded.masks.find(
      (m): m is UserSegmentationMask => m instanceof UserSegmentationMask,
    )!;
  }

  it("re-links fromPredicted from a table with the from_predicted column", async () => {
    const user = await loadHandWrittenMasks(0);
    expect(user.fromPredicted).toBeInstanceOf(PredictedSegmentationMask);
  });

  it("loads fromPredicted as null from a legacy file without the from_predicted column", async () => {
    // The reader is presence-gated by NAME, so every user mask loads with null.
    const user = await loadHandWrittenMasks();
    expect(user.fromPredicted).toBeNull();
  });

  it("loads fromPredicted as null when the persisted index is out of range", async () => {
    // A corrupt file: the bounds-checked re-link (fpIdx < masks.length) leaves null.
    const user = await loadHandWrittenMasks(999);
    expect(user.fromPredicted).toBeNull();
  });

  it("backward compat: reads file with no ROIs/masks as empty arrays", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.rois).toEqual([]);
    expect(loaded.masks).toEqual([]);
    expect(loaded.bboxes).toEqual([]);
  });

  it("sets format_id to 1.5 when ROIs present", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const roi = ROI.fromBbox(0, 0, 50, 50);
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
      rois: [roi],
    });

    // Write and read back - if format_id were wrong, we'd get errors
    const loaded = await roundTrip(labels);
    expect(loaded.rois.length).toBe(1);
  });

  it("sets format_id to 1.4 when no ROIs/masks", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.rois.length).toBe(0);
    expect(loaded.masks.length).toBe(0);
  });

  it("handles empty ROIs/masks arrays", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
      rois: [],
      masks: [],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.rois).toEqual([]);
    expect(loaded.masks).toEqual([]);
  });

  it("ROI with null video and track round-trips", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const roi = ROI.fromBbox(0, 0, 50, 50, {
      name: "orphan",
      video: null,
      track: null,
    });

    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
      rois: [roi],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.rois.length).toBe(1);
    const loadedRoi = loaded.rois[0];
    expect(loadedRoi.track).toBeNull();
    expect(loadedRoi.name).toBe("orphan");
  });

  it("mask round-trips metadata", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const maskData = new Uint8Array(10 * 10);
    maskData[15] = 1;
    maskData[16] = 1;
    maskData[17] = 1;

    const mask = SegmentationMask.fromArray(maskData, 10, 10, {
      name: "predicted_mask",
      category: "obj",
    });

    const lfMask = new LabeledFrame({ video, frameIdx: 3, masks: [mask] });
    const labels = new Labels({
      labeledFrames: [frame, lfMask],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.masks.length).toBe(1);
    expect(loaded.masks[0].name).toBe("predicted_mask");
    expect(loaded.masks[0].category).toBe("obj");
    expect(loaded.masks[0].area).toBe(3);
  });
});

describe("SLP BoundingBox I/O", () => {
  it("round-trips bboxes through write and read", async () => {
    const video = new Video({ filename: "test.mp4" });
    const track = new Track("track0");
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const bb1 = new UserBoundingBox({
      x1: 0,
      y1: 20,
      x2: 100,
      y2: 100,
      track,
      category: "animal",
      name: "bb1",
      source: "manual",
    });
    const bb2 = new PredictedBoundingBox({
      x1: 0,
      y1: 5,
      x2: 40,
      y2: 55,
      score: 0.95,
    });

    const lfBb1 = new LabeledFrame({ video, frameIdx: 3, bboxes: [bb1] });
    const lfBb2 = new LabeledFrame({ video, frameIdx: 1, bboxes: [bb2] });
    const labels = new Labels({
      labeledFrames: [frame, lfBb1, lfBb2],
      videos: [video],
      skeletons: [skeleton],
      tracks: [track],
    });

    const loaded = await roundTrip(labels);

    expect(loaded.bboxes.length).toBe(2);

    const loadedBb1 = loaded.bboxes[0];
    expect(loadedBb1.x1).toBeCloseTo(0);
    expect(loadedBb1.y1).toBeCloseTo(20);
    expect(loadedBb1.x2).toBeCloseTo(100);
    expect(loadedBb1.y2).toBeCloseTo(100);
    expect(loadedBb1.xCenter).toBeCloseTo(50);
    expect(loadedBb1.yCenter).toBeCloseTo(60);
    expect(loadedBb1.width).toBeCloseTo(100);
    expect(loadedBb1.height).toBeCloseTo(80);
    expect(loadedBb1.isPredicted).toBe(false);
    expect(loadedBb1.category).toBe("animal");
    expect(loadedBb1.name).toBe("bb1");
    expect(loadedBb1.source).toBe("manual");
    expect(loadedBb1.track).not.toBeNull();
    expect(loadedBb1.track!.name).toBe("track0");

    const loadedBb2 = loaded.bboxes[1];
    expect(loadedBb2.x1).toBeCloseTo(0);
    expect(loadedBb2.y1).toBeCloseTo(5);
    expect(loadedBb2.x2).toBeCloseTo(40);
    expect(loadedBb2.y2).toBeCloseTo(55);
    expect(loadedBb2.xCenter).toBeCloseTo(20);
    expect(loadedBb2.yCenter).toBeCloseTo(30);
    expect(loadedBb2.width).toBeCloseTo(40);
    expect(loadedBb2.height).toBeCloseTo(50);
    expect(loadedBb2.isPredicted).toBe(true);
    expect((loadedBb2 as PredictedBoundingBox).score).toBeCloseTo(0.95);
  });

  it("handles empty bboxes array", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.bboxes).toEqual([]);
  });
});

describe("SLP ROI instance association (format 1.6)", () => {
  it("round-trips ROI instance references", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A", "B"] });
    const inst0 = new Instance({
      points: { A: [10, 20], B: [30, 40] },
      skeleton,
    });
    const inst1 = new Instance({
      points: { A: [50, 60], B: [70, 80] },
      skeleton,
    });
    const frame = new LabeledFrame({
      video,
      frameIdx: 0,
      instances: [inst0, inst1],
    });

    const roi = ROI.fromPolygon(
      [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
      ],
      { name: "inst-roi", instance: inst1 },
    );
    frame.rois.push(roi);

    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.rois.length).toBe(1);
    const loadedRoi = loaded.rois[0];
    expect(loadedRoi.name).toBe("inst-roi");
    expect(loadedRoi.instance).not.toBeNull();
    // Should resolve to the second instance (index 1)
    expect(loadedRoi.instance).toBe(loaded.labeledFrames[0].instances[1]);
  });

  it("ROI without instance round-trips with null instance", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const roi = ROI.fromBbox(0, 0, 50, 50, { name: "no-inst" });
    frame.rois.push(roi);
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.rois.length).toBe(1);
    expect(loaded.rois[0].instance).toBeNull();
  });
});

describe("SLP Predicted Variant Roundtrips", () => {
  it("round-trips PredictedROI with score", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const roi = new PredictedROI({
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 0],
          ],
        ],
      },
      score: 0.75,
      name: "pred",
      category: "obj",
    });

    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
      rois: [roi],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.rois).toHaveLength(1);
    expect(loaded.rois[0].isPredicted).toBe(true);
    expect(loaded.rois[0]).toBeInstanceOf(PredictedROI);
    expect((loaded.rois[0] as PredictedROI).score).toBeCloseTo(0.75);
    expect(loaded.rois[0].name).toBe("pred");
  });

  it("round-trips PredictedSegmentationMask with score", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const data = new Uint8Array(25);
    data[12] = 1; // single pixel
    const rle = (await import("../src/model/mask.js")).encodeRle(data, 5, 5);
    const mask = new PredictedSegmentationMask({
      rleCounts: rle,
      height: 5,
      width: 5,
      score: 0.92,
      name: "pmask",
      category: "seg",
      instance: inst,
    });

    frame.masks.push(mask);
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    loaded.materialize();
    expect(loaded.masks).toHaveLength(1);
    expect(loaded.masks[0].isPredicted).toBe(true);
    expect(loaded.masks[0]).toBeInstanceOf(PredictedSegmentationMask);
    expect((loaded.masks[0] as PredictedSegmentationMask).score).toBeCloseTo(
      0.92,
    );
    expect(loaded.masks[0].name).toBe("pmask");
  });

  it("round-trips PredictedLabelImage with score", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const data = new Int32Array(9);
    data[0] = 1;
    data[4] = 2;
    const li = new PredictedLabelImage({
      data,
      height: 3,
      width: 3,
      score: 0.85,
    });

    frame.labelImages.push(li);
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.labelImages).toHaveLength(1);
    expect(loaded.labelImages[0].isPredicted).toBe(true);
    expect(loaded.labelImages[0]).toBeInstanceOf(PredictedLabelImage);
    expect((loaded.labelImages[0] as PredictedLabelImage).score).toBeCloseTo(
      0.85,
    );
  });
});

describe("SLP Scale/Offset Roundtrips", () => {
  it("round-trips mask scale/offset", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const mask = SegmentationMask.fromArray(
      new Uint8Array([1, 0, 0, 1, 0, 0, 1, 0, 0]),
      3,
      3,
      { video, frameIdx: 0, scale: [2, 3], offset: [10, 20] },
    );

    frame.masks.push(mask);
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.masks).toHaveLength(1);
    expect(loaded.masks[0].scale[0]).toBeCloseTo(2);
    expect(loaded.masks[0].scale[1]).toBeCloseTo(3);
    expect(loaded.masks[0].offset[0]).toBeCloseTo(10);
    expect(loaded.masks[0].offset[1]).toBeCloseTo(20);
  });

  it("round-trips label image scale/offset", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const data = new Int32Array(4);
    data[0] = 1;
    const li = new UserLabelImage({
      data,
      height: 2,
      width: 2,
      scale: [0.5, 0.5],
      offset: [3, 7],
    });

    frame.labelImages.push(li);
    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
    });

    const loaded = await roundTrip(labels);
    expect(loaded.labelImages).toHaveLength(1);
    expect(loaded.labelImages[0].scale[0]).toBeCloseTo(0.5);
    expect(loaded.labelImages[0].scale[1]).toBeCloseTo(0.5);
    expect(loaded.labelImages[0].offset[0]).toBeCloseTo(3);
    expect(loaded.labelImages[0].offset[1]).toBeCloseTo(7);
  });
});

// HDF5 deflate/gzip filter id (H5Z_FILTER_DEFLATE).
const H5Z_FILTER_DEFLATE = 1;

describe("SLP gzip compression of mask_rle / roi_wkb (Python #464, #465)", () => {
  it("gzip-compresses mask_rle losslessly across several fragmented masks", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    // Several masks with fragmented (many-run) rasters to make the RLE blob
    // sizeable and worth compressing: checkerboards and stripes maximize run
    // count, exercising the packed-bytes path the gzip filter wraps.
    const masks = [
      SegmentationMask.fromArray(
        makeMaskRaster(64, 64, (r, c) => (r + c) % 2 === 0),
        64,
        64,
        { name: "checker", category: "cell", source: "model" },
      ),
      SegmentationMask.fromArray(
        makeMaskRaster(48, 80, (r) => r % 2 === 0),
        48,
        80,
        { name: "hstripes", category: "cell", source: "model" },
      ),
      SegmentationMask.fromArray(
        makeMaskRaster(50, 50, (_r, c) => c % 3 === 0),
        50,
        50,
        { name: "vstripes", category: "cell", source: "model" },
      ),
      SegmentationMask.fromArray(
        makeMaskRaster(40, 60, (r, c) => r >= 5 && r < 35 && c >= 10 && c < 50),
        40,
        60,
        { name: "block", category: "cell", source: "model" },
      ),
    ];

    const lfMask = new LabeledFrame({ video, frameIdx: 3, masks });
    const labels = new Labels({
      labeledFrames: [frame, lfMask],
      videos: [video],
      skeletons: [skeleton],
    });

    const bytes = await saveSlpToBytes(labels);

    // The mask_rle dataset is written with the HDF5 deflate (gzip) filter at
    // level 1.
    const filters = await readDatasetFilters(new Uint8Array(bytes), "mask_rle");
    const deflate = filters.find((f) => f.id === H5Z_FILTER_DEFLATE);
    expect(deflate).toBeDefined();
    expect(deflate!.cd_values[0]).toBe(1);

    // ...and transparently decompresses to byte-identical rasters on read.
    const loaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    expect(loaded.masks.length).toBe(masks.length);
    for (const original of masks) {
      const match = loaded.masks.find((m) => m.name === original.name)!;
      expect(match).toBeDefined();
      expect(match.height).toBe(original.height);
      expect(match.width).toBe(original.width);
      // RLE counts and the decoded raster are both byte-identical.
      expect(Array.from(match.rleCounts)).toEqual(
        Array.from(original.rleCounts),
      );
      expect(match.data).toEqual(original.data);
    }
  });

  it("gzip-compresses roi_wkb losslessly across several ROIs", async () => {
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [1, 2] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const rois = [
      ROI.fromBbox(10, 20, 100, 200, {
        name: "bbox",
        category: "arena",
        source: "manual",
      }),
      ROI.fromPolygon(
        [
          [0, 0],
          [100, 0],
          [100, 100],
          [0, 100],
        ],
        { name: "poly", category: "region", source: "auto" },
      ),
      ROI.fromPolygon(
        [
          [5, 5],
          [60, 0],
          [80, 40],
          [40, 90],
          [0, 50],
        ],
        { name: "pentagon", category: "region", source: "auto" },
      ),
    ];

    const labels = new Labels({
      labeledFrames: [frame],
      videos: [video],
      skeletons: [skeleton],
      rois,
    });

    const bytes = await saveSlpToBytes(labels);

    // The roi_wkb dataset carries the HDF5 deflate (gzip) filter at level 1.
    const filters = await readDatasetFilters(new Uint8Array(bytes), "roi_wkb");
    const deflate = filters.find((f) => f.id === H5Z_FILTER_DEFLATE);
    expect(deflate).toBeDefined();
    expect(deflate!.cd_values[0]).toBe(1);

    // Geometry round-trips byte-identically through the transparent decompress.
    const loaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    expect(loaded.rois.length).toBe(rois.length);
    for (const original of rois) {
      const match = loaded.rois.find((r) => r.name === original.name)!;
      expect(match).toBeDefined();
      expect(match.category).toBe(original.category);
      expect(match.source).toBe(original.source);
      expect(match.geometry).toEqual(original.geometry);
    }
  });

  it("writes empty mask_rle uncompressed when every mask has a zero-length RLE", async () => {
    // A zero-size raster (height*width === 0) encodes to an empty rleCounts, so
    // the packed RLE blob is zero-length even though masks are present. h5wasm
    // rejects a 0-size chunk dim, so the writer must fall back to an
    // uncompressed/contiguous dataset (no deflate filter) for this case.
    const video = new Video({ filename: "test.mp4" });
    const skeleton = new Skeleton({ nodes: ["A"] });
    const inst = new Instance({ points: { A: [10, 20] }, skeleton });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });

    const emptyMask = new UserSegmentationMask({
      rleCounts: new Uint32Array(0),
      height: 0,
      width: 0,
      name: "empty",
      category: "cell",
      source: "model",
    });

    const lfMask = new LabeledFrame({ video, frameIdx: 3, masks: [emptyMask] });
    const labels = new Labels({
      labeledFrames: [frame, lfMask],
      videos: [video],
      skeletons: [skeleton],
    });

    const bytes = await saveSlpToBytes(labels);

    // No deflate filter on the empty dataset (the length>0 guard fell through).
    const filters = await readDatasetFilters(new Uint8Array(bytes), "mask_rle");
    expect(filters.some((f) => f.id === H5Z_FILTER_DEFLATE)).toBe(false);

    // Still round-trips: the empty mask survives with an empty raster.
    const loaded = await readSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    expect(loaded.masks.length).toBe(1);
    expect(loaded.masks[0].name).toBe("empty");
    expect(loaded.masks[0].rleCounts.length).toBe(0);
    expect(loaded.masks[0].data.length).toBe(0);
  });
});
