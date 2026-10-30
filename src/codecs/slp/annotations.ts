/**
 * ROIs, bounding boxes, masks, label images and per-detection links, built from the
 * columns and byte buffers of their tables.
 *
 * The eager and lazy readers (`read.ts`) and the streaming reader (`read-streaming.ts`)
 * get the same tables through different HDF5 APIs. Everything after that lives here, so
 * they build the same objects and refuse the same files. This module has no dependency
 * on any HDF5 library.
 */
import { inflate } from "pako";
import {
  annotationProblems,
  indexOutOfRange,
  reportTableProblems,
  spanOutOfRange,
} from "./columnar.js";
import {
  type BoundingBox,
  PredictedBoundingBox,
  UserBoundingBox,
} from "../../model/bbox.js";
import type { Centroid } from "../../model/centroid.js";
import { Embedding } from "../../model/embedding.js";
import type { Identity } from "../../model/identity.js";
import type {
  Instance,
  PredictedInstance,
  Track,
} from "../../model/instance.js";
import {
  type LabelImage,
  type LabelImageObjectInfo,
  PredictedLabelImage,
  UserLabelImage,
} from "../../model/label-image.js";
import { LabeledFrame } from "../../model/labeled-frame.js";
import type { LazyDataStore, LazyFrameList } from "../../model/lazy.js";
import {
  PredictedSegmentationMask,
  type SegmentationMask,
  UserSegmentationMask,
} from "../../model/mask.js";
import {
  AnnotationType,
  decodeWkb,
  PredictedROI,
  type ROI,
  UserROI,
} from "../../model/roi.js";
import type { Video } from "../../model/video.js";
import { buildVideoIdMap } from "../../model/video-id-map.js";

const textDecoder = new TextDecoder();

/** A table's columns by name, as the readers hand them over. */
export type Rows = Record<string, any[]>;

/** What the annotation tables need to know about the rest of the file. */
export interface TableContext {
  labelsPath: string;
  /** Number of rows in `/instances`. */
  nInstances: number;
}

/** `[annotation, video index, frame index]`, as read from a table's row. */
export type Routed<T> = [T, number, number];

/** Every annotation of a file, in table order, before they reach their frames. */
export interface AnnotationTuples {
  rois: Routed<ROI>[];
  bboxes: Routed<BoundingBox>[];
  masks: Routed<SegmentationMask>[];
  centroids: Routed<Centroid>[];
  labelImages: Routed<LabelImage>[];
}

/** A dataset's values as bytes, whether they came back typed or as a plain array. */
export function toBytes(value: unknown): Uint8Array {
  return value instanceof Uint8Array
    ? value
    : new Uint8Array((value as ArrayLike<number> | null | undefined) ?? []);
}

/**
 * The list of strings in an attribute, which older files keep as JSON text or bytes.
 * `raw` is the attribute as a reader returns it, with or without a `{ value }` wrapper.
 */
export function parseStringListAttr(raw: unknown): string[] {
  if (!raw) return [];
  const value = (raw as { value?: unknown }).value ?? raw;
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return [];
    }
  }
  if (value instanceof Uint8Array) {
    try {
      return JSON.parse(textDecoder.decode(value));
    } catch {
      return [];
    }
  }
  if (Array.isArray(value)) return value.map(String);
  return [];
}

/** A dataset of strings, as the plain strings it holds (`null` if it isn't a list). */
export function stringsFromDataset(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  return value.map((v: unknown) =>
    typeof v === "string" ? v : String(v ?? ""),
  );
}

/** The string columns of a table, wherever its layout keeps them. */
export interface StringColumns {
  categories: ArrayLike<string>;
  names: ArrayLike<string>;
  sources: ArrayLike<string>;
}

export function buildRois(
  table: {
    rows: Rows;
    columnar: boolean;
    /** The geometry buffer the `wkb_start`/`wkb_end` columns index into. */
    wkb: Uint8Array;
  } & StringColumns,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
  instances?: Array<Instance | PredictedInstance>,
): {
  rois: Routed<ROI>[];
  migratedBboxes: Routed<BoundingBox>[];
} {
  const { rows: roisData, wkb: wkbFlat, categories, names, sources } = table;
  const annotationTypes = roisData.annotation_type ?? [];

  if (table.columnar) {
    reportTableProblems(context.labelsPath, [
      ...annotationProblems("rois", roisData, {
        nVideos: videos.length,
        nTracks: tracks.length,
        nInstances: context.nInstances,
      }),
      ...(spanOutOfRange(roisData.wkb_start, roisData.wkb_end, wkbFlat.length)
        ? [`/rois wkb ranges fall outside the buffer (${wkbFlat.length})`]
        : []),
    ]);
  }

  const videoIndices = roisData.video ?? [];
  const frameIndices = roisData.frame_idx ?? [];
  const trackIndices = roisData.track ?? [];
  const scores = roisData.score ?? [];
  const wkbStarts = roisData.wkb_start ?? [];
  const wkbEnds = roisData.wkb_end ?? [];
  const instanceIndices = roisData.instance ?? [];
  // v1.9+ column (may not exist in older files)
  const isPredictedCol = roisData.is_predicted ?? [];
  const trackingScoresCol = roisData.tracking_score ?? [];

  const rois: Routed<ROI>[] = [];
  const migratedBboxes: Routed<BoundingBox>[] = [];

  for (let i = 0; i < annotationTypes.length; i++) {
    const wkbStart = Number(wkbStarts[i]);
    const wkbEnd = Number(wkbEnds[i]);
    const wkbBytes = wkbFlat.slice(wkbStart, wkbEnd);
    const geometry = decodeWkb(wkbBytes);

    const videoIdx = Number(videoIndices[i]);
    const video =
      videoIdx >= 0 && videoIdx < videos.length ? videos[videoIdx] : null;

    const frameIdxVal = Number(frameIndices[i]);

    const trackIdx = Number(trackIndices[i]);
    const track =
      trackIdx >= 0 && trackIdx < tracks.length ? tracks[trackIdx] : null;

    const annotType = Number(annotationTypes[i]);

    const isPred =
      isPredictedCol.length > i ? Number(isPredictedCol[i]) === 1 : false;

    const roiTsVal =
      trackingScoresCol.length > i ? Number(trackingScoresCol[i]) : Number.NaN;
    const roiTrackingScore = Number.isNaN(roiTsVal) ? null : roiTsVal;

    // Migration: annotation_type === 1 (BOUNDING_BOX) -> BoundingBox object
    // Skip predicted ROIs during bbox migration (only migrate user-type box-shaped ROIs)
    if (annotType === AnnotationType.BOUNDING_BOX && !isPred) {
      const tmpRoi = new UserROI({
        geometry,
        name: names[i] ?? "",
        category: categories[i] ?? "",
        source: sources[i] ?? "",
        video,
        track,
      });
      const b = tmpRoi.bounds;
      const scoreVal = Number(scores[i]);
      const bboxScore = Number.isNaN(scoreVal) ? null : scoreVal;

      const bboxOptions = {
        x1: b.minX,
        y1: b.minY,
        x2: b.maxX,
        y2: b.maxY,
        track,
        trackingScore: roiTrackingScore,
        category: categories[i] ?? "",
        name: names[i] ?? "",
        source: sources[i] ?? "",
      };

      let bbox: BoundingBox;
      if (bboxScore !== null) {
        bbox = new PredictedBoundingBox({ ...bboxOptions, score: bboxScore });
      } else {
        bbox = new UserBoundingBox(bboxOptions);
      }

      // Format >= 1.6: resolve instance references for migrated bboxes
      if (instanceIndices.length > 0) {
        const instIdx = Number(instanceIndices[i]);
        if (instances && instIdx >= 0 && instances[instIdx]) {
          bbox.instance = instances[instIdx];
        } else if (instIdx >= 0) {
          bbox._instanceIdx = instIdx;
        }
      }

      migratedBboxes.push([bbox, videoIdx, frameIdxVal]);
    } else {
      const roiOptions = {
        geometry,
        name: names[i] ?? "",
        category: categories[i] ?? "",
        source: sources[i] ?? "",
        video,
        track,
        trackingScore: roiTrackingScore,
      };

      let roi: ROI;
      if (isPred) {
        const scoreVal = Number(scores[i]);
        roi = new PredictedROI({
          ...roiOptions,
          score: Number.isNaN(scoreVal) ? 0 : scoreVal,
        });
      } else {
        roi = new UserROI(roiOptions);
      }

      // Format >= 1.6: resolve instance references
      if (instanceIndices.length > 0) {
        const instIdx = Number(instanceIndices[i]);
        if (instances && instIdx >= 0 && instances[instIdx]) {
          roi.instance = instances[instIdx];
        } else if (instIdx >= 0) {
          roi._instanceIdx = instIdx;
        }
      }

      rois.push([roi, videoIdx, frameIdxVal]);
    }
  }

  return { rois, migratedBboxes };
}

/** How many rows a `/bboxes` table has, in its center-based or corner-based columns. */
export function bboxRowCount(rows: Rows): number {
  const xCenters = rows.x_center ?? [];
  return xCenters.length > 0 ? xCenters.length : (rows.x1 ?? []).length;
}

export function buildBboxes(
  table: { rows: Rows } & StringColumns,
  tracks: Track[],
): Routed<BoundingBox>[] {
  const { rows: bboxesData, categories, names, sources } = table;

  // Legacy format: center-based columns (x_center, y_center, width, height)
  const xCenters = bboxesData.x_center ?? [];
  const isLegacy = xCenters.length > 0;

  // New format: corner-based columns (x1, y1, x2, y2)
  const x1s = bboxesData.x1 ?? [];
  const count = bboxRowCount(bboxesData);

  const yCenters = bboxesData.y_center ?? [];
  const widths = bboxesData.width ?? [];
  const heights = bboxesData.height ?? [];
  const y1s = bboxesData.y1 ?? [];
  const x2s = bboxesData.x2 ?? [];
  const y2s = bboxesData.y2 ?? [];
  const angles = bboxesData.angle ?? [];
  const videoIndices = bboxesData.video ?? [];
  const frameIndices = bboxesData.frame_idx ?? [];
  const trackIndices = bboxesData.track ?? [];
  const bboxScores = bboxesData.score ?? [];
  const instanceIndices = bboxesData.instance ?? [];
  const trackingScores = bboxesData.tracking_score ?? [];
  const isPredictedCol = bboxesData.is_predicted ?? [];

  const bboxes: Routed<BoundingBox>[] = [];
  for (let i = 0; i < count; i++) {
    const videoIdx = Number(videoIndices[i]);

    const frameIdxVal = Number(frameIndices[i]);

    const trackIdx = Number(trackIndices[i]);
    const track =
      trackIdx >= 0 && trackIdx < tracks.length ? tracks[trackIdx] : null;

    const scoreVal = Number(bboxScores[i]);
    const instanceIdx = Number(instanceIndices[i]);

    // Convert legacy center-based to corner-based
    let bx1: number, by1: number, bx2: number, by2: number;
    if (isLegacy) {
      const cx = Number(xCenters[i]);
      const cy = Number(yCenters[i]);
      const w = Number(widths[i]);
      const h = Number(heights[i]);
      bx1 = cx - w / 2;
      by1 = cy - h / 2;
      bx2 = cx + w / 2;
      by2 = cy + h / 2;
    } else {
      bx1 = Number(x1s[i]);
      by1 = Number(y1s[i]);
      bx2 = Number(x2s[i]);
      by2 = Number(y2s[i]);
    }

    const tsVal =
      trackingScores.length > i ? Number(trackingScores[i]) : Number.NaN;
    const trackingScore = Number.isNaN(tsVal) ? null : tsVal;

    const options = {
      x1: bx1,
      y1: by1,
      x2: bx2,
      y2: by2,
      angle: Number(angles[i]),
      track,
      trackingScore,
      category: categories[i] ?? "",
      name: names[i] ?? "",
      source: sources[i] ?? "",
    };

    const isPred =
      isPredictedCol.length > i
        ? Number(isPredictedCol[i]) === 1
        : !Number.isNaN(scoreVal);
    let bbox: BoundingBox;
    if (isPred) {
      bbox = new PredictedBoundingBox({ ...options, score: scoreVal });
    } else {
      bbox = new UserBoundingBox(options);
    }

    if (instanceIdx >= 0) {
      bbox._instanceIdx = instanceIdx;
    }

    bboxes.push([bbox, videoIdx, frameIdxVal]);
  }
  return bboxes;
}

/** The index table of a score-map table, with the bytes it indexes into. */
export interface ScoreMapTable {
  rows: Rows;
  columnar: boolean;
  data: Uint8Array;
}

/**
 * Decompress score maps. Returns a Map from annotation index to
 * { scoreMap, height, width }.
 */
function buildScoreMaps(
  table: "masks/score_maps" | "label_images/score_maps",
  index: ScoreMapTable | null,
  nOwners: number,
  labelsPath: string,
): Map<number, { scoreMap: Float32Array; height: number; width: number }> {
  const result = new Map<
    number,
    { scoreMap: Float32Array; height: number; width: number }
  >();
  if (!index) return result;

  const indexData = index.rows;
  const idxCol = indexData.mask_idx ?? indexData.li_idx ?? [];
  const starts = indexData.data_start ?? [];
  const ends = indexData.data_end ?? [];
  const smHeights = indexData.height ?? [];
  const smWidths = indexData.width ?? [];
  const dataFlat = index.data;

  if (index.columnar) {
    const owner = table.split("/")[0];
    reportTableProblems(labelsPath, [
      ...(indexOutOfRange(idxCol, nOwners, false)
        ? [`/${table} index is not one of the ${nOwners} /${owner}`]
        : []),
      ...(spanOutOfRange(starts, ends, dataFlat.length)
        ? [`/${table} ranges fall outside the buffer (${dataFlat.length})`]
        : []),
    ]);
  }

  for (let i = 0; i < idxCol.length; i++) {
    const annotIdx = Number(idxCol[i]);
    const start = Number(starts[i]);
    const end = Number(ends[i]);
    const h = Number(smHeights[i]);
    const w = Number(smWidths[i]);

    const compressed = dataFlat.slice(start, end);
    const decompressed = inflate(compressed);
    const expectedBytes = h * w * 4; // Float32 = 4 bytes per element
    if (decompressed.byteLength !== expectedBytes) {
      throw new Error(
        `Score map decompression size mismatch: expected ${expectedBytes} bytes, got ${decompressed.byteLength}`,
      );
    }
    const scoreMap = new Float32Array(
      decompressed.buffer.slice(
        decompressed.byteOffset,
        decompressed.byteOffset + decompressed.byteLength,
      ),
    );

    result.set(annotIdx, { scoreMap, height: h, width: w });
  }
  return result;
}

export function buildMasks(
  table: {
    rows: Rows;
    columnar: boolean;
    /** The run-length buffer the `rle_start`/`rle_end` columns index into. */
    rle: Uint8Array;
    scoreMaps: ScoreMapTable | null;
  } & StringColumns,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
): Routed<SegmentationMask>[] {
  const { rows: masksData, rle: rleFlat, categories, names, sources } = table;
  const heights = masksData.height ?? [];

  if (table.columnar) {
    reportTableProblems(context.labelsPath, [
      ...annotationProblems("masks", masksData, {
        nVideos: videos.length,
        nTracks: tracks.length,
        nInstances: context.nInstances,
      }),
      ...(indexOutOfRange(masksData.from_predicted, heights.length)
        ? ["/masks from_predicted is not a mask row"]
        : []),
      ...(spanOutOfRange(masksData.rle_start, masksData.rle_end, rleFlat.length)
        ? [`/masks rle ranges fall outside the buffer (${rleFlat.length})`]
        : []),
    ]);
  }

  const widths = masksData.width ?? [];
  const videoIndices = masksData.video ?? [];
  const frameIndices = masksData.frame_idx ?? [];
  const trackIndices = masksData.track ?? [];
  const rleStarts = masksData.rle_start ?? [];
  const rleEnds = masksData.rle_end ?? [];
  // v1.9+ columns (may not exist in older files)
  const isPredictedCol = masksData.is_predicted ?? [];
  const scoreCol = masksData.score ?? [];
  const instanceCol = masksData.instance ?? [];
  const maskTrackingScoreCol = masksData.tracking_score ?? [];
  // v2.4+: from_predicted provenance index into the flat mask list. Absent in
  // pre-2.4 files (presence-gated by name) -> [] -> per-row fallback -1.
  const fromPredictedCol = masksData.from_predicted ?? [];

  // v2.1+: spatial metadata columns (default to 1.0/0.0 for old files)
  const scaleXCol = masksData.scale_x ?? [];
  const scaleYCol = masksData.scale_y ?? [];
  const offsetXCol = masksData.offset_x ?? [];
  const offsetYCol = masksData.offset_y ?? [];

  const scoreMaps = buildScoreMaps(
    "masks/score_maps",
    table.scoreMaps,
    heights.length,
    context.labelsPath,
  );

  const masks: Routed<SegmentationMask>[] = [];
  // Deferred from_predicted re-link: collect (userMaskIdx, srcIdx) pairs and
  // resolve after ALL masks are built, since a source prediction can appear
  // later in the flat list (mirrors instance from_predicted).
  const fromPredictedPairs: Array<[number, number]> = [];
  for (let i = 0; i < heights.length; i++) {
    const rleStart = Number(rleStarts[i]);
    const rleEnd = Number(rleEnds[i]);
    const rleRaw = rleFlat.slice(rleStart, rleEnd);

    // Convert packed uint8 bytes back to Uint32Array (4 bytes per count, little-endian)
    const numCounts = rleRaw.byteLength / 4;
    const rleCounts = new Uint32Array(numCounts);
    const rleView = new DataView(
      rleRaw.buffer,
      rleRaw.byteOffset,
      rleRaw.byteLength,
    );
    for (let j = 0; j < numCounts; j++) {
      rleCounts[j] = rleView.getUint32(j * 4, true);
    }

    const videoIdx = Number(videoIndices[i]);

    const frameIdxVal = Number(frameIndices[i]);

    const trackIdx = Number(trackIndices[i]);
    const track =
      trackIdx >= 0 && trackIdx < tracks.length ? tracks[trackIdx] : null;

    const scaleX = scaleXCol.length > i ? Number(scaleXCol[i]) : 1;
    const scaleY = scaleYCol.length > i ? Number(scaleYCol[i]) : 1;
    const offsetX = offsetXCol.length > i ? Number(offsetXCol[i]) : 0;
    const offsetY = offsetYCol.length > i ? Number(offsetYCol[i]) : 0;

    const maskTsVal =
      maskTrackingScoreCol.length > i
        ? Number(maskTrackingScoreCol[i])
        : Number.NaN;
    const maskTrackingScore = Number.isNaN(maskTsVal) ? null : maskTsVal;

    const baseOptions = {
      rleCounts,
      height: Number(heights[i]),
      width: Number(widths[i]),
      name: names[i] ?? "",
      category: categories[i] ?? "",
      source: sources[i] ?? "",
      track,
      trackingScore: maskTrackingScore,
      scale: [scaleX, scaleY] as [number, number],
      offset: [offsetX, offsetY] as [number, number],
    };

    // Determine if predicted based on is_predicted column or NaN score fallback
    const isPred =
      isPredictedCol.length > i ? Number(isPredictedCol[i]) === 1 : false;
    let mask: SegmentationMask;

    if (isPred) {
      const scoreVal = scoreCol.length > i ? Number(scoreCol[i]) : 0;
      const sm = scoreMaps.get(i);
      mask = new PredictedSegmentationMask({
        ...baseOptions,
        score: scoreVal,
        scoreMap: sm?.scoreMap ?? null,
      });
    } else {
      mask = new UserSegmentationMask(baseOptions);
      // Collect from_predicted link for deferred re-link (user masks only).
      const fpIdx =
        fromPredictedCol.length > i ? Number(fromPredictedCol[i]) : -1;
      if (fpIdx >= 0) fromPredictedPairs.push([i, fpIdx]);
    }

    // Resolve instance reference
    const instIdx = instanceCol.length > i ? Number(instanceCol[i]) : -1;
    if (instIdx >= 0) {
      mask._instanceIdx = instIdx;
    }

    masks.push([mask, videoIdx, frameIdxVal]);
  }

  // Deferred from_predicted re-link pass. Bounds-checked so out-of-range or -1
  // indices leave fromPredicted at its constructor default (null).
  for (const [i, fpIdx] of fromPredictedPairs) {
    if (fpIdx >= 0 && fpIdx < masks.length) {
      (masks[i][0] as UserSegmentationMask).fromPredicted = masks[
        fpIdx
      ][0] as PredictedSegmentationMask;
    }
  }

  return masks;
}

/** The rows of `/label_images/objects`, with the string columns they don't hold in 2.x. */
export interface LabelImageObjectsTable {
  rows: Rows;
  categories: ArrayLike<string>;
  names: ArrayLike<string>;
}

// Pixel data is eagerly materialized into a standalone Int32Array per
// LabelImage; the HDF5 file is closed by the reader before Labels is returned.
// label-image .data therefore has no dependency on the source file lifetime —
// the Python issue addressed by PR #419 (h5py Dataset closure invalidated by
// Labels.__del__) does not apply here.
export function buildLabelImages(
  table: {
    rows: Rows;
    columnar: boolean;
    sources: ArrayLike<string>;
    /** The pixel data: a blob (1-D) or frames (3-D). `null` if the file has none. */
    data: { value: unknown; shape: number[] } | null;
    objects: LabelImageObjectsTable | null;
    scoreMaps: ScoreMapTable | null;
  },
  videos: Video[],
  tracks: Track[],
  context: TableContext,
  instances?: Array<Instance | PredictedInstance>,
): Routed<LabelImage>[] {
  const { labelsPath } = context;
  const { rows: liData, sources, data, objects: objTable } = table;
  const videoIndices = liData.video ?? [];

  const frameIndices = liData.frame_idx ?? [];
  const heights = liData.height ?? [];
  const widths = liData.width ?? [];
  const nObjectsList = liData.n_objects ?? [];
  const objectsStarts = liData.objects_start ?? [];
  const dataStarts = liData.data_start ?? [];
  const dataEnds = liData.data_end ?? [];

  // v1.9+ columns (may not exist in older files)
  const isPredictedCol = liData.is_predicted ?? [];
  const liScoreCol = liData.score ?? [];

  // v2.1+: spatial metadata columns (default to 1.0/0.0 for old files)
  const liScaleXCol = liData.scale_x ?? [];
  const liScaleYCol = liData.scale_y ?? [];
  const liOffsetXCol = liData.offset_x ?? [];
  const liOffsetYCol = liData.offset_y ?? [];

  // Read pixel data: either blob (1D, v1.8-v2.1) or chunked (3D, v2.2+)
  if (!data) {
    // A 2.x table without pixel data has never been readable.
    if (table.columnar) {
      reportTableProblems(labelsPath, ["/label_images has no pixel data"]);
    }
    return [];
  }
  const dataShape = data.shape ?? [];
  const isChunked = dataShape.length === 3;
  // For blob format: flat Uint8Array of concatenated zlib-compressed frames
  // For chunked format: 3D Int32Array [T, H, W] (decompression handled by HDF5 lib)
  let dataFlat: Uint8Array = new Uint8Array(0);
  let dataChunked: any = null;
  if (isChunked) {
    dataChunked = data.value; // 3D array or flat typed array with shape metadata
  } else {
    dataFlat = toBytes(data.value);
  }

  // The objects table may not exist if all label images have 0 objects
  const objData = objTable?.rows ?? {};
  const objLabelIds = objData.label_id ?? [];
  const objTrackIndices = objData.track ?? [];
  const objInstanceIndices = objData.instance ?? [];
  const objCategories: ArrayLike<string> = objTable?.categories ?? [];
  const objNames: ArrayLike<string> = objTable?.names ?? [];
  const objScoreCol = objData.score ?? [];
  const objTrackingScoreCol = objData.tracking_score ?? [];

  if (table.columnar) {
    const nObjects = objLabelIds.length;
    const problems: string[] = [
      ...(indexOutOfRange(videoIndices, videos.length)
        ? [`/label_images video is not one of the ${videos.length} videos`]
        : []),
      ...(objTable
        ? annotationProblems("label_images/objects", objTable.rows, {
            nVideos: videos.length,
            nTracks: tracks.length,
            nInstances: context.nInstances,
          })
        : []),
    ];
    for (let i = 0; i < videoIndices.length; i++) {
      const start = Number(objectsStarts[i]);
      const end = start + Number(nObjectsList[i]);
      if (start < 0 || end < start || end > nObjects) {
        problems.push(
          `/label_images object ranges fall outside /label_images/objects (${nObjects} rows)`,
        );
        break;
      }
    }
    if (dataShape.length === 3) {
      if (dataShape[0] < videoIndices.length) {
        problems.push(
          `/label_images has ${videoIndices.length} rows but ${dataShape[0]} frames of data`,
        );
      }
    } else if (spanOutOfRange(dataStarts, dataEnds, dataShape[0] ?? 0)) {
      problems.push(
        `/label_images data ranges fall outside the buffer (${dataShape[0] ?? 0})`,
      );
    }
    reportTableProblems(labelsPath, problems);
  }

  const liScoreMaps = buildScoreMaps(
    "label_images/score_maps",
    table.scoreMaps,
    videoIndices.length,
    labelsPath,
  );

  const labelImages: Routed<LabelImage>[] = [];
  for (let i = 0; i < videoIndices.length; i++) {
    const videoIdx = Number(videoIndices[i]);
    const frameIdxVal = Number(frameIndices[i]);
    const height = Number(heights[i]);
    const width = Number(widths[i]);

    // Extract pixel data from blob or chunked format
    let pixelData: Int32Array;
    if (isChunked && dataChunked) {
      // v2.2+: 3D chunked dataset [T, H, W] — frames stored sequentially by the
      // Python writer, so loop index i maps directly to the i-th frame slice.
      const frameSize = height * width;
      if (dataChunked instanceof Int32Array) {
        // Flat typed array with shape [T, H, W] — slice by frame index
        pixelData = new Int32Array(
          dataChunked.buffer,
          dataChunked.byteOffset + i * frameSize * 4,
          frameSize,
        );
      } else if (ArrayBuffer.isView(dataChunked)) {
        // Other typed array — convert to Int32Array
        const offset = i * frameSize;
        pixelData = new Int32Array(frameSize);
        for (let p = 0; p < frameSize; p++) {
          pixelData[p] = (dataChunked as any)[offset + p];
        }
      } else {
        // Fallback: array of arrays or similar
        pixelData = new Int32Array(frameSize);
      }
    } else {
      // v1.8-v2.1: blob format — decompress from flat byte array
      const dataStart = Number(dataStarts[i]);
      const dataEnd = Number(dataEnds[i]);
      const compressed = dataFlat.slice(dataStart, dataEnd);
      const decompressed = inflate(compressed);
      // Convert bytes back to Int32Array (little-endian, matches native).
      // Use buffer.slice() to guarantee 4-byte alignment for Int32Array.
      pixelData = new Int32Array(
        decompressed.buffer.slice(
          decompressed.byteOffset,
          decompressed.byteOffset + decompressed.byteLength,
        ),
      );
    }

    // Build objects map
    const nObj = Number(nObjectsList[i]);
    const objStart = Number(objectsStarts[i]);
    const objects = new Map<number, LabelImageObjectInfo>();
    const deferredInstances = new Map<number, number>();

    for (let j = objStart; j < objStart + nObj; j++) {
      const labelId = Number(objLabelIds[j]);
      const trackIdx = Number(objTrackIndices[j]);
      const track =
        trackIdx >= 0 && trackIdx < tracks.length ? tracks[trackIdx] : null;
      const instIdx = Number(objInstanceIndices[j]);
      let instance: Instance | null = null;

      if (instances && instIdx >= 0 && instances[instIdx]) {
        instance = instances[instIdx] as Instance;
      } else if (instIdx >= 0) {
        deferredInstances.set(labelId, instIdx);
      }

      const objScore = objScoreCol.length > j ? Number(objScoreCol[j]) : null;
      const objTsVal =
        objTrackingScoreCol.length > j ? Number(objTrackingScoreCol[j]) : null;

      objects.set(labelId, {
        track,
        category: objCategories[j] ?? "",
        name: objNames[j] ?? "",
        instance,
        score: objScore !== null && !Number.isNaN(objScore) ? objScore : null,
        trackingScore:
          objTsVal !== null && !Number.isNaN(objTsVal) ? objTsVal : null,
        _instanceIdx: instIdx >= 0 ? instIdx : -1,
      });
    }

    // Determine if predicted
    const isPred =
      isPredictedCol.length > i ? Number(isPredictedCol[i]) === 1 : false;

    // v2.1+: spatial metadata
    const liScaleX = liScaleXCol.length > i ? Number(liScaleXCol[i]) : 1;
    const liScaleY = liScaleYCol.length > i ? Number(liScaleYCol[i]) : 1;
    const liOffsetX = liOffsetXCol.length > i ? Number(liOffsetXCol[i]) : 0;
    const liOffsetY = liOffsetYCol.length > i ? Number(liOffsetYCol[i]) : 0;
    const liScale: [number, number] = [liScaleX, liScaleY];
    const liOffset: [number, number] = [liOffsetX, liOffsetY];

    let li: LabelImage;
    if (isPred) {
      const liScore = liScoreCol.length > i ? Number(liScoreCol[i]) : 0;
      const sm = liScoreMaps.get(i);
      li = new PredictedLabelImage({
        data: pixelData,
        height,
        width,
        objects,
        source: sources[i] ?? "",
        score: liScore,
        scoreMap: sm?.scoreMap ?? null,
        scale: liScale,
        offset: liOffset,
      });
    } else {
      li = new UserLabelImage({
        data: pixelData,
        height,
        width,
        objects,
        source: sources[i] ?? "",
        scale: liScale,
        offset: liOffset,
      });
    }

    if (deferredInstances.size > 0) {
      li._objectInstanceIdxs = deferredInstances;
    }

    labelImages.push([li, videoIdx, frameIdxVal]);
  }
  return labelImages;
}

/** A detection that can carry a per-detection identity/embedding (SLP 2.5). */
interface IdentityBearingRead {
  identity?: Identity | null;
  identityScore?: number | null;
  identityEmbedding?: Embedding | null;
  category?: string | null;
  categoryScore?: number | null;
  categoryEmbedding?: Embedding | null;
}

/**
 * The rows of a per-detection links table — `/identity/links` (SLP 2.5) or
 * `/categories/links` (SLP 2.7) — as `owner_type → owner_id → [idx, score|null]`
 * (NaN score → null). `i8`/`u8` columns are `Number()`-coerced.
 */
export function groupLinks(
  cols: Rows,
  idxField: string,
  scoreField: string,
): Map<number, Map<number, [number, number | null]>> {
  const result = new Map<number, Map<number, [number, number | null]>>();
  const ot = cols.owner_type ?? [];
  const oid = cols.owner_id ?? [];
  const idx = cols[idxField] ?? [];
  const sc = cols[scoreField] ?? [];
  for (let i = 0; i < ot.length; i += 1) {
    const ownerType = Number(ot[i]);
    const ownerId = Number(oid[i]);
    const score = Number(sc[i]);
    let sub = result.get(ownerType);
    if (!sub) {
      sub = new Map();
      result.set(ownerType, sub);
    }
    sub.set(ownerId, [Number(idx[i]), Number.isNaN(score) ? null : score]);
  }
  return result;
}

/**
 * One `/embeddings` triple — a `(rows, dim)` float matrix + parallel join columns — as
 * `owner_type → owner_id → Embedding`. `i8` columns are `Number()`-coerced.
 */
export function groupEmbeddings(
  flat: ArrayLike<number>,
  rows: number,
  dim: number,
  ot: ArrayLike<unknown>,
  oid: ArrayLike<unknown>,
): Map<number, Map<number, Embedding>> {
  const result = new Map<number, Map<number, Embedding>>();
  for (let i = 0; i < rows; i += 1) {
    const ownerType = Number(ot[i]);
    const ownerId = Number(oid[i]);
    const vec = new Array<number>(dim);
    for (let j = 0; j < dim; j += 1) vec[j] = Number(flat[i * dim + j]);
    let sub = result.get(ownerType);
    if (!sub) {
      sub = new Map();
      result.set(ownerType, sub);
    }
    sub.set(ownerId, new Embedding(vec));
  }
  return result;
}

/**
 * Attach per-detection identity + score + embedding onto a modality's ordered
 * detection list (index == `owner_id`), from the `/identity/links` + `/embeddings`
 * maps for one owner type. Mutates the detection objects in place.
 */
function attachIdentityToOwners(
  owners: IdentityBearingRead[],
  identities: Identity[],
  links: Map<number, [number, number | null]> | undefined,
  embs: Map<number, Embedding> | undefined,
): void {
  if (links) {
    for (const [ownerId, [idx, score]] of links) {
      const det = owners[ownerId];
      if (det && idx >= 0 && idx < identities.length) {
        det.identity = identities[idx];
        det.identityScore = score;
      }
    }
  }
  if (embs) {
    for (const [ownerId, emb] of embs) {
      const det = owners[ownerId];
      if (det) det.identityEmbedding = emb;
    }
  }
}

/**
 * Attach per-detection category (string) + score + embedding onto a modality's
 * ordered detection list (index == `owner_id`), from the `/categories` catalog +
 * `/categories/links` + category embeddings for one owner type.
 */
function attachCategoryToOwners(
  owners: IdentityBearingRead[],
  catalog: string[],
  links: Map<number, [number, number | null]> | undefined,
  embs: Map<number, Embedding> | undefined,
): void {
  if (links) {
    for (const [ownerId, [idx, score]] of links) {
      const det = owners[ownerId];
      if (det && idx >= 0 && idx < catalog.length) {
        det.category = catalog[idx];
        det.categoryScore = score;
      }
    }
  }
  if (embs) {
    for (const [ownerId, emb] of embs) {
      const det = owners[ownerId];
      if (det) det.categoryEmbedding = emb;
    }
  }
}

/** All per-detection identity (SLP 2.5) + category (SLP 2.7) join maps + the
 * category catalog, read once. Empty maps on pre-2.5/2.7 files. */
export interface PerDetectionMaps {
  idLinks: Map<number, Map<number, [number, number | null]>>;
  idEmbs: Map<number, Map<number, Embedding>>;
  catLinks: Map<number, Map<number, [number, number | null]>>;
  catEmbs: Map<number, Map<number, Embedding>>;
  categoryCatalog: string[];
}

/** Attach identity + category for one owner type onto its ordered detection list. */
export function attachOwnerType(
  maps: PerDetectionMaps,
  ownerType: number,
  owners: IdentityBearingRead[],
  identities: Identity[],
): void {
  attachIdentityToOwners(
    owners,
    identities,
    maps.idLinks.get(ownerType),
    maps.idEmbs.get(ownerType),
  );
  attachCategoryToOwners(
    owners,
    maps.categoryCatalog,
    maps.catLinks.get(ownerType),
    maps.catEmbs.get(ownerType),
  );
}

/**
 * Attach identity + category to everything that isn't an instance, each owner type by
 * its table order. Instances are built by the reader or the lazy store, so the reader
 * handles them.
 */
export function attachAnnotationOwners(
  maps: PerDetectionMaps,
  annotations: AnnotationTuples,
  identities: Identity[],
): void {
  attachOwnerType(
    maps,
    2,
    annotations.centroids.map((t) => t[0]),
    identities,
  );
  attachOwnerType(
    maps,
    3,
    annotations.masks.map((t) => t[0]),
    identities,
  );
  attachOwnerType(
    maps,
    4,
    annotations.bboxes.map((t) => t[0]),
    identities,
  );
  attachOwnerType(
    maps,
    5,
    annotations.rois.map((t) => t[0]),
    identities,
  );
}

/**
 * Put each annotation on its frame, making a frame for one that isn't in `/frames`
 * (e.g., TrackMate SLPs), and link it to its instance. Returns the ROIs that belong to
 * no frame, which are static ROIs.
 */
export function distributeAnnotations(
  labeledFrames: LabeledFrame[],
  videos: Video[],
  annotations: AnnotationTuples,
  allInstances: Array<Instance | PredictedInstance>,
): ROI[] {
  const {
    rois: roiTuples,
    bboxes: bboxTuples,
    masks: maskTuples,
    centroids: centroidTuples,
    labelImages: labelImageTuples,
  } = annotations;
  const staticRois: ROI[] = [];
  if (
    !roiTuples.length &&
    !bboxTuples.length &&
    !maskTuples.length &&
    !centroidTuples.length &&
    !labelImageTuples.length
  ) {
    return staticRois;
  }

  // Distribute annotations into LabeledFrames using routing tuples
  const frameMap = new Map<string, LabeledFrame>();
  for (const lf of labeledFrames) {
    const vidIdx = videos.indexOf(lf.video);
    frameMap.set(`${vidIdx}:${lf.frameIdx}`, lf);
  }
  const getOrCreateFrame = (vidIdx: number, frameIdx: number): LabeledFrame => {
    const key = `${vidIdx}:${frameIdx}`;
    let lf = frameMap.get(key);
    if (!lf) {
      lf = new LabeledFrame({ video: videos[vidIdx], frameIdx });
      frameMap.set(key, lf);
      labeledFrames.push(lf);
    }
    return lf;
  };

  const distributeTuples = <T>(
    tuples: Routed<T>[],
    push: (lf: LabeledFrame, ann: T) => void,
  ): void => {
    for (const [ann, vidIdx, frameIdx] of tuples) {
      if (vidIdx >= 0 && vidIdx < videos.length && frameIdx >= 0) {
        push(getOrCreateFrame(vidIdx, frameIdx), ann);
      }
    }
  };

  // Distribute ROIs — undistributed ones are static ROIs
  for (const [roi, vidIdx, frameIdx] of roiTuples) {
    if (vidIdx >= 0 && vidIdx < videos.length && frameIdx >= 0) {
      getOrCreateFrame(vidIdx, frameIdx).rois.push(roi);
    } else {
      staticRois.push(roi);
    }
  }

  distributeTuples(bboxTuples, (lf, b) => lf.bboxes.push(b));
  distributeTuples(maskTuples, (lf, m) => lf.masks.push(m));
  distributeTuples(centroidTuples, (lf, c) => lf.centroids.push(c));
  distributeTuples(labelImageTuples, (lf, li) => lf.labelImages.push(li));

  // Resolve deferred instance references (_instanceIdx) to live instances.
  // Mirrors Labels.materialize() for eager mode so bboxes/centroids/masks/
  // labelImages get their .instance set on read (ROIs are already resolved
  // in buildRois since it receives `instances`).
  const resolveInstanceRef = (ann: {
    _instanceIdx: number | null;
    instance: Instance | null;
  }): void => {
    if (
      ann._instanceIdx !== null &&
      ann._instanceIdx >= 0 &&
      allInstances[ann._instanceIdx]
    ) {
      ann.instance = allInstances[ann._instanceIdx];
      ann._instanceIdx = null;
    }
  };
  for (const lf of labeledFrames) {
    for (const b of lf.bboxes) resolveInstanceRef(b);
    for (const c of lf.centroids) resolveInstanceRef(c);
    for (const m of lf.masks) resolveInstanceRef(m);
    for (const r of lf.rois) resolveInstanceRef(r);
    for (const li of lf.labelImages) {
      if (li._objectInstanceIdxs) {
        for (const [labelId, instIdx] of li._objectInstanceIdxs) {
          const obj = li.objects.get(labelId);
          if (obj && instIdx >= 0 && allInstances[instIdx]) {
            obj.instance = allInstances[instIdx];
          }
        }
        li._objectInstanceIdxs = null;
      }
    }
  }
  return staticRois;
}

/**
 * Hand annotations to a lazy store, which attaches them as frames materialize. They
 * stay on the store, not on `Labels`. An annotation whose frame isn't in `/frames`
 * (e.g., TrackMate SLPs) gets a non-lazy frame of its own.
 */
export function addAnnotationsToStore(
  store: LazyDataStore,
  lazyFrames: LazyFrameList,
  annotations: AnnotationTuples,
): void {
  // Build per-frame annotation dicts for lazy materialization
  const buildAnnByFrame = <T>(
    tuples: Routed<T>[],
  ): { byFrame: Map<string, T[]>; undistributed: T[] } => {
    const byFrame = new Map<string, T[]>();
    const undistributed: T[] = [];
    for (const [ann, vidIdx, frameIdx] of tuples) {
      if (vidIdx >= 0 && frameIdx >= 0) {
        const key = `${vidIdx}:${frameIdx}`;
        const list = byFrame.get(key);
        if (list) list.push(ann);
        else byFrame.set(key, [ann]);
      } else {
        undistributed.push(ann);
      }
    }
    return { byFrame, undistributed };
  };

  const cResult = buildAnnByFrame(annotations.centroids);
  const bResult = buildAnnByFrame(annotations.bboxes);
  const mResult = buildAnnByFrame(annotations.masks);
  const rResult = buildAnnByFrame(annotations.rois);
  const liResult = buildAnnByFrame(annotations.labelImages);

  store._centroidByFrame = cResult.byFrame;
  store._bboxByFrame = bResult.byFrame;
  store._maskByFrame = mResult.byFrame;
  store._roiByFrame = rResult.byFrame;
  store._labelImageByFrame = liResult.byFrame;

  store._undistributedCentroids = cResult.undistributed;
  store._undistributedBboxes = bResult.undistributed;
  store._undistributedMasks = mResult.undistributed;
  store._undistributedRois = rResult.undistributed;
  store._undistributedLabelImages = liResult.undistributed;

  const allAnnKeys = new Set<string>();
  for (const dict of [
    store._centroidByFrame,
    store._bboxByFrame,
    store._maskByFrame,
    store._labelImageByFrame,
    store._roiByFrame,
  ]) {
    for (const key of dict.keys()) allAnnKeys.add(key);
  }
  if (allAnnKeys.size === 0) return;

  // Check for annotation-only frames not in /frames (e.g., TrackMate SLPs)
  const { framesData, videos } = store;
  const frameKeys = new Set<string>();
  const frameVideoIds = framesData.video ?? [];
  const frameFrameIdxs = framesData.frame_idx ?? [];
  // Annotations are keyed by video index, which a sparse `video` id is not.
  const videoIdToIndex = buildVideoIdMap(framesData, videos);
  for (let i = 0; i < (framesData.frame_id ?? []).length; i++) {
    const rawVideoId = Number(frameVideoIds[i]);
    const videoIndex = videoIdToIndex.get(rawVideoId) ?? rawVideoId;
    frameKeys.add(`${videoIndex}:${Number(frameFrameIdxs[i])}`);
  }

  // Create non-lazy frames for annotations without matching /frames entries
  for (const key of [...allAnnKeys].sort()) {
    if (frameKeys.has(key)) continue;
    const [vidIdxStr, fidxStr] = key.split(":");
    const vidIdx = Number(vidIdxStr);
    const fidx = Number(fidxStr);
    if (vidIdx >= 0 && vidIdx < videos.length) {
      lazyFrames._supplementary.push(
        new LabeledFrame({
          video: videos[vidIdx],
          frameIdx: fidx,
          centroids: store._centroidByFrame.get(key) ?? [],
          bboxes: store._bboxByFrame.get(key) ?? [],
          masks: store._maskByFrame.get(key) ?? [],
          labelImages: store._labelImageByFrame.get(key) ?? [],
          rois: store._roiByFrame.get(key) ?? [],
        }),
      );
    }
  }
}
