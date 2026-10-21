import { Labels } from "../../model/labels.js";
import { type Instance, PredictedInstance } from "../../model/instance.js";
import type { Track } from "../../model/instance.js";
import { LabeledFrame } from "../../model/labeled-frame.js";
import type {
  RecordingSession,
  Camera,
  InstanceGroup,
  FrameGroup,
} from "../../model/camera.js";
import type { Skeleton } from "../../model/skeleton.js";
import { SuggestionFrame } from "../../model/suggestions.js";
import type { Video } from "../../model/video.js";
import { CropVideoBackend } from "../../video/crop-backend.js";
import {
  encodeBitmapToPng,
  encodeImageDataToPng,
  isImageBitmapLike,
  isImageDataLike,
} from "../../video/image-decode.js";
import { getH5Module, getH5FileSystem, ensureH5StagingDir } from "./h5.js";
import { type ROI, type PredictedROI, encodeWkb } from "../../model/roi.js";
import type {
  SegmentationMask,
  UserSegmentationMask,
  PredictedSegmentationMask,
} from "../../model/mask.js";
import type { BoundingBox, PredictedBoundingBox } from "../../model/bbox.js";
import type { Centroid, PredictedCentroid } from "../../model/centroid.js";
import { normalizeCentroidSource } from "../../model/centroid.js";
import type {
  LabelImage,
  PredictedLabelImage,
} from "../../model/label-image.js";
import { deflate } from "pako";
import type { Identity } from "../../model/identity.js";
import { PredictedInstance3D } from "../../model/instance3d.js";
import type { LazyDataStore } from "../../model/lazy.js";
import { buildVideoIdMap } from "../../model/video-id-map.js";

// File writer hook — registered by h5-node.ts (imported as side-effect from Node entry point).
let _writeToFile:
  | ((filename: string, bytes: Uint8Array) => Promise<void>)
  | null = null;

/**
 * Register a file writer for Node.js environments.
 * Called as a side-effect when the Node entry point imports h5-node.ts.
 * @internal
 */
export function _registerFileWriter(
  writer: (filename: string, bytes: Uint8Array) => Promise<void>,
): void {
  _writeToFile = writer;
}

const FORMAT_ID = 1.4;
const textEncoder = new TextEncoder();

/** Write a string as a fixed-length HDF5 string attribute (H5T_STRING).
 *  h5py reads fixed-length strings as `bytes`, so Python's `.decode()` works.
 *  Using `S<n>` dtype avoids variable-length strings (returned as `str`)
 *  and uint8 arrays (returned as `numpy.ndarray`). */
function setStringAttr(target: any, name: string, value: string): void {
  const byteLength = textEncoder.encode(value).length;
  target.create_attribute(name, value, null, `S${byteLength}`);
}

/** Write a string array as a JSON-encoded string attribute dataset at root level. */
function writeStringDataset(file: any, name: string, values: string[]): void {
  const json = JSON.stringify(values);
  const bytes = textEncoder.encode(json);
  file.create_dataset({
    name,
    data: bytes,
    shape: [bytes.length],
    dtype: "<B",
  });
  const ds = file.get(name);
  setStringAttr(ds, "json", json);
}

const SPAWNED_ON = 0;

export type SlpWriteOptions = {
  embed?: boolean | string;
  restoreOriginalVideos?: boolean;
  /**
   * Called as frame images are read and encoded for embedding from a
   * continuous video (`current` of `total` frames). Already-embedded videos
   * copy their stored images in one go and don't report.
   */
  onProgress?: (current: number, total: number, message?: string) => void;
};

/**
 * A planned embedded video output: copy raw blobs, or (legacy) encode.
 *
 * `kind: "raw"` copies an already-embedded video's stored encoded blobs verbatim
 * via `getFrameBuffer` (no decode / re-encode) — the fast, lossless path that
 * mirrors Python's re-save of a `.pkg.slp`. `kind: "encode"` is the new-embed
 * of a continuous video: read each selected frame via `getFrame`, then store its
 * bytes directly (`frameToBytes`) when the backend already returns encoded bytes,
 * or PNG-encode it when the backend returns raw pixels (an `ImageBitmap` /
 * `ImageData` — the MediaBunny / Mp4Box mp4 case). Works in both browser and
 * Node.
 */
type EmbedPlanEntry = {
  kind: "raw" | "encode";
  videoIndex: number;
  video: Video;
  frameNumbers: number[]; // source frame numbers to write, in order
  format: string;
  channelOrder: string;
  /** For "encode": pre-collected bytes keyed by frame number. */
  frameData?: Map<number, Uint8Array>;
};
type EmbedPlan = Map<number, EmbedPlanEntry>;

/** True for a video whose stored encoded blobs we can copy verbatim. */
function isRawCopyable(video: Video): boolean {
  return !!(video.hasEmbeddedImages && video.backend?.getFrameBuffer);
}

function writeSlpToFile(
  file: any,
  labels: Labels,
  plan?: EmbedPlan | null,
): void {
  writeMetadata(file, labels);

  if (plan && plan.size > 0) {
    writeEmbeddedVideosJson(file, labels, plan);
  } else {
    writeVideos(file, labels.videos);
  }
  writeVideoCrops(file, labels.videos);

  writeTracks(file, labels.tracks);
  writeSuggestions(file, labels.suggestions, labels.videos);
  writeIdentities(file, labels.identities);
  writeSessions(
    file,
    labels.sessions,
    labels.videos,
    labels.labeledFrames,
    labels.identities,
  );
  writeLabeledFrames(file, labels);
  writeNegativeFrames(file, labels);
  const allInstances = labels.labeledFrames.flatMap((f) => f.instances);

  // Build flat annotation lists with routing contexts from parent LabeledFrames
  const allRois: ROI[] = [];
  const roiCtx: [number, number][] = [];
  const allMasks: SegmentationMask[] = [];
  const maskCtx: [number, number][] = [];
  const allBboxes: BoundingBox[] = [];
  const bboxCtx: [number, number][] = [];
  const allCentroids: Centroid[] = [];
  const centroidCtx: [number, number][] = [];
  const allLabelImages: LabelImage[] = [];
  const liCtx: [number, number][] = [];

  for (const lf of labels.labeledFrames) {
    const vidIdx = labels.videos.indexOf(lf.video);
    for (const r of lf.rois) {
      allRois.push(r);
      roiCtx.push([vidIdx, lf.frameIdx]);
    }
    for (const m of lf.masks) {
      allMasks.push(m);
      maskCtx.push([vidIdx, lf.frameIdx]);
    }
    for (const b of lf.bboxes) {
      allBboxes.push(b);
      bboxCtx.push([vidIdx, lf.frameIdx]);
    }
    for (const c of lf.centroids) {
      allCentroids.push(c);
      centroidCtx.push([vidIdx, lf.frameIdx]);
    }
    for (const li of lf.labelImages) {
      allLabelImages.push(li);
      liCtx.push([vidIdx, lf.frameIdx]);
    }
  }
  // Static ROIs
  for (const r of labels._staticRois) {
    allRois.push(r);
    roiCtx.push([r.video ? labels.videos.indexOf(r.video) : -1, -1]);
  }

  writeRois(file, allRois, labels.videos, labels.tracks, allInstances, roiCtx);
  writeMasks(
    file,
    allMasks,
    labels.videos,
    labels.tracks,
    allInstances,
    maskCtx,
  );
  writeBboxes(
    file,
    allBboxes,
    labels.videos,
    labels.tracks,
    allInstances,
    bboxCtx,
  );
  writeCentroids(
    file,
    allCentroids,
    labels.videos,
    labels.tracks,
    allInstances,
    centroidCtx,
  );
  writeLabelImages(
    file,
    allLabelImages,
    labels.videos,
    labels.tracks,
    allInstances,
    liCtx,
  );

  // Per-detection re-ID identity (SLP 2.5): /identity/links + /embeddings. The flat
  // per-modality lists above are in the SAME global order the modality datasets were
  // written, so a detection's array index == its owner_id (matching Python).
  const detections: DetectionCollections = {
    instance: allInstances,
    centroid: allCentroids,
    mask: allMasks,
    bbox: allBboxes,
    roi: allRois,
  };
  writeIdentityLinks(file, labels.identities, detections);
  writeCategoryLinks(file, detections);
  writeEmbeddings(file, detections);
}

/**
 * Sentinel error thrown by {@link makeLazySourceLabels} when the lazy source
 * fast path cannot safely swap videos in place (multi-camera sessions
 * referencing a video that's getting swapped). The dispatch layer catches
 * this and falls back to materializing the labels.
 */
class LazySourceFallback extends Error {
  constructor() {
    super("lazy source mode requires materialization");
    this.name = "LazySourceFallback";
  }
}

/**
 * Build a shallow Labels copy with `videos[]` swapped to source paths,
 * suitable for the lazy fast-path writer when `embed: "source"` is set.
 *
 * The lazy data store is keyed by `videoIdx` (not video object identity),
 * so we don't need to remap any annotations — only the `videos[]` array,
 * the `suggestions` (which reference videos by identity), and we need to
 * verify no `RecordingSession` references a video that's being swapped.
 *
 * Note: the eager source-mode path (`saveSlpToBytes`) does NOT remap
 * suggestions or check sessions today. The lazy path here is more correct;
 * fixing eager is a separate follow-up.
 *
 * @throws LazySourceFallback if any session references a swapped video.
 */
function makeLazySourceLabels(labels: Labels): Labels {
  const restoredVideos = labels.videos.map((v) => v.sourceVideo ?? v);
  const videoMap = new Map<Video, Video>();
  for (let i = 0; i < labels.videos.length; i++) {
    if (labels.videos[i] !== restoredVideos[i]) {
      videoMap.set(labels.videos[i], restoredVideos[i]);
    }
  }

  // No swaps needed — just return the original labels.
  if (videoMap.size === 0) return labels;

  // Multi-camera sessions referencing a swapped video are too complex for
  // the lazy fast path. Bail out so the dispatch layer can materialize.
  for (const session of labels.sessions) {
    for (const v of session.videoByCamera.values()) {
      if (videoMap.has(v)) {
        throw new LazySourceFallback();
      }
    }
  }

  const remappedSuggestions = labels.suggestions.map((s) => {
    const newVideo = videoMap.get(s.video);
    if (!newVideo) return s;
    return new SuggestionFrame({
      video: newVideo,
      frameIdx: s.frameIdx,
      group: s.group,
      metadata: s.metadata,
    });
  });

  const out = new Labels({
    videos: restoredVideos,
    skeletons: labels.skeletons,
    tracks: labels.tracks,
    suggestions: remappedSuggestions,
    sessions: labels.sessions, // safe: verified no swapped refs
    identities: labels.identities,
    provenance: labels.provenance,
    rois: labels._staticRois,
  });
  out._lazyFrameList = labels._lazyFrameList;
  out._lazyDataStore = labels._lazyDataStore;
  return out;
}

/**
 * Rebuild row-major `number[][]` from a lazy store's parsed column data
 * (`Record<string, any[]>`, from the HDF5 reader), so the pose datasets can be
 * written with the same helpers the eager writer uses ({@link createMatrixDataset}
 * / {@link createCompoundDataset}) for byte-identical output.
 *
 * TODO: a column-aware fast path could avoid the row rebuild entirely.
 */
function lazyColumnsToRows(
  columns: Record<string, any[]>,
  fieldNames: string[],
): number[][] {
  const rowCount = (columns[fieldNames[0]] ?? []).length;
  const colCount = fieldNames.length;
  const rows: number[][] = [];
  for (let i = 0; i < rowCount; i++) {
    const row = new Array<number>(colCount);
    for (let j = 0; j < colCount; j++) {
      const v = (columns[fieldNames[j]] ?? [])[i];
      row[j] = v === undefined || v === null ? 0 : Number(v);
    }
    rows.push(row);
  }
  return rows;
}

function writeLazyMatrixDataset(
  file: any,
  name: string,
  columns: Record<string, any[]>,
  fieldNames: string[],
  dtype: string,
): void {
  // resizable=true: the lazy fast-path writer's label tables must also be
  // in-place-editable on re-save (matches the eager writer). Only `frames` (a
  // flat int matrix) uses this now; the pose tables go through
  // writeLazyCompoundDataset.
  createMatrixDataset(
    file,
    name,
    lazyColumnsToRows(columns, fieldNames),
    fieldNames,
    dtype,
    true,
  );
}

/** Lazy-store analog of {@link createCompoundDataset}: columns → compound. */
function writeLazyCompoundDataset(
  file: any,
  name: string,
  columns: Record<string, any[]>,
  fields: CompoundField[],
): void {
  const fieldNames = fields.map(([fieldName]) => fieldName);
  createCompoundDataset(
    file,
    name,
    lazyColumnsToRows(columns, fieldNames),
    fields,
  );
}

function writeLazyFramesAndInstances(file: any, store: LazyDataStore): void {
  writeLazyMatrixDataset(
    file,
    "frames",
    store.framesData,
    ["frame_id", "video", "frame_idx", "instance_id_start", "instance_id_end"],
    "<i8",
  );
  writeLazyCompoundDataset(
    file,
    "instances",
    store.instancesData,
    INSTANCE_COMPOUND_FIELDS,
  );
  writeLazyCompoundDataset(
    file,
    "points",
    store.pointsData,
    POINT_COMPOUND_FIELDS,
  );
  writeLazyCompoundDataset(
    file,
    "pred_points",
    store.predPointsData,
    PRED_POINT_COMPOUND_FIELDS,
  );
}

function writeLazyNegativeFrames(file: any, store: LazyDataStore): void {
  if (store.negativeFrames.size === 0) return;
  const rows: number[][] = [];
  for (const key of store.negativeFrames) {
    const [vidStr, fidxStr] = key.split(":");
    rows.push([Number(vidStr), Number(fidxStr)]);
  }
  // Deterministic ordering for byte-stable output.
  rows.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  createCompoundDataset(
    file,
    "negative_frames",
    rows,
    NEGATIVE_FRAME_COMPOUND_FIELDS,
  );
}

/**
 * Lazy fast-path SLP writer. Mirrors Python's `_write_labels_lazy`.
 *
 * Reuses metadata writers (which only depend on eager `Labels` fields like
 * `videos`/`tracks`/`suggestions`/`identities`/`sessions`/`provenance`) and
 * pulls frames/instances/points/pred_points/negative_frames + per-type
 * annotation maps directly from `LazyDataStore`. No `LabeledFrame` or
 * `Instance` objects are constructed in this path.
 *
 * Limitations (matching Python):
 *   - `writeSessions` is called with an empty `labeledFrames` list, so any
 *     `FrameGroup` references that resolve via labeled-frame index are lost.
 *   - Instance associations on annotations are persisted via the stored
 *     `_instanceIdx` field (no live `instance` references in lazy mode).
 *
 * @throws Error if `labels.isLazy` is false or the lazy store is missing.
 */
function writeSlpToFileLazy(file: any, labels: Labels): void {
  const store = labels._lazyDataStore;
  if (!labels.isLazy || !store) {
    throw new Error(
      "writeSlpToFileLazy requires lazy Labels with a data store",
    );
  }

  writeMetadata(file, labels);
  writeVideos(file, labels.videos);
  writeVideoCrops(file, labels.videos);
  writeTracks(file, labels.tracks);
  writeSuggestions(file, labels.suggestions, labels.videos);
  writeIdentities(file, labels.identities);
  // Sessions: pass empty labeledFrames since they aren't materialized.
  // Frame-group refs resolved by lf index will be lost — matches Python.
  writeSessions(file, labels.sessions, labels.videos, [], labels.identities);

  writeLazyFramesAndInstances(file, store);
  writeLazyNegativeFrames(file, store);

  // Build per-type (annotation, context) pairs from the lazy store maps.
  // Per-frame Map keys are "vidIdx:frameIdx" strings (see read.ts).
  const allRois: ROI[] = [];
  const roiCtx: [number, number][] = [];
  const allMasks: SegmentationMask[] = [];
  const maskCtx: [number, number][] = [];
  const allBboxes: BoundingBox[] = [];
  const bboxCtx: [number, number][] = [];
  const allCentroids: Centroid[] = [];
  const centroidCtx: [number, number][] = [];
  const allLabelImages: LabelImage[] = [];
  const liCtx: [number, number][] = [];

  const collectFrameBound = <T>(
    byFrame: Map<string, T[]>,
    out: T[],
    ctxOut: [number, number][],
  ): void => {
    for (const [key, list] of byFrame) {
      const [vidStr, fidxStr] = key.split(":");
      const vidIdx = Number(vidStr);
      const fidx = Number(fidxStr);
      for (const ann of list) {
        out.push(ann);
        ctxOut.push([vidIdx, fidx]);
      }
    }
  };

  collectFrameBound(store._roiByFrame, allRois, roiCtx);
  collectFrameBound(store._maskByFrame, allMasks, maskCtx);
  collectFrameBound(store._bboxByFrame, allBboxes, bboxCtx);
  collectFrameBound(store._centroidByFrame, allCentroids, centroidCtx);
  collectFrameBound(store._labelImageByFrame, allLabelImages, liCtx);

  // Undistributed ROIs (static ROIs): preserve the video association by
  // looking up videos.indexOf(roi.video) instead of using -1. The other
  // four undistributed lists shouldn't have annotations with .video after
  // PR #94 (those classes no longer carry video), so they always use -1.
  // Mirrors the fix from Python sleap-io PR #414 (talmolab/sleap-io); the
  // JS port already had this in place via Side fix B and is covered by
  // tests/lazy-write.test.ts "preserves static ROI video association".
  for (const roi of store._undistributedRois) {
    allRois.push(roi);
    const vidIdx = roi.video ? labels.videos.indexOf(roi.video) : -1;
    roiCtx.push([vidIdx, -1]);
  }
  for (const m of store._undistributedMasks) {
    allMasks.push(m);
    maskCtx.push([-1, -1]);
  }
  for (const b of store._undistributedBboxes) {
    allBboxes.push(b);
    bboxCtx.push([-1, -1]);
  }
  for (const c of store._undistributedCentroids) {
    allCentroids.push(c);
    centroidCtx.push([-1, -1]);
  }
  for (const li of store._undistributedLabelImages) {
    allLabelImages.push(li);
    liCtx.push([-1, -1]);
  }

  // Pass undefined for `instances` so each writer uses the _instanceIdx
  // fallback (Side fix A) for any annotations with deferred instance links.
  writeRois(file, allRois, labels.videos, labels.tracks, undefined, roiCtx);
  writeMasks(file, allMasks, labels.videos, labels.tracks, [], maskCtx);
  writeBboxes(file, allBboxes, labels.videos, labels.tracks, [], bboxCtx);
  writeCentroids(
    file,
    allCentroids,
    labels.videos,
    labels.tracks,
    [],
    centroidCtx,
  );
  writeLabelImages(
    file,
    allLabelImages,
    labels.videos,
    labels.tracks,
    [],
    liCtx,
  );
}

/**
 * Serialize Labels to SLP format and return the bytes.
 * Works in both Node.js and browser environments.
 *
 * When `embed` is set, video frames are read from their backends and stored
 * directly in the SLP file as HDF5 datasets (video0/video, video1/video, etc.).
 * The video backends must be open and able to return frame data.
 *
 * Supported embed modes:
 * - `true` or `"all"` - Embed all labeled frames
 * - `"user"` - Embed only user-labeled frames ({@link Labels.userLabeledFrames}:
 *   user instances, negative frames, user centroids/boxes/masks/ROIs)
 * - `"suggestions"` - Embed only suggestion frames
 * - `"user+suggestions"` - Embed user-labeled frames and suggestion frames
 * - `"all+suggestions"` - Embed all labeled frames AND all suggestion frames
 * - `"source"` - Restore original video paths (no embedding)
 *
 * ALREADY-EMBEDDED videos (a `.pkg.slp` loaded with open backends) are ALWAYS
 * preserved: their full stored set of encoded image blobs is copied VERBATIM
 * (no decode/re-encode — see {@link writeRawEmbeddedVideo}), regardless of the
 * `embed` mode, UNLESS `embed:"source"` is passed to externalize them. So even
 * a bare `saveSlpToBytes(labels)` (no options) cannot silently drop embedded
 * images — the fix for the #213 re-save data loss. As a backstop, the raw-copy
 * path THROWS (rather than writing a stripped file) if it planned to copy N
 * frames but could read fewer than N blobs.
 */
// ===========================================================================
// Streaming / incremental SLP writer (issue #207)
//
// A write-side companion to `readSlpStreaming({ lazy })`: build an SLP file by
// appending pose frames/instances/points in bounded windows to *resizable* HDF5
// datasets (create-empty → `resize` + `write_slice` per window), rather than
// materializing the whole `Labels` graph or a full row matrix. Also supports
// MERGING N per-camera `LazyDataStore`s into one combined multi-video `.slp`,
// remapping video/instance/point/track ids as each store is appended — the
// downstream LUCID export path (one combined multi-camera file, memory-bounded).
//
// Scope (v1): pose tables only (frames/instances/points/pred_points) +
// negative_frames + all header metadata (videos/tracks/skeletons/suggestions/
// identities/sessions/provenance). Per-frame annotation overlays (masks/rois/
// bboxes/centroids/label-images), an edit overlay of corrected frames, and a
// `FileSystemWritableFileStream` sink are follow-ups.
// ===========================================================================

/** Frames per append window — bounds the per-window typed-array allocation. */
const DEFAULT_WRITE_WINDOW_FRAMES = 5000;
/** HDF5 chunk row count for the appendable pose datasets. */
const WRITE_CHUNK_ROWS = 8192;
/**
 * Per-window byte budget for the streamed raw embedded-blob copy: blobs
 * accumulate up to this many bytes before one `resize` + `write_slice` flush,
 * bounding peak JS memory to ~one window instead of the whole concatenation.
 */
const EMBED_WRITE_WINDOW_BYTES = 32 * 1024 * 1024;
/** HDF5 chunk length (elements) for the 1-D embedded video byte dataset. */
const EMBED_VIDEO_CHUNK_BYTES = 1 << 20;

const FRAMES_FIELDS = [
  "frame_id",
  "video",
  "frame_idx",
  "instance_id_start",
  "instance_id_end",
];
const INSTANCES_FIELDS = [
  "instance_id",
  "instance_type",
  "frame_id",
  "skeleton",
  "track",
  "from_predicted",
  "score",
  "point_id_start",
  "point_id_end",
  "tracking_score",
];
const POINTS_FIELDS = ["x", "y", "visible", "complete"];
const PRED_POINTS_FIELDS = ["x", "y", "visible", "complete", "score"];
const NEGATIVE_FRAMES_FIELDS = ["video_id", "frame_idx"];

/** Read a numeric column cell with a default (mirrors the lazy reader). */
function numAt(col: any[] | undefined, i: number, def = 0): number {
  const v = col?.[i];
  return v === undefined || v === null ? def : Number(v);
}

/**
 * Create an empty, row-resizable 2D dataset (`[0, cols]`, maxshape
 * `[null, cols]`, chunked) and stamp its `field_names` attribute — the
 * append-mode analog of {@link createMatrixDataset}.
 */
function createAppendableMatrixDataset(
  file: any,
  name: string,
  fieldNames: string[],
  dtype: string,
): void {
  const colCount = fieldNames.length;
  file.create_dataset({
    name,
    data: new Float64Array(0),
    shape: [0, colCount],
    maxshape: [null, colCount],
    chunks: [WRITE_CHUNK_ROWS, colCount],
    dtype,
  });
  setStringAttr(file.get(name), "field_names", JSON.stringify(fieldNames));
}

/** Resize a matrix dataset by `rowCount` rows and write `flat` into the tail. */
function appendMatrixRows(
  file: any,
  name: string,
  flat: Float64Array,
  rowCount: number,
  colCount: number,
  currentRows: number,
): void {
  if (rowCount === 0) return;
  const ds = file.get(name);
  ds.resize([currentRows + rowCount, colCount]);
  ds.write_slice(
    [
      [currentRows, currentRows + rowCount],
      [0, colCount],
    ],
    flat,
  );
}

/** Static header (everything except the pose frames) for {@link openSlpWriter}. */
export interface SlpWriteHeader {
  skeletons: Skeleton[];
  videos: Video[];
  tracks?: Track[];
  suggestions?: SuggestionFrame[];
  identities?: Identity[];
  sessions?: RecordingSession[];
  provenance?: Record<string, unknown>;
}

/** Per-store id offsets applied while appending (see {@link SlpStreamWriter.appendStore}). */
export interface AppendStoreOptions {
  /** Added to each frame's (remapped) video index. */
  videoIndexOffset?: number;
  /** Added to each instance's non-null track index. */
  trackOffset?: number;
  /** Added to each instance's skeleton index. */
  skeletonOffset?: number;
  /** Frames per append window (defaults to {@link DEFAULT_WRITE_WINDOW_FRAMES}). */
  windowFrames?: number;
}

/**
 * A chunked byte sink for {@link SlpStreamWriter.writeToSink} — the output-side
 * companion to the append writer. Satisfied by a browser
 * `FileSystemWritableFileStream` (OPFS / save-file-picker) and by Node's
 * `fs.WriteStream`, so the finished file need not be resident as one big
 * `Uint8Array`.
 */
export interface SlpWriteSink {
  write(chunk: Uint8Array): unknown | Promise<unknown>;
  close?(): unknown | Promise<unknown>;
}

/** Emscripten-FS methods used for chunked reads (present on h5wasm's `module.FS`). */
interface ChunkedFs {
  open?: (path: string, flags: string) => unknown;
  read?: (
    stream: unknown,
    buffer: Uint8Array,
    offset: number,
    length: number,
    position: number,
  ) => number;
  close?: (stream: unknown) => void;
  stat?: (path: string) => { size: number };
}

/** An annotation with the instance link the writers read via fallback. */
type AnnLinked = { instance: Instance | null; _instanceIdx: number | null };
/** Accumulated annotations of one type: objects + `[vid,frame]` ctx + global inst idx. */
interface AnnBucket<T> {
  anns: T[];
  ctx: [number, number][];
  inst: number[];
}
function newAnnBucket<T>(): AnnBucket<T> {
  return { anns: [], ctx: [], inst: [] };
}

/**
 * Incremental SLP writer. Open with {@link openSlpWriter} (which writes the
 * header and creates the resizable pose datasets), append one or more
 * {@link LazyDataStore}s with {@link appendStore}, then {@link close} to get the
 * file bytes. Ids are rebased per store so multiple stores concatenate into one
 * consistent multi-video file.
 */
export class SlpStreamWriter {
  private file: any;
  private module: any;
  private memPath: string;
  private videos: Video[];
  private skeletons: Skeleton[];
  private tracks: Track[];
  private frameRows = 0;
  private instRows = 0;
  private pointRows = 0;
  private predPointRows = 0;
  private negativeFrames = new Set<string>();
  private closed = false;
  // Combined `${videoIndex}:${frameIdx}` keys already written, for dedup across
  // appendStore/appendFrames. First write wins: a later append that repeats a
  // key skips that frame entirely. Append overlays (appendFrames) BEFORE bulk
  // stores so the overlay's frame is the one kept. See #208 follow-up.
  private writtenFrames = new Set<string>();

  // Per-frame annotations (masks/rois/bboxes/centroids/label-images) accumulated
  // across appends and written once at finalize — they're far fewer than pose
  // points, so they are not windowed. `ctx` is `[videoIndex, frameIdx]`; `inst`
  // is the resolved GLOBAL instance index (or -1) for the annotation's link.
  private pendingRois = newAnnBucket<ROI>();
  private pendingMasks = newAnnBucket<SegmentationMask>();
  private pendingBboxes = newAnnBucket<BoundingBox>();
  private pendingCentroids = newAnnBucket<Centroid>();
  // Label images carry instance links PER OBJECT (not on the top-level object),
  // so only their video/frame context is remapped here; each object's
  // `_instanceIdx` passes through (correct single-store; may be off in a
  // multi-store merge — a niche edge, tracked on #207).
  private pendingLabelImages: {
    anns: LabelImage[];
    ctx: [number, number][];
  } = { anns: [], ctx: [] };

  /** @internal — use {@link openSlpWriter}. */
  constructor(
    module: any,
    file: any,
    memPath: string,
    header: { videos: Video[]; skeletons: Skeleton[]; tracks: Track[] },
  ) {
    this.module = module;
    this.file = file;
    this.memPath = memPath;
    this.videos = header.videos;
    this.skeletons = header.skeletons;
    this.tracks = header.tracks;
  }

  /**
   * Append every frame of `store` (in windows), rebasing its ids onto the
   * running file so the store's videos/instances/points/tracks land at the
   * offsets given in `opts`. Point coordinates and all per-instance fields are
   * copied verbatim from the store's columns — no `Instance`/`LabeledFrame`
   * object is constructed. The store's frame/instance/point tables are assumed
   * ordered by frame (the SLP on-disk invariant).
   *
   * A `(video, frameIdx)` already written (by an earlier append) is SKIPPED — so
   * append overlays via {@link appendFrames} first for them to win. Frame /
   * instance / point ids are assigned from running OUTPUT counters (not the
   * store's positions), so skips leave no gaps; cross-references
   * (`from_predicted`, annotation instance links) are remapped through the
   * skipped ranges. `store` must not itself contain duplicate `(video, frameIdx)`.
   */
  appendStore(store: LazyDataStore, opts?: AppendStoreOptions): void {
    if (this.closed) throw new Error("SlpStreamWriter is closed");
    const windowFrames = opts?.windowFrames ?? DEFAULT_WRITE_WINDOW_FRAMES;
    const vOff = opts?.videoIndexOffset ?? 0;
    const tOff = opts?.trackOffset ?? 0;
    const kOff = opts?.skeletonOffset ?? 0;

    const fd = store.framesData;
    const idn = store.instancesData;
    const nFrames = (fd.frame_id ?? fd.video ?? []).length;
    if (nFrames === 0) return;

    // Remap this store's raw `frames.video` ids to its own video indices (#204)
    // before offsetting into the combined `videos` array.
    const videoIdMap = buildVideoIdMap(fd, store.videos);
    const remapVideo = (raw: number) => vOff + (videoIdMap.get(raw) ?? raw);

    // Per-store OUTPUT bases: the first output row/instance/point this store
    // contributes. Ids are `base + <running output counter>`, so skipped frames
    // leave no gaps.
    const frameBase = this.frameRows;
    const instBase = this.instRows;
    const pointBase = this.pointRows;
    const predBase = this.predPointRows;

    const px = store.pointsData.x,
      py = store.pointsData.y,
      pv = store.pointsData.visible,
      pc = store.pointsData.complete;
    const qx = store.predPointsData.x,
      qy = store.predPointsData.y,
      qv = store.predPointsData.visible,
      qc = store.predPointsData.complete,
      qs = store.predPointsData.score;

    // Pre-pass: the instance ranges of frames that will be SKIPPED (their
    // `(video, frameIdx)` was already written). Sorted by construction (frames
    // are in order). `outIdxOf` maps a store instance index to its output
    // position — identity when nothing is skipped, so the no-overlap path is
    // unchanged — or `null` when that instance falls in a skipped frame.
    const skippedRanges: Array<[number, number]> = [];
    const skippedKeys = new Set<string>();
    for (let r = 0; r < nFrames; r++) {
      const key = `${remapVideo(numAt(fd.video, r))}:${numAt(fd.frame_idx, r)}`;
      if (this.writtenFrames.has(key)) {
        skippedRanges.push([
          numAt(fd.instance_id_start, r),
          numAt(fd.instance_id_end, r),
        ]);
        skippedKeys.add(key);
      }
    }
    const outIdxOf = (idx: number): number | null => {
      if (skippedRanges.length === 0) return idx;
      let shift = 0;
      for (const [s, e] of skippedRanges) {
        if (idx < s) break; // ranges sorted; the rest are past idx
        if (idx < e) return null; // idx is inside a skipped frame → dangling
        shift += e - s;
      }
      return idx - shift;
    };

    // Running OUTPUT counters for this store (across all its windows): emitted
    // frames, instances, and re-packed user / predicted points.
    let outFrames = 0;
    let outInsts = 0;
    let userWritten = 0;
    let predWritten = 0;

    try {
      for (let wStart = 0; wStart < nFrames; wStart += windowFrames) {
        const wEnd = Math.min(wStart + windowFrames, nFrames);
        const wFrames = wEnd - wStart;

        // Pass 1: size the window's buffers at max (skips only shrink usage; the
        // actually-filled slice is written below).
        let nInst = 0;
        let nUserPts = 0;
        let nPredPts = 0;
        for (let r = wStart; r < wEnd; r++) {
          const iStart = numAt(fd.instance_id_start, r);
          const iEnd = numAt(fd.instance_id_end, r);
          nInst += iEnd - iStart;
          for (let j = iStart; j < iEnd; j++) {
            const cnt =
              numAt(idn.point_id_end, j) - numAt(idn.point_id_start, j);
            if (numAt(idn.instance_type, j) === 1) nPredPts += cnt;
            else nUserPts += cnt;
          }
        }

        const framesFlat = new Float64Array(wFrames * 5);
        const instFlat = new Float64Array(nInst * 10);
        const userPtsFlat = new Float64Array(nUserPts * 4);
        const predPtsFlat = new Float64Array(nPredPts * 5);
        let fi = 0;
        let ii = 0;
        let upi = 0;
        let ppi = 0;

        // Pass 2: emit rows for non-skipped frames, assigning ids from the
        // running output counters.
        for (let r = wStart; r < wEnd; r++) {
          const combinedVideo = remapVideo(numAt(fd.video, r));
          const frameIdx = numAt(fd.frame_idx, r);
          const key = `${combinedVideo}:${frameIdx}`;
          const iStart = numAt(fd.instance_id_start, r);
          const iEnd = numAt(fd.instance_id_end, r);

          if (this.writtenFrames.has(key)) continue; // dedup: first write wins

          const newFrameId = frameBase + outFrames;
          outFrames++;
          framesFlat[fi++] = newFrameId;
          framesFlat[fi++] = combinedVideo;
          framesFlat[fi++] = frameIdx;
          framesFlat[fi++] = instBase + outInsts; // instance range start
          framesFlat[fi++] = instBase + outInsts + (iEnd - iStart); // end

          if (store.negativeFrames.has(`${numAt(fd.video, r)}:${frameIdx}`)) {
            this.negativeFrames.add(`${combinedVideo}:${frameIdx}`);
          }

          for (let j = iStart; j < iEnd; j++) {
            const type = numAt(idn.instance_type, j);
            const ptStart = numAt(idn.point_id_start, j);
            const ptEnd = numAt(idn.point_id_end, j);
            const trk = numAt(idn.track, j, -1);
            const fp = numAt(idn.from_predicted, j, -1);
            const fpOut = fp >= 0 ? outIdxOf(fp) : null;

            instFlat[ii++] = instBase + outInsts; // instance_id
            instFlat[ii++] = type; // instance_type
            instFlat[ii++] = newFrameId; // frame_id
            instFlat[ii++] = numAt(idn.skeleton, j) + kOff; // skeleton
            instFlat[ii++] = trk >= 0 ? trk + tOff : -1; // track
            instFlat[ii++] = fpOut != null ? instBase + fpOut : -1; // from_predicted
            instFlat[ii++] = numAt(idn.score, j); // score
            if (type === 1) {
              // point_id range = position in the re-packed pred-point stream.
              instFlat[ii++] = predBase + predWritten;
              instFlat[ii++] = predBase + predWritten + (ptEnd - ptStart);
              for (let p = ptStart; p < ptEnd; p++) {
                predPtsFlat[ppi++] = numAt(qx, p);
                predPtsFlat[ppi++] = numAt(qy, p);
                predPtsFlat[ppi++] = numAt(qv, p);
                predPtsFlat[ppi++] = numAt(qc, p);
                predPtsFlat[ppi++] = numAt(qs, p);
              }
              predWritten += ptEnd - ptStart;
            } else {
              instFlat[ii++] = pointBase + userWritten;
              instFlat[ii++] = pointBase + userWritten + (ptEnd - ptStart);
              for (let p = ptStart; p < ptEnd; p++) {
                userPtsFlat[upi++] = numAt(px, p);
                userPtsFlat[upi++] = numAt(py, p);
                userPtsFlat[upi++] = numAt(pv, p);
                userPtsFlat[upi++] = numAt(pc, p);
              }
              userWritten += ptEnd - ptStart;
            }
            instFlat[ii++] = numAt(idn.tracking_score, j); // tracking_score
            outInsts++;
          }
          this.writtenFrames.add(key);
        }

        // Write only the actually-filled slice of each (max-sized) buffer.
        const emFrames = fi / 5;
        const emInsts = ii / 10;
        const emUserPts = upi / 4;
        const emPredPts = ppi / 5;
        appendMatrixRows(
          this.file,
          "frames",
          framesFlat.subarray(0, fi),
          emFrames,
          5,
          this.frameRows,
        );
        this.frameRows += emFrames;
        appendMatrixRows(
          this.file,
          "instances",
          instFlat.subarray(0, ii),
          emInsts,
          10,
          this.instRows,
        );
        this.instRows += emInsts;
        appendMatrixRows(
          this.file,
          "points",
          userPtsFlat.subarray(0, upi),
          emUserPts,
          4,
          this.pointRows,
        );
        this.pointRows += emUserPts;
        appendMatrixRows(
          this.file,
          "pred_points",
          predPtsFlat.subarray(0, ppi),
          emPredPts,
          5,
          this.predPointRows,
        );
        this.predPointRows += emPredPts;
      }
      // Per-frame + undistributed annotations (masks/rois/…), video + instance
      // remapped, skipping any that belong to a skipped (overlaid) frame.
      this.collectStoreAnnotations(
        store,
        vOff,
        instBase,
        outIdxOf,
        skippedKeys,
      );
    } catch (e) {
      // A mid-write failure leaves a partial file — release it rather than leak.
      this.dispose();
      throw e;
    }
  }

  /**
   * Append a batch of already-materialized `LabeledFrame`s — the write-side of
   * an edit overlay: user-corrected or newly-added frames layered onto a lazy
   * stream. Each frame's `video`/`skeleton`/`track` is resolved against this
   * writer's header (by identity); `from_predicted` links are resolved among the
   * batch. Intended for a bounded batch (the corrected subset), so it is not
   * windowed. Interleave freely with {@link appendStore}.
   */
  appendFrames(frames: LabeledFrame[]): void {
    if (this.closed) throw new Error("SlpStreamWriter is closed");
    if (frames.length === 0) return;
    try {
      const frameBase = this.frameRows;
      const instBase = this.instRows;
      const pointBase = this.pointRows;
      const predBase = this.predPointRows;

      // Build the batch with LOCAL 0-based ids, then offset onto the file.
      const framesRows: number[][] = [];
      const instRows: number[][] = [];
      const userPts: number[][] = [];
      const predPts: number[][] = [];
      const localId = new Map<Instance, number>();
      const links: Array<[number, PredictedInstance]> = [];

      for (const frame of frames) {
        const videoIndex = Math.max(0, this.videos.indexOf(frame.video));
        const key = `${videoIndex}:${frame.frameIdx}`;
        if (this.writtenFrames.has(key)) continue; // dedup: first write wins
        this.writtenFrames.add(key);
        const localFrameId = framesRows.length;
        const instanceStart = instRows.length;
        for (const inst of frame.instances) {
          const localInstId = instRows.length;
          localId.set(inst as Instance, localInstId);
          const skeletonId = Math.max(0, this.skeletons.indexOf(inst.skeleton));
          const trackId = inst.track ? this.tracks.indexOf(inst.track) : -1;
          const trackingScore = inst.trackingScore ?? 0;
          let type = 0;
          let score = 0;
          let ptStart = 0;
          let ptEnd = 0;
          if (inst instanceof PredictedInstance) {
            type = 1;
            score = inst.score ?? 0;
            ptStart = predPts.length;
            emitInstancePoints(predPts, inst, true);
            ptEnd = predPts.length;
          } else {
            ptStart = userPts.length;
            emitInstancePoints(userPts, inst as Instance, false);
            ptEnd = userPts.length;
            if (inst.fromPredicted)
              links.push([localInstId, inst.fromPredicted]);
          }
          instRows.push([
            localInstId,
            type,
            localFrameId,
            skeletonId,
            trackId,
            -1, // from_predicted (patched below)
            score,
            ptStart,
            ptEnd,
            trackingScore,
          ]);
        }
        framesRows.push([
          localFrameId,
          videoIndex,
          frame.frameIdx,
          instanceStart,
          instRows.length,
        ]);
        if (frame.isNegative) {
          this.negativeFrames.add(`${videoIndex}:${frame.frameIdx}`);
        }
        this.collectFrameAnnotations(frame, videoIndex, instBase, localId);
      }

      // Resolve from_predicted to a GLOBAL instance id (or -1 if the source is
      // not in this batch — a dangling link, matching the eager writer).
      for (const [local, src] of links) {
        const srcLocal = localId.get(src as unknown as Instance);
        instRows[local][5] = srcLocal != null ? instBase + srcLocal : -1;
      }

      const nF = framesRows.length;
      const nI = instRows.length;
      const nU = userPts.length;
      const nP = predPts.length;

      const framesFlat = new Float64Array(nF * 5);
      for (let i = 0; i < nF; i++) {
        const r = framesRows[i];
        const o = i * 5;
        framesFlat[o] = frameBase + r[0];
        framesFlat[o + 1] = r[1];
        framesFlat[o + 2] = r[2];
        framesFlat[o + 3] = instBase + r[3];
        framesFlat[o + 4] = instBase + r[4];
      }
      const instFlat = new Float64Array(nI * 10);
      for (let i = 0; i < nI; i++) {
        const r = instRows[i];
        const o = i * 10;
        const ptBase = r[1] === 1 ? predBase : pointBase;
        instFlat[o] = instBase + r[0];
        instFlat[o + 1] = r[1];
        instFlat[o + 2] = frameBase + r[2];
        instFlat[o + 3] = r[3];
        instFlat[o + 4] = r[4];
        instFlat[o + 5] = r[5]; // already global (or -1)
        instFlat[o + 6] = r[6];
        instFlat[o + 7] = ptBase + r[7];
        instFlat[o + 8] = ptBase + r[8];
        instFlat[o + 9] = r[9];
      }
      const userFlat = new Float64Array(nU * 4);
      for (let i = 0; i < nU; i++) {
        const r = userPts[i];
        const o = i * 4;
        userFlat[o] = r[0];
        userFlat[o + 1] = r[1];
        userFlat[o + 2] = r[2];
        userFlat[o + 3] = r[3];
      }
      const predFlat = new Float64Array(nP * 5);
      for (let i = 0; i < nP; i++) {
        const r = predPts[i];
        const o = i * 5;
        predFlat[o] = r[0];
        predFlat[o + 1] = r[1];
        predFlat[o + 2] = r[2];
        predFlat[o + 3] = r[3];
        predFlat[o + 4] = r[4];
      }

      appendMatrixRows(this.file, "frames", framesFlat, nF, 5, this.frameRows);
      this.frameRows += nF;
      appendMatrixRows(this.file, "instances", instFlat, nI, 10, this.instRows);
      this.instRows += nI;
      appendMatrixRows(this.file, "points", userFlat, nU, 4, this.pointRows);
      this.pointRows += nU;
      appendMatrixRows(
        this.file,
        "pred_points",
        predFlat,
        nP,
        5,
        this.predPointRows,
      );
      this.predPointRows += nP;
    } catch (e) {
      this.dispose();
      throw e;
    }
  }

  /**
   * Collect a store's per-frame + undistributed annotations, remapping the
   * video index (`+vOff`) and the instance link (through `outIdxOf`, `+instBase`)
   * onto the combined file. Annotations on a SKIPPED (overlaid) frame — whose
   * combined `"vid:frameIdx"` key is in `skippedKeys` — are dropped. The store's
   * map keys are `"videoIndex:frameIdx"` (array-index video).
   */
  private collectStoreAnnotations(
    store: LazyDataStore,
    vOff: number,
    instBase: number,
    outIdxOf: (idx: number) => number | null,
    skippedKeys: Set<string>,
  ): void {
    const resolve = (idx: number | null | undefined): number => {
      if (idx == null || idx < 0) return -1;
      const o = outIdxOf(idx);
      return o != null ? instBase + o : -1;
    };
    const fromMap = <T extends AnnLinked>(
      map: Map<string, T[]>,
      bucket: AnnBucket<T>,
    ): void => {
      for (const [key, list] of map) {
        const sep = key.indexOf(":");
        const vid = vOff + Number(key.slice(0, sep));
        const fidx = Number(key.slice(sep + 1));
        if (skippedKeys.has(`${vid}:${fidx}`)) continue; // overlaid frame wins
        for (const ann of list) {
          bucket.anns.push(ann);
          bucket.ctx.push([vid, fidx]);
          bucket.inst.push(resolve(ann._instanceIdx));
        }
      }
    };
    fromMap(store._roiByFrame, this.pendingRois);
    fromMap(store._maskByFrame, this.pendingMasks);
    fromMap(store._bboxByFrame, this.pendingBboxes);
    fromMap(store._centroidByFrame, this.pendingCentroids);
    for (const [key, list] of store._labelImageByFrame) {
      const sep = key.indexOf(":");
      const vid = vOff + Number(key.slice(0, sep));
      const fidx = Number(key.slice(sep + 1));
      if (skippedKeys.has(`${vid}:${fidx}`)) continue;
      for (const li of list) {
        this.pendingLabelImages.anns.push(li);
        this.pendingLabelImages.ctx.push([vid, fidx]);
      }
    }

    const undist = <T extends AnnLinked>(
      list: T[],
      bucket: AnnBucket<T>,
      vidFor: (ann: T) => number,
    ): void => {
      for (const ann of list) {
        bucket.anns.push(ann);
        bucket.ctx.push([vidFor(ann), -1]);
        bucket.inst.push(resolve(ann._instanceIdx));
      }
    };
    // Static ROIs keep their video association (via the combined videos array).
    undist(store._undistributedRois, this.pendingRois, (roi) =>
      (roi as ROI).video ? this.videos.indexOf((roi as ROI).video!) : -1,
    );
    undist(store._undistributedMasks, this.pendingMasks, () => -1);
    undist(store._undistributedBboxes, this.pendingBboxes, () => -1);
    undist(store._undistributedCentroids, this.pendingCentroids, () => -1);
    for (const li of store._undistributedLabelImages) {
      this.pendingLabelImages.anns.push(li);
      this.pendingLabelImages.ctx.push([-1, -1]);
    }
  }

  /**
   * Collect a materialized frame's annotations, resolving each annotation's live
   * `instance` to its GLOBAL index via the batch's local map (`+instBase`).
   * Annotations without a live `instance` in the batch drop their link (-1).
   */
  private collectFrameAnnotations(
    frame: LabeledFrame,
    vid: number,
    instBase: number,
    localId: Map<Instance, number>,
  ): void {
    const resolve = (inst: Instance | null | undefined): number => {
      if (!inst) return -1;
      const l = localId.get(inst);
      return l != null ? instBase + l : -1;
    };
    const push = <T extends AnnLinked>(
      list: T[],
      bucket: AnnBucket<T>,
    ): void => {
      for (const ann of list) {
        bucket.anns.push(ann);
        bucket.ctx.push([vid, frame.frameIdx]);
        bucket.inst.push(resolve(ann.instance));
      }
    };
    push(frame.rois, this.pendingRois);
    push(frame.masks, this.pendingMasks);
    push(frame.bboxes, this.pendingBboxes);
    push(frame.centroids, this.pendingCentroids);
    for (const li of frame.labelImages) {
      this.pendingLabelImages.anns.push(li);
      this.pendingLabelImages.ctx.push([vid, frame.frameIdx]);
    }
  }

  /**
   * Write all accumulated annotation datasets. The per-type writers read the
   * instance link off each object, so the resolved GLOBAL index is applied via a
   * contained mutate→write→restore (source annotations are left unchanged).
   */
  private writePendingAnnotations(): void {
    const writeBucket = <T extends AnnLinked>(
      bucket: AnnBucket<T>,
      fn: (
        file: any,
        anns: T[],
        videos: Video[],
        tracks: Track[],
        instances: never[],
        ctx: [number, number][],
      ) => void,
    ): void => {
      if (bucket.anns.length === 0) return;
      const saved = bucket.anns.map(
        (a) => [a.instance, a._instanceIdx] as const,
      );
      try {
        for (let i = 0; i < bucket.anns.length; i++) {
          bucket.anns[i].instance = null;
          bucket.anns[i]._instanceIdx =
            bucket.inst[i] >= 0 ? bucket.inst[i] : null;
        }
        fn(this.file, bucket.anns, this.videos, this.tracks, [], bucket.ctx);
      } finally {
        for (let i = 0; i < bucket.anns.length; i++) {
          bucket.anns[i].instance = saved[i][0];
          bucket.anns[i]._instanceIdx = saved[i][1];
        }
      }
    };
    writeBucket(this.pendingRois, writeRois);
    writeBucket(this.pendingMasks, writeMasks);
    writeBucket(this.pendingBboxes, writeBboxes);
    writeBucket(this.pendingCentroids, writeCentroids);
    if (this.pendingLabelImages.anns.length > 0) {
      writeLabelImages(
        this.file,
        this.pendingLabelImages.anns,
        this.videos,
        this.tracks,
        [],
        this.pendingLabelImages.ctx,
      );
    }
  }

  /** Write pending `negative_frames` and close the HDF5 file (shared finalize). */
  private finalizeFile(): void {
    this.writePendingAnnotations();
    if (this.negativeFrames.size > 0) {
      const rows: number[][] = [];
      for (const key of this.negativeFrames) {
        const [v, f] = key.split(":");
        rows.push([Number(v), Number(f)]);
      }
      rows.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      createMatrixDataset(
        this.file,
        "negative_frames",
        rows,
        ["video_id", "frame_idx"],
        "<i8",
      );
    }
    this.file.close();
  }

  /** Finalize the file (writes `negative_frames` if any) and return its bytes. */
  close(): Uint8Array {
    if (this.closed) throw new Error("SlpStreamWriter is already closed");
    this.closed = true;
    this.finalizeFile();
    const fs = getH5FileSystem(this.module);
    const bytes = fs.readFile!(this.memPath);
    fs.unlink!(this.memPath);
    return bytes;
  }

  /**
   * Finalize and stream the file to `sink` in chunks, then unlink it — the
   * output-side companion to {@link appendStore}, so the finished file never has
   * to be resident as one big `Uint8Array` on the JS side. Reads the in-memory
   * HDF5 file with chunked FS reads when available (falling back to a single
   * read). `sink.close()` is awaited if present.
   *
   * @param opts.chunkBytes Chunk size (default 8 MiB).
   */
  async writeToSink(
    sink: SlpWriteSink,
    opts?: { chunkBytes?: number },
  ): Promise<void> {
    if (this.closed) throw new Error("SlpStreamWriter is already closed");
    this.closed = true;
    this.finalizeFile();

    const fs = getH5FileSystem(this.module);
    const cfs = fs as unknown as ChunkedFs;
    const chunkBytes = Math.max(
      1,
      Math.trunc(opts?.chunkBytes ?? 8 * 1024 * 1024),
    );
    try {
      if (cfs.open && cfs.read && cfs.close && cfs.stat) {
        const size = Number(cfs.stat(this.memPath).size);
        const stream = cfs.open(this.memPath, "r");
        try {
          const buf = new Uint8Array(chunkBytes);
          let pos = 0;
          while (pos < size) {
            const want = Math.min(chunkBytes, size - pos);
            const n = cfs.read(stream, buf, 0, want, pos);
            if (n <= 0) break;
            await sink.write(buf.slice(0, n)); // copy — buf is reused
            pos += n;
          }
        } finally {
          cfs.close(stream);
        }
      } else {
        // No chunked FS reads: read once, then hand out slices.
        const bytes = fs.readFile!(this.memPath);
        for (let pos = 0; pos < bytes.length; pos += chunkBytes) {
          await sink.write(
            bytes.slice(pos, Math.min(pos + chunkBytes, bytes.length)),
          );
        }
      }
      if (sink.close) await sink.close();
    } finally {
      fs.unlink!(this.memPath);
    }
  }

  /**
   * Release the underlying HDF5 file WITHOUT producing bytes — call to clean up
   * a writer that will not be finished (e.g. after an error). Idempotent and
   * best-effort (never throws); the file/MEMFS path are closed and unlinked.
   */
  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    try {
      this.file.close();
    } catch {
      // best-effort
    }
    try {
      getH5FileSystem(this.module).unlink?.(this.memPath);
    } catch {
      // best-effort
    }
  }
}

/**
 * Open a streaming SLP writer: create the in-memory HDF5 file, write the header
 * (skeletons/videos/tracks/suggestions/identities/sessions/provenance), and
 * create the resizable `frames`/`instances`/`points`/`pred_points` datasets.
 * Frame data is appended later via {@link SlpStreamWriter.appendStore}.
 */
export async function openSlpWriter(
  header: SlpWriteHeader,
): Promise<SlpStreamWriter> {
  const module = await getH5Module();
  ensureH5StagingDir(module);
  const memPath = `/tmp/sleap_stream_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  const file = new module.File(memPath, "w");

  try {
    // Reuse the eager metadata writers via a frameless header `Labels` (they read
    // only header fields). Sessions are written with an empty labeled-frame list,
    // so frame-group refs that resolve by labeled-frame index are dropped — matches
    // the lazy fast-path (`writeSlpToFileLazy`).
    const headerLabels = new Labels({
      labeledFrames: [],
      videos: header.videos,
      skeletons: header.skeletons,
      tracks: header.tracks ?? [],
      suggestions: header.suggestions ?? [],
      sessions: header.sessions ?? [],
      identities: header.identities ?? [],
      provenance: header.provenance ?? {},
    });

    writeMetadata(file, headerLabels);
    writeVideos(file, headerLabels.videos);
    writeVideoCrops(file, headerLabels.videos);
    writeTracks(file, headerLabels.tracks);
    writeSuggestions(file, headerLabels.suggestions, headerLabels.videos);
    writeIdentities(file, headerLabels.identities);
    writeSessions(
      file,
      headerLabels.sessions,
      headerLabels.videos,
      [],
      headerLabels.identities,
    );

    // dtype chars matter: h5wasm keys off the type CHAR and IGNORES the trailing
    // digit, so "<f8" resolves to float32 (char 'f'=4 bytes) and "<i8" to int32
    // (char 'i'=4). See the notes above INSTANCE_COMPOUND_FIELDS and
    // SESSION_FRAME_GROUP_FIELDS.
    //
    // `frames`: this is actually int32 ("<q" is h5wasm's int64 code). Harmless
    // below 2^31 frame/instance rows; the Python reader casts positionally.
    createAppendableMatrixDataset(file, "frames", FRAMES_FIELDS, "<i8");
    // `instances`: MUST be true float64 ("<d"), NOT "<f8" (== float32 in h5wasm).
    // This table holds integer bookkeeping columns (instance_id, frame_id,
    // skeleton, track, point_id_start/point_id_end, …); float32 cannot represent
    // odd integers above 2^24 (16,777,216), so on files with >~16.7M point rows the
    // point_id_start/end spans round to even → instances read back with the wrong
    // node count → points misattributed to the wrong skeleton nodes, silently
    // (#231). "<d" (char 'd'=8 bytes) is exact to 2^53. The eager/lazy writers dodge
    // this via HDF5 compound columns (#218); this streaming path stays a float
    // matrix, so it must use true float64 here.
    createAppendableMatrixDataset(file, "instances", INSTANCES_FIELDS, "<d");
    // `points`/`pred_points` hold only coordinates, visibility flags, and scores —
    // no large integer ids — so float32 ("<f8") is an intentional, acceptable
    // choice here (contrast `instances` above, which MUST be "<d"). Remember "<f8"
    // == float32 in h5wasm (only the type char is read).
    createAppendableMatrixDataset(file, "points", POINTS_FIELDS, "<f8");
    createAppendableMatrixDataset(
      file,
      "pred_points",
      PRED_POINTS_FIELDS,
      "<f8",
    );
  } catch (e) {
    // Don't leak the MEMFS file if header setup fails before a writer exists.
    try {
      file.close();
    } catch {
      // best-effort
    }
    try {
      getH5FileSystem(module).unlink?.(memPath);
    } catch {
      // best-effort
    }
    throw e;
  }

  return new SlpStreamWriter(module, file, memPath, {
    videos: header.videos,
    skeletons: header.skeletons,
    tracks: header.tracks ?? [],
  });
}

/**
 * Structural signature of a skeleton list — node names, edges, and symmetries —
 * for cross-store equality. Instances from every store are read back against the
 * combined (store 0's) skeleton, so topology must match, not just node names.
 */
function skeletonSignature(skeletons: Skeleton[]): string {
  return JSON.stringify(
    skeletons.map((s) => ({
      nodes: s.nodeNames,
      edges: s.edges.map((e) => [e.source.name, e.destination.name]),
      symmetries: s.symmetryNames,
    })),
  );
}

/**
 * Merge N per-camera {@link LazyDataStore}s into one combined multi-video
 * `.slp`, streaming each store's frames in bounded windows (peak memory ≪ the
 * whole graph). The combined `videos` and `tracks` are the concatenation of the
 * stores' (video index and track index remapped accordingly); all stores must
 * share a structurally-identical skeleton list (the multi-camera case), which
 * becomes the combined skeleton set. Session graph / identities / suggestions /
 * provenance for the combined file are supplied via `options`.
 *
 * @returns the SLP file bytes (round-trips via `readSlpStreaming` / `readSlp`).
 */
export interface MergeStoresOptions {
  sessions?: RecordingSession[];
  identities?: Identity[];
  suggestions?: SuggestionFrame[];
  provenance?: Record<string, unknown>;
  windowFrames?: number;
}

/**
 * Validate the stores share a skeleton, open a writer over the concatenated
 * videos/tracks, and append every store (windowed) with cumulative offsets.
 * Returns the OPEN writer — the caller finalizes via `close()` (bytes) or
 * `writeToSink()` (streamed output). On error the partial file is disposed.
 */
async function buildMergedWriter(
  stores: LazyDataStore[],
  options?: MergeStoresOptions,
): Promise<SlpStreamWriter> {
  if (stores.length === 0) {
    throw new Error("merging SLP stores requires at least one store");
  }
  const sig = skeletonSignature(stores[0].skeletons);
  for (let i = 1; i < stores.length; i++) {
    if (skeletonSignature(stores[i].skeletons) !== sig) {
      throw new Error(
        `merging SLP stores: store ${i} has a different skeleton than store 0; ` +
          "all stores must share the same skeletons to merge into one file.",
      );
    }
  }

  const videos: Video[] = stores.flatMap((s) => s.videos);
  const tracks: Track[] = stores.flatMap((s) => s.tracks);

  const writer = await openSlpWriter({
    skeletons: stores[0].skeletons,
    videos,
    tracks,
    suggestions: options?.suggestions,
    identities: options?.identities,
    sessions: options?.sessions,
    provenance: options?.provenance,
  });

  try {
    let videoOffset = 0;
    let trackOffset = 0;
    for (const store of stores) {
      writer.appendStore(store, {
        videoIndexOffset: videoOffset,
        trackOffset,
        skeletonOffset: 0, // shared skeletons (validated)
        windowFrames: options?.windowFrames,
      });
      videoOffset += store.videos.length;
      trackOffset += store.tracks.length;
    }
  } catch (e) {
    writer.dispose();
    throw e;
  }

  return writer;
}

/**
 * Merge N per-camera {@link LazyDataStore}s into one combined multi-video
 * `.slp`, streaming each store's frames in bounded windows (peak memory ≪ the
 * whole graph). The combined `videos` and `tracks` are the concatenation of the
 * stores' (video index and track index remapped accordingly); all stores must
 * share a structurally-identical skeleton list (the multi-camera case), which
 * becomes the combined skeleton set. Session graph / identities / suggestions /
 * provenance for the combined file are supplied via `options`.
 *
 * @returns the SLP file bytes (round-trips via `readSlpStreaming` / `readSlp`).
 */
export async function saveSlpMergedFromStores(
  stores: LazyDataStore[],
  options?: MergeStoresOptions,
): Promise<Uint8Array> {
  const writer = await buildMergedWriter(stores, options);
  return writer.close();
}

/**
 * Like {@link saveSlpMergedFromStores}, but streams the combined file to `sink`
 * in chunks instead of returning bytes — so neither the input `Labels` graph nor
 * the whole output is ever fully resident. Use with a browser
 * `FileSystemWritableFileStream` (OPFS / save picker) or Node `fs.WriteStream`.
 */
export async function saveSlpMergedToSink(
  stores: LazyDataStore[],
  sink: SlpWriteSink,
  options?: MergeStoresOptions & { chunkBytes?: number },
): Promise<void> {
  const writer = await buildMergedWriter(stores, options);
  await writer.writeToSink(sink, { chunkBytes: options?.chunkBytes });
}

export async function saveSlpToBytes(
  labels: Labels,
  options?: SlpWriteOptions,
): Promise<Uint8Array> {
  return writeSlpBytes(labels, options, { includeEmbeddedData: true });
}

/**
 * Main-thread half of the streaming pkg.slp writer (Phase 2 / Task 2.1, see
 * spike/write-bseam-device): write the small structure ONLY — labels,
 * skeletons, tracks, suggestions, metadata, and `videos_json` (marking
 * embedded videos per the same plan `saveSlpToBytes` would use) — but SKIP
 * the big `video{i}/video` embedded-image datasets (and their sibling
 * `frame_numbers`/`frame_sizes`). A Web Worker later opens the resulting file
 * in append mode and adds those datasets directly to disk via the raw-copy
 * path (see `buildSerializableEmbedPlan`), so a desktop save of a many-GB
 * pkg.slp never needs to hold the embedded image bytes in the wasm heap on
 * the main thread.
 *
 * Shares its setup (embed-mode resolution, lazy materialization, the
 * `embed:"source"` rebuild, and embedding plan computation) with
 * `saveSlpToBytes` via the private `writeSlpBytes` helper — the two functions
 * differ only in whether `writeEmbeddedVideoData` runs afterward.
 *
 * @returns SLP bytes containing the structure only. NOT a directly-loadable
 *   embedded pkg.slp on its own if the plan embeds anything (the videos_json
 *   claims embedded images that aren't actually present yet) — it is an
 *   intermediate artifact meant to be completed by the worker's append step.
 */
export async function saveSlpStructureToBytes(
  labels: Labels,
  options?: SlpWriteOptions,
): Promise<Uint8Array> {
  return writeSlpBytes(labels, options, { includeEmbeddedData: false });
}

async function writeSlpBytes(
  labels: Labels,
  options: SlpWriteOptions | undefined,
  { includeEmbeddedData }: { includeEmbeddedData: boolean },
): Promise<Uint8Array> {
  const embedMode = options?.embed ?? false;

  // Auto-preserve already-embedded videos unless the caller externalizes them.
  // An already-embedded video is ALWAYS raw-copied (preserved) unless the caller
  // explicitly passes embed:"source" (which restores the external source video),
  // so even a bare saveSlpToBytes(labels) keeps its stored images (fixes #213).
  const hasEmbeddedToPreserve =
    embedMode !== "source" && labels.videos.some(isRawCopyable);
  const doEmbed =
    (embedMode !== false && embedMode !== "source") || hasEmbeddedToPreserve;

  // Lazy fast path: skip materialization only when we don't need frame data.
  // Any real embed OR a preserve of already-embedded images needs frame bytes,
  // so materialize and continue with the eager path. Mirrors Python's dispatch.
  if (labels.isLazy) {
    if (doEmbed) {
      labels.materialize();
    } else {
      let lazyWriteLabels: Labels = labels;
      let proceedWithFastPath = true;

      if (embedMode === "source") {
        try {
          lazyWriteLabels = makeLazySourceLabels(labels);
        } catch (e) {
          if (e instanceof LazySourceFallback) {
            // Multi-camera session references would break under the swap.
            // Fall back to materialization + the eager source-mode path.
            labels.materialize();
            proceedWithFastPath = false;
          } else {
            throw e;
          }
        }
      }

      if (proceedWithFastPath) {
        const module = await getH5Module();
        ensureH5StagingDir(module);
        const memPath = `/tmp/sleap_output_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
        const file = new module.File(memPath, "w");
        try {
          writeSlpToFileLazy(file, lazyWriteLabels);
        } finally {
          file.close();
        }
        const fs = getH5FileSystem(module);
        const bytes = fs.readFile!(memPath);
        fs.unlink!(memPath);
        return bytes;
      }
    }
  }

  // Source mode: restore original video paths before writing
  let writeLabels = labels;
  if (embedMode === "source") {
    const restoredVideos = labels.videos.map((video) => {
      if (video.sourceVideo) return video.sourceVideo;
      return video;
    });
    writeLabels = new Labels({
      labeledFrames: labels.labeledFrames.map((frame) => {
        const videoIdx = labels.videos.indexOf(frame.video);
        const restoredVideo =
          videoIdx >= 0 ? restoredVideos[videoIdx] : frame.video;
        return new LabeledFrame({
          video: restoredVideo,
          frameIdx: frame.frameIdx,
          instances: frame.instances,
          centroids: frame.centroids,
          bboxes: frame.bboxes,
          masks: frame.masks,
          labelImages: frame.labelImages,
          rois: frame.rois,
        });
      }),
      videos: restoredVideos,
      skeletons: labels.skeletons,
      tracks: labels.tracks,
      suggestions: labels.suggestions,
      sessions: labels.sessions,
      provenance: labels.provenance,
      rois: labels._staticRois,
      identities: labels.identities,
    });
  }

  const module = await getH5Module();
  ensureH5StagingDir(module);
  const memPath = `/tmp/sleap_output_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;

  // Plan which videos to embed (raw-copy preserved ones + any new-embed target).
  // The plan reads raw blobs from `labels.videos` (the ORIGINAL videos), not the
  // source-mode `writeLabels` copy — but source mode never embeds (plan is null).
  const plan = doEmbed
    ? await planEmbedding(labels, embedMode, options?.onProgress)
    : null;

  const fs = getH5FileSystem(module);
  const file = new module.File(memPath, "w");
  try {
    try {
      writeSlpToFile(file, writeLabels, plan);
      if (includeEmbeddedData && plan && plan.size > 0) {
        await writeEmbeddedVideoData(file, labels, plan);
      }
    } finally {
      file.close();
    }
    return fs.readFile!(memPath);
  } finally {
    try {
      fs.unlink!(memPath);
    } catch {
      // best-effort cleanup
    }
  }
}

export async function writeSlp(
  filename: string,
  labels: Labels,
  options?: SlpWriteOptions,
): Promise<void> {
  const bytes = await saveSlpToBytes(labels, options);

  if (_writeToFile) {
    await _writeToFile(filename, bytes);
  } else {
    throw new Error(
      "writeSlp requires a Node.js environment for file I/O. " +
        "Use saveSlpToBytes() to get the SLP data as a Uint8Array in the browser.",
    );
  }
}

/**
 * Build the exact JSON string written into the `/metadata` `json` attribute by
 * {@link writeMetadata}. Exposed so the in-place saver can re-emit that attribute
 * (via {@link writeLabelTablesInPlace}) without re-serializing the whole file.
 * The caller decides WHEN metadata actually changed and passes this into
 * {@link buildLabelTableUpdate}'s `metadataJson` option; a value-only pose edit
 * leaves it unset so the attribute is untouched.
 */
export function buildMetadataJson(labels: Labels): string {
  const { skeletons, nodes } = serializeSkeletons(labels.skeletons);
  const metadata = {
    version: "2.0.0",
    skeletons,
    nodes,
    videos: [],
    tracks: [],
    suggestions: [],
    negative_anchors: {},
    provenance: labels.provenance ?? {},
  };
  return JSON.stringify(metadata);
}

function writeMetadata(file: any, labels: Labels): void {
  const metadataJson = buildMetadataJson(labels);

  const hasRoiInstance = labels.rois.some((roi) => roi.instance !== null);
  const hasIdentities = (labels.identities?.length ?? 0) > 0;
  const hasPredicted =
    labels.rois.some((r) => r.isPredicted) ||
    labels.masks.some((m) => m.isPredicted) ||
    (labels.labelImages ?? []).some((li) => li.isPredicted);
  const hasMaskInstances = labels.masks.some(
    (m) =>
      m.instance !== null || (m._instanceIdx != null && m._instanceIdx >= 0),
  );
  let formatId =
    (labels.bboxes?.length ?? 0) > 0
      ? 2.0
      : hasPredicted || hasMaskInstances
        ? 1.9
        : (labels.labelImages?.length ?? 0) > 0
          ? 1.8
          : hasRoiInstance
            ? 1.6
            : labels.rois.length > 0 || labels.masks.length > 0
              ? 1.5
              : FORMAT_ID;
  if (hasIdentities) {
    formatId = Math.max(formatId, 1.9);
  }

  // v2.1: spatial transform metadata on masks or label images
  const hasSpatialTransform =
    labels.masks.some((m) => m.hasSpatialTransform) ||
    (labels.labelImages ?? []).some((li) => li.hasSpatialTransform);
  if (hasSpatialTransform) {
    formatId = Math.max(formatId, 2.1);
  }

  // v2.3: virtual on-read video crops (/video_crops dataset). Bumped ONLY when a
  // video is cropped, so uncropped files keep their lower format_id. 2.3 is
  // purely additive — old readers (<= 2.2) skip /video_crops and read the
  // uncropped videos_json safely. Port of Python write_metadata (slp.py:1891-1895).
  if (labels.videos.some((v) => v._cropTuple() !== null)) {
    formatId = Math.max(formatId, 2.3);
  }

  // v2.4: persisted mask from_predicted provenance link. Bumped ONLY when some
  // user mask actually records a *resolvable* link — i.e. its source prediction
  // is itself in the saved mask list (object identity), mirroring the persisted
  // from_predicted index being >= 0. A dangling link (source omitted from the
  // saved frames) writes -1 and must NOT bump the format. The from_predicted
  // column is always written; reads are presence-gated so pre-2.4 files load the
  // link as null.
  const savedMasks = labels.masks;
  const savedMaskSet = new Set<SegmentationMask>(savedMasks);
  if (
    savedMasks.some(
      (m) =>
        (m as UserSegmentationMask).fromPredicted != null &&
        savedMaskSet.has(
          (m as UserSegmentationMask).fromPredicted as SegmentationMask,
        ),
    )
  ) {
    formatId = Math.max(formatId, 2.4);
  }

  // The slim `sessions_json` shape itself (calibration keyed `cam_<i>`, camcorder
  // map keyed by integer index) is a convergence to Python's existing format and
  // does not, on its own, bump `format_id`. `format_id` is a namespace shared with
  // Python (2.5 identity + re-ID embeddings, 2.6 events, 2.7 categories).
  //
  // v2.5: per-detection re-ID identity (/identity/links + /embeddings). Bumped only
  // when a detection actually carries an identity link or embedding; the identity
  // CATALOG alone (identities present but unlinked) does not trigger it.
  if (hasPerDetectionIdentity(labels)) {
    formatId = Math.max(formatId, 2.5);
  }

  // v2.7: class/category subsystem (/categories catalog + /categories/links +
  // category_* embeddings), gated on a detection actually carrying a category.
  if (hasPerDetectionCategory(labels)) {
    formatId = Math.max(formatId, 2.7);
  }

  // v2.8: columnar RecordingSession frame-group data (the `/session_data` group).
  // Bumped ONLY when a session carries frame groups — session-free / single-view
  // files write no group and stay byte-identical below 2.8. Gated on the same
  // predicate `writeSessions` uses to write the group, so the version and the
  // on-disk group never disagree (mirrors Python's group-presence gate).
  if (sessionsHaveFrameGroups(labels.sessions)) {
    formatId = Math.max(formatId, 2.8);
  }

  file.create_group("metadata");
  const metadataGroup = file.get("metadata");
  metadataGroup.create_attribute("format_id", formatId);
  setStringAttr(metadataGroup, "json", metadataJson);
}

function serializeSkeletons(skeletons: Skeleton[]): {
  skeletons: any[];
  nodes: Array<{ name: string }>;
} {
  const nodes: Array<{ name: string }> = [];
  const nodeIndex = new Map<string, number>();

  for (const skeleton of skeletons) {
    for (const nodeName of skeleton.nodeNames) {
      if (!nodeIndex.has(nodeName)) {
        nodeIndex.set(nodeName, nodes.length);
        nodes.push({ name: nodeName });
      }
    }
  }

  const serialized = skeletons.map((skeleton) => {
    const links: Array<Record<string, any>> = [];
    // Track py/id assignments for edge types (jsonpickle convention).
    // First occurrence of each edge type gets py/reduce; subsequent get py/id.
    const edgeTypePyId: Record<number, number> = {};
    let nextPyId = 1;
    let edgeInsertIdx = 0;

    function makeEdgeType(typeVal: number): any {
      if (edgeTypePyId[typeVal] != null) {
        return { "py/id": edgeTypePyId[typeVal] };
      }
      edgeTypePyId[typeVal] = nextPyId++;
      return {
        "py/reduce": [
          { "py/type": "sleap.skeleton.EdgeType" },
          { "py/tuple": [typeVal] },
        ],
      };
    }

    for (const edge of skeleton.edges) {
      const source = nodeIndex.get(edge.source.name) ?? 0;
      const target = nodeIndex.get(edge.destination.name) ?? 0;
      links.push({
        edge_insert_idx: edgeInsertIdx++,
        key: 0,
        source,
        target,
        type: makeEdgeType(1),
      });
    }

    for (const [left, right] of skeleton.symmetryNames) {
      const source = nodeIndex.get(left) ?? 0;
      const target = nodeIndex.get(right) ?? 0;
      links.push({ key: 0, source, target, type: makeEdgeType(2) });
    }

    // Build per-skeleton node index list (global indices of this skeleton's nodes)
    const skeletonNodeIds = skeleton.nodeNames.map(
      (name) => nodeIndex.get(name) ?? 0,
    );

    return {
      directed: true,
      graph: {
        name: skeleton.name ?? "",
        num_edges_inserted: skeleton.edges.length,
      },
      links,
      multigraph: true,
      nodes: skeletonNodeIds.map((id) => ({ id })),
    };
  });

  return { skeletons: serialized, nodes };
}

function writeVideos(file: any, videos: Video[]): void {
  const payload = videos.map((video) => JSON.stringify(serializeVideo(video)));
  file.create_dataset({ name: "videos_json", data: payload });
}

function serializeVideo(video: Video): Record<string, unknown> {
  const backend = { ...(video.backendMetadata ?? {}) } as Record<
    string,
    unknown
  >;
  // ImageVideo (image sequence): the canonical SLP format stores the full
  // ordered list under `filenames` (plural) and only the FIRST image under the
  // scalar `filename` (Python `video_to_dict`, and sleap-io.js's own reader in
  // parsers.ts `resolveVideoFilename`). Emitting the whole list under `filename`
  // makes Python's `make_video` do `Path(list)` and crash (#221).
  if (Array.isArray(video.filename)) {
    backend.filenames = video.filename;
    backend.filename = video.filename[0] ?? "";
  } else if (backend.filename == null) {
    backend.filename = video.filename;
  }

  // Crop unwrap (SLP 2.3): a cropped Video serializes as its UNCROPPED inner so
  // videos_json describes the full frame and old readers never hit an unknown
  // wrapper type. The crop rides /video_crops (see writeVideoCrops); it must NOT
  // enter videos_json. Port of Python video_to_dict (slp.py:503-580).
  const liveBackend = video.backend;
  if (liveBackend instanceof CropVideoBackend) {
    const inner = liveBackend.inner;
    if (inner instanceof CropVideoBackend) {
      // A nested (un-flattened) crop-of-crop can't be represented by the
      // single-crop-per-video /video_crops schema. wrap() only nests when fills
      // differ or the outer rect exceeds the inner frame.
      throw new Error(
        "Cannot serialize a nested crop-of-crop video: the /video_crops format " +
          "stores a single crop per video. Flatten the crop (use matching fills " +
          "and an in-bounds region) before saving.",
      );
    }
    // Serialize the inner's full-frame shape/dataset/fps, not the cropped facade.
    const innerShape =
      inner.shape ??
      (backend.source_shape as number[] | undefined) ??
      video.sourceVideo?.shape;
    if (innerShape != null) backend.shape = [...innerShape];
    else delete backend.shape;
    if (inner.dataset != null) backend.dataset = inner.dataset;
    if (inner.fps != null) backend.fps = inner.fps;
  } else if (liveBackend == null && "crop" in backend) {
    // Closed cropped path: backendMetadata carries the CROPPED shape plus a crop
    // record. Restore the UNCROPPED source shape so videos_json describes the
    // full frame; refuse to emit a self-inconsistent entry when unavailable.
    let srcShape: number[] | null = null;
    if (video.sourceVideo?.shape != null) {
      srcShape = [...video.sourceVideo.shape];
    } else if (backend.source_shape != null) {
      srcShape = [...(backend.source_shape as number[])];
    }
    if (srcShape == null) {
      throw new Error(
        "Cannot serialize closed cropped video: the uncropped source shape is " +
          "unavailable (no source_video and no recorded source_shape), so " +
          "videos_json cannot describe the full frame.",
      );
    }
    backend.shape = srcShape;
  } else {
    if (backend.dataset == null && liveBackend?.dataset)
      backend.dataset = liveBackend.dataset;
    if (backend.shape == null && liveBackend?.shape)
      backend.shape = liveBackend.shape;
    if (backend.fps == null && liveBackend?.fps != null)
      backend.fps = liveBackend.fps;
  }

  // Strip crop keys from videos_json regardless of path (they ride /video_crops).
  delete backend.crop;
  delete backend.crop_fill;
  delete backend.source_shape;

  const entry: Record<string, unknown> = {
    filename: video.filename,
    backend,
  };

  // Serialize the full source_video lineage (filename + backend incl. shape +
  // any deeper chain), not just its filename, so a reload can recover the
  // source's frame extent. Mirrors Python `video_to_dict` (recursive). See #160.
  if (video.sourceVideo) {
    entry.source_video = serializeVideo(video.sourceVideo);
  }

  return entry;
}

/**
 * The full source-video dict to persist for a video that is being embedded, or
 * `null` when there is nothing to record. Prefers an explicit `sourceVideo`
 * (its lineage, incl. shape); otherwise, for a not-yet-embedded video, records
 * the video itself as its own source (mirrors Python `process_and_embed_frames`,
 * where `source_video = video.source_video or video`). An already-embedded video
 * with no known source contributes nothing.
 */
function sourceVideoDict(video: Video): Record<string, unknown> | null {
  if (video.sourceVideo) return serializeVideo(video.sourceVideo);
  if (!video.hasEmbeddedImages) return serializeVideo(video);
  return null;
}

/** Python `METADATA_ATTR_SIZE_LIMIT`: the HDF5 64 KB attribute ceiling. */
const SOURCE_VIDEO_ATTR_LIMIT = 64000;

/**
 * Write a source video's metadata JSON into its `{group}/source_video` HDF5
 * group, the authoritative location an embedded video's source is read from
 * (Python `_write_source_video_json`). Normally a `json` string *attribute*;
 * if the blob would exceed the 64 KB attribute ceiling it goes to a `json`
 * *dataset* in the same group instead (readers check the dataset first).
 */
function writeSourceVideoJson(
  file: any,
  groupPath: string,
  sourceDict: Record<string, unknown>,
): void {
  const blob = JSON.stringify(sourceDict);
  file.create_group(`${groupPath}/source_video`);
  if (new TextEncoder().encode(blob).length <= SOURCE_VIDEO_ATTR_LIMIT) {
    setStringAttr(file.get(`${groupPath}/source_video`), "json", blob);
    return;
  }
  file.create_dataset({ name: `${groupPath}/source_video/json`, data: [blob] });
}

/**
 * Write the top-level `/video_crops` dataset (SLP format 2.3) when at least one
 * video carries a virtual crop. Port of Python `write_video_crops`
 * (slp.py:683-711).
 *
 * Emits one `{ video, crop: [x1,y1,x2,y2], fill }` entry per cropped video in
 * `videos` order, encoded as a single compact JSON string in a length-1 vlen
 * string array — the only h5wasm-producible form accepted by Python's
 * `read_video_crops` (the scalar `np.bytes_` form Python writes is read-equivalent).
 * The dataset is omitted entirely when no video is cropped so uncropped files
 * stay byte-identical and old readers never see an unknown dataset.
 */
function writeVideoCrops(file: any, videos: Video[]): void {
  const crops: Array<Record<string, unknown>> = [];
  for (let i = 0; i < videos.length; i++) {
    const rect = videos[i]._cropTuple();
    if (rect == null) continue;
    crops.push({ video: i, crop: [...rect], fill: videos[i]._cropFill() });
  }
  if (crops.length === 0) return;
  // JSON.stringify uses no spaces by default, matching Python's
  // json.dumps(separators=(",",":")) byte-for-byte.
  const payload = JSON.stringify(crops);
  file.create_dataset({ name: "video_crops", data: [payload] });
}

/**
 * Serialize `tracks` to the exact per-track JSON strings written into the
 * `tracks_json` dataset. Shared by {@link writeTracks} and the in-place
 * writability gate ({@link buildExpectedSidecars}) so the gate's comparison to
 * the on-disk `tracks_json` is byte-exact.
 */
export function buildTracksJson(tracks: Array<{ name: string }>): string[] {
  return tracks.map((track) => JSON.stringify([SPAWNED_ON, track.name]));
}

/**
 * Serialize `suggestions` to the exact per-suggestion JSON strings written into
 * the `suggestions_json` dataset. Shared by {@link writeSuggestions} and the
 * in-place writability gate so the comparison is byte-exact.
 */
export function buildSuggestionsJson(
  suggestions: SuggestionFrame[],
  videos: Video[],
): string[] {
  return suggestions.map((suggestion) =>
    JSON.stringify({
      video: String(videos.indexOf(suggestion.video)),
      frame_idx: suggestion.frameIdx,
      group: suggestion.group ?? "default",
    }),
  );
}

/**
 * A cheap per-video signature — filename(s), shape, and embedded-flag — that
 * uniquely identifies a video WITHOUT reproducing the full `videos_json`
 * serialization (which depends on backend metadata and is easy to diff
 * spuriously). Used by the in-place writability gate to refuse an edit that
 * added/removed/repointed a video (which would leave `videos_json` — and the
 * embedded `video{i}` groups — stale relative to the rewritten frames table).
 * The caller builds the on-disk side from the videos it parsed on open, and the
 * expected side (via {@link buildExpectedSidecars}) from the current labels.
 */
export function buildVideoSignatures(videos: Video[]): string[] {
  return videos.map((v) =>
    JSON.stringify({
      filename: v.filename,
      shape: v.shape ?? null,
      embedded: v.hasEmbeddedImages,
    }),
  );
}

function writeTracks(file: any, tracks: Array<{ name: string }>): void {
  file.create_dataset({ name: "tracks_json", data: buildTracksJson(tracks) });
}

function writeSuggestions(
  file: any,
  suggestions: SuggestionFrame[],
  videos: Video[],
): void {
  file.create_dataset({
    name: "suggestions_json",
    data: buildSuggestionsJson(suggestions, videos),
  });
}

function writeIdentities(file: any, identities: Identity[]): void {
  if (!identities.length) return;
  // Legacy JS-native `identities_json` (typed metadata; kept for older JS readers).
  const payload = identities.map((identity) => {
    const d: Record<string, unknown> = { name: identity.name };
    if (identity.color != null) d.color = identity.color;
    for (const [key, value] of Object.entries(identity.metadata)) {
      if (key !== "name" && key !== "color") {
        d[key] = value;
      }
    }
    return JSON.stringify(d);
  });
  file.create_dataset({ name: "identities_json", data: payload });

  // Python-compatible /identity group (SLP 2.5): a native vlen `name` catalog plus
  // an optional entity-attribute-value metadata table. Written ALONGSIDE
  // identities_json (dual-write) so both Python and older JS readers interoperate.
  // Python has no first-class `color`, so color is folded into metadata["color"]
  // (docs/formats/slp.md). Values are stringified to match Python's string-only EAV.
  file.create_group("identity");
  file.create_dataset({
    name: "identity/name",
    data: identities.map((i) => i.name),
  });
  const metaOwner: number[] = [];
  const metaKey: string[] = [];
  const metaVal: string[] = [];
  identities.forEach((identity, idx) => {
    const meta: Record<string, unknown> = { ...identity.metadata };
    if (identity.color != null && meta.color == null)
      meta.color = identity.color;
    for (const [k, v] of Object.entries(meta)) {
      metaOwner.push(idx);
      metaKey.push(String(k));
      metaVal.push(String(v));
    }
  });
  if (metaOwner.length) {
    file.create_dataset({
      name: "identity/meta_owner",
      data: new Int32Array(metaOwner),
      shape: [metaOwner.length],
      dtype: "<i", // int32 (see the dtype-char note above)
    });
    file.create_dataset({ name: "identity/meta_key", data: metaKey });
    file.create_dataset({ name: "identity/meta_val", data: metaVal });
  }
}

// ---- Per-detection identity/embedding owner types (SLP 2.5) --------------------
// Shared with Python (slp.py OWNER_*). Value 1 is unused; label images get none.
const OWNER_INSTANCE = 0;
const OWNER_CENTROID = 2;
const OWNER_MASK = 3;
const OWNER_BBOX = 4;
const OWNER_ROI = 5;

/** A detection that can carry per-detection identity + category (SLP 2.5/2.7). */
interface IdentityBearing {
  identity?: Identity | null;
  identityScore?: number | null;
  identityEmbedding?: { vector: number[] } | null;
  category?: string | null;
  categoryScore?: number | null;
  categoryEmbedding?: { vector: number[] } | null;
}

/** The five detection collections, each in the SAME global order its modality
 * dataset was written (array index == owner_id). */
interface DetectionCollections {
  instance: IdentityBearing[];
  centroid: IdentityBearing[];
  mask: IdentityBearing[];
  bbox: IdentityBearing[];
  roi: IdentityBearing[];
}

/** Owner-type → ordered detection list (owner_id = index within each list). */
function ownerCollections(
  c: DetectionCollections,
): Array<[number, IdentityBearing[]]> {
  return [
    [OWNER_INSTANCE, c.instance],
    [OWNER_CENTROID, c.centroid],
    [OWNER_MASK, c.mask],
    [OWNER_BBOX, c.bbox],
    [OWNER_ROI, c.roi],
  ];
}

/**
 * Write per-detection identity links (SLP 2.5) to `/identity/links`: one
 * `(owner_type, owner_id, identity_idx, identity_score)` row per detection carrying
 * an `Identity` registered in the catalog. `owner_id` is the detection's global
 * per-modality index (matching how the modality datasets — and Python — are
 * written). Emitted as a flat-2-D `<d` matrix + `field_names` (the h5wasm compound
 * stand-in); the coordinated Python reader (sleap-io#547) converts it. No-op when
 * there are no identities or no linked detections.
 */
function writeIdentityLinks(
  file: any,
  identities: Identity[],
  c: DetectionCollections,
): void {
  if (!identities.length) return;
  const idToIdx = new Map<Identity, number>(identities.map((id, i) => [id, i]));
  const rows: number[][] = [];
  for (const [ownerType, list] of ownerCollections(c)) {
    list.forEach((det, ownerId) => {
      if (det.identity == null) return;
      const idx = idToIdx.get(det.identity);
      if (idx == null) {
        console.warn(
          `Detection references an Identity ("${det.identity.name}") not found in Labels.identities — identity link dropped on save.`,
        );
        return;
      }
      rows.push([ownerType, ownerId, idx, det.identityScore ?? Number.NaN]);
    });
  }
  if (!rows.length) return;
  // The /identity group is created by writeIdentities (runs earlier when
  // identities exist); create defensively in case it does not.
  if (!file.get("identity")) file.create_group("identity");
  createMatrixDataset(
    file,
    "identity/links",
    rows,
    ["owner_type", "owner_id", "identity_idx", "identity_score"],
    "<d",
  );
}

/**
 * Write one `/embeddings` triple: a `(N, D)` float `vectors` matrix (chunked+gzip,
 * no field_names) + parallel `owner_type` / `owner_id` join columns. Used for both
 * the identity vectors (`vectors`/`owner_type`/`owner_id`) and the independent
 * category vectors (`category_vectors`/`category_owner_type`/`category_owner_id`).
 * All vectors in a triple must share one dimension. No-op when none present.
 */
function writeEmbeddingTriple(
  file: any,
  c: DetectionCollections,
  names: [string, string, string],
  get: (d: IdentityBearing) => { vector: number[] } | null | undefined,
): void {
  const vectors: number[][] = [];
  const ownerTypes: number[] = [];
  const ownerIds: number[] = [];
  let dim = -1;
  for (const [ownerType, list] of ownerCollections(c)) {
    list.forEach((det, ownerId) => {
      const emb = get(det);
      if (emb == null) return;
      if (dim === -1) dim = emb.vector.length;
      if (emb.vector.length !== dim) {
        throw new Error(
          `All ${names[0]} must share one dimension; got ${emb.vector.length} and ${dim}.`,
        );
      }
      vectors.push(emb.vector);
      ownerTypes.push(ownerType);
      ownerIds.push(ownerId);
    });
  }
  if (!vectors.length) return;
  if (!file.get("embeddings")) file.create_group("embeddings");
  createGzipFloatMatrix(file, `embeddings/${names[0]}`, vectors, dim);
  file.create_dataset({
    name: `embeddings/${names[1]}`,
    data: new Float64Array(ownerTypes),
    shape: [ownerTypes.length],
    dtype: "<d",
  });
  file.create_dataset({
    name: `embeddings/${names[2]}`,
    data: new Float64Array(ownerIds),
    shape: [ownerIds.length],
    dtype: "<d",
  });
}

/**
 * Write per-detection re-ID (SLP 2.5) + category (SLP 2.7) embeddings to the
 * `/embeddings` group as two independent triples sharing the same owner join.
 * Written whenever a detection carries the corresponding embedding (lossless
 * preserve). No-op when none present.
 */
function writeEmbeddings(file: any, c: DetectionCollections): void {
  writeEmbeddingTriple(
    file,
    c,
    ["vectors", "owner_type", "owner_id"],
    (d) => d.identityEmbedding,
  );
  writeEmbeddingTriple(
    file,
    c,
    ["category_vectors", "category_owner_type", "category_owner_id"],
    (d) => d.categoryEmbedding,
  );
}

/** Distinct non-empty category strings across all detections, in first-seen order
 * (the `/categories/name` catalog). JS categories are plain strings, so the catalog
 * is derived from the strings themselves (no per-category metadata). */
function collectCategoryCatalog(c: DetectionCollections): string[] {
  const seen = new Set<string>();
  const catalog: string[] = [];
  for (const [, list] of ownerCollections(c)) {
    for (const det of list) {
      const cat = det.category;
      if (cat && !seen.has(cat)) {
        seen.add(cat);
        catalog.push(cat);
      }
    }
  }
  return catalog;
}

/**
 * Write the class/category subsystem (SLP 2.7): the `/categories/name` catalog
 * (distinct category strings) + `/categories/links` (owner_type, owner_id,
 * category_idx, category_score) — one row per detection with a non-empty category,
 * flat-2-D `<d`+field_names (the Python reader converts it, sleap-io#548). No-op
 * when no detection carries a category.
 */
function writeCategoryLinks(file: any, c: DetectionCollections): void {
  const catalog = collectCategoryCatalog(c);
  if (!catalog.length) return;
  const catToIdx = new Map<string, number>(catalog.map((n, i) => [n, i]));
  const rows: number[][] = [];
  for (const [ownerType, list] of ownerCollections(c)) {
    list.forEach((det, ownerId) => {
      const cat = det.category;
      if (!cat) return;
      rows.push([
        ownerType,
        ownerId,
        catToIdx.get(cat) as number,
        det.categoryScore ?? Number.NaN,
      ]);
    });
  }
  if (!rows.length) return;
  file.create_group("categories");
  file.create_dataset({ name: "categories/name", data: catalog });
  createMatrixDataset(
    file,
    "categories/links",
    rows,
    ["owner_type", "owner_id", "category_idx", "category_score"],
    "<d",
  );
}

/** Whether any detection satisfies `pred` across all five modalities (used to gate
 * the identity 2.5 / category 2.7 format bumps). */
function anyDetection(
  labels: Labels,
  pred: (d: IdentityBearing) => boolean,
): boolean {
  const any = (list: IdentityBearing[]) => list.some(pred);
  return (
    labels.labeledFrames.some((f) => any(f.instances as IdentityBearing[])) ||
    any(labels.masks as IdentityBearing[]) ||
    any(labels.centroids as IdentityBearing[]) ||
    any(labels.bboxes as IdentityBearing[]) ||
    any(labels.rois as IdentityBearing[])
  );
}

/** Whether any detection carries a per-detection identity or embedding (format 2.5). */
function hasPerDetectionIdentity(labels: Labels): boolean {
  return anyDetection(
    labels,
    (d) => d.identity != null || d.identityEmbedding != null,
  );
}

/** Whether any detection carries a category or category embedding (format 2.7). */
function hasPerDetectionCategory(labels: Labels): boolean {
  return anyDetection(
    labels,
    (d) =>
      (d.category != null && d.category !== "") || d.categoryEmbedding != null,
  );
}

// ---- SLP 2.8 columnar /session_data field names (snake_case, Python-compatible) --
// h5wasm cannot create HDF5 compound datasets, so these struct tables are written as
// flat 2-D float64 matrices + a `field_names` attribute (exactly as points/instances/
// frames are), and the coordinated Python reader rebuilds the structured array via
// `_read_dataset_from_open_file`. float64 losslessly holds every index (≤ 2^53), the
// `-1` sentinels, and the `NaN` scores; the Python reader's int()/float()/isnan casts
// tolerate the f8 backing. points_3d / pred_points_3d are PLAIN float64 matrices with
// NO field_names (Python slices them as `block[:, :3]`).
//
// NOTE: the on-disk dtype MUST be "<d" (float64), NOT "<f8": h5wasm's dtype parser
// keys off the type CHAR and ignores the digit, so "<f8" resolves to float32 (char
// 'f'=4 bytes) and "<i8" to int32 (char 'i'=4) — which would silently truncate 3D
// precision and, worse, CORRUPT any index past 2^24. "<d" = float64 (char 'd'=8).
const SESSION_FRAME_GROUP_FIELDS = ["frame_idx", "ig_start", "ig_end"];
const SESSION_INSTANCE_GROUP_FIELDS = [
  "identity_idx",
  "score",
  "instance_3d_score",
  "pts3d_start",
  "pts3d_end",
  "pts3d_predicted",
  "member_start",
  "member_end",
];
const SESSION_INSTANCE_GROUP_MEMBER_FIELDS = ["camera", "lf", "inst"];

/**
 * Whether any session carries a frame group with ≥1 instance group. Gates both the
 * `/session_data` group and the format 2.8 bump — session-free / single-view files
 * write no group and stay byte-identical.
 */
export function sessionsHaveFrameGroups(sessions: RecordingSession[]): boolean {
  return sessions.some((s) =>
    [...s.frameGroups.values()].some((fg) => fg.instanceGroups.length > 0),
  );
}

/**
 * Coerce an inline/legacy 3-D point row (possibly `null`, or with `null` coords) to a
 * fixed-width float row, mapping every missing entry to `NaN` — never `Number(null)`
 * (=== 0), which would move a missing keypoint to the origin. Ports Python's
 * `_inline_3d_to_array`; `NaN` then round-trips natively in the float dataset.
 */
function coerce3dRow(row: unknown, width: number): number[] {
  const out = new Array<number>(width);
  if (row == null) {
    out.fill(Number.NaN);
    return out;
  }
  const arr = row as ArrayLike<unknown>;
  for (let i = 0; i < width; i++) {
    const v = arr[i];
    out[i] = v == null ? Number.NaN : Number(v);
  }
  return out;
}

/**
 * Build the slim calibration + camera→video map that stays inline in `sessions_json`
 * (the O(cameras) data). Unchanged from the legacy serializer so the on-disk shape of
 * these fields is identical; only the per-frame payload moves to `/session_data`.
 */
function sessionCalibrationDict(
  session: RecordingSession,
  videos: Video[],
): {
  calibration: Record<string, unknown>;
  camcorder_to_video_idx_map: Record<string, number>;
} {
  const calibration: Record<string, unknown> = {
    metadata: session.cameraGroup.metadata ?? {},
  };
  // Key calibration by `cam_<index>` to match Python `camera_group_to_dict`. The key
  // is cosmetic on read (cameras resolve by positional index); the camcorder map
  // keys by bare integer index, exactly as Python does.
  session.cameraGroup.cameras.forEach((camera, idx) => {
    const key = `cam_${idx}`;
    const camData: Record<string, unknown> = {
      name: camera.name ?? key,
      rotation: camera.rvec,
      translation: camera.tvec,
      matrix: camera.matrix,
      distortions: camera.distortions,
    };
    // Always write `size`, using "" when unknown — matches Python `camera_to_dict`
    // (`"" if camera.size is None else list(size)`). Python's `make_camera` does a
    // bare `camera_dict.pop("size")`, so omitting the key KeyErrors on read.
    camData.size = camera.size && camera.size.length === 2 ? camera.size : "";
    calibration[key] = camData;
  });

  const camcorder_to_video_idx_map: Record<string, number> = {};
  for (const [camera, video] of session.videoByCamera.entries()) {
    const camIdx = session.cameraGroup.cameras.indexOf(camera);
    const videoIndex = videos.indexOf(video);
    if (camIdx >= 0 && videoIndex >= 0) {
      camcorder_to_video_idx_map[String(camIdx)] = videoIndex;
    }
  }
  return { calibration, camcorder_to_video_idx_map };
}

/**
 * Resolve an instance group's members to `(cameraIdx, lfIdx, instIdx)` rows — the
 * columnarized `camcorder_to_lf_and_inst_idx_map`. Mirrors the legacy serializer's
 * hybrid derivation: concrete instances resolved via `labeledFrameIndex` first (so
 * edits/reorders are reflected), falling back to the stored index refs so an
 * untouched lazy group serializes with ZERO frame materialization.
 */
function instanceGroupMemberRows(
  group: InstanceGroup,
  session: RecordingSession,
  frameGroup: FrameGroup,
  labeledFrameIndex: Map<LabeledFrame, number>,
): Array<[number, number, number]> {
  const concreteInst = group._instanceByCamera;
  const instRefs = group._instanceRefsByCamera;
  const lfByCamera =
    concreteInst && labeledFrameIndex.size > 0
      ? frameGroup.labeledFrameByCamera
      : undefined;
  const cameras = session.cameraGroup.cameras;
  const rows: Array<[number, number, number]> = [];
  const seen = new Set<Camera>([
    ...(concreteInst?.keys() ?? []),
    ...(instRefs?.keys() ?? []),
  ]);
  for (const camera of seen) {
    const camIdx = cameras.indexOf(camera);
    if (camIdx < 0) continue;
    let pair: [number, number] | undefined;
    const instance = concreteInst?.get(camera);
    if (instance && lfByCamera) {
      const lf = lfByCamera.get(camera);
      const lfIdx = lf !== undefined ? labeledFrameIndex.get(lf) : undefined;
      const instIdx = lf ? lf.instances.indexOf(instance as Instance) : -1;
      if (lfIdx !== undefined && instIdx >= 0) pair = [lfIdx, instIdx];
    }
    if (!pair && instRefs?.has(camera)) pair = instRefs.get(camera);
    if (pair) rows.push([camIdx, pair[0], pair[1]]);
  }
  return rows;
}

/**
 * Create a chunked + gzip 2-D float64 dataset with NO `field_names` attribute — the
 * Python reader slices `points_3d`/`pred_points_3d` as a plain matrix (`block[:, :3]`),
 * and a `field_names` attr would make it rebuild a structured array and break that.
 * gzip requires a non-zero chunk dim, so callers must guard on `rows.length > 0`.
 */
function createGzipFloatMatrix(
  file: any,
  name: string,
  rows: number[][],
  ncols: number,
): void {
  const n = rows.length;
  const data = new Float64Array(n * ncols);
  for (let i = 0; i < n; i++) {
    const r = rows[i];
    const off = i * ncols;
    for (let j = 0; j < ncols; j++) data[off + j] = r[j];
  }
  file.create_dataset({
    name,
    data,
    shape: [n, ncols],
    dtype: "<d", // float64 — see the dtype-char note above ("<f8" would be float32)
    chunks: [Math.min(WRITE_CHUNK_ROWS, Math.max(1, n)), ncols],
    compression: "gzip",
    compression_opts: 1,
  });
}

/**
 * Write `RecordingSession` metadata (SLP 2.8 columnar layout).
 *
 * `sessions_json` holds only the slim per-session blob (calibration +
 * `camcorder_to_video_idx_map` + session metadata + an `fg_start`/`fg_end` half-open
 * range into `session_data/frame_groups`). The unbounded per-frame payload — frame
 * groups, instance groups, the columnarized member map, and 3-D points — goes into
 * the `/session_data` group, written only when a session has frame groups (so
 * session-free / single-view files stay byte-identical and below format 2.8).
 *
 * `labeledFrames` may be empty (the lazy / streaming-header paths): member rows then
 * derive from the stored index refs rather than concrete instances.
 */
function writeSessions(
  file: any,
  sessions: RecordingSession[],
  videos: Video[],
  labeledFrames: LabeledFrame[],
  identities?: Identity[],
): void {
  const labeledFrameIndex = new Map<LabeledFrame, number>();
  labeledFrames.forEach((lf, idx) => {
    labeledFrameIndex.set(lf, idx);
  });

  // Columnar accumulators for /session_data (small: fixed-width ints/floats per frame
  // group / instance group / member, plus the 3-D point rows).
  const fgRows: number[][] = []; // [frame_idx, ig_start, ig_end]
  const igRows: number[][] = []; // SESSION_INSTANCE_GROUP_FIELDS
  const memberRows: number[][] = []; // [camera, lf, inst]
  const fgMeta: string[] = [];
  const igMeta: string[] = [];
  const pts3dRows: number[][] = []; // [x, y, z]
  const predPts3dRows: number[][] = []; // [x, y, z, score]

  const sessionsJson: string[] = [];
  for (const session of sessions) {
    const { calibration, camcorder_to_video_idx_map } = sessionCalibrationDict(
      session,
      videos,
    );

    const fgStart = fgRows.length;
    for (const frameGroup of session.frameGroups.values()) {
      if (!frameGroup.instanceGroups.length) continue;

      const igStart = igRows.length;
      for (const group of frameGroup.instanceGroups) {
        const memberStart = memberRows.length;
        for (const m of instanceGroupMemberRows(
          group,
          session,
          frameGroup,
          labeledFrameIndex,
        )) {
          memberRows.push(m);
        }
        const memberEnd = memberRows.length;

        // Identity → catalog index (-1 when unset / not registered).
        let identityIdx = -1;
        if (group.identity && identities) {
          identityIdx = identities.indexOf(group.identity);
          if (identityIdx < 0) {
            console.warn(
              `InstanceGroup references an Identity ("${group.identity.name}") not found in Labels.identities — identity will be dropped on save.`,
            );
          }
        }

        const score = group.score != null ? group.score : Number.NaN;

        // 3-D points → points_3d / pred_points_3d row range (NaN = missing keypoint).
        let pts3dStart = -1;
        let pts3dEnd = -1;
        let pts3dPredicted = 0;
        let i3dScore = Number.NaN;
        const inst3d = group.instance3d;
        const points3d = inst3d?.points ?? group.points;
        if (points3d) {
          const isPred =
            inst3d instanceof PredictedInstance3D && inst3d.pointScores != null;
          if (isPred) {
            const scores = (inst3d as PredictedInstance3D)
              .pointScores as number[];
            pts3dStart = predPts3dRows.length;
            points3d.forEach((row, i) => {
              const xyz = coerce3dRow(row, 3);
              const s = scores[i];
              predPts3dRows.push([
                xyz[0],
                xyz[1],
                xyz[2],
                s == null ? Number.NaN : Number(s),
              ]);
            });
            pts3dEnd = predPts3dRows.length;
            pts3dPredicted = 1;
          } else {
            pts3dStart = pts3dRows.length;
            for (const row of points3d) pts3dRows.push(coerce3dRow(row, 3));
            pts3dEnd = pts3dRows.length;
          }
          if (inst3d?.score != null) i3dScore = inst3d.score;
        }

        igRows.push([
          identityIdx,
          score,
          i3dScore,
          pts3dStart,
          pts3dEnd,
          pts3dPredicted,
          memberStart,
          memberEnd,
        ]);
        igMeta.push(
          group.metadata && Object.keys(group.metadata).length
            ? JSON.stringify(group.metadata)
            : "",
        );
      }
      const igEnd = igRows.length;
      fgRows.push([frameGroup.frameIdx, igStart, igEnd]);
      fgMeta.push(
        frameGroup.metadata && Object.keys(frameGroup.metadata).length
          ? JSON.stringify(frameGroup.metadata)
          : "",
      );
    }
    const fgEnd = fgRows.length;

    sessionsJson.push(
      JSON.stringify({
        calibration,
        camcorder_to_video_idx_map,
        metadata: session.metadata ?? {},
        fg_start: fgStart,
        fg_end: fgEnd,
      }),
    );
  }

  file.create_dataset({ name: "sessions_json", data: sessionsJson });

  // Columnar /session_data group — only when some session has frame groups.
  if (fgRows.length > 0) {
    file.create_group("session_data");
    createMatrixDataset(
      file,
      "session_data/frame_groups",
      fgRows,
      SESSION_FRAME_GROUP_FIELDS,
      "<d",
    );
    createMatrixDataset(
      file,
      "session_data/instance_groups",
      igRows,
      SESSION_INSTANCE_GROUP_FIELDS,
      "<d",
    );
    createMatrixDataset(
      file,
      "session_data/instance_group_members",
      memberRows,
      SESSION_INSTANCE_GROUP_MEMBER_FIELDS,
      "<d",
    );
    if (pts3dRows.length > 0) {
      createGzipFloatMatrix(file, "session_data/points_3d", pts3dRows, 3);
    }
    if (predPts3dRows.length > 0) {
      createGzipFloatMatrix(
        file,
        "session_data/pred_points_3d",
        predPts3dRows,
        4,
      );
    }
    // Per-row metadata JSON blobs, presence-guarded (omitted when all empty).
    if (fgMeta.some((s) => s.length > 0)) {
      file.create_dataset({
        name: "session_data/frame_group_meta",
        data: fgMeta,
      });
    }
    if (igMeta.some((s) => s.length > 0)) {
      file.create_dataset({
        name: "session_data/instance_group_meta",
        data: igMeta,
      });
    }
  }
}

// One-time warning keys for the point-count enforcement below (avoids log spam
// across 100k+ frames when an upstream produced malformed instances).
const _spanWarned = new Set<string>();

/**
 * Push exactly `nNodes` point rows for `instance` into `target`, enforcing the SLP
 * invariant that an instance's point count equals its skeleton's node count. Python
 * `read_instances` requires `point_id_end - point_id_start == n_nodes`; an instance
 * that somehow holds a different number of points (e.g. built from a coordinate array
 * whose length ≠ the skeleton — see `pointsFromArray`) would otherwise write an
 * inconsistent span that breaks the Python reader (luc3d#161). Missing points are
 * padded as invisible `NaN` rows and any extras are dropped, warning once.
 *
 * `predicted` selects a 5-col row (x, y, visible, complete, score) vs 4-col.
 */
function emitInstancePoints(
  target: number[][],
  instance: Instance | PredictedInstance,
  predicted: boolean,
): void {
  const nNodes = instance.skeleton.nodeNames.length;
  const pts = instance.points;
  if (pts.length !== nNodes && !_spanWarned.has(instance.skeleton.name ?? "")) {
    _spanWarned.add(instance.skeleton.name ?? "");
    console.warn(
      `Instance has ${pts.length} point(s) but its skeleton "${instance.skeleton.name ?? "?"}" has ${nNodes} node(s); ` +
        "padding/truncating to the node count so the written SLP stays self-consistent (a mismatch would break the Python reader).",
    );
  }
  for (let i = 0; i < nNodes; i++) {
    const p = pts[i];
    if (p) {
      const row = [p.xy[0], p.xy[1], p.visible ? 1 : 0, p.complete ? 1 : 0];
      if (predicted) row.push((p as { score?: number }).score ?? 0);
      target.push(row);
    } else {
      target.push(
        predicted
          ? [Number.NaN, Number.NaN, 0, 0, 0]
          : [Number.NaN, Number.NaN, 0, 0],
      );
    }
  }
}

/** The five mutable label tables as `number[][]` matrices (shared row builder). */
export interface LabelTableRows {
  frames: number[][];
  instances: number[][];
  points: number[][];
  predPoints: number[][];
  negativeFrames: number[][];
}

/** Build the `negative_frames` rows (`[video_id, frame_idx]`), in frame order. */
function buildNegativeFrameRows(labels: Labels): number[][] {
  const rows: number[][] = [];
  for (const frame of labels.labeledFrames) {
    if (!frame.isNegative) continue;
    const videoIndex = Math.max(0, labels.videos.indexOf(frame.video));
    rows.push([videoIndex, frame.frameIdx]);
  }
  return rows;
}

/**
 * Build the `frames`/`instances`/`points`/`pred_points`/`negative_frames` row
 * matrices from `labels`, exactly as the eager writer would serialize them.
 *
 * Shared by {@link writeLabeledFrames} (the full save path) and
 * {@link buildLabelTableUpdate} (the in-place save path) so both emit
 * byte-identical columns using the same field-name constants. Contains no HDF5
 * I/O — pure `Labels` → `number[][]`.
 */
export function buildLabelTableRows(labels: Labels): LabelTableRows {
  const frames: number[][] = [];
  const instances: number[][] = [];
  const points: number[][] = [];
  const predPoints: number[][] = [];
  const instanceIndex = new Map<Instance, number>();
  const predictedLinks: Array<[number, PredictedInstance]> = [];

  for (const labeledFrame of labels.labeledFrames) {
    const frameId = frames.length;
    const instanceStart = instances.length;
    const videoIndex = Math.max(0, labels.videos.indexOf(labeledFrame.video));

    for (const instance of labeledFrame.instances) {
      const instanceId = instances.length;
      instanceIndex.set(instance as Instance, instanceId);

      const skeletonId = Math.max(
        0,
        labels.skeletons.indexOf(instance.skeleton),
      );
      const trackId = instance.track
        ? labels.tracks.indexOf(instance.track)
        : -1;
      const trackingScore = instance.trackingScore ?? 0;
      const fromPredicted = -1;
      let score = 0;
      let pointStart = 0;
      let pointEnd = 0;

      if (instance instanceof PredictedInstance) {
        score = instance.score ?? 0;
        pointStart = predPoints.length;
        emitInstancePoints(predPoints, instance, true);
        pointEnd = predPoints.length;
      } else {
        pointStart = points.length;
        emitInstancePoints(points, instance as Instance, false);
        pointEnd = points.length;
        if (instance.fromPredicted) {
          predictedLinks.push([instanceId, instance.fromPredicted]);
        }
      }

      instances.push([
        instanceId,
        instance instanceof PredictedInstance ? 1 : 0,
        frameId,
        skeletonId,
        trackId,
        fromPredicted,
        score,
        pointStart,
        pointEnd,
        trackingScore,
      ]);
    }

    const instanceEnd = instances.length;
    frames.push([
      frameId,
      videoIndex,
      labeledFrame.frameIdx,
      instanceStart,
      instanceEnd,
    ]);
  }

  for (const [instanceId, fromPredictedInstance] of predictedLinks) {
    const fromIndex = instanceIndex.get(fromPredictedInstance as Instance);
    if (fromIndex != null) {
      instances[instanceId][5] = fromIndex;
    } else {
      instances[instanceId][5] = -1;
    }
  }

  return {
    frames,
    instances,
    points,
    predPoints,
    negativeFrames: buildNegativeFrameRows(labels),
  };
}

function writeLabeledFrames(file: any, labels: Labels): void {
  const { frames, instances, points, predPoints } = buildLabelTableRows(labels);

  // `frames` stays a flat int matrix (Python reads it positionally), written
  // CHUNKED + unlimited (resizable=true) so it can be resized in place on an edit
  // re-save. The pose tables are HDF5 COMPOUND with integer id columns — so files
  // open in Python sleap-io / the PyQt GUI (#218) — AND chunked + unlimited, so
  // the fast in-place edit-save can patch them via write_slice / resize without
  // rewriting the embedded video groups. See createCompoundDataset /
  // writeLabelTablesInPlace.
  createMatrixDataset(file, "frames", frames, FRAMES_FIELDS, "<i8", true);
  createCompoundDataset(file, "instances", instances, INSTANCE_COMPOUND_FIELDS);
  createCompoundDataset(file, "points", points, POINT_COMPOUND_FIELDS);
  createCompoundDataset(
    file,
    "pred_points",
    predPoints,
    PRED_POINT_COMPOUND_FIELDS,
  );
}

function writeNegativeFrames(file: any, labels: Labels): void {
  const rows = buildNegativeFrameRows(labels);
  if (!rows.length) return;
  createCompoundDataset(
    file,
    "negative_frames",
    rows,
    NEGATIVE_FRAME_COMPOUND_FIELDS,
  );
}

// ===========================================================================
// Fast in-place label save (edit re-save)
//
// Re-save an existing `.pkg.slp` by patching ONLY the small label tables
// (frames/instances/points/pred_points/negative_frames) + the /metadata `json`
// attribute, NEVER reading or writing the multi-GB embedded `video{i}/video`
// groups. On a network share this makes an edit-save ~100x faster (no re-copy of
// the embedded images). Feasibility + primitives were validated against h5wasm:
// `write_slice` (same-shape overwrite) works on flat AND compound datasets with
// ZERO file growth; `resize` (H5Dset_extent) grows/shrinks chunked+unlimited
// datasets in place. Python-written files store points `visible`/`complete` as an
// HDF5 enum/bool (typeClass 8) that h5wasm cannot write — such files must fall
// back to a full re-save (see checkInPlaceWritable).
// ===========================================================================

/** HDF5 datatype class id for an enum member (bool/enum) — not writable by h5wasm. */
const H5T_ENUM_CLASS = 8;
/** HDF5 datatype class id for a float member. */
const H5T_FLOAT_CLASS = 1;

/** One label table as logical rows + its column (field) names. */
export interface LabelTable {
  fields: string[];
  rows: number[][];
}

/**
 * The small, mutable structure of a `.pkg.slp`, ready to patch in place. Each
 * table is `{ fields, rows }` (logical `number[][]`, layout-agnostic — the writer
 * packs it into whatever on-disk layout the target dataset uses). `metadataJson`
 * is set only when metadata actually changed (caller decides) so a value-only
 * pose edit leaves the `/metadata` attribute untouched.
 */
export interface LabelTableUpdate {
  frames: LabelTable;
  instances: LabelTable;
  points: LabelTable;
  predPoints: LabelTable;
  negativeFrames?: LabelTable;
  metadataJson?: string;
}

/**
 * Build a {@link LabelTableUpdate} from `labels` — the in-place counterpart to
 * the eager writer. Reuses {@link buildLabelTableRows} so the columns are
 * byte-identical to a full save. `negativeFrames` is always included (possibly
 * with 0 rows) so an in-place save can also empty an existing `negative_frames`
 * table when the user removed all negative frames. Pass `opts.metadataJson`
 * (e.g. from {@link buildMetadataJson}) only when metadata changed.
 */
export function buildLabelTableUpdate(
  labels: Labels,
  opts?: { metadataJson?: string },
): LabelTableUpdate {
  const r = buildLabelTableRows(labels);
  const update: LabelTableUpdate = {
    frames: { fields: FRAMES_FIELDS, rows: r.frames },
    instances: { fields: INSTANCES_FIELDS, rows: r.instances },
    points: { fields: POINTS_FIELDS, rows: r.points },
    predPoints: { fields: PRED_POINTS_FIELDS, rows: r.predPoints },
    negativeFrames: { fields: NEGATIVE_FRAMES_FIELDS, rows: r.negativeFrames },
  };
  if (opts?.metadataJson !== undefined) update.metadataJson = opts.metadataJson;
  return update;
}

/** One on-disk compound member's identity (for the writability gate). */
export interface OnDiskMember {
  name: string;
  /** HDF5 datatype class id (`H5Tget_class`): 0=int, 1=float, 8=enum. */
  typeClass: number;
  /** Member size in bytes. */
  size?: number;
  /** Whether the integer member is signed (h5wasm reports float as unsigned). */
  signed?: boolean;
}

/** On-disk shape/layout of one label table, as seen by the writability gate. */
export interface OnDiskTable {
  /** Current row count (axis-0 extent). */
  rows: number;
  /** Column count (matrix width, or compound member count). */
  cols: number;
  /** Storage layout: a 2-D numeric matrix, or a 1-D compound record dataset. */
  layout: "flat" | "compound";
  /** Whether the dataset is chunked (⇒ resizable on axis 0). */
  chunked?: boolean;
  /** Compound members (only for `layout: "compound"`). */
  members?: OnDiskMember[];
}

/** The on-disk state of the label tables the gate inspects. */
export interface OnDiskTables {
  frames?: OnDiskTable;
  instances?: OnDiskTable;
  points?: OnDiskTable;
  predPoints?: OnDiskTable;
  negativeFrames?: OnDiskTable;
}

/** Result of {@link checkInPlaceWritable}. */
export type InPlaceWritable = { ok: true } | { ok: false; reason: string };

/** Minimal `getDatasetMeta` shape consumed by {@link onDiskTableFromMeta}. */
export interface DatasetMetaLike {
  shape?: number[];
  metadata?: {
    chunks?: unknown;
    compound_type?: {
      members?: Array<{
        name: string;
        type: number;
        size?: number;
        signed?: boolean;
      }>;
    };
  };
}

/**
 * Convert a `getDatasetMeta` result (from the streaming reader) into an
 * {@link OnDiskTable} for the writability gate. Returns `undefined` when the
 * dataset is absent. Detects compound vs flat from the presence of
 * `compound_type.members`, and resizability from the presence of `chunks`.
 */
export function onDiskTableFromMeta(
  meta: DatasetMetaLike | null | undefined,
): OnDiskTable | undefined {
  if (!meta || !Array.isArray(meta.shape)) return undefined;
  const md = meta.metadata;
  const members = md?.compound_type?.members;
  const chunked = !!md?.chunks;
  if (members && members.length > 0) {
    return {
      rows: meta.shape[0] ?? 0,
      cols: members.length,
      layout: "compound",
      chunked,
      members: members.map((m) => ({
        name: m.name,
        typeClass: m.type,
        size: m.size,
        signed: m.signed,
      })),
    };
  }
  return {
    rows: meta.shape[0] ?? 0,
    cols: meta.shape[1] ?? 1,
    layout: "flat",
    chunked,
  };
}

/**
 * The JSON sidecar datasets/attribute that the in-place writer does NOT rewrite
 * but which the label tables reference by index — so an edit that changed any of
 * them would leave the (rewritten) tables pointing into stale sidecar data. The
 * gate compares these to prove the save is confined to the label tables (+ the
 * `/metadata` json attr). `tracks_json` / `suggestions_json` are the on-disk
 * per-element JSON string arrays; `metadataJson` is the `/metadata` `json`
 * attribute string. `null` means the dataset/attr is absent.
 *
 * The video SET is covered SEMANTICALLY via `videos` (per-video signatures, see
 * {@link buildVideoSignatures}) rather than a byte-exact `videos_json` compare.
 *
 * NOT covered here (the in-place path is only safe for edits that leave these
 * unchanged; the caller must guarantee it, and the app's post-write verify is the
 * final backstop): `identities_json`, `sessions_json`, and the annotation
 * datasets (rois/masks/bboxes/centroids/label_images) — their serializers are
 * complex and reproducing them risks spurious refusals. Extending the gate to
 * those is a follow-up.
 */
export interface OnDiskSidecars {
  tracksJson: string[] | null;
  suggestionsJson: string[] | null;
  metadataJson: string | null;
  /** Per-video signatures (see {@link buildVideoSignatures}). */
  videos: string[] | null;
}

/**
 * Serialize the sidecars a `labels` would write, for exact comparison against
 * {@link OnDiskSidecars} in {@link checkInPlaceWritable}. Reuses the same
 * serializers the full writer uses ({@link buildTracksJson},
 * {@link buildSuggestionsJson}, {@link buildMetadataJson}) so the comparison is
 * byte-exact.
 */
export function buildExpectedSidecars(labels: Labels): OnDiskSidecars {
  return {
    tracksJson: buildTracksJson(labels.tracks),
    suggestionsJson: buildSuggestionsJson(labels.suggestions, labels.videos),
    metadataJson: buildMetadataJson(labels),
    videos: buildVideoSignatures(labels.videos),
  };
}

/** Element-wise string-array equality (null treated as empty). */
function sameStringArray(a: string[] | null, b: string[] | null): boolean {
  const aa = a ?? [];
  const bb = b ?? [];
  if (aa.length !== bb.length) return false;
  for (let i = 0; i < aa.length; i++) if (aa[i] !== bb[i]) return false;
  return true;
}

/**
 * Decide whether `update` can be written into `onDisk` IN PLACE, or whether the
 * caller must fall back to a full re-save. NOT writable when:
 *  - any table's compound members include an HDF5 enum/bool (typeClass 8) —
 *    Python-written points store `visible`/`complete` this way and h5wasm cannot
 *    write them; OR
 *  - a table's row count changes (structural edit) but the on-disk dataset is
 *    contiguous (not chunked ⇒ not resizable) — an old file written before the
 *    chunked-table change (value-only edits are fine on contiguous); OR
 *  - the track set/order or the suggestions changed — the in-place writer does
 *    NOT rewrite `tracks_json` / `suggestions_json`, so the (rewritten) instances
 *    table's `track` indices would point into a STALE `tracks_json` → silent
 *    track/suggestion corruption on reload (the HIGH bug this guards); OR
 *  - the VIDEO SET changed (a video was added/removed/repointed) — the in-place
 *    writer does NOT rewrite `videos_json` or the embedded `video{i}` groups, so
 *    the (rewritten) frames table's `video` indices would point into a stale
 *    video list. Compared semantically via per-video signatures
 *    ({@link buildVideoSignatures}), not a byte-exact `videos_json` diff; OR
 *  - the `/metadata` json changed (e.g. a skeleton or provenance edit) but
 *    `update.metadataJson` was not set to carry that change into the attr.
 *
 * A pose/point/visibility/score/track-membership-preserving edit still passes.
 *
 * CALLER CONTRACT:
 *  - `onDisk` MUST describe every label table that actually exists on disk
 *    (frames/instances/points/pred_points/negative_frames) — an omitted table
 *    skips its resizability check here; the post-resize assertion inside
 *    {@link writeLabelTablesInPlace} is the loud backstop if one is missed.
 *  - This gate proves confinement for the label tables + tracks + suggestions +
 *    videos + metadata ONLY. It does NOT check `identities_json`, `sessions_json`,
 *    or the annotation datasets (rois/masks/bboxes/centroids/label_images). The
 *    in-place path therefore ASSUMES those are unchanged; the caller (app) MUST
 *    NOT route an edit that touches them to in-place (route it to a full re-save),
 *    and its post-write verify is the final backstop.
 * Pure function (no I/O): the caller reads `sidecars.onDisk` from the file (the
 * videos it parsed on open) and builds `sidecars.expected` via
 * {@link buildExpectedSidecars}.
 */
export function checkInPlaceWritable(
  update: LabelTableUpdate,
  onDisk: OnDiskTables,
  sidecars: { onDisk: OnDiskSidecars; expected: OnDiskSidecars },
): InPlaceWritable {
  const checks: Array<
    [string, LabelTable | undefined, OnDiskTable | undefined]
  > = [
    ["frames", update.frames, onDisk.frames],
    ["instances", update.instances, onDisk.instances],
    ["points", update.points, onDisk.points],
    ["pred_points", update.predPoints, onDisk.predPoints],
    ["negative_frames", update.negativeFrames, onDisk.negativeFrames],
  ];
  for (const [name, table, od] of checks) {
    if (!od) continue; // absent on disk → writer creates it (resizable)
    if (od.layout === "compound" && od.members) {
      const enumMember = od.members.find((m) => m.typeClass === H5T_ENUM_CLASS);
      if (enumMember) {
        return {
          ok: false,
          reason: `table "${name}" member "${enumMember.name}" is an HDF5 enum/bool (typeClass ${H5T_ENUM_CLASS}); h5wasm cannot write it — re-save the whole file instead`,
        };
      }
    }
    if (table && table.rows.length !== od.rows && !od.chunked) {
      return {
        ok: false,
        reason: `table "${name}" row count changed (${od.rows} → ${table.rows.length}) but the on-disk dataset is contiguous (not resizable) — re-save the whole file instead`,
      };
    }
  }

  // Confinement: the label tables index into tracks_json / suggestions_json /
  // metadata, which the in-place writer does NOT rewrite (except the /metadata
  // attr, and only when update.metadataJson is set). If any changed, an in-place
  // save would desync — refuse so the caller full-rewrites.
  if (
    !sameStringArray(sidecars.onDisk.tracksJson, sidecars.expected.tracksJson)
  ) {
    return {
      ok: false,
      reason:
        "the track set/order changed; an in-place save cannot update tracks_json (the instances table would point into a stale track list) — re-save the whole file instead",
    };
  }
  if (
    !sameStringArray(
      sidecars.onDisk.suggestionsJson,
      sidecars.expected.suggestionsJson,
    )
  ) {
    return {
      ok: false,
      reason:
        "the suggestions changed; an in-place save cannot update suggestions_json — re-save the whole file instead",
    };
  }
  if (!sameStringArray(sidecars.onDisk.videos, sidecars.expected.videos)) {
    return {
      ok: false,
      reason:
        "the video set changed (a video was added, removed, or repointed); an in-place save cannot update videos_json or the embedded video groups (the frames table would point into a stale video list) — re-save the whole file instead",
    };
  }
  if (
    sidecars.onDisk.metadataJson !== sidecars.expected.metadataJson &&
    update.metadataJson === undefined
  ) {
    return {
      ok: false,
      reason:
        "the /metadata json changed (e.g. a skeleton or provenance edit) but update.metadataJson was not set to carry it — set update.metadataJson, or re-save the whole file instead",
    };
  }

  return { ok: true };
}

/**
 * Pack one logical column (`values`, extracted from the update rows) into the
 * TypedArray that matches an on-disk compound `member`'s dtype, honoring
 * h5wasm's single-char member dtype (int/float, width, signedness). Int64
 * members become Big(U)Int64Array (values are exact ≤ 2^53). Throws on an enum
 * member (the gate should have refused the file first).
 */
function memberColumnTypedArray(
  member: { name: string; type: number; size?: number; signed?: boolean },
  values: number[],
): ArrayBufferView {
  const size = member.size ?? 8;
  if (member.type === H5T_ENUM_CLASS) {
    throw new Error(
      `cannot write enum/bool compound member "${member.name}" in place`,
    );
  }
  if (member.type === H5T_FLOAT_CLASS) {
    return size === 4 ? Float32Array.from(values) : Float64Array.from(values);
  }
  // Integer member. h5wasm reports floats as signed:false, so default ints to
  // signed unless explicitly unsigned (matches readCompoundColumnsManual).
  const signed = member.signed !== false;
  if (size === 8) {
    const big = values.map((v) => BigInt(Math.trunc(v)));
    return signed ? BigInt64Array.from(big) : BigUint64Array.from(big);
  }
  if (size === 1) return (signed ? Int8Array : Uint8Array).from(values);
  if (size === 2) return (signed ? Int16Array : Uint16Array).from(values);
  return (signed ? Int32Array : Uint32Array).from(values);
}

/**
 * Overwrite one already-open dataset (`diskName`) with `table`'s rows, resizing
 * first when the row count changed. Handles BOTH on-disk layouts:
 *  - flat 2-D matrix: `write_slice([[0,n],[0,cols]], Float64Array)` (h5wasm
 *    converts f64 → the dataset's `<i8`/`<f8` dtype, mirroring how it was
 *    written); and
 *  - 1-D compound: a per-member column `Map` keyed by the on-disk member name,
 *    each column packed to the member's dtype.
 * Creates the dataset (chunked/resizable, flat) if it is absent AND non-empty.
 * NEVER touches any `video{i}` group.
 */
function writeOneTableInPlace(
  file: any,
  diskName: string,
  table: LabelTable,
  dtype: string,
): void {
  const rowCount = table.rows.length;
  const colCount = table.fields.length;
  const ds = file.get(diskName);
  if (!ds) {
    // Absent on disk: create it fresh (resizable, matching the eager writer).
    // Skip an empty table so we don't add a dataset a file never had.
    if (rowCount === 0) return;
    createMatrixDataset(file, diskName, table.rows, table.fields, dtype, true);
    return;
  }

  const md = ds.metadata;
  const members = md?.compound_type?.members as
    | Array<{ name: string; type: number; size?: number; signed?: boolean }>
    | undefined;
  const isCompound = !!(members && members.length > 0);
  const curRows = ds.shape?.[0] ?? 0;

  if (curRows !== rowCount) {
    ds.resize(isCompound ? [rowCount] : [rowCount, colCount]);
    // h5wasm SILENTLY no-ops resize() on a non-resizable (contiguous) dataset
    // rather than throwing — which would leave stale ghost rows / a short write.
    // Read the shape back and abort loudly so the caller full-rewrites instead
    // of corrupting the file. (The gate should have refused this, but this is the
    // backstop if the caller passed an incomplete OnDiskTables.)
    const newRows = ds.shape?.[0] ?? -1;
    if (newRows !== rowCount) {
      throw new Error(
        `in-place resize of "${diskName}" did not take effect (had ${curRows} rows, wanted ${rowCount}, got ${newRows}); the dataset is not resizable — re-save the whole file instead`,
      );
    }
  }
  if (rowCount === 0) return; // resized to empty; nothing to write

  if (isCompound) {
    const fieldIndex = new Map(table.fields.map((f, i) => [f, i]));
    const columns = new Map<string, ArrayBufferView>();
    for (const m of members!) {
      const ci = fieldIndex.get(m.name);
      if (ci === undefined) {
        throw new Error(
          `in-place update for "${diskName}" is missing compound member "${m.name}"`,
        );
      }
      const colVals = table.rows.map((row) => row[ci]);
      columns.set(m.name, memberColumnTypedArray(m, colVals));
    }
    ds.write_slice([[0, rowCount]], columns);
  } else {
    const flat = new Float64Array(rowCount * colCount);
    for (let i = 0; i < rowCount; i++) {
      const row = table.rows[i];
      const off = i * colCount;
      for (let j = 0; j < colCount; j++) flat[off + j] = row[j];
    }
    ds.write_slice(
      [
        [0, rowCount],
        [0, colCount],
      ],
      flat,
    );
  }
}

/**
 * Patch the label tables of an ALREADY-OPEN, writable h5wasm `File` (opened r+ /
 * mode `"a"`) in place from `update`, then (if `update.metadataJson` is set)
 * replace the `/metadata` `json` attribute. Works on any open h5wasm File —
 * Node FS, MEMFS, or the bridged write device (the streaming worker).
 *
 * DATA SAFETY: only the frames/instances/points/pred_points/negative_frames
 * datasets and the `/metadata` attribute are ever referenced — the embedded
 * `video{i}/video` groups are never opened, read, or written. The caller is
 * responsible for having gated with {@link checkInPlaceWritable} (and, ideally,
 * a post-write verify) before calling this: the gate proves the edit is confined
 * to these tables (tracks/suggestions/metadata unchanged or carried). As a
 * backstop for an incomplete gate, each per-table resize is asserted to have
 * actually taken effect (h5wasm silently no-ops resize on a non-resizable
 * dataset) — a resize that can't happen THROWS here rather than writing ghost
 * rows, so the caller falls back to a full re-save.
 *
 * Does NOT flush or close the file — the caller owns the file lifecycle.
 */
export function writeLabelTablesInPlace(
  file: any,
  update: LabelTableUpdate,
): void {
  // The `dtype` here is only consulted when a table is ABSENT on disk and gets
  // created fresh as a flat matrix (writeOneTableInPlace → createMatrixDataset);
  // for an existing dataset the on-disk layout/dtype wins and this arg is ignored.
  // Even so, `instances` uses "<d" (true float64), NEVER "<f8" (== float32 in
  // h5wasm — only the type char is read): its id / point_id columns must survive
  // integers past 2^24 (#231). frames stays int32 ("<i8"); points/pred_points are
  // coordinate-only, so float32 ("<f8") is acceptable.
  writeOneTableInPlace(file, "frames", update.frames, "<i8");
  writeOneTableInPlace(file, "instances", update.instances, "<d");
  writeOneTableInPlace(file, "points", update.points, "<f8");
  writeOneTableInPlace(file, "pred_points", update.predPoints, "<f8");
  if (update.negativeFrames) {
    writeOneTableInPlace(file, "negative_frames", update.negativeFrames, "<i8");
  }

  if (update.metadataJson !== undefined) {
    const metadataGroup = file.get("metadata");
    if (metadataGroup) {
      // The json attr is a fixed-length S-string; a new value of a different
      // length can't overwrite in place, so delete + recreate. (Attribute
      // storage may relocate within the object header — harmless to dataset
      // bytes, but noted as a risk for very large metadata.)
      try {
        metadataGroup.delete_attribute("json");
      } catch {
        // no existing attr — fine
      }
      setStringAttr(metadataGroup, "json", update.metadataJson);
    }
  }
}

/**
 * Plan which videos to embed and how. An already-embedded video is ALWAYS
 * raw-copied (its FULL stored set preserved via `getFrameBuffer`, no decode)
 * regardless of `embedMode`; only a NEW embed of a continuous video takes the
 * legacy getFrame+encode path — and only when a real embed mode was requested.
 * (`embedMode === "source"` never reaches here — the caller externalizes.)
 */
async function planEmbedding(
  labels: Labels,
  embedMode: boolean | string,
  onProgress?: SlpWriteOptions["onProgress"],
): Promise<EmbedPlan> {
  const plan: EmbedPlan = new Map();
  const isRealMode = embedMode !== false && embedMode !== "source";
  // Progress spans every frame read from a continuous video, across videos.
  const toEncode = labels.videos.map((video, vi) =>
    isRealMode && video.backend && !isRawCopyable(video)
      ? selectEmbedFrames(labels, vi, embedMode)
      : [],
  );
  const progress = {
    done: 0,
    total: toEncode.reduce((n, frames) => n + frames.length, 0),
    report: onProgress,
  };
  for (let vi = 0; vi < labels.videos.length; vi++) {
    const video = labels.videos[vi];
    if (isRawCopyable(video)) {
      // Preserve the FULL stored set regardless of embedMode (semantics A).
      await video.backend!.ensureLoaded?.();
      const frameNumbers = video.embeddedFrameIndices ?? [];
      if (frameNumbers.length === 0) continue;
      plan.set(vi, {
        kind: "raw",
        videoIndex: vi,
        video,
        frameNumbers,
        format: video.backend!.embeddedFormat ?? "png",
        channelOrder: video.backend!.embeddedChannelOrder ?? "RGB",
      });
    } else if (isRealMode && video.backend) {
      // New-embed of a continuous video: legacy getFrame+encode path. Frames a
      // backend returns as raw pixels (ImageBitmap/ImageData — the MediaBunny/
      // Mp4Box mp4 case) are PNG-encoded by collectEncodedFrames; when that
      // happens the stored `format` MUST be "png" (the encoded bytes ARE PNG),
      // overriding any codec-ish `backendMetadata.format`, or the reader would
      // mis-decode.
      const { frameData, encodedFormat } = await collectEncodedFrames(
        video,
        toEncode[vi],
        progress,
      );
      if (frameData.size === 0) continue;
      const frameNumbers = [...frameData.keys()].sort((a, b) => a - b);
      plan.set(vi, {
        kind: "encode",
        videoIndex: vi,
        video,
        frameNumbers,
        format:
          encodedFormat ?? (video.backendMetadata?.format as string) ?? "png",
        channelOrder: (video.backendMetadata?.channel_order as string) ?? "RGB",
        frameData,
      });
    }
  }
  return plan;
}

/**
 * Serializable projection of one raw-copyable {@link EmbedPlanEntry}, safe to
 * pass across a Web Worker boundary via `structuredClone` (Task 1.1 of the
 * streaming pkg.slp writer). Contains only plain data — no `Video`/backend
 * instances, functions, or open file handles — so a worker holding ONLY this
 * entry plus read access to the SOURCE file (via
 * {@link SerializableEmbedPlan.sourcePath}) can locate and copy the video's
 * stored embedded images into the destination file without ever touching the
 * live `Labels` graph.
 */
export type SerializableEmbedEntry = {
  /**
   * Destination video index — this video's position in `labels.videos` of the
   * file being WRITTEN. NOT assumed to equal the video's index in the SOURCE
   * file (videos can be added/removed/reordered between load and save); the
   * actual source location is carried separately in `sourceDataset`/
   * `sourceGroup` below.
   */
  videoIndex: number;
  /**
   * The SOURCE file's HDF5 dataset path holding the raw encoded blobs (e.g.
   * `"video0/video"`), read directly off the live backend's `dataset`
   * property (`VideoBackend.dataset`) — the same path `Hdf5VideoBackend` /
   * `StreamingHdf5VideoBackend` use internally for `getFrameBuffer`
   * (delegated through `CropVideoBackend` for a cropped video). This is the
   * authoritative source location; it is never derived from `videoIndex`.
   */
  sourceDataset: string;
  /**
   * The SOURCE file's HDF5 GROUP containing `sourceDataset` and its sibling
   * `frame_numbers` / `frame_sizes` / `source_video` datasets/group — derived
   * by stripping a trailing `"/video"` from `sourceDataset` (mirrors
   * `StreamingHdf5VideoBackend.ensureLoaded`'s `groupPath` derivation). The
   * worker reads `{sourceGroup}/frame_numbers` etc. from the SOURCE file and
   * writes the copy under `video{videoIndex}` in the destination.
   */
  sourceGroup: string;
  /** Stored image format (`VideoBackend.embeddedFormat`), e.g. "png"/"jpg". */
  format: string;
  /**
   * Stored channel order (`VideoBackend.embeddedChannelOrder`), e.g.
   * "RGB"/"BGR".
   */
  channelOrder: string;
  /**
   * The FULL stored source frame numbers being preserved, in storage order
   * (the raw path always copies the entire stored set regardless of embed
   * mode — semantics A; see {@link planEmbedding}). Always non-empty:
   * `planEmbedding` skips 0-frame videos before they can reach this builder.
   */
  frameNumbers: number[];
  /**
   * Plain-object `source_video` lineage (from {@link sourceVideoDict}), or
   * `null` when this video has no separate source to record. The worker
   * writes this into the destination's `{group}/source_video` HDF5 group
   * using the same JSON/attr logic as {@link writeSourceVideoJson}, without
   * needing the live `Labels`/`Video` object.
   */
  sourceVideoJson: Record<string, unknown> | null;
};

/**
 * Fully `structuredClone`-safe embed plan handed to the streaming writer's Web
 * Worker (Task 1.1). Scope is a straight RE-SAVE of an already-embedded
 * `pkg.slp` via the raw-copy path ONLY — see
 * {@link buildSerializableEmbedPlan}.
 */
export type SerializableEmbedPlan = {
  /**
   * On-disk path of the SOURCE pkg.slp that every entry's `sourceDataset` /
   * `sourceGroup` are relative to. A single shared path is sufficient because
   * this task's scope is re-saving ONE already-embedded file; a future
   * "merge multiple pkg.slp sources into one" scenario would need to move
   * this onto each entry instead.
   */
  sourcePath: string;
  entries: SerializableEmbedEntry[];
};

/**
 * Project a `planEmbedding` result into a worker-transportable
 * {@link SerializableEmbedPlan}, for the RE-SAVE/raw-copy path ONLY (Task 1.1
 * of the streaming pkg.slp writer). Reuses `planEmbedding` for video
 * selection/frameNumbers (no re-derivation of that logic), then strips each
 * raw entry down to plain data a worker can act on without the live `Labels`/
 * `Video`/backend objects.
 *
 * Throws if `planEmbedding` selected a `kind: "encode"` (new-embed of a
 * continuous video) entry for any video — the streaming writer's worker half
 * only knows how to copy an already-embedded video's stored blobs; new-embed
 * is a deferred follow-up (see the {@link EmbedPlanEntry} doc).
 *
 * @param labels Labels to save (already-open embedded video backends).
 * @param embedMode Same `embed` option as `saveSlpToBytes`/`planEmbedding`.
 * @param sourcePath On-disk path of the SOURCE pkg.slp; the worker opens it
 *   via its own read bridge to copy the embedded groups from.
 * @throws Error if any selected video would use the new-embed ("encode")
 *   path, or a raw-copyable video's backend exposes no `dataset` to locate
 *   its source embedded group by.
 */
export async function buildSerializableEmbedPlan(
  labels: Labels,
  embedMode: boolean | string,
  sourcePath: string,
): Promise<SerializableEmbedPlan> {
  const plan = await planEmbedding(labels, embedMode);
  const entries: SerializableEmbedEntry[] = [];
  for (const entry of plan.values()) {
    if (entry.kind !== "raw") {
      throw new Error(
        `buildSerializableEmbedPlan: video${entry.videoIndex} would use the ` +
          `new-embed ("encode") path, which the streaming writer's worker ` +
          "half does not support - this path is re-save/raw-copy of an " +
          "already-embedded pkg.slp ONLY (new-embed of a continuous video is " +
          "a deferred follow-up).",
      );
    }
    const sourceDataset =
      entry.video.backend?.dataset ??
      (entry.video.backendMetadata.dataset as string | undefined);
    if (!sourceDataset) {
      throw new Error(
        `buildSerializableEmbedPlan: video${entry.videoIndex} is raw-copyable ` +
          'but its backend exposes no "dataset" path, so the worker has no ' +
          "way to locate its source embedded group.",
      );
    }
    const sourceGroup = sourceDataset.endsWith("/video")
      ? sourceDataset.slice(0, -"/video".length)
      : sourceDataset;
    entries.push({
      videoIndex: entry.videoIndex,
      sourceDataset,
      sourceGroup,
      format: entry.format,
      channelOrder: entry.channelOrder,
      frameNumbers: [...entry.frameNumbers],
      sourceVideoJson: sourceVideoDict(entry.video),
    });
  }
  return { sourcePath, entries };
}

/**
 * The frame indices of video `videoIndex` whose images `embedMode` embeds,
 * sorted. `"user"` / `"user+suggestions"` take every user-labeled frame
 * (`isUserLabeled`: user instances, negative frames, user centroids/boxes/
 * masks/ROIs), as sleap-io's `user_labeled_frames`.
 */
function selectEmbedFrames(
  labels: Labels,
  videoIndex: number,
  embedMode: boolean | string,
): number[] {
  const mode = embedMode === true ? "all" : String(embedMode).toLowerCase();
  const frameIndices = new Set<number>();
  // "all" and "all+suggestions" both include every labeled frame; the latter
  // ALSO includes suggestion-only frames (added in the suggestion loop below).
  const includeAllLabeled = mode === "all" || mode === "all+suggestions";
  const includeSuggestions =
    mode === "suggestions" ||
    mode === "user+suggestions" ||
    mode === "all+suggestions";

  for (const frame of labels.labeledFrames) {
    if (labels.videos.indexOf(frame.video) !== videoIndex) continue;
    let include = false;
    if (includeAllLabeled) {
      include = true;
    } else if (mode === "user" || mode === "user+suggestions") {
      // Any user labeling — instances, negative flag, user centroids/boxes/
      // masks/ROIs — as sleap-io's `user_labeled_frames` (slp.py embed).
      include = frame.isUserLabeled;
    } // "suggestions": added below
    if (include) frameIndices.add(frame.frameIdx);
  }

  if (includeSuggestions) {
    for (const suggestion of labels.suggestions) {
      if (labels.videos.indexOf(suggestion.video) !== videoIndex) continue;
      frameIndices.add(suggestion.frameIdx);
    }
  }
  return Array.from(frameIndices).sort((a, b) => a - b);
}

/** Read attempts per frame before giving up — a decoder may drop a read. */
const EMBED_READ_ATTEMPTS = 3;

/**
 * Collect encoded frame bytes for a NEW embed of one continuous video, reading
 * each of `sortedFrames` (from {@link selectEmbedFrames}) via `getFrame` +
 * {@link frameToBytes} (the legacy path). Reads are `pinned` and retried, and a
 * frame inside the video that still can't be read fails the save rather than
 * leaving a package that silently lacks its image.
 */
async function collectEncodedFrames(
  video: Video,
  sortedFrames: number[],
  progress: {
    done: number;
    total: number;
    report?: SlpWriteOptions["onProgress"];
  },
): Promise<{
  frameData: Map<number, Uint8Array>;
  encodedFormat: string | null;
}> {
  const frameData = new Map<number, Uint8Array>();
  if (!video?.backend) return { frameData, encodedFormat: null };

  // True once any frame was re-encoded from raw pixels (an ImageBitmap or
  // ImageData) — those bytes are PNG, so the recorded `format` MUST be "png".
  // A frame returned as ready-made encoded bytes (e.g. an already-embedded
  // source) is stored verbatim and does NOT flip this.
  let encodedToPng = false;
  const frameCount = video.shape?.[0];
  const unreadable: number[] = [];
  for (const frameIdx of sortedFrames) {
    // `pinned`: the app may keep showing other frames meanwhile, and the mp4/
    // AVI decoders otherwise cancel an older read for the newest one.
    let frame: Awaited<ReturnType<Video["getFrame"]>> = null;
    for (let attempt = 0; attempt < EMBED_READ_ATTEMPTS && !frame; attempt++) {
      frame = await video.getFrame(frameIdx, { pinned: true, prefetch: false });
    }
    progress.done += 1;
    progress.report?.(progress.done, progress.total, "Embedding frames");
    if (!frame) {
      // A frame past the video's end has no image to embed; one inside it
      // should — writing the package without it would silently lose it.
      if (frameCount == null || frameIdx < frameCount)
        unreadable.push(frameIdx);
      continue;
    }
    let bytes = frameToBytes(frame);
    if (!bytes) {
      // The MediaBunny/Mp4Box (and image/seq) backends return raw pixels
      // (ImageBitmap / ImageData) rather than encoded bytes; PNG-encode them
      // so the frame is actually embedded instead of silently dropped.
      if (isImageBitmapLike(frame)) {
        bytes = await encodeBitmapToPng(frame as ImageBitmap);
        encodedToPng = true;
      } else if (isImageDataLike(frame)) {
        bytes = await encodeImageDataToPng(frame);
        encodedToPng = true;
      }
    }
    if (bytes) frameData.set(frameIdx, bytes);
  }
  if (unreadable.length > 0) {
    const name = Array.isArray(video.filename)
      ? video.filename[0]
      : video.filename;
    const list =
      unreadable.slice(0, 10).join(", ") +
      (unreadable.length > 10 ? ", …" : "");
    throw new Error(
      `Couldn't read ${unreadable.length} frame image(s) from "${name}" to embed ` +
        `(frame ${list}). The package was not written.`,
    );
  }
  return { frameData, encodedFormat: encodedToPng ? "png" : null };
}

/**
 * Convert a video frame to Uint8Array bytes for embedding.
 */
function frameToBytes(frame: unknown): Uint8Array | null {
  if (frame instanceof Uint8Array) return frame;
  if (frame instanceof ArrayBuffer) return new Uint8Array(frame);
  if (ArrayBuffer.isView(frame)) {
    const view = frame as ArrayBufferView;
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  }
  return null;
}

/**
 * Write ONLY the `videos_json` dataset for a plan-driven embed: each planned
 * video gets an embedded-pointer entry (`filename:"."`, backend pointing at
 * `video{vi}/video` with the planned format/channel_order + crop-aware inner
 * shape/fps + source_video lineage); every other video is serialized normally.
 * The per-group image datasets are written separately by
 * {@link writeEmbeddedVideoData}.
 */
function writeEmbeddedVideosJson(
  file: any,
  labels: Labels,
  plan: EmbedPlan,
): void {
  const payload = labels.videos.map((video, videoIndex) => {
    const entry = plan.get(videoIndex);
    if (entry) {
      // This video is being embedded - update metadata
      const backend: Record<string, unknown> = {
        filename: ".",
        dataset: `video${videoIndex}/video`,
        format: entry.format,
        channel_order: entry.channelOrder,
      };
      // For a cropped video, videos_json must describe the UNCROPPED inner frame
      // (the crop rides /video_crops and is re-applied once on read); read the
      // shape/fps from the inner backend, not the cropped facade.
      const inner =
        video.backend instanceof CropVideoBackend
          ? video.backend.inner
          : video.backend;
      const innerShape =
        inner?.shape ??
        (video.backendMetadata?.source_shape as number[] | undefined);
      if (innerShape) backend.shape = innerShape;
      if (inner?.fps != null) backend.fps = inner.fps;

      const outEntry: Record<string, unknown> = {
        filename: ".",
        backend,
      };
      // Preserve the full source_video lineage (filename + backend incl. shape
      // + deeper chain), mirroring newer Python which nests it in videos_json as
      // well as the authoritative HDF5 group written below. See #160.
      const srcDict = sourceVideoDict(video);
      if (srcDict) outEntry.source_video = srcDict;
      return JSON.stringify(outEntry);
    }
    return JSON.stringify(serializeVideo(video));
  });
  file.create_dataset({ name: "videos_json", data: payload });
}

/**
 * Write the per-video embedded image datasets (`video{vi}/video`,
 * `frame_numbers`, `frame_sizes`, `source_video`) for each planned video. Raw
 * entries copy the stored encoded blobs verbatim via `getFrameBuffer` (no
 * decode/re-encode); encode entries write the pre-collected bytes.
 *
 * Peak memory: the raw path STREAMS stored blobs into a resizable 1-D `<B`
 * dataset in bounded byte windows (peak ~= 2x one window during a flush — the
 * window's blobs plus their concatenated buffer — plus the h5wasm MEMFS copy),
 * rather than holding every blob plus a full concatenated buffer at once. The
 * encode path keeps accumulate-then-write — its bytes are already fully in JS
 * memory (`frameData`), so there is nothing to stream. Either way the on-disk
 * `{group}/video` layout is byte-identical to a single concatenated write; the
 * reader slices it back into frames via `frame_sizes`.
 *
 * Backstop (raw path only): if a video planned N frames but fewer blobs could
 * be read, throw rather than silently write a file with the images stripped —
 * exactly the #213 data-loss this fix prevents.
 */
async function writeEmbeddedVideoData(
  file: any,
  labels: Labels,
  plan: EmbedPlan,
): Promise<void> {
  for (const entry of plan.values()) {
    const group = `video${entry.videoIndex}`;
    file.create_group(group);

    // Write the source video lineage into the authoritative
    // `{group}/source_video` HDF5 group — the location Python reads for an
    // embedded `.pkg.slp` (it ignores videos_json's source_video for embedded
    // videos). Mirrors Python `write_videos`' lineage pass. See #160.
    const srcDict = sourceVideoDict(labels.videos[entry.videoIndex]);
    if (srcDict) writeSourceVideoJson(file, group, srcDict);

    // Write `{group}/video` and collect the frame numbers/sizes actually
    // written. Raw entries stream stored blobs (low peak); encode entries write
    // their pre-collected in-memory bytes in one shot.
    const { writtenFns, sizes } =
      entry.kind === "raw"
        ? await writeRawEmbeddedVideo(file, group, entry)
        : writeEncodedEmbeddedVideo(file, group, entry);

    // Backstop (raw path only): a raw copy's frameNumbers IS the exact stored
    // set (embeddedFrameIndices), so every frame must read a blob — a partial
    // read is anomalous data loss. Refuse to write a stripped file rather than
    // silently drop images (the #213 data-loss this fix prevents). Note
    // frameNumbers.length is always > 0 for a raw entry (planEmbedding skips
    // 0-frame videos). The encode path keeps today's lenient behavior ->
    // deferred follow-up.
    if (entry.kind === "raw" && writtenFns.length < entry.frameNumbers.length) {
      throw new Error(
        `embedding video${entry.videoIndex}: read ${writtenFns.length} of ` +
          `${entry.frameNumbers.length} planned frame(s) - refusing to write a ` +
          `file with dropped images.`,
      );
    }

    const ds = file.get(`${group}/video`);
    if (ds) {
      setStringAttr(ds, "format", entry.format);
      setStringAttr(ds, "channel_order", entry.channelOrder);
    }
    file.create_dataset({
      name: `${group}/frame_numbers`,
      data: writtenFns,
      shape: [writtenFns.length],
      dtype: "<i4",
    });
    file.create_dataset({
      name: `${group}/frame_sizes`,
      data: sizes,
      shape: [sizes.length],
      dtype: "<i4",
    });
  }
}

/**
 * Raw path: stream a video's stored encoded blobs into a resizable 1-D `<B`
 * dataset at `{group}/video`, flushing to the dataset tail in bounded byte
 * windows so peak JS memory is ~2x one window during a flush (the window's blobs
 * plus their concatenated buffer) rather than every blob plus a full
 * concatenated buffer. The on-disk bytes are the in-order concatenation of the
 * blobs — byte-identical to an accumulate-then-write. A blob that reads back
 * null/empty is skipped (the caller's backstop then refuses to write a file
 * with dropped images). Returns the frame numbers/sizes actually written.
 */
async function writeRawEmbeddedVideo(
  file: any,
  group: string,
  entry: EmbedPlanEntry,
): Promise<{ writtenFns: number[]; sizes: number[] }> {
  file.create_dataset({
    name: `${group}/video`,
    data: new Uint8Array(0),
    shape: [0],
    maxshape: [null],
    chunks: [EMBED_VIDEO_CHUNK_BYTES],
    dtype: "<B",
  });
  const vds = file.get(`${group}/video`);
  const sizes: number[] = [];
  const writtenFns: number[] = [];
  let total = 0;
  let win: Uint8Array[] = [];
  let winBytes = 0;
  const flush = () => {
    if (winBytes === 0) return;
    const buf = new Uint8Array(winBytes);
    let o = 0;
    for (const b of win) {
      buf.set(b, o);
      o += b.length;
    }
    vds.resize([total + winBytes]);
    vds.write_slice([[total, total + winBytes]], buf);
    total += winBytes;
    win = [];
    winBytes = 0;
  };
  for (const fn of entry.frameNumbers) {
    const blob = await entry.video.getFrameBuffer(fn);
    if (!blob || blob.length === 0) continue;
    win.push(blob);
    winBytes += blob.length;
    sizes.push(blob.length);
    writtenFns.push(fn);
    if (winBytes >= EMBED_WRITE_WINDOW_BYTES) flush();
  }
  flush();
  return { writtenFns, sizes };
}

/**
 * Encode path: write a video's pre-collected frame bytes (`entry.frameData`,
 * already fully in memory) as a single concatenated 1-D `<B` dataset at
 * `{group}/video`. Accumulate-then-write is appropriate here — there is nothing
 * to stream. Returns the frame numbers/sizes written.
 */
function writeEncodedEmbeddedVideo(
  file: any,
  group: string,
  entry: EmbedPlanEntry,
): { writtenFns: number[]; sizes: number[] } {
  const blobs: Uint8Array[] = [];
  const writtenFns: number[] = [];
  for (const fn of entry.frameNumbers) {
    const blob = entry.frameData!.get(fn) ?? null;
    if (!blob || blob.length === 0) continue;
    blobs.push(blob);
    writtenFns.push(fn);
  }
  const total = blobs.reduce((n, b) => n + b.length, 0);
  const combined = new Uint8Array(total);
  let off = 0;
  for (const b of blobs) {
    combined.set(b, off);
    off += b.length;
  }
  file.create_dataset({
    name: `${group}/video`,
    data: combined,
    shape: [combined.length],
    dtype: "<B",
  });
  const sizes = blobs.map((b) => b.length);
  return { writtenFns, sizes };
}

/**
 * Write a 2-D matrix dataset with a `field_names` attribute.
 *
 * When `resizable` is true the dataset is created CHUNKED with an unlimited
 * axis-0 `maxshape` (mirroring {@link createAppendableMatrixDataset}) so it can
 * later be grown/shrunk in place via `resize` + `write_slice` — the on-disk
 * pre-requisite for the fast in-place label save ({@link writeLabelTablesInPlace}).
 * The logical content is byte-for-byte identical to the contiguous layout; only
 * the storage layout changes. Used for the mutable label tables
 * (frames/instances/points/pred_points/negative_frames); the annotation tables
 * stay contiguous (`resizable` defaults to false).
 */
function createMatrixDataset(
  file: any,
  name: string,
  rows: number[][],
  fieldNames: string[],
  dtype: string,
  resizable = false,
): void {
  const rowCount = rows.length;
  const colCount = fieldNames.length;
  // Pre-allocate typed array to avoid intermediate .flat() allocation
  const TypedArray = dtype.includes("i")
    ? dtype.includes("4")
      ? Int32Array
      : Float64Array
    : Float64Array;
  const data = new TypedArray(rowCount * colCount);
  for (let i = 0; i < rowCount; i++) {
    const row = rows[i];
    const offset = i * colCount;
    for (let j = 0; j < colCount; j++) {
      data[offset + j] = row[j];
    }
  }
  if (resizable) {
    // Chunk rows cap the per-chunk allocation for large tables while staying
    // small for tiny files (chunk >= 1 is required by HDF5, even for 0 rows).
    const chunkRows = Math.max(1, Math.min(WRITE_CHUNK_ROWS, rowCount || 1));
    file.create_dataset({
      name,
      data,
      shape: [rowCount, colCount],
      maxshape: [null, colCount],
      chunks: [chunkRows, colCount],
      dtype,
    });
  } else {
    file.create_dataset({ name, data, shape: [rowCount, colCount], dtype });
  }
  const dataset = file.get(name);
  setStringAttr(dataset, "field_names", JSON.stringify(fieldNames));
}

/** One member of a compound (structured) dataset: `[name, HDF5 dtype]`. */
type CompoundField = [name: string, dtype: string];

// Per-member (name, dtype) layout for the pose tables written as HDF5 COMPOUND
// datasets — matching Python sleap-io's on-disk dtypes (see `io/slp.py`
// `instance_dtype`/`point_dtype`/`predicted_point_dtype` + the negative-frames
// dtype). Writing these as a flat float matrix made Python read the id/count
// columns (skeleton, track, …) back as `numpy.float32`, so `skeletons[id]`
// raised `TypeError: list indices must be integers` (#218) — the values were
// fine, only the on-disk dtype was wrong.
//
// NB: h5wasm dtype strings are SINGLE-CHAR — the trailing digits are ignored
// (`<f8` is really float32!), and `u`/numpy widths aren't accepted. The chars:
// `q`/`Q` = (u)int64, `i`/`I` = (u)int32, `b`/`B` = (u)int8, `f` = float32,
// `d` = float64. Python stores `visible`/`complete` as bool (`?`); HDF5-bool
// isn't writable through h5wasm, so we store them as uint8 (Python casts them
// back to bool on read).
const INSTANCE_COMPOUND_FIELDS: CompoundField[] = [
  ["instance_id", "<q"], // i8
  ["instance_type", "<B"], // u1
  ["frame_id", "<Q"], // u8
  ["skeleton", "<I"], // u4
  ["track", "<i"], // i4 (carries -1)
  ["from_predicted", "<q"], // i8 (carries -1)
  ["score", "<f"], // f4
  ["point_id_start", "<Q"], // u8
  ["point_id_end", "<Q"], // u8
  ["tracking_score", "<f"], // f4
];
const POINT_COMPOUND_FIELDS: CompoundField[] = [
  ["x", "<d"], // f8
  ["y", "<d"], // f8
  ["visible", "<B"], // bool -> u1
  ["complete", "<B"], // bool -> u1
];
const PRED_POINT_COMPOUND_FIELDS: CompoundField[] = [
  ["x", "<d"], // f8
  ["y", "<d"], // f8
  ["visible", "<B"], // bool -> u1
  ["complete", "<B"], // bool -> u1
  ["score", "<d"], // f8
];
const NEGATIVE_FRAME_COMPOUND_FIELDS: CompoundField[] = [
  ["video_id", "<I"], // u4
  ["frame_idx", "<Q"], // u8
];

/**
 * Write `rows` as an HDF5 COMPOUND (structured) dataset — one struct per row,
 * each member with its own dtype — the layout Python sleap-io expects for the
 * pose tables. `fields` gives `[memberName, hdf5Dtype]` in column order;
 * `rows[i][j]` supplies member j of row i. h5wasm takes the columns as a
 * per-member `Map` and packs them into an array-of-structures; passing plain
 * number columns lets it widen 64-bit int members to BigInt itself.
 *
 * A `field_names` attribute is stamped too — Python ignores it (it reads the
 * compound member names), and it keeps this dataset shape identical to the old
 * matrix output for any JS reader that keys off the attribute.
 */
function createCompoundDataset(
  file: any,
  name: string,
  rows: number[][],
  fields: CompoundField[],
): void {
  const rowCount = rows.length;
  const columns = new Map<string, number[]>();
  for (let j = 0; j < fields.length; j++) {
    const col = new Array<number>(rowCount);
    for (let i = 0; i < rowCount; i++) col[i] = rows[i][j];
    columns.set(fields[j][0], col);
  }
  // Emit the compound pose tables CHUNKED + unlimited (1-D `maxshape: [null]`) so
  // a later edit can be re-saved IN PLACE (`write_slice` for value edits,
  // `resize` + `write_slice` for structural edits) without rewriting the
  // multi-GB embedded video groups — the reconciliation of #218 (compound, for
  // Python interop) with the in-place edit-save (which requires resizable tables;
  // see writeLabelTablesInPlace / checkInPlaceWritable). chunk >= 1 is required by
  // HDF5 even for 0 rows.
  const chunkRows = Math.max(1, Math.min(WRITE_CHUNK_ROWS, rowCount || 1));
  file.create_dataset({
    name,
    data: columns,
    shape: [rowCount],
    maxshape: [null],
    chunks: [chunkRows],
    dtype: fields,
  });
  setStringAttr(
    file.get(name),
    "field_names",
    JSON.stringify(fields.map(([fieldName]) => fieldName)),
  );
}

function writeRois(
  file: any,
  rois: ROI[],
  videos: Video[],
  tracks: Array<{ name: string }>,
  instances?: Array<Instance | PredictedInstance>,
  contexts?: [number, number][],
): void {
  if (!rois.length) return;

  const rows: number[][] = [];
  const wkbChunks: Uint8Array[] = [];
  let wkbOffset = 0;
  const categories: string[] = [];
  const names: string[] = [];
  const sources: string[] = [];
  const hasInstances = instances && instances.length > 0;

  for (let i = 0; i < rois.length; i++) {
    const roi = rois[i];
    const wkb = encodeWkb(roi.geometry);
    const wkbStart = wkbOffset;
    const wkbEnd = wkbOffset + wkb.length;
    wkbChunks.push(wkb);
    wkbOffset = wkbEnd;

    const videoIdx = contexts
      ? contexts[i][0]
      : roi.video
        ? videos.indexOf(roi.video)
        : -1;
    const frameIdx = contexts ? contexts[i][1] : -1;
    const trackIdx = roi.track ? tracks.indexOf(roi.track as any) : -1;
    // Fall back to stored _instanceIdx when no live instance link (e.g., lazy mode).
    const instanceIdx =
      hasInstances && roi.instance
        ? instances.indexOf(roi.instance)
        : (roi._instanceIdx ?? -1);
    const score = roi.isPredicted ? (roi as PredictedROI).score : Number.NaN;
    const isPredicted = roi.isPredicted ? 1 : 0;
    const trackingScore = roi.trackingScore ?? Number.NaN;

    rows.push([
      0,
      videoIdx,
      frameIdx,
      trackIdx,
      score,
      trackingScore,
      wkbStart,
      wkbEnd,
      instanceIdx,
      isPredicted,
    ]);
    categories.push(roi.category);
    names.push(roi.name);
    sources.push(roi.source);
  }

  createMatrixDataset(
    file,
    "rois",
    rows,
    [
      "annotation_type",
      "video",
      "frame_idx",
      "track",
      "score",
      "tracking_score",
      "wkb_start",
      "wkb_end",
      "instance",
      "is_predicted",
    ],
    "<f8",
  );

  // Write string metadata as datasets at root level (v1.9+)
  writeStringDataset(file, "roi_categories", categories);
  writeStringDataset(file, "roi_names", names);
  writeStringDataset(file, "roi_sources", sources);

  // Write concatenated WKB bytes
  const totalWkb = wkbChunks.reduce((sum, c) => sum + c.length, 0);
  const wkbFlat = new Uint8Array(totalWkb);
  let offset = 0;
  for (const chunk of wkbChunks) {
    wkbFlat.set(chunk, offset);
    offset += chunk.length;
  }
  // gzip-compress the packed WKB blob (Python #465). Transparent on read:
  // h5wasm/.value and Python h5py natively decompress the HDF5 deflate filter.
  // h5wasm throws "cannot specify compression without chunks" and rejects a
  // 0-size chunk dim, so only enable the filter when there are bytes to write;
  // writeRois early-returns on empty input, but the length>0 guard is the safe
  // pattern (mirrors Python keeping empty datasets uncompressed).
  if (wkbFlat.length > 0) {
    file.create_dataset({
      name: "roi_wkb",
      data: wkbFlat,
      shape: [wkbFlat.length],
      dtype: "<B",
      chunks: [wkbFlat.length],
      compression: "gzip",
      compression_opts: 1,
    });
  } else {
    file.create_dataset({
      name: "roi_wkb",
      data: wkbFlat,
      shape: [wkbFlat.length],
      dtype: "<B",
    });
  }
}

function writeMasks(
  file: any,
  masks: SegmentationMask[],
  videos: Video[],
  tracks: Array<{ name: string }>,
  instances: (Instance | PredictedInstance)[],
  contexts?: [number, number][],
): void {
  if (!masks.length) return;

  const rows: number[][] = [];
  const rleChunks: Uint8Array[] = [];
  let rleOffset = 0;
  const categories: string[] = [];
  const names: string[] = [];
  const sources: string[] = [];

  // Score map collection
  const scoreMapIndexRows: number[][] = [];
  const scoreMapChunks: Uint8Array[] = [];
  let smOffset = 0;

  // Map each mask object to its position in the flat list so a user mask's
  // `fromPredicted` link can be resolved to a global index (mirrors instance
  // from_predicted). Keyed by the object itself (object identity) — the JS
  // analog of Python's id(mask).
  const maskIdToIdx = new Map<SegmentationMask, number>();
  masks.forEach((m, i) => {
    maskIdToIdx.set(m, i);
  });

  for (let i = 0; i < masks.length; i++) {
    const mask = masks[i];
    // Convert Uint32Array RLE counts to bytes (little-endian)
    const rleBytes = new Uint8Array(mask.rleCounts.length * 4);
    const view = new DataView(rleBytes.buffer);
    for (let j = 0; j < mask.rleCounts.length; j++) {
      view.setUint32(j * 4, mask.rleCounts[j], true);
    }
    const rleStart = rleOffset;
    const rleEnd = rleOffset + rleBytes.length;
    rleChunks.push(rleBytes);
    rleOffset = rleEnd;

    const videoIdx = contexts ? contexts[i][0] : -1;
    const frameIdx = contexts ? contexts[i][1] : -1;
    const trackIdx = mask.track ? tracks.indexOf(mask.track as any) : -1;
    const score = mask.isPredicted
      ? (mask as PredictedSegmentationMask).score
      : Number.NaN;
    const isPredicted = mask.isPredicted ? 1 : 0;
    const instanceIdx = mask.instance
      ? instances.indexOf(mask.instance as Instance)
      : (mask._instanceIdx ?? -1);
    const maskTrackingScore = mask.trackingScore ?? Number.NaN;

    // Resolve the from_predicted provenance link to a global index into the
    // flat mask list. -1 sentinel when there is no link or the source
    // prediction is absent from the saved list (Map miss). Only user masks
    // carry fromPredicted; predicted masks always resolve to -1.
    const fromPredictedSrc =
      (mask as UserSegmentationMask).fromPredicted ?? null;
    const fromPredictedIdx =
      fromPredictedSrc != null ? (maskIdToIdx.get(fromPredictedSrc) ?? -1) : -1;

    rows.push([
      mask.height,
      mask.width,
      2,
      videoIdx,
      frameIdx,
      trackIdx,
      score,
      rleStart,
      rleEnd,
      isPredicted,
      instanceIdx,
      maskTrackingScore,
      mask.scale[0],
      mask.scale[1],
      mask.offset[0],
      mask.offset[1],
      fromPredictedIdx,
    ]);
    categories.push(mask.category);
    names.push(mask.name);
    sources.push(mask.source);

    // Collect score maps for predicted masks
    if (mask.isPredicted) {
      const pm = mask as PredictedSegmentationMask;
      if (pm.scoreMap) {
        const smBytes = new Uint8Array(
          pm.scoreMap.buffer,
          pm.scoreMap.byteOffset,
          pm.scoreMap.byteLength,
        );
        const compressed = deflate(smBytes);
        const smH = pm.scoreMap.length / mask.width;
        if (!Number.isInteger(smH)) {
          throw new Error(
            `Score map size ${pm.scoreMap.length} not divisible by width ${mask.width}`,
          );
        }
        scoreMapIndexRows.push([
          i,
          smOffset,
          smOffset + compressed.length,
          smH,
          mask.width,
        ]);
        scoreMapChunks.push(compressed);
        smOffset += compressed.length;
      }
    }
  }

  createMatrixDataset(
    file,
    "masks",
    rows,
    [
      "height",
      "width",
      "annotation_type",
      "video",
      "frame_idx",
      "track",
      "score",
      "rle_start",
      "rle_end",
      "is_predicted",
      "instance",
      "tracking_score",
      "scale_x",
      "scale_y",
      "offset_x",
      "offset_y",
      "from_predicted",
    ],
    "<f8",
  );

  // Write string metadata as datasets at root level (v1.9+)
  writeStringDataset(file, "mask_categories", categories);
  writeStringDataset(file, "mask_names", names);
  writeStringDataset(file, "mask_sources", sources);

  // Write concatenated RLE bytes
  const totalRle = rleChunks.reduce((sum, c) => sum + c.length, 0);
  const rleFlat = new Uint8Array(totalRle);
  let offset = 0;
  for (const chunk of rleChunks) {
    rleFlat.set(chunk, offset);
    offset += chunk.length;
  }
  // gzip-compress the packed RLE blob (Python #464). Transparent on read:
  // h5wasm/.value and Python h5py natively decompress the HDF5 deflate filter.
  // A fully-empty mask raster encodes to a zero-length RLE, so rleFlat can be
  // empty even when masks are present; h5wasm rejects a 0-size chunk dim and
  // throws "cannot specify compression without chunks", so only enable the
  // filter when there are bytes (mirrors Python keeping empty mask_rle raw).
  if (rleFlat.length > 0) {
    file.create_dataset({
      name: "mask_rle",
      data: rleFlat,
      shape: [rleFlat.length],
      dtype: "<B",
      chunks: [rleFlat.length],
      compression: "gzip",
      compression_opts: 1,
    });
  } else {
    file.create_dataset({
      name: "mask_rle",
      data: rleFlat,
      shape: [rleFlat.length],
      dtype: "<B",
    });
  }

  // Write score maps
  if (scoreMapIndexRows.length > 0) {
    createMatrixDataset(
      file,
      "mask_score_map_index",
      scoreMapIndexRows,
      ["mask_idx", "data_start", "data_end", "height", "width"],
      "<f8",
    );
    const totalSm = scoreMapChunks.reduce((sum, c) => sum + c.length, 0);
    const smFlat = new Uint8Array(totalSm);
    let smOff = 0;
    for (const chunk of scoreMapChunks) {
      smFlat.set(chunk, smOff);
      smOff += chunk.length;
    }
    file.create_dataset({
      name: "mask_score_maps",
      data: smFlat,
      shape: [smFlat.length],
      dtype: "<B",
    });
  }
}

function writeBboxes(
  file: any,
  bboxes: BoundingBox[],
  _videos: Video[],
  tracks: Array<{ name: string }>,
  instances: (Instance | PredictedInstance)[],
  contexts?: [number, number][],
): void {
  if (!bboxes.length) return;

  const rows: number[][] = [];
  const categories: string[] = [];
  const names: string[] = [];
  const sources: string[] = [];

  for (let i = 0; i < bboxes.length; i++) {
    const bbox = bboxes[i];
    const videoIdx = contexts ? contexts[i][0] : -1;
    const frameIdx = contexts ? contexts[i][1] : -1;
    const trackIdx = bbox.track ? tracks.indexOf(bbox.track as any) : -1;
    const score = bbox.isPredicted
      ? (bbox as PredictedBoundingBox).score
      : Number.NaN;
    // Fall back to stored _instanceIdx when no live instance link (e.g., lazy mode).
    const instanceIdx = bbox.instance
      ? instances.indexOf(bbox.instance as Instance)
      : (bbox._instanceIdx ?? -1);

    const trackingScore = bbox.trackingScore ?? Number.NaN;

    rows.push([
      bbox.x1,
      bbox.y1,
      bbox.x2,
      bbox.y2,
      bbox.angle,
      videoIdx,
      frameIdx,
      trackIdx,
      score,
      instanceIdx,
      trackingScore,
    ]);
    categories.push(bbox.category);
    names.push(bbox.name);
    sources.push(bbox.source);
  }

  createMatrixDataset(
    file,
    "bboxes",
    rows,
    [
      "x1",
      "y1",
      "x2",
      "y2",
      "angle",
      "video",
      "frame_idx",
      "track",
      "score",
      "instance",
      "tracking_score",
    ],
    "<f8",
  );

  // Write string metadata as datasets at root level (v1.9+)
  writeStringDataset(file, "bbox_categories", categories);
  writeStringDataset(file, "bbox_names", names);
  writeStringDataset(file, "bbox_sources", sources);
}

function writeLabelImages(
  file: any,
  labelImages: LabelImage[],
  _videos: Video[],
  tracks: Track[],
  instances: (Instance | PredictedInstance)[],
  contexts?: [number, number][],
): void {
  if (!labelImages.length) return;

  // Verify little-endian platform (label image data is stored as raw LE bytes)
  const endianCheck = new Uint8Array(new Uint16Array([0x0102]).buffer);
  if (endianCheck[0] !== 0x02) {
    throw new Error("LabelImage I/O requires a little-endian platform.");
  }

  const rows: number[][] = [];
  const compressedChunks: Uint8Array[] = [];
  let dataOffset = 0;
  const objectRows: number[][] = [];
  const objectCategories: string[] = [];
  const objectNames: string[] = [];
  const sources: string[] = [];
  let objectsOffset = 0;

  // Score map collection
  const smIndexRows: number[][] = [];
  const smChunks: Uint8Array[] = [];
  let smOffset = 0;

  for (let liIdx = 0; liIdx < labelImages.length; liIdx++) {
    const li = labelImages[liIdx];
    const videoIdx = contexts ? contexts[liIdx][0] : -1;
    const frameIdx = contexts ? contexts[liIdx][1] : -1;

    // Compress pixel data: Int32Array -> raw bytes -> zlib
    const pixelBytes = new Uint8Array(
      li.data.buffer,
      li.data.byteOffset,
      li.data.byteLength,
    );
    const compressed = deflate(pixelBytes);
    const dataStart = dataOffset;
    const dataEnd = dataOffset + compressed.length;
    compressedChunks.push(compressed);
    dataOffset = dataEnd;

    const isPredicted = li.isPredicted ? 1 : 0;
    const liScore = li.isPredicted
      ? (li as PredictedLabelImage).score
      : Number.NaN;

    // Per-object entries
    const objectsStart = objectsOffset;
    for (const [labelId, info] of li.objects) {
      const trackIdx = info.track ? tracks.indexOf(info.track as Track) : -1;
      // PR #386: fall back to _instanceIdx when instance is null
      let instanceIdx = li._objectInstanceIdxs?.get(labelId) ?? -1;
      if (info.instance) {
        const found = instances.indexOf(info.instance as Instance);
        if (found >= 0) instanceIdx = found;
      } else if (info._instanceIdx != null && info._instanceIdx >= 0) {
        instanceIdx = info._instanceIdx;
      }
      const objScore = info.score != null ? info.score : Number.NaN;
      const objTrackingScore =
        info.trackingScore != null ? info.trackingScore : Number.NaN;
      objectRows.push([
        labelId,
        trackIdx,
        instanceIdx,
        objScore,
        objTrackingScore,
      ]);
      objectCategories.push(info.category);
      objectNames.push(info.name);
      objectsOffset++;
    }

    rows.push([
      videoIdx,
      frameIdx,
      li.height,
      li.width,
      li.nObjects,
      objectsStart,
      dataStart,
      dataEnd,
      isPredicted,
      liScore,
      li.scale[0],
      li.scale[1],
      li.offset[0],
      li.offset[1],
    ]);
    sources.push(li.source);

    // Collect score maps for predicted label images
    if (li.isPredicted) {
      const pli = li as PredictedLabelImage;
      if (pli.scoreMap) {
        const smBytes = new Uint8Array(
          pli.scoreMap.buffer,
          pli.scoreMap.byteOffset,
          pli.scoreMap.byteLength,
        );
        const smCompressed = deflate(smBytes);
        const smH = pli.scoreMap.length / li.width;
        if (!Number.isInteger(smH)) {
          throw new Error(
            `Score map size ${pli.scoreMap.length} not divisible by width ${li.width}`,
          );
        }
        smIndexRows.push([
          liIdx,
          smOffset,
          smOffset + smCompressed.length,
          smH,
          li.width,
        ]);
        smChunks.push(smCompressed);
        smOffset += smCompressed.length;
      }
    }
  }

  // Write main metadata table
  createMatrixDataset(
    file,
    "label_images",
    rows,
    [
      "video",
      "frame_idx",
      "height",
      "width",
      "n_objects",
      "objects_start",
      "data_start",
      "data_end",
      "is_predicted",
      "score",
      "scale_x",
      "scale_y",
      "offset_x",
      "offset_y",
    ],
    "<f8",
  );

  // Write string metadata as datasets at root level (v1.9+)
  writeStringDataset(file, "label_image_sources", sources);

  // Write objects table (if any objects exist)
  if (objectRows.length > 0) {
    createMatrixDataset(
      file,
      "label_image_objects",
      objectRows,
      ["label_id", "track", "instance", "score", "tracking_score"],
      "<f8",
    );
    // Write string metadata as datasets at root level (v1.9+)
    writeStringDataset(file, "label_image_obj_categories", objectCategories);
    writeStringDataset(file, "label_image_obj_names", objectNames);
  }

  // Write concatenated compressed pixel data
  const totalData = compressedChunks.reduce((sum, c) => sum + c.length, 0);
  const dataFlat = new Uint8Array(totalData);
  let offset = 0;
  for (const chunk of compressedChunks) {
    dataFlat.set(chunk, offset);
    offset += chunk.length;
  }
  file.create_dataset({
    name: "label_image_data",
    data: dataFlat,
    shape: [dataFlat.length],
    dtype: "<B",
  });

  // Write score maps
  if (smIndexRows.length > 0) {
    createMatrixDataset(
      file,
      "label_image_score_map_index",
      smIndexRows,
      ["li_idx", "data_start", "data_end", "height", "width"],
      "<f8",
    );
    const totalSm = smChunks.reduce((sum, c) => sum + c.length, 0);
    const smFlat = new Uint8Array(totalSm);
    let smOff = 0;
    for (const chunk of smChunks) {
      smFlat.set(chunk, smOff);
      smOff += chunk.length;
    }
    file.create_dataset({
      name: "label_image_score_maps",
      data: smFlat,
      shape: [smFlat.length],
      dtype: "<B",
    });
  }
}

function writeCentroids(
  file: any,
  centroids: Centroid[],
  _videos: Video[],
  tracks: Array<{ name: string }>,
  instances: (Instance | PredictedInstance)[],
  contexts?: [number, number][],
): void {
  if (!centroids.length) return;

  const n = centroids.length;
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);
  const video = new Int32Array(n);
  const frameIdx = new Int32Array(n);
  const track = new Int32Array(n);
  const instance = new Int32Array(n);
  const isPredicted = new Uint8Array(n);
  const score = new Float64Array(n);
  const trackingScore = new Float64Array(n);
  const categories: string[] = [];
  const names: string[] = [];
  const sources: string[] = [];

  for (let i = 0; i < n; i++) {
    const c = centroids[i];
    x[i] = c.x;
    y[i] = c.y;
    z[i] = c.z ?? Number.NaN;
    video[i] = contexts ? contexts[i][0] : -1;
    frameIdx[i] = contexts ? contexts[i][1] : -1;
    track[i] = c.track ? tracks.indexOf(c.track as any) : -1;
    // Fall back to stored _instanceIdx when no live instance link (e.g., lazy mode).
    instance[i] = c.instance
      ? instances.indexOf(c.instance as Instance)
      : (c._instanceIdx ?? -1);
    isPredicted[i] = c.isPredicted ? 1 : 0;
    score[i] = c.isPredicted ? (c as PredictedCentroid).score : Number.NaN;
    trackingScore[i] = c.trackingScore ?? Number.NaN;
    categories.push(c.category);
    names.push(c.name);
    // Persist the canonical snake_case source for Python/PyQt interop. Normally
    // `c.source` is already snake_case (set by `fromInstance`/reader), but
    // normalize defensively in case a caller built the Centroid with a legacy
    // camelCase source directly.
    sources.push(normalizeCentroidSource(c.source));
  }

  // Match Python sleap-io's on-disk layout: a `/centroids` GROUP with one named
  // dataset per field (NOT a flat matrix), so Python/PyQt sleap-io can read
  // JS-written centroids. (Previously JS wrote a 2-D matrix dataset + a
  // `field_names` attr, which Python's group-expecting reader rejects with
  // "Field names only allowed for compound types".)
  file.create_group("centroids");
  const col = (name: string, data: ArrayLike<number>, dtype: string) =>
    file.create_dataset({ name: `centroids/${name}`, data, shape: [n], dtype });
  col("x", x, "<d");
  col("y", y, "<d");
  col("z", z, "<d");
  col("video", video, "<i4");
  col("frame_idx", frameIdx, "<i4");
  col("track", track, "<i4");
  col("instance", instance, "<i4");
  col("is_predicted", isPredicted, "<B");
  col("score", score, "<d");
  col("tracking_score", trackingScore, "<d");
  file.create_dataset({ name: "centroids/category", data: categories });
  file.create_dataset({ name: "centroids/name", data: names });
  file.create_dataset({ name: "centroids/source", data: sources });
}
