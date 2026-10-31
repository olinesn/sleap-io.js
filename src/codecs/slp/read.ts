import {
  checkOpenedAsHdf5,
  openH5File,
  type OpenH5Options,
  type SlpSource,
  getH5EmscriptenModule,
} from "./h5.js";
import { readCompoundColumnsManual } from "./h5-compound.js";
import {
  addAnnotationsToStore,
  attachAnnotationOwners,
  attachOwnerType,
  type AnnotationTuples,
  bboxRowCount,
  buildBboxes,
  buildLabelImages,
  buildMasks,
  buildRois,
  distributeAnnotations,
  groupEmbeddings,
  groupLinks,
  parseStringListAttr,
  type LabelImageObjectsTable,
  type PerDetectionMaps,
  type Routed,
  type ScoreMapTable,
  stringsFromDataset,
  type TableContext,
  toBytes,
} from "./annotations.js";
import {
  checkDatasetFilters,
  type ColumnarTableName,
  indexOutOfRange,
  isH5Group,
  legacyTablePath,
  normalizeColumnarTable,
  normalizePoseTables,
  type PoseTables,
  type RawTable,
  readH5GroupColumns,
  reportTableProblems,
  sessionProblems,
  tableBufferPath,
  tableMembers,
} from "./columnar.js";
import {
  attrToNumber,
  attrToString,
  checkSlpFormat,
  datasetValueToString,
  parseMetadataJson,
  parseProvenance,
  missingMetadataJsonError,
  parseJsonEntry,
  sessionsReadError,
  reconstructColumnarFrameGroups,
  type SessionData,
  type SessionPointMatrix,
  parseSkeletons,
  resolveCameraKey,
  normalizeCameraSize,
  reconstructInstance3D,
  resolveIdentity,
  resolveVideoFilename,
} from "./parsers.js";
import { buildSourceVideoFromDict } from "./source-video.js";
import { Labels } from "../../model/labels.js";
import { LabeledFrame } from "../../model/labeled-frame.js";
import { Instance, PredictedInstance, Track } from "../../model/instance.js";
import { Skeleton } from "../../model/skeleton.js";
import { SuggestionFrame } from "../../model/suggestions.js";
import { Video, type VideoBackendError } from "../../model/video.js";
import {
  createVideoBackend,
  UnsupportedVideoFormatError,
  isImageSource,
} from "../../video/factory.js";
import {
  resolveVideoSource,
  posixDirname,
  posixBasename,
} from "../../video/path-resolve.js";
import { getFsResolver } from "../../model/matching.js";
import { isUrl } from "../../io/remote.js";
import { CropVideoBackend } from "../../video/crop-backend.js";
import { resolveSourceFrameCount } from "./frame-count.js";
import type { CropRect } from "../../transform/points.js";
import type { Fill } from "../../transform/frame.js";
import {
  Camera,
  CameraGroup,
  FrameGroup,
  InstanceGroup,
  injectSessionFrameResolver,
  RecordingSession,
} from "../../model/camera.js";
import { Identity } from "../../model/identity.js";
import type { Embedding } from "../../model/embedding.js";
import { LazyDataStore, LazyFrameList } from "../../model/lazy.js";
import { buildVideoIdMap } from "../../model/video-id-map.js";
import {
  markNegativeFrames,
  negativeFrameKeys,
} from "../../model/negative-frames.js";
import type { ROI } from "../../model/roi.js";
import type { SegmentationMask } from "../../model/mask.js";
import type { BoundingBox } from "../../model/bbox.js";
import { type Centroid } from "../../model/centroid.js";
import { buildCentroidTuples } from "./centroids.js";
import type { LabelImage } from "../../model/label-image.js";

const textDecoder = new TextDecoder();

/**
 * Ordered stage labels for the eager reader. `total` is derived from
 * `EAGER_STAGES.length` so the count can never drift from the labels — the
 * single source of truth for the eager path's progress. These are coarser than
 * the streaming reader's stages (frames/instances/points are read together as
 * one "Building labeled frames" step), which is fine: the consumer always gets
 * a consistent (current, total, message).
 */
const EAGER_STAGES = [
  "Reading metadata",
  "Reading tracks",
  "Reading videos",
  "Reading suggestions",
  "Building labeled frames",
  "Reading identities",
  "Reading sessions",
  "Reading annotations",
] as const;

export async function readSlp(
  source: SlpSource,
  options?: {
    openVideos?: boolean;
    h5?: OpenH5Options;
    /**
     * Capture the verbatim, deep-cloned `sessions_json` dict onto each
     * `RecordingSession.rawJson` (deprecated, transitional). Default `false`:
     * the faithful typed session model is the source of truth, and skipping the
     * clone avoids doubling session memory in `Labels.copy()`. See
     * `RecordingSession.rawJson`.
     */
    rawSessions?: boolean;
    /**
     * Optional progress callback fired as loading advances through its stages.
     * `current` counts completed stages out of `total`; `message` labels the
     * stage about to run. Matches the (current, total, message?) convention used
     * elsewhere in the library (Labels.merge, RenderOptions.onProgress). The
     * stages are coarser than the streaming reader's, but the contract is the
     * same. Reporting is a pure side effect and never alters the loaded Labels.
     */
    onProgress?: (current: number, total: number, message?: string) => void;
  },
): Promise<Labels> {
  // `total` is derived from the stage list (single source of truth) so the
  // count cannot drift from the labels. `report(i)` fires the i-th stage label.
  const total = EAGER_STAGES.length;
  const onProgress = options?.onProgress;
  const report = (i: number) => onProgress?.(i, total, EAGER_STAGES[i]);

  const { file, close } = await openH5File(source, options?.h5);
  // Remote auth headers to persist onto video backends (so existence probes /
  // URL-backed reopens stay authenticated). Only present when loading a remote
  // `.slp` with headers.
  const remote = {
    urlHeaders: options?.h5?.headers,
  };
  try {
    checkOpenedAsHdf5(file, source, options?.h5);
    report(0); // Reading metadata
    const labelsPath =
      typeof source === "string"
        ? source
        : (options?.h5?.filenameHint ?? "slp-data.slp");
    const metadataGroup = file.get("metadata");
    // A missing `metadata` group is treated the same as a missing `json`
    // attribute: both indicate a truncated/corrupt file. Mirrors Python
    // sleap-io PR #446, where `read_metadata` catches the `KeyError` from
    // BOTH cases and maps them to the same helpful `ValueError`.
    if (!metadataGroup) {
      throw missingMetadataJsonError(labelsPath);
    }

    const metadataAttrs =
      (metadataGroup as unknown as { attrs?: Record<string, any> }).attrs ?? {};
    const formatId = checkSlpFormat(metadataAttrs, labelsPath);
    // Throws the same helpful error if the `metadata` group exists but its
    // required `json` attribute is missing/empty (truncated/corrupt file);
    // mirrors Python sleap-io PR #446.
    const metadataJson = parseMetadataJson(
      metadataAttrs["json"],
      labelsPath,
    ) as Record<string, unknown> | null;

    const skeletons = parseSkeletons(metadataJson);
    report(1); // Reading tracks
    const tracks = readTracks(file.get("tracks_json"));
    const videoCrops = readVideoCrops(file);
    // Hold the bar at the "Reading videos" stage while videos open; the
    // per-video sub-reporter surfaces "Opening videos (i/n)" when openVideos is
    // true (probing embedded backends can be slow). Mirrors the streaming path.
    report(2); // Reading videos
    const openVideos = options?.openVideos ?? true;
    const videos = await readVideos(
      file.get("videos_json"),
      labelsPath,
      openVideos,
      file,
      formatId,
      videoCrops,
      openVideos
        ? (i, n) => onProgress?.(2, total, `Opening videos (${i}/${n})`)
        : undefined,
      remote,
    );
    report(3); // Reading suggestions
    const suggestions = readSuggestions(file.get("suggestions_json"), videos);

    report(4); // Building labeled frames
    // Low-level Module for the fast columnar compound read (falls back to
    // `.value` when unavailable). Fetched once; null on browsers without the
    // raw surface, which is fine — normalizeStructDataset then uses `.value`.
    const emscripten = await getH5EmscriptenModule();
    const pose = readPoseTables(file, labelsPath, videos.length, emscripten);
    const { framesData, instancesData, pointsData, predPointsData } = pose;

    // One entry per /instances row, so links to instances index it by row.
    const allInstances: Array<Instance | PredictedInstance> = [];
    const labeledFrames = buildLabeledFrames({
      framesData,
      instancesData,
      pointsData,
      predPointsData,
      skeletons,
      tracks,
      videos,
      formatId,
      instancesByRow: allInstances,
    });

    // Read negative frames (video ids resolved like /frames', see helper)
    markNegativeFrames(
      labeledFrames,
      readNegativeFrames(file, pose),
      framesData,
      videos,
    );

    report(5); // Reading identities
    const identities = readIdentityCatalog(file);
    report(6); // Reading sessions
    const nFrameRows = (framesData.frame_id ?? []).length;
    const sessions = readSessions(
      file.get("sessions_json"),
      videos,
      skeletons,
      identities,
      options?.rawSessions ?? false,
      readSessionDataEager(file, labelsPath, nFrameRows, emscripten),
    );
    report(7); // Reading annotations
    const tableContext: TableContext = {
      labelsPath,
      nInstances: (instancesData.instance_type ?? []).length,
    };
    const annotations = readAnnotations(
      file,
      videos,
      tracks,
      tableContext,
      allInstances,
    );

    // Per-detection re-ID identity (SLP 2.5) + category (SLP 2.7): attach onto each
    // modality's ordered detection list (index == owner_id). No-op on older files.
    const pdMaps = readPerDetectionMaps(file, labelsPath, emscripten);
    attachOwnerType(pdMaps, 0, allInstances, identities);
    attachAnnotationOwners(pdMaps, annotations, identities);

    const staticRois = distributeAnnotations(
      labeledFrames,
      videos,
      annotations,
      allInstances,
    );

    onProgress?.(total, total, "Finalizing");
    const labels = new Labels({
      labeledFrames,
      videos,
      skeletons,
      tracks,
      suggestions,
      sessions,
      identities,
      provenance: readProvenance(file, metadataJson),
      rois: staticRois,
    });
    // Sessions are read before the frame store exists; wire the lazy frame
    // resolver now so ref-backed grouping resolves against labels.labeledFrames.
    injectSessionFrameResolver(labels);
    return labels;
  } finally {
    close();
  }
}

/**
 * Ordered stage labels for the lazy reader. As with EAGER_STAGES, `total` is
 * derived from `LAZY_STAGES.length` (single source of truth). The lazy path
 * does NOT materialize labeled frames up front — it reads the raw frame/
 * instance/point columns into a LazyDataStore — so the "Reading frame data"
 * stage reads raw data rather than constructing LabeledFrames. The stages
 * otherwise mirror the eager reader's real steps.
 */
const LAZY_STAGES = [
  "Reading metadata",
  "Reading tracks",
  "Reading videos",
  "Reading suggestions",
  "Reading frame data",
  "Reading identities",
  "Reading sessions",
  "Reading annotations",
] as const;

/**
 * Read an SLP file in lazy mode. Frames are not materialized until accessed.
 * Returns a Labels object with a LazyFrameList that loads frames on demand.
 */
export async function readSlpLazy(
  source: SlpSource,
  options?: {
    openVideos?: boolean;
    h5?: OpenH5Options;
    /**
     * Capture the verbatim, deep-cloned `sessions_json` dict onto each
     * `RecordingSession.rawJson` (deprecated, transitional). Default `false`.
     * See `readSlp` and `RecordingSession.rawJson`.
     */
    rawSessions?: boolean;
    /**
     * Optional progress callback fired as loading advances through its stages.
     * `current` counts completed stages out of `total`; `message` labels the
     * stage about to run. Matches the (current, total, message?) convention used
     * elsewhere in the library (Labels.merge, RenderOptions.onProgress). The
     * lazy path's stages are coarser than the streaming reader's. Reporting is a
     * pure side effect and never alters the loaded Labels.
     */
    onProgress?: (current: number, total: number, message?: string) => void;
  },
): Promise<Labels> {
  // `total` is derived from the stage list (single source of truth) so the
  // count cannot drift from the labels. `report(i)` fires the i-th stage label.
  const total = LAZY_STAGES.length;
  const onProgress = options?.onProgress;
  const report = (i: number) => onProgress?.(i, total, LAZY_STAGES[i]);

  const { file, close } = await openH5File(source, options?.h5);
  const remote = {
    urlHeaders: options?.h5?.headers,
  };
  try {
    checkOpenedAsHdf5(file, source, options?.h5);
    report(0); // Reading metadata
    const labelsPath =
      typeof source === "string"
        ? source
        : (options?.h5?.filenameHint ?? "slp-data.slp");
    const metadataGroup = file.get("metadata");
    // A missing `metadata` group is treated the same as a missing `json`
    // attribute: both indicate a truncated/corrupt file. Mirrors Python
    // sleap-io PR #446 (and the eager/streaming readers).
    if (!metadataGroup) {
      throw missingMetadataJsonError(labelsPath);
    }

    const metadataAttrs =
      (metadataGroup as unknown as { attrs?: Record<string, any> }).attrs ?? {};
    const formatId = checkSlpFormat(metadataAttrs, labelsPath);
    // Throws the same helpful error if the `metadata` group exists but its
    // required `json` attribute is missing/empty (truncated/corrupt file);
    // mirrors Python sleap-io PR #446.
    const metadataJson = parseMetadataJson(
      metadataAttrs["json"],
      labelsPath,
    ) as Record<string, unknown> | null;

    const skeletons = parseSkeletons(metadataJson);
    report(1); // Reading tracks
    const tracks = readTracks(file.get("tracks_json"));
    const videoCrops = readVideoCrops(file);
    // Hold the bar at the "Reading videos" stage while videos open; the
    // per-video sub-reporter surfaces "Opening videos (i/n)" when openVideos is
    // true. Mirrors the streaming and eager paths.
    report(2); // Reading videos
    const openVideos = options?.openVideos ?? true;
    const videos = await readVideos(
      file.get("videos_json"),
      labelsPath,
      openVideos,
      file,
      formatId,
      videoCrops,
      openVideos
        ? (i, n) => onProgress?.(2, total, `Opening videos (${i}/${n})`)
        : undefined,
      remote,
    );
    report(3); // Reading suggestions
    const suggestions = readSuggestions(file.get("suggestions_json"), videos);

    report(4); // Reading frame data
    // Read raw data but don't build frames yet. Use the fast columnar compound
    // read (falls back to `.value`); see normalizeStructDataset / h5-compound.ts.
    const emscripten = await getH5EmscriptenModule();
    const pose = readPoseTables(file, labelsPath, videos.length, emscripten);
    const { framesData, instancesData, pointsData, predPointsData } = pose;

    // Read negative frames (raw video ids; the store resolves them per frame)
    const negativeFrames = readNegativeFrames(file, pose);

    const store = new LazyDataStore({
      framesData,
      instancesData,
      pointsData,
      predPointsData,
      skeletons,
      tracks,
      videos,
      formatId,
      negativeFrames,
    });

    const lazyFrames = new LazyFrameList(store);

    // Read sessions — grouping is captured as index refs only (no frame
    // materialization); refs resolve lazily via the injected frame resolver.
    report(5); // Reading identities
    const identities = readIdentityCatalog(file);
    report(6); // Reading sessions
    const sessions = readSessions(
      file.get("sessions_json"),
      videos,
      skeletons,
      identities,
      options?.rawSessions ?? false,
      readSessionDataEager(
        file,
        labelsPath,
        (framesData.frame_id ?? []).length,
        emscripten,
      ),
    );
    report(7); // Reading annotations
    const tableContext: TableContext = {
      labelsPath,
      nInstances: (instancesData.instance_type ?? []).length,
    };
    const annotations = readAnnotations(file, videos, tracks, tableContext);

    // Per-detection identity (SLP 2.5) + category (SLP 2.7). Instances materialize
    // lazily, so the OWNER_INSTANCE maps + catalogs are handed to the store and
    // attached in materializeFrame; the other modalities are concrete and attached
    // now.
    const pdMaps = readPerDetectionMaps(file, labelsPath, emscripten);
    store.identities = identities;
    store._instanceIdentityLinks = pdMaps.idLinks.get(0);
    store._instanceEmbeddings = pdMaps.idEmbs.get(0);
    store.categoryCatalog = pdMaps.categoryCatalog;
    store._instanceCategoryLinks = pdMaps.catLinks.get(0);
    store._instanceCategoryEmbeddings = pdMaps.catEmbs.get(0);
    attachAnnotationOwners(pdMaps, annotations, identities);

    addAnnotationsToStore(store, lazyFrames, annotations);

    // Don't pass annotations to Labels — they live on the store
    const labels = new Labels({
      videos,
      skeletons,
      tracks,
      suggestions,
      sessions,
      identities,
      provenance: readProvenance(file, metadataJson),
    });

    // Replace the eager labeledFrames with lazy proxy
    labels._lazyFrameList = lazyFrames;
    labels._lazyDataStore = store;

    // Wire the lazy frame resolver AFTER _lazyFrameList is attached so ref-backed
    // grouping resolves a single frame on demand (never the full table).
    injectSessionFrameResolver(labels);

    onProgress?.(total, total, "Finalizing");
    return labels;
  } finally {
    close();
  }
}

// parseJsonAttr and parseSkeletons are imported from parsers.ts

/** A table's group of columns (SLP 3.0), or `null` if it isn't stored as one. */
function findTableGroup(file: any, table: string): any {
  let path = "";
  let node: any = file;
  for (const part of table.split("/")) {
    path = path ? `${path}/${part}` : part;
    if (!isH5Group(node) || !node.keys().includes(part)) return null;
    node = file.get(path);
  }
  return isH5Group(node) ? node : null;
}

/**
 * Read one table, whichever layout it is in: as columns the rest of this file can
 * index (`columnar` says it was a group of them), or `null` if the file doesn't have
 * it. Mirrors Python's `_read_table`.
 */
function readTable(
  file: any,
  table: ColumnarTableName,
  labelsPath: string,
  module?: unknown,
): { rows: Record<string, any[]>; columnar: boolean } | null {
  const group = findTableGroup(file, table);
  if (group) {
    const raw = readH5GroupColumns(group, table, tableMembers(table));
    return {
      rows: normalizeColumnarTable(labelsPath, table, raw) as Record<
        string,
        any[]
      >,
      columnar: true,
    };
  }
  const dataset = file.get(legacyTablePath(table));
  return dataset
    ? { rows: normalizeStructDataset(dataset, module), columnar: false }
    : null;
}

/** Read and normalize `/frames`, `/instances`, `/points` and `/pred_points`. */
function readPoseTables(
  file: any,
  labelsPath: string,
  nVideos: number,
  module: unknown,
): PoseTables {
  const raw = (path: string): RawTable => {
    const node = file.get(path);
    return isH5Group(node)
      ? { ...readH5GroupColumns(node, path), columnar: true }
      : { columns: normalizeStructDataset(node, module), columnar: false };
  };
  return normalizePoseTables(labelsPath, nVideos, {
    frames: raw("frames"),
    instances: raw("instances"),
    points: raw("points"),
    predPoints: raw("pred_points"),
  });
}

/** The raw negative frame keys: from a 3.0 `/frames`, else `/negative_frames`. */
function readNegativeFrames(file: any, pose: PoseTables): Set<string> {
  if (pose.negativeFrames) return pose.negativeFrames;
  const dataset = file.get("negative_frames");
  return dataset
    ? negativeFrameKeys(normalizeStructDataset(dataset))
    : new Set<string>();
}

/** The file's provenance: its own dataset if it has one, else the metadata's. */
function readProvenance(
  file: any,
  metadataJson: Record<string, unknown> | null,
): Record<string, unknown> {
  const dataset = file.get("provenance_json");
  return parseProvenance(dataset ? dataset.value : undefined, metadataJson);
}

function readTracks(dataset: any): Track[] {
  if (!dataset) return [];
  const values = dataset.value ?? [];
  const tracks: Track[] = [];
  for (const entry of values) {
    let parsed = entry;
    if (typeof entry === "string") {
      try {
        parsed = JSON.parse(entry);
      } catch {
        parsed = entry;
      }
    }
    if (Array.isArray(parsed)) {
      tracks.push(new Track(String(parsed[1] ?? parsed[0])));
    } else if (parsed?.name) {
      tracks.push(new Track(String(parsed.name)));
    } else {
      tracks.push(new Track(String(parsed)));
    }
  }
  return tracks;
}

/** A single `/video_crops` entry: the crop rect and OOB fill for one video. */
interface VideoCropEntry {
  crop: CropRect;
  fill: Fill;
}

/**
 * Read the top-level `/video_crops` dataset (SLP format 2.3) into a map keyed by
 * video index. Port of Python `read_video_crops` (slp.py:649-680).
 *
 * The dataset holds a single JSON string (the array form written by
 * {@link writeVideoCrops}, or the scalar `np.bytes_` form Python writes) listing
 * `{ video, crop: [x1,y1,x2,y2], fill }` entries, one per cropped video. It is
 * absent on old/uncropped files, in which case an empty map is returned so the
 * loader falls back to the uncropped source. h5wasm surfaces the value as a
 * plain string (Python's scalar S-dtype), a length-1 array of strings (the JS
 * vlen-array form), or raw bytes — all three are normalized here.
 */
export function readVideoCrops(file: any): Map<number, VideoCropEntry> {
  const out = new Map<number, VideoCropEntry>();
  const keys = file.keys?.() ?? [];
  if (!keys.includes("video_crops")) return out;
  const ds = file.get("video_crops");
  if (!ds) return out;

  // Python writes /video_crops as a scalar `|S<n>` (np.bytes_) JSON dataset;
  // decode every backend's representation through the shared, throw-safe helper.
  let json: string | undefined;
  try {
    json = datasetValueToString((ds as { value?: unknown }).value);
  } catch {
    json = undefined;
  }
  if (json === undefined) return out;

  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return out;
  }
  if (!Array.isArray(parsed)) return out;
  for (const entry of parsed as Array<Record<string, unknown>>) {
    if (!entry || typeof entry !== "object") continue;
    const videoIdx = Number(entry.video);
    const cropArr = entry.crop as number[] | undefined;
    if (!Array.isArray(cropArr) || cropArr.length !== 4) continue;
    const crop: CropRect = [
      Number(cropArr[0]),
      Number(cropArr[1]),
      Number(cropArr[2]),
      Number(cropArr[3]),
    ];
    const fillRaw = entry.fill;
    const fill: Fill = Array.isArray(fillRaw)
      ? (fillRaw as number[]).map((v) => Number(v))
      : Number(fillRaw ?? 0);
    out.set(videoIdx, { crop, fill });
  }
  return out;
}

/**
 * Read a video's `source_video` metadata JSON from its `{group}/source_video`
 * HDF5 group, or `null` when the group is absent. Checks a `json` *dataset*
 * first (where oversized metadata is stored) then the `json` *attribute* (the
 * normal case), mirroring Python `_read_source_video_json`. h5wasm sync path.
 */
export function readSourceVideoGroupJson(
  file: any,
  groupPath: string,
): Record<string, unknown> | null {
  const grp = file.get(`${groupPath}/source_video`);
  if (!grp) return null;
  let raw: string | undefined;
  // Prefer the spilled `json` DATASET (oversized metadata, e.g. an ImageVideo
  // source with thousands of filenames — Python writes it as a scalar `|S<n>`
  // via np.bytes_). The read is guarded: h5wasm `.value` can throw on unusual
  // scalar/large datasets, and a throw here would otherwise abort the whole
  // file open (this runs inside readVideos with no surrounding try/catch), so
  // fall back to the `json` attribute rather than failing the load.
  try {
    const ds = file.get(`${groupPath}/source_video/json`);
    if (ds) raw = datasetValueToString((ds as { value?: unknown }).value);
  } catch {
    raw = undefined;
  }
  if (raw === undefined) {
    raw = attrToString((grp.attrs ?? {}).json);
  }
  if (raw === undefined) return null;
  try {
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
}

async function readVideos(
  dataset: any,
  labelsPath: string,
  openVideos: boolean,
  file: any,
  formatId: number,
  videoCrops?: Map<number, VideoCropEntry>,
  onVideoProgress?: (current: number, total: number) => void,
  remote?: {
    urlHeaders?: Record<string, string>;
  },
): Promise<Video[]> {
  if (!dataset) return [];
  const urlHeaders = remote?.urlHeaders;
  const values = dataset.value ?? [];
  const videos: Video[] = [];

  for (let videoIndex = 0; videoIndex < values.length; videoIndex++) {
    onVideoProgress?.(videoIndex + 1, values.length);
    const entry = values[videoIndex];
    if (!entry) continue;
    const parsed =
      typeof entry === "string"
        ? JSON.parse(entry)
        : JSON.parse(textDecoder.decode(entry));
    const backendMeta = parsed.backend ?? {};
    let filename = resolveVideoFilename(backendMeta, parsed);
    let datasetPath = backendMeta.dataset ?? null;
    let embedded = false;

    if (filename === ".") {
      embedded = true;
      filename = labelsPath;
    }

    // Auto-detect dataset path when embedded but not specified in metadata
    if (embedded && !datasetPath) {
      datasetPath = findVideoDataset(file, videoIndex);
    }

    // Read format, channel_order, and frames from HDF5 dataset attributes when
    // available. Matches Python sleap-io behavior (video_reading.py:987-1006).
    // The `frames` attribute records the source video's total frame count for
    // embedded videos — the dataset itself only holds a subset (e.g. labeled
    // frames), but downstream consumers (seekbar, re-embed) need the source
    // range. Falls back to JSON shape[0] when absent.
    let format = backendMeta.format as string | undefined;
    let channelOrderFromAttrs: string | undefined;
    let frameCountFromAttrs: number | undefined;
    let heightFromAttrs: number | undefined;
    let widthFromAttrs: number | undefined;
    let channelsFromAttrs: number | undefined;
    if (datasetPath) {
      const videoDs = file.get(datasetPath);
      if (videoDs) {
        const attrs =
          (videoDs as { attrs?: Record<string, unknown> }).attrs ?? {};
        if (!format) {
          format = attrToString(attrs.format);
        }
        channelOrderFromAttrs = attrToString(attrs.channel_order);
        const framesNum = attrToNumber(attrs.frames);
        if (framesNum !== undefined && framesNum > 0) {
          frameCountFromAttrs = framesNum;
        }
        // Height/width/channels from dataset attrs, used when JSON metadata is
        // missing (common for pkg.slp). Mirrors read-streaming.ts.
        const h = attrToNumber(attrs.height);
        if (h !== undefined && h > 0) heightFromAttrs = h;
        const w = attrToNumber(attrs.width);
        if (w !== undefined && w > 0) widthFromAttrs = w;
        const c = attrToNumber(attrs.channels);
        if (c !== undefined && c > 0) channelsFromAttrs = c;
      }
    }

    // Read frame_numbers up front so the source frame count can be resolved
    // synchronously (the dataset itself only holds a subset; see below). Reused
    // by the backend below so it is not read twice.
    const frameNumbers = datasetPath ? readFrameNumbers(file, datasetPath) : [];

    // Compose shape. Dimensions prefer the videos_json shape, falling back to
    // the dataset attrs (pkg.slp files often carry height/width/channels as
    // attrs with no JSON shape). The source frame count (seekbar extent) comes
    // from the `frames` attr, then videos_json, then max(frame_numbers)+1 — the
    // last keeps multi-video pkg.slp files written without a `frames` attr
    // (older PyQt SLEAP) resolving an extent instead of reporting none. Mirrors
    // read-streaming.ts so the eager and streaming readers stay in parity.
    const jsonShape = backendMeta.shape as number[] | undefined;
    const height = jsonShape?.[1] ?? heightFromAttrs;
    const width = jsonShape?.[2] ?? widthFromAttrs;
    const channels = jsonShape?.[3] ?? channelsFromAttrs;
    const frameCount = resolveSourceFrameCount({
      framesAttr: frameCountFromAttrs,
      jsonFrameCount: jsonShape?.[0],
      frameNumbers,
    });
    const shape: [number, number, number, number] | undefined =
      height && width && channels
        ? [frameCount ?? 0, height, width, channels]
        : undefined;

    // Determine channel order with priority:
    // 1. JSON metadata (backendMeta.channel_order)
    // 2. HDF5 dataset attribute (channelOrderFromAttrs)
    // 3. Legacy fallback based on format_id (BGR for < 1.4)
    const channelOrder =
      (backendMeta.channel_order as string | undefined) ??
      channelOrderFromAttrs ??
      (formatId < 1.4 ? "BGR" : "RGB");

    // Resolve external (non-embedded) source paths against the labels-file
    // directory so a project moved between machines — or whose media now lives
    // in a subfolder next to the `.slp` — still opens. Uses the injected
    // FsResolver for existence checks (Node, Tauri, and browser-injected all
    // share one policy); with no resolver, or for URLs / embedded videos, the
    // stored source is used verbatim. See issue #213 and video/path-resolve.ts.
    let openFilename: string | string[] = filename;
    let sourceMissing = false;
    if (openVideos && !embedded) {
      const fsResolver = getFsResolver();
      const sourceIsUrl = typeof filename === "string" && isUrl(filename);
      if (fsResolver && !sourceIsUrl && !isUrl(labelsPath)) {
        const resolved = await resolveVideoSource(
          filename,
          posixDirname(labelsPath),
          fsResolver,
        );
        openFilename = resolved.filename;
        sourceMissing = resolved.firstMissing;
      }
    }

    let backend = null;
    let backendError: VideoBackendError | null = null;
    if (openVideos) {
      if (sourceMissing && isImageSource(filename)) {
        // The resolver confirms the first image is unreachable. Do NOT hand back
        // a backend that looks healthy: an ImageVideo given a stored `shape`
        // skips the up-front frame-0 decode, so it would open "successfully"
        // while unable to read a single frame (issue #213). Record the reason so
        // consumers can surface a locate/repair affordance instead of silently
        // rendering blank frames.
        const firstStored = Array.isArray(filename)
          ? (filename[0] ?? "")
          : filename;
        backendError = {
          kind: "image-sequence",
          message:
            `Image sequence not found: could not locate ` +
            `"${posixBasename(firstStored)}" relative to the labels directory ` +
            `"${posixDirname(labelsPath)}". The stored path "${firstStored}" ` +
            `was likely saved on another machine; move or locate the image folder.`,
        };
      } else {
        try {
          backend = await createVideoBackend(openFilename, {
            dataset: datasetPath ?? undefined,
            embedded,
            frameNumbers,
            frameSizes: readFrameSizes(file, datasetPath),
            format,
            channelOrder,
            shape,
            fps: backendMeta.fps,
            // Persist the remote `.slp` auth headers onto remote/embedded video
            // backends so reopens / existence probes stay authenticated. The
            // factory only uses them for URL-backed backends; embedded/local
            // backends ignore them.
            headers: urlHeaders,
          });
        } catch (err) {
          // Resilient load: a single video whose backend can't be built — an
          // image-sequence with missing files, an unsupported `.avi`/`.mpeg`, or
          // a decode failure — must NOT abort the whole project load. Leave the
          // backend null and record the reason so the consumer can show an
          // actionable message / resolver instead of the load throwing.
          backend = null;
          backendError = {
            kind:
              err instanceof UnsupportedVideoFormatError
                ? "unsupported-format"
                : isImageSource(filename)
                  ? "image-sequence"
                  : "decode",
            message: err instanceof Error ? err.message : String(err),
          };
        }
      }
    }

    // Reconstruct the source_video lineage WITH its recorded shape (and any
    // deeper chain) so a reloaded embedded subset can resolve its source's full
    // frame extent via _getEffectiveShape — the prerequisite for the
    // embedded-subset -> restore-original matching workflow (#160). Embedded
    // videos store their source in the authoritative `{group}/source_video`
    // HDF5 group (older `.pkg.slp` omit it from videos_json entirely); prefer it
    // and fall back to a nested videos_json `source_video` (non-embedded videos,
    // and newer files that carry both).
    let sourceVideo: Video | null = null;
    if (embedded && datasetPath) {
      const groupPath = datasetPath.endsWith("/video")
        ? datasetPath.slice(0, -6)
        : datasetPath;
      const svDict = readSourceVideoGroupJson(file, groupPath);
      if (svDict) sourceVideo = buildSourceVideoFromDict(svDict, labelsPath);
    }
    if (!sourceVideo && parsed.source_video) {
      sourceVideo = buildSourceVideoFromDict(
        parsed.source_video as Record<string, unknown>,
        labelsPath,
      );
    }

    // Crop reconstruction (SLP 2.3): the backend just built from videos_json is
    // the UNCROPPED inner. If a /video_crops entry exists for this index, wrap it
    // in a CropVideoBackend (open path) and ALWAYS seed crop/source_shape/shape/
    // crop_fill on backendMetadata so the closed (openVideos=false) path reports
    // the cropped view too. Port of Python make_video (slp.py:380-416).
    const cropEntry = videoCrops?.get(videoIndex);
    let backendMetadata: Record<string, unknown> =
      shape !== backendMeta.shape ? { ...backendMeta, shape } : backendMeta;
    if (cropEntry) {
      const [cx1, cy1, cx2, cy2] = cropEntry.crop;
      if (openVideos && backend) {
        // Each reloaded tile owns its freshly-reconstructed inner (ownsInner
        // defaults to true). We do NOT share one inner across sibling mosaic
        // tiles here: on read each video entry rebuilds a private inner, so
        // sharing would leave the inner unowned (leaked — never closed) for no
        // decode savings. The Hdf5VideoBackend is a thin per-frame slicer over
        // the already-shared open h5 `file`, so duplicating it per tile is
        // cheap. Live mosaic decode-sharing is opt-in and in-memory only, via
        // Video.crop({ shareDecode: true }). Mirrors Python slp.py make_video
        // (slp.py:393-401).
        backend = CropVideoBackend.wrap({
          inner: backend,
          crop: cropEntry.crop,
          fill: cropEntry.fill,
          // ownsInner: true (default) — see comment above.
        });
      }
      // Copy before overwriting so a shared metadata dict is never mutated.
      backendMetadata = { ...backendMetadata };
      const innerShape = backendMetadata.shape as number[] | undefined;
      if (innerShape && innerShape.length === 4) {
        backendMetadata.source_shape = [...innerShape];
        backendMetadata.shape = [
          innerShape[0],
          cy2 - cy1,
          cx2 - cx1,
          innerShape[3],
        ];
      }
      backendMetadata.crop = [...cropEntry.crop];
      backendMetadata.crop_fill = cropEntry.fill;
    }

    const video = new Video({
      filename,
      backend,
      backendError,
      backendMetadata,
      sourceVideo,
      openBackend: openVideos,
      embedded,
    });
    // Persist remote auth headers so a later Video.exists() probe (or a
    // URL-backed backend reopen) stays authenticated. Mirrors Python
    // HDF5Video._url_headers. Only meaningful for videos backed by a remote
    // `.slp` (urlHeaders is undefined for local loads).
    if (urlHeaders) {
      video._setUrlPersistence({ headers: urlHeaders });
    }
    videos.push(video);
  }

  return videos;
}

function readFrameNumbers(file: any, datasetPath: string | null): number[] {
  if (!datasetPath) return [];
  const groupPath = datasetPath.endsWith("/video")
    ? datasetPath.slice(0, -6)
    : datasetPath;
  const frameDataset = file.get(`${groupPath}/frame_numbers`);
  if (!frameDataset) return [];
  const values = frameDataset.value ?? [];
  return Array.from(values).map((v: any) => Number(v));
}

function readFrameSizes(
  file: any,
  datasetPath: string | null,
): number[] | undefined {
  if (!datasetPath) return undefined;
  const groupPath = datasetPath.endsWith("/video")
    ? datasetPath.slice(0, -6)
    : datasetPath;
  const sizesDataset = file.get(`${groupPath}/frame_sizes`);
  if (!sizesDataset) return undefined;
  const values = sizesDataset.value ?? [];
  return Array.from(values).map((v: any) => Number(v));
}

/**
 * Auto-detect video dataset path by scanning HDF5 structure.
 * Tries explicit path first, then scans root keys for video groups.
 */
function findVideoDataset(file: any, videoIndex: number): string | null {
  // Try explicit path first (video0/video, video1/video, etc.)
  const explicitPath = `video${videoIndex}/video`;
  if (file.get(explicitPath)) {
    return explicitPath;
  }

  // Scan root keys for video groups
  const keys = file.keys?.() ?? [];
  for (const key of keys) {
    if (key.startsWith("video")) {
      const candidatePath = `${key}/video`;
      if (file.get(candidatePath)) {
        // For single video case, return first found
        if (videoIndex === 0) {
          return candidatePath;
        }
        // For multi-video, try to match by index from key
        const keyIndex = parseInt(key.slice(5), 10);
        if (!isNaN(keyIndex) && keyIndex === videoIndex) {
          return candidatePath;
        }
      }
    }
  }

  return null;
}

function readSuggestions(dataset: any, videos: Video[]): SuggestionFrame[] {
  if (!dataset) return [];
  const values = dataset.value ?? [];
  const suggestions: SuggestionFrame[] = [];
  for (const entry of values) {
    const parsed =
      typeof entry === "string"
        ? JSON.parse(entry)
        : JSON.parse(textDecoder.decode(entry));
    const videoIndex = Number(parsed.video ?? 0);
    const video = videos[videoIndex];
    if (!video) continue;
    suggestions.push(
      new SuggestionFrame({
        video,
        frameIdx: parsed.frame_idx ?? parsed.frameIdx ?? 0,
        group: parsed.group != null ? String(parsed.group) : undefined,
        metadata: parsed,
      }),
    );
  }
  return suggestions;
}

function readIdentities(dataset: any): Identity[] {
  if (!dataset) return [];
  const values = dataset.value ?? [];
  const identities: Identity[] = [];
  for (const entry of values) {
    const parsed =
      typeof entry === "string"
        ? JSON.parse(entry)
        : JSON.parse(textDecoder.decode(entry));
    const { name, color, ...rest } = parsed;
    identities.push(
      new Identity({
        name: name ?? "",
        color: color ?? undefined,
        metadata: rest,
      }),
    );
  }
  return identities;
}

/** Decode a vlen-string dataset element (string / bytes) to a plain string. */
function decodeStr(v: unknown): string {
  if (typeof v === "string") return v;
  if (v instanceof Uint8Array) return textDecoder.decode(v);
  return String(v ?? "");
}

/**
 * Read the Python-compatible `/identity` group (SLP 2.5): a native vlen `name`
 * catalog + optional EAV metadata (`meta_owner`/`meta_key`/`meta_val`). Color is
 * recovered from `metadata["color"]` back onto `Identity.color`. Returns `[]` when
 * the group is absent. Used as the fallback when a file has no legacy
 * `identities_json` (i.e. a Python-written file); JS-written files carry both and
 * prefer the typed `identities_json`.
 */
function readIdentityGroup(file: any): Identity[] {
  if (!file.get("identity") || !file.get("identity/name")) return [];
  const names = (file.get("identity/name").value ?? []).map(decodeStr);
  const metadata: Record<string, unknown>[] = names.map(() => ({}));
  const ownerDs = file.get("identity/meta_owner");
  if (ownerDs) {
    const owners = Array.from(ownerDs.value ?? []).map(Number);
    const keys = (file.get("identity/meta_key")?.value ?? []).map(decodeStr);
    const vals = (file.get("identity/meta_val")?.value ?? []).map(decodeStr);
    owners.forEach((o: number, i: number) => {
      if (metadata[o]) metadata[o][keys[i]] = vals[i];
    });
  }
  return names.map((name: string, i: number) => {
    const meta = metadata[i];
    const color = typeof meta.color === "string" ? meta.color : undefined;
    const { color: _color, ...rest } = meta;
    return new Identity({ name, color, metadata: rest });
  });
}

/**
 * Read the identity catalog, preferring the legacy typed `identities_json` (JS-native,
 * present on JS-written files) and falling back to the Python `/identity` group (on
 * Python-written files). Both are written by the JS writer (dual-write).
 */
function readIdentityCatalog(file: any): Identity[] {
  const json = file.get("identities_json");
  if (json) return readIdentities(json);
  return readIdentityGroup(file);
}

/**
 * Read a per-detection links table — `/identity/links` (SLP 2.5) or
 * `/categories/links` (SLP 2.7) — a genuine compound (Python) or a
 * flat-2-D+`field_names` (JS) `(owner_type, owner_id, <idxField>, <scoreField>)`
 * table via `normalizeStructDataset` (handles both). Returns
 * `owner_type → owner_id → [idx, score|null]` (NaN score → null); empty when the
 * dataset is absent. `i8`/`u8` columns are `Number()`-coerced.
 */
function readLinksTable(
  file: any,
  table: "identity/links" | "categories/links",
  idxField: string,
  scoreField: string,
  labelsPath: string,
  emscripten?: unknown,
): Map<number, Map<number, [number, number | null]>> {
  const read = readTable(file, table, labelsPath, emscripten);
  if (!read) return new Map();
  const cols = read.rows;
  const catalog = file.get(`${table.split("/")[0]}/name`);
  if (
    read.columnar &&
    catalog &&
    indexOutOfRange(cols[idxField] ?? [], catalog.shape[0], false)
  ) {
    reportTableProblems(labelsPath, [
      `/${table} ${idxField} is not one of the ${catalog.shape[0]} catalog entries`,
    ]);
  }
  return groupLinks(cols, idxField, scoreField);
}

const readIdentityLinks = (
  file: any,
  labelsPath: string,
  emscripten?: unknown,
) =>
  readLinksTable(
    file,
    "identity/links",
    "identity_idx",
    "identity_score",
    labelsPath,
    emscripten,
  );

const readCategoryLinks = (
  file: any,
  labelsPath: string,
  emscripten?: unknown,
) =>
  readLinksTable(
    file,
    "categories/links",
    "category_idx",
    "category_score",
    labelsPath,
    emscripten,
  );

/** Read the `/categories/name` catalog (SLP 2.7) as plain strings. */
function readCategoryCatalog(file: any): string[] {
  const ds = file.get("categories/name");
  if (!ds) return [];
  return (ds.value ?? []).map(decodeStr);
}

/**
 * Read one `/embeddings` triple — `(vectors, owner_type, owner_id)` (identity, SLP
 * 2.5) or `(category_vectors, category_owner_type, category_owner_id)` (category,
 * SLP 2.7) — a plain `(N, D)` float matrix + parallel join columns. Returns
 * `owner_type → owner_id → Embedding`; empty when absent. `i8` columns are
 * `Number()`-coerced.
 */
function readEmbeddingTriple(
  file: any,
  vecName: string,
  otName: string,
  oidName: string,
): Map<number, Map<number, Embedding>> {
  const vecDs = file.get(`embeddings/${vecName}`);
  if (!vecDs) return new Map();
  const shape = vecDs.shape as number[] | undefined;
  const rows = shape?.[0] ?? 0;
  const dim = shape && shape.length >= 2 ? shape[1] : 0;
  if (!rows || !dim) return new Map();
  return groupEmbeddings(
    vecDs.value as ArrayLike<number>,
    rows,
    dim,
    file.get(`embeddings/${otName}`)?.value ?? [],
    file.get(`embeddings/${oidName}`)?.value ?? [],
  );
}

const readEmbeddings = (file: any) =>
  readEmbeddingTriple(file, "vectors", "owner_type", "owner_id");

const readCategoryEmbeddings = (file: any) =>
  readEmbeddingTriple(
    file,
    "category_vectors",
    "category_owner_type",
    "category_owner_id",
  );

function readPerDetectionMaps(
  file: any,
  labelsPath: string,
  emscripten?: unknown,
): PerDetectionMaps {
  return {
    idLinks: readIdentityLinks(file, labelsPath, emscripten),
    idEmbs: readEmbeddings(file),
    catLinks: readCategoryLinks(file, labelsPath, emscripten),
    catEmbs: readCategoryEmbeddings(file),
    categoryCatalog: readCategoryCatalog(file),
  };
}

/**
 * Read the columnar `/session_data` group (SLP 2.8) into an in-memory
 * {@link SessionData} for the eager (h5wasm) reader, or `null` when the group is
 * absent (legacy ≤2.7 files) or missing a required struct table. Struct tables go
 * through {@link readTable} (so a 3.0 group of columns, a JS-written
 * flat-2D+`field_names` matrix and a Python-written compound all resolve to column
 * records); the `points_3d` / `pred_points_3d` float matrices are read raw (flat buffer
 * + column count).
 */
function readSessionDataEager(
  file: any,
  labelsPath: string,
  nFrames: number,
  emscripten?: unknown,
): SessionData | null {
  if (!file.get("session_data")) return null;
  const struct = (name: string) =>
    readTable(
      file,
      `session_data/${name}` as ColumnarTableName,
      labelsPath,
      emscripten,
    );
  const frameGroups = struct("frame_groups");
  const instanceGroups = struct("instance_groups");
  const members = struct("instance_group_members");
  if (!frameGroups || !instanceGroups || !members) return null;

  const matrix = (
    name: string,
    ncolsDefault: number,
  ): SessionPointMatrix | null => {
    const ds = file.get(`session_data/${name}`);
    if (!ds) return null;
    const shape = ds.shape as number[] | undefined;
    return {
      flat: ds.value as ArrayLike<number>,
      ncols: shape && shape.length >= 2 ? shape[1] : ncolsDefault,
    };
  };
  const meta = (name: string): unknown[] | null => {
    const ds = file.get(`session_data/${name}`);
    if (!ds) return null;
    const v = ds.value;
    if (Array.isArray(v)) return v;
    return v != null ? Array.from(v as ArrayLike<unknown>) : null;
  };

  const sessionData: SessionData = {
    frameGroups: frameGroups.rows,
    instanceGroups: instanceGroups.rows,
    members: members.rows,
    points3d: matrix("points_3d", 3),
    predPoints3d: matrix("pred_points_3d", 4),
    frameGroupMeta: meta("frame_group_meta"),
    instanceGroupMeta: meta("instance_group_meta"),
  };
  if (frameGroups.columnar && instanceGroups.columnar && members.columnar) {
    reportTableProblems(
      labelsPath,
      sessionProblems(sessionData, {
        nIdentities: file.get("identity/name")?.shape?.[0] ?? 0,
        nFrames,
      }),
    );
  }
  return sessionData;
}

function readSessions(
  dataset: any,
  videos: Video[],
  skeletons: Skeleton[],
  identities?: Identity[],
  captureRaw: boolean = false,
  sessionData?: SessionData | null,
): RecordingSession[] {
  if (!dataset) return [];
  const values = dataset.value ?? [];
  // Present-but-empty is a legitimate "no sessions" state; a present dataset whose
  // entries read back blank/unparseable is the h5wasm vlen ceiling (sleap-io.js#220)
  // and must fail loud rather than silently drop calibration + grouping + 3D.
  if (values.length === 0) return [];
  const sessions: RecordingSession[] = [];
  for (const entry of values) {
    let parsed: Record<string, unknown>;
    try {
      parsed = parseJsonEntry(entry) as Record<string, unknown>;
    } catch (err) {
      throw sessionsReadError(values.length, entry, err);
    }
    if (!parsed || typeof parsed !== "object") {
      throw sessionsReadError(values.length, entry);
    }
    const cameraGroup = new CameraGroup();
    const cameraMap = new Map<string, Camera>();
    const calibration = asRecord(parsed.calibration);
    for (const [key, data] of Object.entries(calibration)) {
      if (key === "metadata") continue;
      const cameraData = asRecord(data);
      const camera = new Camera({
        name: (cameraData.name as string | undefined) ?? key,
        rvec: (cameraData.rotation as number[] | undefined) ?? [0, 0, 0],
        tvec: (cameraData.translation as number[] | undefined) ?? [0, 0, 0],
        matrix: cameraData.matrix as number[][] | undefined,
        distortions: cameraData.distortions as number[] | undefined,
        size: normalizeCameraSize(cameraData.size),
      });
      cameraGroup.cameras.push(camera);
      cameraMap.set(String(key), camera);
    }
    cameraGroup.metadata =
      (calibration.metadata as Record<string, unknown> | undefined) ?? {};

    const session = new RecordingSession({
      cameraGroup,
      metadata: (parsed.metadata as Record<string, unknown> | undefined) ?? {},
    });
    // Optionally retain the verbatim parsed sessions_json dict (deprecated,
    // opt-in via `rawSessions`). Deep-cloned so it is an INDEPENDENT snapshot:
    // the object model reuses `parsed`'s nested metadata/calibration objects by
    // reference, so without the clone a consumer mutating `rawJson` would
    // silently alter what `serializeSession` writes to disk (and vice versa).
    // Never re-written to disk; see RecordingSession.rawJson.
    if (captureRaw) session.rawJson = structuredClone(parsed);
    const map = asRecord(parsed.camcorder_to_video_idx_map);
    for (const [cameraKey, videoIdx] of Object.entries(map)) {
      const camera = resolveCameraKey(
        cameraKey,
        cameraMap,
        cameraGroup.cameras,
      );
      const video = videos[Number(videoIdx)];
      if (camera && video) {
        session.addVideo(video, camera);
      }
    }

    // SLP 2.8 columnar path: reconstruct frame groups from the /session_data tables
    // via the session's fg_start/fg_end range. Dispatched on the presence of both
    // fg_start (slim sessions_json) and the loaded /session_data group.
    const fgStart = parsed.fg_start;
    if (fgStart != null && sessionData) {
      const frameGroupMap = reconstructColumnarFrameGroups(
        cameraGroup.cameras,
        skeletons,
        identities,
        sessionData,
        Number(fgStart),
        Number(parsed.fg_end),
      );
      for (const [k, v] of frameGroupMap) session.frameGroups.set(k, v);
      sessions.push(session);
      continue;
    }

    // Legacy (≤2.7) inline path: frame groups nested in the sessions_json blob.
    const frameGroups = Array.isArray(parsed.frame_group_dicts)
      ? parsed.frame_group_dicts
      : [];
    for (const group of frameGroups) {
      const groupRecord = asRecord(group);
      const frameIdx =
        (groupRecord.frame_idx as number | undefined) ??
        (groupRecord.frameIdx as number | undefined) ??
        0;
      const instanceGroups: InstanceGroup[] = [];
      const instanceGroupList = Array.isArray(groupRecord.instance_groups)
        ? groupRecord.instance_groups
        : [];
      for (const instanceGroup of instanceGroupList) {
        const instanceGroupRecord = asRecord(instanceGroup);
        // Concrete instances are built ONLY for the JS-inline format (camera key
        // -> point dict). The Python/camcorder format is stored as index refs and
        // resolved lazily via the injected frame resolver — no frame
        // materialization at read time.
        let instanceByCamera: Map<Camera, Instance> | undefined;
        const instancesRecord = asRecord(instanceGroupRecord.instances);
        for (const [cameraKey, points] of Object.entries(instancesRecord)) {
          const camera = resolveCameraKey(
            cameraKey,
            cameraMap,
            cameraGroup.cameras,
          );
          if (!camera) {
            console.warn(
              `Camera key "${cameraKey}" not found in session calibration — skipping 2D instance data for this camera.`,
            );
            continue;
          }
          const skeleton = skeletons[0] ?? new Skeleton({ nodes: [] });
          if (!instanceByCamera) instanceByCamera = new Map<Camera, Instance>();
          instanceByCamera.set(
            camera,
            new Instance({
              points: points as Record<string, number[]>,
              skeleton,
            }),
          );
        }

        // Capture verbatim index refs (camera -> [lfIdx, instIdx]) from the
        // Python-canonical camcorder map, as NUMBERS (the fixture stores them as
        // strings). Kept even when concrete instances exist so an untouched
        // round-trip re-derives / falls back to them losslessly on write.
        let instanceRefsByCamera: Map<Camera, [number, number]> | undefined;
        const lfInstMap = asRecord(
          instanceGroupRecord.camcorder_to_lf_and_inst_idx_map,
        );
        for (const [camIdx, value] of Object.entries(lfInstMap)) {
          const camera = resolveCameraKey(
            camIdx,
            cameraMap,
            cameraGroup.cameras,
          );
          if (!camera) continue;
          const pair = value as unknown as [unknown, unknown];
          if (!instanceRefsByCamera)
            instanceRefsByCamera = new Map<Camera, [number, number]>();
          instanceRefsByCamera.set(camera, [Number(pair[0]), Number(pair[1])]);
        }

        const instance3d = reconstructInstance3D(
          instanceGroupRecord,
          skeletons,
        );
        const identity = resolveIdentity(instanceGroupRecord, identities);

        instanceGroups.push(
          new InstanceGroup({
            instanceByCamera,
            instanceRefsByCamera,
            score: instanceGroupRecord.score as number | undefined,
            instance3d,
            identity,
            metadata:
              (instanceGroupRecord.metadata as
                | Record<string, unknown>
                | undefined) ?? {},
          }),
        );
      }

      // Capture labeled-frame index refs (camera -> lfIdx) verbatim; resolved
      // lazily on access. No frame materialization at read time.
      let labeledFrameRefsByCamera: Map<Camera, number> | undefined;
      const labeledFrameMap = asRecord(groupRecord.labeled_frame_by_camera);
      for (const [cameraKey, labeledFrameIdx] of Object.entries(
        labeledFrameMap,
      )) {
        const camera = resolveCameraKey(
          cameraKey,
          cameraMap,
          cameraGroup.cameras,
        );
        if (!camera) {
          console.warn(
            `Camera key "${cameraKey}" not found in session calibration — skipping labeled frame mapping.`,
          );
          continue;
        }
        if (!labeledFrameRefsByCamera)
          labeledFrameRefsByCamera = new Map<Camera, number>();
        labeledFrameRefsByCamera.set(camera, Number(labeledFrameIdx));
      }

      // If no labeled_frame_by_camera, reconstruct refs from
      // camcorder_to_lf_and_inst_idx_map.
      if (!labeledFrameRefsByCamera) {
        for (const instanceGroup of instanceGroupList) {
          const igRecord = asRecord(instanceGroup);
          const lfInstMap = asRecord(igRecord.camcorder_to_lf_and_inst_idx_map);
          for (const [camIdx, value] of Object.entries(lfInstMap)) {
            const camera = resolveCameraKey(
              camIdx,
              cameraMap,
              cameraGroup.cameras,
            );
            if (!camera) continue;
            const pair = value as unknown as [unknown, unknown];
            if (!labeledFrameRefsByCamera)
              labeledFrameRefsByCamera = new Map<Camera, number>();
            labeledFrameRefsByCamera.set(camera, Number(pair[0]));
          }
        }
      }

      session.frameGroups.set(
        Number(frameIdx),
        new FrameGroup({
          frameIdx: Number(frameIdx),
          instanceGroups,
          labeledFrameRefsByCamera,
          metadata:
            (groupRecord.metadata as Record<string, unknown> | undefined) ?? {},
        }),
      );
    }
    sessions.push(session);
  }
  return sessions;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object") {
    return value as Record<string, unknown>;
  }
  return {};
}

function readAttrString(dataset: any, name: string): string[] {
  const attrs = (dataset as { attrs?: Record<string, any> }).attrs ?? {};
  return parseStringListAttr(attrs[name]);
}

/** Read every annotation table, in table order. */
function readAnnotations(
  file: any,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
  instances?: Array<Instance | PredictedInstance>,
): AnnotationTuples {
  const { rois, bboxes } = readRoisAndBboxes(
    file,
    videos,
    tracks,
    context,
    instances,
  );
  return {
    rois,
    bboxes,
    masks: readMasks(file, videos, tracks, context),
    centroids: readCentroids(file, videos, tracks),
    labelImages: readLabelImages(file, videos, tracks, context, instances),
  };
}

function readRoisAndBboxes(
  file: any,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
  instances?: Array<Instance | PredictedInstance>,
): { rois: Routed<ROI>[]; bboxes: Routed<BoundingBox>[] } {
  const { rois, migratedBboxes } = readRoisWithMigration(
    file,
    videos,
    tracks,
    context,
    instances,
  );
  let bboxes = readBboxes(file, tracks, context.labelsPath);
  if (bboxes.length === 0 && migratedBboxes.length > 0) {
    bboxes = migratedBboxes;
  }
  return { rois, bboxes };
}

function readRoisWithMigration(
  file: any,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
  instances?: Array<Instance | PredictedInstance>,
): {
  rois: Routed<ROI>[];
  migratedBboxes: Routed<BoundingBox>[];
} {
  const table = readTable(file, "rois", context.labelsPath);
  if (!table) return { rois: [], migratedBboxes: [] };
  const rows = table.rows;
  if (!(rows.annotation_type ?? []).length) {
    return { rois: [], migratedBboxes: [] };
  }

  const wkbDs = file.get(tableBufferPath("rois", table.columnar) as string);
  // A 2.x table without its geometry buffer has never been readable.
  if (!wkbDs && !table.columnar) return { rois: [], migratedBboxes: [] };

  // v3.0: columns of the table; v1.9+: string datasets; fallback to JSON attrs
  const roisDs = table.columnar ? null : file.get("rois");
  return buildRois(
    {
      rows,
      columnar: table.columnar,
      wkb: toBytes(wkbDs?.value),
      categories: table.columnar
        ? rows.category
        : readStringMetadata(file, "roi_categories", roisDs, "categories"),
      names: table.columnar
        ? rows.name
        : readStringMetadata(file, "roi_names", roisDs, "names"),
      sources: table.columnar
        ? rows.source
        : readStringMetadata(file, "roi_sources", roisDs, "sources"),
    },
    videos,
    tracks,
    context,
    instances,
  );
}

function readBboxes(
  file: any,
  tracks: Track[],
  labelsPath: string,
): Routed<BoundingBox>[] {
  const bboxesDs = file.get("bboxes");
  if (!bboxesDs) return [];
  // Python has written /bboxes as a group of columns since format 2.0; sleap-io.js
  // writes a flat matrix with sibling string datasets.
  const columnar = isH5Group(bboxesDs);
  const rows = columnar
    ? (normalizeColumnarTable(
        labelsPath,
        "bboxes",
        readH5GroupColumns(bboxesDs, "bboxes"),
      ) as Record<string, any[]>)
    : normalizeStructDataset(bboxesDs);
  if (!bboxRowCount(rows)) return [];

  // v2.0+ groups: columns; v1.9+: string datasets at root level; fallback to JSON
  // attrs on dataset
  return buildBboxes(
    {
      rows,
      categories: columnar
        ? rows.category
        : readStringMetadata(file, "bbox_categories", bboxesDs, "categories"),
      names: columnar
        ? rows.name
        : readStringMetadata(file, "bbox_names", bboxesDs, "names"),
      sources: columnar
        ? rows.source
        : readStringMetadata(file, "bbox_sources", bboxesDs, "sources"),
    },
    tracks,
  );
}

/**
 * Read string metadata from a root-level dataset or fall back to a JSON attribute.
 * v1.9+ writes string datasets; older files use JSON-encoded attrs.
 */
function readStringMetadata(
  file: any,
  datasetPath: string,
  dataset: any,
  attrName: string,
): string[] {
  const ds = file.get(datasetPath);
  if (ds) {
    // Try the "json" attribute on the string dataset first
    const jsonAttr = readAttrString(ds, "json");
    if (jsonAttr.length > 0) return jsonAttr;
    // Fall back to raw value
    const strings = stringsFromDataset(ds.value);
    if (strings) return strings;
  }
  return readAttrString(dataset, attrName);
}

/** Read a score-map index and the bytes it indexes into, if the file has them. */
function readScoreMaps(
  file: any,
  table: "masks/score_maps" | "label_images/score_maps",
  labelsPath: string,
): ScoreMapTable | null {
  const index = readTable(file, table, labelsPath);
  if (!index) return null;
  const dataDs = file.get(tableBufferPath(table, index.columnar) as string);
  if (!dataDs && !index.columnar) return null;
  return {
    rows: index.rows,
    columnar: index.columnar,
    data: toBytes(dataDs?.value),
  };
}

function readMasks(
  file: any,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
): Routed<SegmentationMask>[] {
  const table = readTable(file, "masks", context.labelsPath);
  if (!table) return [];
  const rows = table.rows;
  if (!(rows.height ?? []).length) return [];

  const rleDs = file.get(tableBufferPath("masks", table.columnar) as string);
  // A 2.x table without its RLE buffer has never been readable.
  if (!rleDs && !table.columnar) return [];

  // v3.0: columns of the table; v1.9+: string datasets at root level; fallback to
  // JSON attrs on mask dataset
  const masksDs = table.columnar ? null : file.get("masks");
  return buildMasks(
    {
      rows,
      columnar: table.columnar,
      rle: toBytes(rleDs?.value),
      categories: table.columnar
        ? rows.category
        : readStringMetadata(file, "mask_categories", masksDs, "categories"),
      names: table.columnar
        ? rows.name
        : readStringMetadata(file, "mask_names", masksDs, "names"),
      sources: table.columnar
        ? rows.source
        : readStringMetadata(file, "mask_sources", masksDs, "sources"),
      scoreMaps: readScoreMaps(file, "masks/score_maps", context.labelsPath),
    },
    videos,
    tracks,
    context,
  );
}

function readLabelImages(
  file: any,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
  instances?: Array<Instance | PredictedInstance>,
): Routed<LabelImage>[] {
  const { labelsPath } = context;
  const table = readTable(file, "label_images", labelsPath);
  if (!table) return [];
  const rows = table.rows;
  if (!(rows.video ?? []).length) return [];

  // v3.0: a column of the table; v1.9+: string datasets; fallback to JSON attrs
  const sources = table.columnar
    ? rows.source
    : readStringMetadata(
        file,
        "label_image_sources",
        file.get("label_images"),
        "sources",
      );

  const dataDs = file.get(
    tableBufferPath("label_images", table.columnar) as string,
  );
  const base = { rows, columnar: table.columnar, sources };
  if (!dataDs) {
    return buildLabelImages(
      { ...base, data: null, objects: null, scoreMaps: null },
      videos,
      tracks,
      context,
      instances,
    );
  }

  // Read objects table (may not exist if all label images have 0 objects)
  let objects: LabelImageObjectsTable | null = null;
  const objTable = readTable(file, "label_images/objects", labelsPath);
  if (objTable) {
    // v3.0: columns of the table; v1.9+: string datasets at root level
    const objDs = objTable.columnar ? null : file.get("label_image_objects");
    objects = {
      rows: objTable.rows,
      categories: objTable.columnar
        ? objTable.rows.category
        : readStringMetadata(
            file,
            "label_image_obj_categories",
            objDs,
            "categories",
          ),
      names: objTable.columnar
        ? objTable.rows.name
        : readStringMetadata(file, "label_image_obj_names", objDs, "names"),
    };
  }

  return buildLabelImages(
    {
      ...base,
      data: { value: dataDs.value, shape: dataDs.shape ?? [] },
      objects,
      scoreMaps: readScoreMaps(file, "label_images/score_maps", labelsPath),
    },
    videos,
    tracks,
    context,
    instances,
  );
}

function normalizeStructDataset(
  dataset: any,
  module?: unknown,
): Record<string, any[]> {
  if (!dataset) return {};
  checkDatasetFilters(dataset, String(dataset.path ?? "").replace(/^\//, ""));

  // Fast path: read fixed-size compound tables (frames/instances/points/
  // pred_points) column-wise straight from the record blob, skipping h5wasm's
  // per-row/per-member `Dataset.value` materialization (the dominant open cost
  // on large point tables). Returns null — and we fall through to `.value` —
  // for anything it can't read exactly. See ./h5-compound.ts.
  if (module) {
    const fast = readCompoundColumnsManual(module, dataset);
    if (fast) return fast as Record<string, any[]>;
  }

  const raw = dataset.value;
  if (!raw) return {};

  const fieldNames = getFieldNames(dataset);

  if (Array.isArray(raw) && raw.length > 0 && Array.isArray(raw[0])) {
    return mapStructuredRows(raw, fieldNames);
  }

  if (
    raw &&
    ArrayBuffer.isView(raw) &&
    Array.isArray(dataset.shape) &&
    dataset.shape.length === 2
  ) {
    const [rowCount, colCount] = dataset.shape as [number, number];
    const flat = raw as unknown as { readonly [i: number]: number };
    if (fieldNames.length) {
      // Fast path: extract each field's column DIRECTLY from the flat typed
      // array by stride, in a single sequential pass. The previous code built
      // an intermediate array-of-rows — one boxed `Array.from(slice)` per row
      // (millions of tiny allocations on a large points table) — then re-mapped
      // it column-by-column; that transpose-and-back dominated large-file loads
      // (~2.7s of an 11s open on a 0.9M-point file). Column j gets `undefined`
      // past colCount and columns past fieldNames are dropped, matching the old
      // `rows.map(row => row[idx])` semantics exactly.
      const cols: any[][] = fieldNames.map(() => new Array(rowCount));
      const fillCols = Math.min(fieldNames.length, colCount);
      for (let i = 0; i < rowCount; i += 1) {
        const base = i * colCount;
        for (let j = 0; j < fillCols; j += 1) {
          cols[j][i] = flat[base + j];
        }
      }
      const data: Record<string, any[]> = {};
      for (let j = 0; j < fieldNames.length; j += 1) {
        data[fieldNames[j]] = cols[j];
      }
      return data;
    }
    // No field names (malformed/legacy): preserve the row-index-keyed fallback.
    const rows: any[][] = [];
    for (let i = 0; i < rowCount; i += 1) {
      const start = i * colCount;
      rows.push(Array.from((raw as any).slice(start, start + colCount)));
    }
    return mapStructuredRows(rows, fieldNames);
  }

  if (raw && typeof raw === "object") {
    return raw;
  }

  return {};
}

function mapStructuredRows(
  rows: any[][],
  fieldNames: string[],
): Record<string, any[]> {
  if (!fieldNames.length) {
    return rows.reduce((acc: Record<string, any[]>, row: any[], idx) => {
      acc[String(idx)] = row;
      return acc;
    }, {});
  }
  const data: Record<string, any[]> = {};
  fieldNames.forEach((field, idx) => {
    data[field] = rows.map((row) => row[idx]);
  });
  return data;
}

function getFieldNames(dataset: any): string[] {
  const fields = dataset.dtype?.fields ? Object.keys(dataset.dtype.fields) : [];
  if (fields.length) return fields;
  const compoundMembers = dataset.metadata?.compound_type?.members;
  if (Array.isArray(compoundMembers) && compoundMembers.length) {
    const names = compoundMembers
      .map((member: { name?: string }) => member.name)
      .filter((name: string | undefined): name is string => !!name);
    if (names.length) return names;
  }
  const attr = dataset.attrs?.field_names ?? dataset.attrs?.fieldNames;
  if (!attr) return [];
  const value = attr.value ?? attr;
  if (Array.isArray(value)) return value.map((entry) => String(entry));
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) return parsed.map((entry) => String(entry));
    } catch {
      return value
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean);
    }
  }
  if (value instanceof Uint8Array) {
    try {
      const parsed = JSON.parse(textDecoder.decode(value));
      if (Array.isArray(parsed)) return parsed.map((entry) => String(entry));
    } catch {
      return [];
    }
  }
  return [];
}

function buildLabeledFrames(options: {
  framesData: Record<string, any[]>;
  instancesData: Record<string, any[]>;
  pointsData: Record<string, any[]>;
  predPointsData: Record<string, any[]>;
  skeletons: Skeleton[];
  tracks: Track[];
  videos: Video[];
  formatId: number;
  /** Filled with each instance at its `/instances` row (unknown types left out). */
  instancesByRow?: Array<Instance | PredictedInstance>;
}): LabeledFrame[] {
  const frames: LabeledFrame[] = [];
  const {
    framesData,
    instancesData,
    pointsData,
    predPointsData,
    skeletons,
    tracks,
    videos,
    formatId,
    instancesByRow,
  } = options;
  const frameIds = framesData.frame_id ?? [];
  const videoIdToIndex = buildVideoIdMap(framesData, videos);
  const instanceById = new Map<number, Instance | PredictedInstance>();
  const fromPredictedPairs: Array<[number, number]> = [];

  for (let frameIdx = 0; frameIdx < frameIds.length; frameIdx += 1) {
    const rawVideoId = Number(framesData.video?.[frameIdx] ?? 0);
    const videoIndex = videoIdToIndex.get(rawVideoId) ?? rawVideoId;
    const frameIndex = Number(framesData.frame_idx?.[frameIdx] ?? 0);
    const instStart = Number(framesData.instance_id_start?.[frameIdx] ?? 0);
    const instEnd = Number(framesData.instance_id_end?.[frameIdx] ?? 0);
    const video = videos[videoIndex];
    if (!video) continue;

    const instances: Array<Instance | PredictedInstance> = [];
    for (let instIdx = instStart; instIdx < instEnd; instIdx += 1) {
      const instanceType = Number(instancesData.instance_type?.[instIdx] ?? 0);
      // Rows of an unknown type are skipped, as in Python.
      if (instanceType !== 0 && instanceType !== 1) continue;
      const skeletonId = Number(instancesData.skeleton?.[instIdx] ?? 0);
      const trackId = Number(instancesData.track?.[instIdx] ?? -1);
      const pointStart = Number(instancesData.point_id_start?.[instIdx] ?? 0);
      const pointEnd = Number(instancesData.point_id_end?.[instIdx] ?? 0);
      const score = Number(instancesData.score?.[instIdx] ?? 0);
      const rawTrackingScore =
        formatId < 1.2
          ? 0
          : Number(instancesData.tracking_score?.[instIdx] ?? 0);
      const trackingScore = Number.isNaN(rawTrackingScore)
        ? 0
        : rawTrackingScore;
      const fromPredicted = Number(
        instancesData.from_predicted?.[instIdx] ?? -1,
      );
      const skeleton = skeletons[skeletonId] ?? skeletons[0];
      const track = trackId >= 0 ? tracks[trackId] : null;

      let instance: Instance | PredictedInstance;
      if (instanceType === 0) {
        // Build straight from the point columns — no intermediate Point[].
        instance = Instance._fromColumns({
          columns: pointsData,
          start: pointStart,
          end: pointEnd,
          skeleton,
          track,
          trackingScore,
        });
        if (formatId < 1.1) {
          instance.points.forEach((point) => {
            point.xy = [point.xy[0] - 0.5, point.xy[1] - 0.5];
          });
        }
        if (fromPredicted >= 0) {
          fromPredictedPairs.push([instIdx, fromPredicted]);
        }
      } else {
        instance = PredictedInstance._fromColumns({
          columns: predPointsData,
          start: pointStart,
          end: pointEnd,
          skeleton,
          track,
          score,
          trackingScore,
        });
        if (formatId < 1.1) {
          instance.points.forEach((point) => {
            point.xy = [point.xy[0] - 0.5, point.xy[1] - 0.5];
          });
        }
      }

      instanceById.set(instIdx, instance);
      if (instancesByRow) instancesByRow[instIdx] = instance;
      instances.push(instance);
    }

    frames.push(new LabeledFrame({ video, frameIdx: frameIndex, instances }));
  }

  for (const [instanceId, fromPredictedId] of fromPredictedPairs) {
    const instance = instanceById.get(instanceId);
    const predicted = instanceById.get(fromPredictedId);
    if (
      instance &&
      predicted instanceof PredictedInstance &&
      instance instanceof Instance
    ) {
      instance.fromPredicted = predicted;
    }
  }

  return frames;
}

function readCentroids(
  file: any,
  _videos: Video[],
  tracks: Track[],
): [Centroid, number, number][] {
  // Two on-disk layouts are supported:
  //  - Python / current JS: a `/centroids` GROUP with one child dataset per
  //    field (x, y, z, video, frame_idx, track, instance, is_predicted, score,
  //    tracking_score) plus string children (category/name/source).
  //  - Legacy JS: a flat `/centroids` matrix dataset + `field_names` attr, with
  //    sibling `centroid_categories`/`centroid_names`/`centroid_sources`.
  let data: Record<string, ArrayLike<number>>;
  let categories: string[];
  let names: string[];
  let sources: string[];

  const centroidsDs = file.get("centroids");
  if (!centroidsDs) return [];
  // Go by the node itself: h5wasm returns a truthy Group for a path through a dataset,
  // so asking for `centroids/x` can't tell the layouts apart.
  if (isH5Group(centroidsDs)) {
    // Group layout (matches Python sleap-io).
    const col = (name: string): ArrayLike<number> =>
      (file.get(`centroids/${name}`)?.value as ArrayLike<number>) ?? [];
    data = {
      x: col("x"),
      y: col("y"),
      z: col("z"),
      video: col("video"),
      frame_idx: col("frame_idx"),
      track: col("track"),
      instance: col("instance"),
      is_predicted: col("is_predicted"),
      score: col("score"),
      tracking_score: col("tracking_score"),
    };
    const strCol = (name: string): string[] => {
      const ds = file.get(`centroids/${name}`);
      if (!ds) return [];
      try {
        return ((ds.value as ArrayLike<unknown>) ?? []).length
          ? Array.from(ds.value as ArrayLike<unknown>, decodeStr)
          : [];
      } catch {
        // Python may store these as vlen-str, which h5wasm can't always read;
        // the centroid position/type/link still round-trips without them.
        return [];
      }
    };
    categories = strCol("category");
    names = strCol("name");
    sources = strCol("source");
  } else {
    data = normalizeStructDataset(centroidsDs);
    categories = readStringMetadata(
      file,
      "centroid_categories",
      centroidsDs,
      "categories",
    );
    names = readStringMetadata(file, "centroid_names", centroidsDs, "names");
    sources = readStringMetadata(
      file,
      "centroid_sources",
      centroidsDs,
      "sources",
    );
  }

  return buildCentroidTuples(data, categories, names, sources, tracks);
}
