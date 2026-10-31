/**
 * Streaming SLP file reader using HTTP range requests.
 *
 * This module provides a streaming alternative to `readSlp` that uses
 * `StreamingH5File` for efficient range request-based file access.
 * Only the data actually needed is downloaded, rather than the entire file.
 *
 * @module
 */

import {
  hasPath,
  openH5Worker,
  type StreamingH5File,
  isStreamingSupported,
  type StreamingH5Source,
} from "./h5-streaming.js";
import {
  addAnnotationsToStore,
  type AnnotationTuples,
  attachAnnotationOwners,
  attachOwnerType,
  bboxRowCount,
  buildBboxes,
  buildLabelImages,
  buildMasks,
  buildRois,
  distributeAnnotations,
  groupEmbeddings,
  groupLinks,
  type LabelImageObjectsTable,
  type LabelImagePixels,
  parseStringListAttr,
  type PerDetectionMaps,
  type Routed,
  type Rows,
  type ScoreMapTable,
  stringsFromDataset,
  type TableContext,
  toBytes,
} from "./annotations.js";
import {
  checkDatasetFilters,
  type ColumnarTableName,
  type Columns,
  indexOutOfRange,
  legacyTablePath,
  normalizeColumnarTable,
  normalizePoseTables,
  type RawTable,
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
  decodeEntryText,
  sessionsReadError,
  reconstructColumnarFrameGroups,
  type SessionData,
  type SessionPointMatrix,
  parseSkeletons,
  parseTracks,
  parseVideosMetadata,
  parseSuggestions,
  resolveCameraKey,
  normalizeCameraSize,
  reconstructInstance3D,
  resolveIdentity,
} from "./parsers.js";
import { buildSourceVideoFromDict } from "./source-video.js";
import { Labels } from "../../model/labels.js";
import { LabeledFrame } from "../../model/labeled-frame.js";
import type { BoundingBox } from "../../model/bbox.js";
import type { Centroid } from "../../model/centroid.js";
import type { LabelImage } from "../../model/label-image.js";
import type { SegmentationMask } from "../../model/mask.js";
import type { ROI } from "../../model/roi.js";
import { buildCentroidTuples } from "./centroids.js";
import { LazyDataStore, LazyFrameList } from "../../model/lazy.js";
import { buildVideoIdMap } from "../../model/video-id-map.js";
import {
  markNegativeFrames,
  negativeFrameKeys,
} from "../../model/negative-frames.js";
import {
  Instance,
  PredictedInstance,
  type Track,
  type PointColumns,
} from "../../model/instance.js";
import { Skeleton } from "../../model/skeleton.js";
import { SuggestionFrame } from "../../model/suggestions.js";
import { Video } from "../../model/video.js";
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
import { StreamingHdf5VideoBackend } from "../../video/streaming-hdf5-video.js";
import { CropVideoBackend } from "../../video/crop-backend.js";
import { resolveSourceFrameCount } from "./frame-count.js";
import type { CropRect } from "../../transform/points.js";
import type { Fill } from "../../transform/frame.js";

/**
 * Options for streaming SLP file loading.
 */
export interface StreamingSlpOptions {
  /** URL hint for h5wasm CDN */
  h5wasmUrl?: string;
  /** Filename hint for the HDF5 file */
  filenameHint?: string;
  /**
   * Extra HTTP request headers (e.g. `{ Authorization: "Bearer …" }`) forwarded
   * to the streaming worker. When non-empty, the worker downloads the file in a
   * buffer (authenticated) rather than using header-free range streaming.
   */
  headers?: Record<string, string>;
  /** Whether to open video backends for embedded videos (default: false) */
  openVideos?: boolean;
  /**
   * Capture the verbatim, deep-cloned `sessions_json` dict onto each
   * `RecordingSession.rawJson` (deprecated, transitional). Default `false`. See
   * `RecordingSession.rawJson`.
   */
  rawSessions?: boolean;
  /**
   * Defer frame materialization. When `true`, the pose tables are wrapped in a
   * `LazyDataStore`/`LazyFrameList` and individual `LabeledFrame`/`Instance`
   * objects are built only on first access (via `labels.frameAt(i)` /
   * `labels.materialize()`), instead of eagerly building the full object graph.
   * Bounds peak memory for very large prediction files. Default `false`.
   */
  lazy?: boolean;
  /**
   * Defer per-video metadata reads. When `true`, embedded videos are built from
   * `videos_json` ALONE — skipping the per-video HDF5 reads (dataset lookup,
   * `getAttrs`, `frame_numbers`, `frame_sizes`, `source_video`) that dominate
   * open time on many-video `pkg.slp` files read over high-latency storage
   * (hundreds of videos × several serial reads each). Shape/channel_order come
   * from JSON (with the `format_id` default fallback), so the videos panel still
   * shows dimensions; backends are NOT built — the caller opens them on demand
   * (e.g. on first view). Default `false`. Independent of `openVideos`; when
   * set, embedded backends are always deferred. Frames render blank until the
   * caller builds a backend on demand.
   */
  lazyVideoMetadata?: boolean;
  /**
   * Optional progress callback fired as loading advances through its stages.
   * `current` counts completed stages out of `total`; `message` labels the
   * stage about to run. Matches the (current, total, message?) convention used
   * elsewhere in the library (Labels.merge, RenderOptions.onProgress).
   */
  onProgress?: (current: number, total: number, message?: string) => void;
}

/**
 * Read an SLP file using a Web Worker for efficient, non-blocking HDF5 access.
 *
 * This function offloads all h5wasm operations to a Web Worker, keeping the
 * main thread responsive. For URLs, it uses HTTP range requests to download
 * only the data needed rather than the entire file.
 *
 * When `openVideos` is true, video backends are created for embedded videos,
 * allowing frame data to be retrieved. The underlying HDF5 file remains open
 * until all video backends are closed.
 *
 * @param source - URL, File, ArrayBuffer, or Uint8Array containing the SLP file
 * @param options - Optional settings
 * @returns Labels object with all annotation data
 *
 * @example
 * ```typescript
 * // Load from URL with video backends
 * const labels = await readSlpStreaming('https://example.com/labels.slp', {
 *   openVideos: true
 * });
 *
 * // Load from File object (file input)
 * const labels = await readSlpStreaming(fileInput.files[0], {
 *   openVideos: true
 * });
 *
 * // Load from ArrayBuffer
 * const labels = await readSlpStreaming(arrayBuffer, {
 *   filenameHint: 'data.slp'
 * });
 * ```
 */
export async function readSlpStreaming(
  source: StreamingH5Source,
  options?: StreamingSlpOptions,
): Promise<Labels> {
  if (!isStreamingSupported()) {
    throw new Error(
      "Streaming HDF5 requires Web Worker support (browser environment)",
    );
  }

  const file = await openH5Worker(source, {
    h5wasmUrl: options?.h5wasmUrl,
    filenameHint: options?.filenameHint,
    headers: options?.headers,
  });

  const openVideos = options?.openVideos ?? false;
  const lazyVideoMetadata = options?.lazyVideoMetadata ?? false;

  // Determine the source path for video resolution
  const sourcePath =
    typeof source === "string"
      ? source
      : typeof File !== "undefined" && source instanceof File
        ? source.name
        : (options?.filenameHint ?? "slp-data.slp");

  let labels: Labels;
  try {
    labels = await readFromStreamingFile(
      file,
      sourcePath,
      options?.filenameHint,
      openVideos,
      options?.onProgress,
      options?.rawSessions ?? false,
      options?.lazy ?? false,
      lazyVideoMetadata,
    );
  } catch (e) {
    // Nothing will read this file now, so end its worker, whatever the options.
    // Don't wait for that or let it fail: the caller needs this error, and a stuck
    // worker never answers.
    file.close().catch(() => {});
    throw e;
  }
  // Keep the file open when video backends need it: eager (openVideos) OR
  // deferred (lazyVideoMetadata), whose per-video backends read frame data on
  // demand from this same StreamingH5File. Only close for pure metadata reads.
  if (!openVideos && !lazyVideoMetadata) {
    await file.close();
  }
  return labels;
}

/**
 * Read Labels from an already-opened StreamingH5File.
 *
 * Exported for direct testing of the streaming metadata-error path: the full
 * streaming reader (`readSlpStreaming`) is browser/Worker-gated and unreachable
 * from the all-Node test suite, so the error-routing here is exercised by
 * driving this function with a minimal `getAttrs`-providing stub.
 */
export async function readFromStreamingFile(
  file: StreamingH5File,
  url: string,
  filenameHint?: string,
  openVideos: boolean = false,
  onProgress?: (current: number, total: number, message?: string) => void,
  rawSessions: boolean = false,
  lazy: boolean = false,
  lazyVideoMetadata: boolean = false,
): Promise<Labels> {
  // Stage-level progress. The orchestration in this function runs on the MAIN
  // thread — only the low-level HDF5 byte reads are delegated to the worker via
  // `file` — so each phase can be surfaced as it starts without any cross-worker
  // messaging. `current` counts completed stages; `message` labels the stage
  // about to run. (Shape matches Labels.merge / RenderOptions.onProgress.)
  //
  // `total` is DERIVED from this ordered label list (single source of truth) so
  // the count can never drift from the stages. report(i) fires STAGES[i].
  const STAGES = [
    "Reading metadata", // 0
    "Reading tracks", // 1
    "Reading video metadata", // 2 (also holds the bar for "Opening videos (i/n)")
    "Reading suggestions", // 3
    "Reading frames", // 4
    "Reading instances", // 5
    "Reading points", // 6
    "Building labeled frames", // 7
    "Reading identities", // 8
    "Reading sessions", // 9
    "Reading annotations", // 10
  ] as const;
  const total = STAGES.length;
  const report = (n: number, message?: string) =>
    onProgress?.(n, total, message ?? STAGES[n]);

  report(0);
  const labelsPath =
    filenameHint ?? url.split("/").pop()?.split("?")[0] ?? "slp-data.slp";
  // A missing `metadata` group surfaces from the worker as a thrown
  // "Path not found: metadata"; treat that the same as a missing `json`
  // attribute (both indicate a truncated/corrupt file) and route it through
  // the same helpful error. Mirrors Python sleap-io PR #446, where
  // `read_metadata` catches the `KeyError` from BOTH cases and maps them to
  // the same `ValueError`, and matches the eager/lazy readers in read.ts.
  let metadataAttrs: Record<string, unknown>;
  try {
    metadataAttrs = await file.getAttrs("metadata");
  } catch (err) {
    // A failed read says nothing about the file, so don't call it corrupt.
    if (!String((err as Error)?.message).startsWith("Path not found"))
      throw err;
    throw missingMetadataJsonError(labelsPath);
  }
  const formatId = checkSlpFormat(metadataAttrs, labelsPath);
  // Throws the same helpful error if the `metadata` group exists but its
  // required `json` attribute is missing/empty (truncated/corrupt file);
  // mirrors Python sleap-io PR #446 and matches the eager/lazy readers.
  const metadataJson = parseMetadataJson(
    metadataAttrs["json"],
    labelsPath,
  ) as Record<string, unknown> | null;

  const skeletons = parseSkeletons(metadataJson);

  report(1);
  const tracks = await readTracksStreaming(file);

  // Read per-video crop records (SLP 2.3; empty on old/uncropped files).
  report(2);
  const videoCrops = await readVideoCropsStreaming(file);

  // Read video metadata (and optionally create backends for embedded videos).
  // When opening videos, the per-video reporter keeps the bar at this stage
  // while the label counts videos (probeShape per embedded backend is slow).
  const videos = await readVideosStreaming(
    file,
    labelsPath,
    openVideos,
    formatId,
    videoCrops,
    openVideos ? (i, n) => report(2, `Opening videos (${i}/${n})`) : undefined,
    lazyVideoMetadata,
  );

  report(3);
  const suggestions = await readSuggestionsStreaming(file, videos);

  // Read frame/instance/point data
  report(4);
  const framesRaw = await readRawTableStreaming(file, "frames");
  report(5);
  const instancesRaw = await readRawTableStreaming(file, "instances");
  report(6);
  const pointsRaw = await readRawTableStreaming(file, "points");
  const predPointsRaw = await readRawTableStreaming(file, "pred_points");
  const pose = normalizePoseTables(labelsPath, videos.length, {
    frames: framesRaw,
    instances: instancesRaw,
    points: pointsRaw,
    predPoints: predPointsRaw,
  });
  const { framesData, instancesData, pointsData, predPointsData } = pose;
  const negativeFrames =
    pose.negativeFrames ??
    negativeFrameKeys(
      file.keys().includes("negative_frames")
        ? await readStructDatasetStrict(file, "negative_frames")
        : {},
    );

  // Build labeled frames eagerly, or (lazy mode) wrap the columnar tables in a
  // LazyDataStore and defer LabeledFrame/Instance construction to first access.
  report(7);
  const lazyStore = lazy
    ? new LazyDataStore({
        framesData,
        instancesData,
        pointsData,
        predPointsData,
        skeletons,
        tracks,
        videos,
        formatId,
        negativeFrames,
      })
    : undefined;
  const lazyFrames = lazyStore ? new LazyFrameList(lazyStore) : undefined;
  // One entry per /instances row, so links to instances index it by row.
  const allInstances: Array<Instance | PredictedInstance> = [];
  const labeledFrames = lazyStore
    ? undefined
    : buildLabeledFrames({
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
  if (labeledFrames) {
    markNegativeFrames(labeledFrames, negativeFrames, framesData, videos);
  }

  // Read identities
  report(8);
  const identities = await readIdentitiesStreaming(file);

  // Read sessions — grouping captured as index refs (no frame materialization).
  report(9);
  const sessions = await readSessionsStreaming(
    file,
    videos,
    skeletons,
    identities,
    rawSessions,
    { labelsPath, nFrames: (framesData.frame_id ?? []).length },
  );

  report(10);
  const annotations = await readAnnotationsStreaming(
    file,
    videos,
    tracks,
    {
      labelsPath,
      nInstances: (instancesData.instance_type ?? []).length,
    },
    labeledFrames ? allInstances : undefined,
  );

  // Per-detection identity (SLP 2.5) + category (SLP 2.7): attach onto each modality's
  // ordered detection list (index == owner_id). Lazy instances don't exist yet, so
  // their maps + catalogs go to the store and attach in materializeFrame.
  const pdMaps = await readPerDetectionMapsStreaming(
    file,
    labelsPath,
    identities.length,
  );
  attachAnnotationOwners(pdMaps, annotations, identities);
  let staticRois: ROI[] = [];
  if (labeledFrames) {
    attachOwnerType(pdMaps, 0, allInstances, identities);
    staticRois = distributeAnnotations(
      labeledFrames,
      videos,
      annotations,
      allInstances,
    );
  }
  if (lazyStore && lazyFrames) {
    lazyStore.identities = identities;
    lazyStore._instanceIdentityLinks = pdMaps.idLinks.get(0);
    lazyStore._instanceEmbeddings = pdMaps.idEmbs.get(0);
    lazyStore.categoryCatalog = pdMaps.categoryCatalog;
    lazyStore._instanceCategoryLinks = pdMaps.catLinks.get(0);
    lazyStore._instanceCategoryEmbeddings = pdMaps.catEmbs.get(0);
    addAnnotationsToStore(lazyStore, lazyFrames, annotations);
  }

  onProgress?.(total, total, "Finalizing");
  // In lazy mode omit `labeledFrames` entirely (mirrors readSlpLazy): the Labels
  // ctor's instance-track registration is guarded by `!_lazyFrameList`, and the
  // proxy is attached below — so tracks come only from the explicit `tracks`.
  const labels = new Labels({
    ...(labeledFrames ? { labeledFrames } : {}),
    videos,
    skeletons,
    tracks,
    suggestions,
    sessions,
    identities,
    provenance: parseProvenance(
      await readProvenanceStreaming(file),
      metadataJson,
    ),
    rois: staticRois,
  });
  if (lazyStore && lazyFrames) {
    // Attach the lazy proxies BEFORE injectSessionFrameResolver so the resolver's
    // frameAt() resolves against the lazy list (a single frame on demand, never
    // the full table). Note: LazyDataStore.materializeFrame indexes videos by the
    // raw video id (no buildVideoIdMap remap) — this matches readSlpLazy and only
    // differs from the eager streaming build for rare sparse/non-contiguous ids.
    labels._lazyFrameList = lazyFrames;
    labels._lazyDataStore = lazyStore;
  }
  // Sessions are read before the frame store exists; wire the lazy frame
  // resolver now so ref-backed grouping resolves against labels.labeledFrames
  // (or, in lazy mode, the attached _lazyFrameList via frameAt).
  injectSessionFrameResolver(labels);
  return labels;
}

/**
 * Read tracks from tracks_json dataset.
 */
async function readTracksStreaming(file: StreamingH5File): Promise<Track[]> {
  if (!file.keys().includes("tracks_json")) return [];

  const data = await file.getDatasetValue("tracks_json");
  const values = normalizeDatasetArray(data.value);
  return parseTracks(values);
}

/** A single `/video_crops` entry: the crop rect and OOB fill for one video. */
interface VideoCropEntry {
  crop: CropRect;
  fill: Fill;
}

/**
 * Read the top-level `/video_crops` dataset (SLP format 2.3) into a map keyed by
 * video index. Streaming mirror of `readVideoCrops` (read.ts). Absent on old/
 * uncropped files, in which case an empty map is returned.
 *
 * Exported for testing the h5wasm payload-form normalization (string, length-1
 * array, raw bytes); not part of the public API.
 */
export async function readVideoCropsStreaming(
  file: StreamingH5File,
): Promise<Map<number, VideoCropEntry>> {
  const out = new Map<number, VideoCropEntry>();
  if (!file.keys().includes("video_crops")) return out;

  const data = await file.getDatasetValue("video_crops");
  try {
    let raw: unknown = data.value;
    // h5wasm surfaces the vlen-string `/video_crops` payload as: (a) a plain
    // string, (b) a length-1 array of strings (the vlen form), or (c) raw
    // bytes (Uint8Array). Mirror the non-streaming readVideoCrops logic exactly:
    // unwrap the length-1 array, then decode raw bytes directly. Do NOT flatten
    // a Uint8Array via Array.from (that would extract a single byte value).
    if (Array.isArray(raw)) raw = raw[0];
    let json: string;
    if (typeof raw === "string") {
      json = raw;
    } else if (raw instanceof Uint8Array) {
      json = new TextDecoder().decode(raw);
    } else if (raw != null) {
      json = String(raw);
    } else {
      return out;
    }

    const parsed = JSON.parse(json);
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
  } catch {
    return out;
  }
}

/**
 * Read a video's `source_video` metadata JSON from its `{group}/source_video`
 * HDF5 group over the streaming worker, or `null` when absent. Checks a `json`
 * *dataset* first (oversized metadata) then the `json` *attribute* (normal
 * case), mirroring the sync reader and Python `_read_source_video_json`.
 */
export async function readSourceVideoGroupJsonStreaming(
  file: Pick<
    StreamingH5File,
    "keys" | "getKeys" | "getAttrs" | "getDatasetValue"
  >,
  groupPath: string,
): Promise<Record<string, unknown> | null> {
  const svPath = `${groupPath}/source_video`;
  if (!(await hasPath(file, svPath))) return null;
  let raw: string | undefined;
  let failedRead: unknown;
  if ((await file.getKeys(svPath)).includes("json")) {
    // Oversized (spilled) metadata: a scalar `|S<n>` dataset. Decode every
    // backend representation through the shared helper (h5wasm string,
    // jsfive length-1 array, raw bytes) rather than the partial inline checks.
    // Scope the guard tightly (mirrors the sync reader) so a dataset-read
    // failure falls back to the `json` attribute instead of dropping the
    // lineage.
    try {
      const { value } = await file.getDatasetValue(`${svPath}/json`);
      raw = datasetValueToString(value);
    } catch (err) {
      failedRead = err;
    }
  }
  if (raw === undefined) {
    const attrs = await file.getAttrs(svPath);
    raw = attrToString(attrs.json);
  }
  if (raw === undefined) {
    // With no attribute to fall back on, the next save would drop the lineage.
    if (failedRead !== undefined) throw failedRead;
    return null;
  }
  try {
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
}

/**
 * Read video metadata from videos_json dataset.
 * When openVideos is true, creates StreamingHdf5VideoBackend for embedded videos.
 */
async function readVideosStreaming(
  file: StreamingH5File,
  labelsPath: string,
  openVideos: boolean = false,
  formatId: number = 1.0,
  videoCrops?: Map<number, VideoCropEntry>,
  onVideoProgress?: (current: number, total: number) => void,
  lazyVideoMetadata: boolean = false,
): Promise<Video[]> {
  if (!file.keys().includes("videos_json")) return [];

  const data = await file.getDatasetValue("videos_json");
  const values = normalizeDatasetArray(data.value);
  const metadataList = parseVideosMetadata(values, labelsPath);

  const videos: Video[] = [];

  for (let videoIndex = 0; videoIndex < metadataList.length; videoIndex++) {
    onVideoProgress?.(videoIndex + 1, metadataList.length);
    const meta = metadataList[videoIndex];

    // Fast path: build the embedded Video from videos_json ALONE, skipping
    // every per-video HDF5 read (dataset lookup, getAttrs, frame_numbers,
    // frame_sizes, source_video). Those serial reads dominate open time on
    // many-video pkg.slp over high-latency storage. No backend is built here —
    // the caller opens one on demand (frames render blank until then). Shape
    // and channel_order come from JSON (with the format_id default fallback),
    // so the videos panel still reports dimensions.
    if (lazyVideoMetadata && meta.embedded) {
      const datasetPath = meta.dataset ?? `video${videoIndex}/video`;
      // Seed the source frame count from videos_json (the embedded/labeled
      // count); the deferred backend corrects it to the true source count from
      // frame_numbers/`frames` attr on first view.
      const frameCount = resolveSourceFrameCount({
        jsonFrameCount: meta.frameCount,
      });
      const shape: [number, number, number, number] | undefined =
        meta.height && meta.width && meta.channels
          ? [frameCount ?? 0, meta.height, meta.width, meta.channels]
          : undefined;
      const channelOrder =
        meta.channelOrder ?? (formatId < 1.4 ? "BGR" : "RGB");
      // Deferred backend: no per-video HDF5 reads now; getFrame() lazily reads
      // frame_numbers/frame_sizes/attrs on first view (ensureLoaded). Holds the
      // still-open StreamingH5File, so readSlpStreaming must NOT close it.
      const backend = new StreamingHdf5VideoBackend({
        filename: Array.isArray(meta.filename)
          ? (meta.filename[0] ?? "")
          : meta.filename,
        h5file: file,
        datasetPath,
        format: meta.format ?? "png",
        channelOrder,
        shape,
        fps: meta.fps,
        deferred: true,
      });
      const sourceVideo = meta.sourceVideo
        ? buildSourceVideoFromDict(meta.sourceVideo, labelsPath)
        : null;
      videos.push(
        new Video({
          filename: meta.filename,
          backend,
          backendMetadata: {
            dataset: datasetPath,
            format: meta.format,
            shape,
            fps: meta.fps,
            channel_order: channelOrder,
            grayscale: meta.grayscale,
          },
          sourceVideo,
          openBackend: false,
          embedded: meta.embedded,
        }),
      );
      continue;
    }

    // Auto-detect dataset path when embedded but not specified in metadata
    let datasetPath: string | undefined = meta.dataset;
    if (meta.embedded && !datasetPath) {
      datasetPath =
        (await findVideoDatasetStreaming(file, videoIndex)) ?? undefined;
    }

    // Read format/channel_order/frames from HDF5 dataset attributes when a
    // dataset path is known. Always probe even when openVideos is false —
    // downstream consumers (e.g. write.ts re-embed at line 1054) read
    // channel_order from Video.backendMetadata to avoid color corruption on
    // save, so metadata-only loads must populate it too. The `frames` attribute
    // records the source video's total frame count for embedded videos, used
    // by the seekbar. Matches Python sleap-io (video_reading.py:987-1006).
    let format = meta.format;
    let channelOrderFromAttrs: string | undefined;
    let frameCountFromAttrs: number | undefined;
    // An external video's dataset lives in its own file, not this one.
    if (datasetPath && (await hasPath(file, datasetPath))) {
      const attrs = await file.getAttrs(datasetPath);
      if (!format) {
        format = attrToString(attrs.format);
      }
      channelOrderFromAttrs = attrToString(attrs.channel_order);
      const framesNum = attrToNumber(attrs.frames);
      if (framesNum !== undefined && framesNum > 0) {
        frameCountFromAttrs = framesNum;
      }
      // Read height/width/channels from dataset attributes when JSON metadata
      // is missing (common for pkg.slp files)
      const readNumAttr = (attr: unknown): number | undefined => {
        if (attr === undefined || attr === null) return undefined;
        const v =
          typeof attr === "object" && attr !== null && "value" in attr
            ? (attr as { value: unknown }).value
            : attr;
        const n = Number(v);
        return Number.isFinite(n) && n > 0 ? n : undefined;
      };
      if (!meta.height) meta.height = readNumAttr(attrs.height);
      if (!meta.width) meta.width = readNumAttr(attrs.width);
      if (!meta.channels) meta.channels = readNumAttr(attrs.channels);
    }

    // For embedded videos we read frame_numbers up front so the source frame
    // count (the seekbar extent) can be resolved synchronously — see below.
    // frame_sizes lets the 1D-concatenated layout (JS writer) byte-range
    // slice exactly, mirroring the sync reader (read.ts) — see issue #135.
    let frameNumbers: number[] = [];
    let frameSizes: number[] | undefined;
    if (openVideos && meta.embedded && datasetPath) {
      frameNumbers = await readFrameNumbersStreaming(file, datasetPath);
      frameSizes = await readFrameSizesStreaming(file, datasetPath);
    }

    // Source frame count: prefer the `frames` attr, then the videos_json
    // count, then max(frame_numbers)+1. The last fallback keeps multi-video
    // pkg.slp files (written without a `frames` attr, e.g. older PyQt SLEAP)
    // resolving a shape at read time instead of relying on an async per-video
    // image-decode probe that races the UI (reporting 0 / "?" / wrong counts).
    const frameCount = resolveSourceFrameCount({
      framesAttr: frameCountFromAttrs,
      jsonFrameCount: meta.frameCount,
      frameNumbers,
    });
    const shape: [number, number, number, number] | undefined =
      meta.height && meta.width && meta.channels
        ? [frameCount ?? 0, meta.height, meta.width, meta.channels]
        : undefined;

    // Determine channel order with priority:
    // 1. JSON metadata (meta.channelOrder)
    // 2. HDF5 dataset attribute (channelOrderFromAttrs)
    // 3. Legacy fallback based on format_id (BGR for < 1.4)
    const channelOrder =
      meta.channelOrder ??
      channelOrderFromAttrs ??
      (formatId < 1.4 ? "BGR" : "RGB");

    // Create streaming backend for embedded videos when openVideos is true
    let backend = null;
    if (openVideos && meta.embedded && datasetPath) {
      backend = new StreamingHdf5VideoBackend({
        // Embedded videos always carry a single string filename (the labels
        // path); the array form is only for image sequences, which never
        // reach this embedded branch. Narrow for the type checker.
        filename: Array.isArray(meta.filename)
          ? (meta.filename[0] ?? "")
          : meta.filename,
        h5file: file,
        datasetPath,
        frameNumbers,
        frameSizes,
        format: format ?? "png",
        channelOrder,
        shape,
        fps: meta.fps,
      });

      // Only probe (decode one image) when height/width are still unknown —
      // i.e. neither the dataset attrs nor videos_json provided them. With a
      // resolved shape this is skipped, so the common pkg.slp path no longer
      // pays a per-video network decode.
      if (!shape || shape[0] === 0) {
        await backend.probeShape(frameCount ?? undefined);
      }
    }

    let videoBackend: typeof backend | CropVideoBackend = backend;
    const backendMetadata: Record<string, unknown> = {
      dataset: datasetPath,
      format,
      shape,
      fps: meta.fps,
      channel_order: channelOrder,
      grayscale: meta.grayscale,
    };

    // Crop reconstruction (SLP 2.3): wrap the uncropped inner backend and seed
    // crop/source_shape/shape/crop_fill so streaming reads report the cropped
    // view identically to the eager reader. Mirrors read.ts readVideos.
    const cropEntry = videoCrops?.get(videoIndex);
    if (cropEntry) {
      const [cx1, cy1, cx2, cy2] = cropEntry.crop;
      if (openVideos && videoBackend) {
        // Each reloaded tile owns its freshly-reconstructed inner (ownsInner
        // defaults to true). We do NOT share one inner across sibling mosaic
        // tiles here: on read each video entry rebuilds a private inner, so
        // sharing would leave the inner unowned (leaked — never closed) for no
        // decode savings. The streaming inner already shares the single
        // StreamingH5File worker across all backends and never closes it
        // (streaming-hdf5-video.ts), so the per-tile wrapper is cheap and its
        // ownsInner=true is harmless (close() only clears the wrapper's tiny
        // legacy cache; it cannot tear down the shared worker file). Live
        // mosaic decode-sharing is opt-in and in-memory only, via
        // Video.crop({ shareDecode: true }). Mirrors Python slp.py make_video
        // (slp.py:393-401).
        videoBackend = CropVideoBackend.wrap({
          inner: videoBackend,
          crop: cropEntry.crop,
          fill: cropEntry.fill,
          // ownsInner: true (default) — see comment above.
        });
      }
      if (shape && shape.length === 4) {
        backendMetadata.source_shape = [...shape];
        backendMetadata.shape = [shape[0], cy2 - cy1, cx2 - cx1, shape[3]];
      }
      backendMetadata.crop = [...cropEntry.crop];
      backendMetadata.crop_fill = cropEntry.fill;
    }

    // Reconstruct the source_video lineage WITH its recorded shape (and any
    // deeper chain), preferring the authoritative `{group}/source_video` HDF5
    // group for embedded videos and falling back to the nested videos_json
    // dict — parity with the eager reader (read.ts) for #160.
    let sourceVideo: Video | null = null;
    if (meta.embedded && datasetPath) {
      const groupPath = datasetPath.endsWith("/video")
        ? datasetPath.slice(0, -6)
        : datasetPath;
      const svDict = await readSourceVideoGroupJsonStreaming(file, groupPath);
      if (svDict) sourceVideo = buildSourceVideoFromDict(svDict, labelsPath);
    }
    if (!sourceVideo && meta.sourceVideo) {
      sourceVideo = buildSourceVideoFromDict(meta.sourceVideo, labelsPath);
    }

    videos.push(
      new Video({
        filename: meta.filename,
        backend: videoBackend,
        backendMetadata,
        sourceVideo,
        openBackend: openVideos && meta.embedded,
        embedded: meta.embedded,
      }),
    );
  }

  return videos;
}

/**
 * Read frame_numbers dataset for a video.
 * Returns the mapping from frame indices to storage indices.
 */
async function readFrameNumbersStreaming(
  file: StreamingH5File,
  datasetPath: string,
): Promise<number[]> {
  // Extract group path from dataset path (e.g., "video0/video" -> "video0")
  const groupPath = datasetPath.endsWith("/video")
    ? datasetPath.slice(0, -6)
    : datasetPath;

  const frameNumbersPath = `${groupPath}/frame_numbers`;

  // Check if dataset exists
  if (!(await hasPath(file, frameNumbersPath))) {
    return [];
  }

  const data = await file.getDatasetValue(frameNumbersPath);
  const values = data.value;

  // Convert to number array
  if (Array.isArray(values)) {
    return values.map((v: unknown) => Number(v));
  }
  if (ArrayBuffer.isView(values)) {
    return Array.from(values as unknown as ArrayLike<number>).map(Number);
  }
  return [];
}

/**
 * Read the frame_sizes dataset for a video, if present.
 *
 * frame_sizes records the encoded byte length of each embedded frame and is
 * written by the JS writer (1D concatenated layout). When available it lets the
 * backend byte-range slice an exact frame instead of scanning. Returns undefined
 * when the dataset is absent (e.g. Python-written 2D padded files).
 */
async function readFrameSizesStreaming(
  file: StreamingH5File,
  datasetPath: string,
): Promise<number[] | undefined> {
  const groupPath = datasetPath.endsWith("/video")
    ? datasetPath.slice(0, -6)
    : datasetPath;

  if (!(await hasPath(file, `${groupPath}/frame_sizes`))) {
    return undefined;
  }

  const data = await file.getDatasetValue(`${groupPath}/frame_sizes`);
  const values = data.value;

  if (Array.isArray(values)) {
    return values.map((v: unknown) => Number(v));
  }
  if (ArrayBuffer.isView(values)) {
    return Array.from(values as unknown as ArrayLike<number>).map(Number);
  }
  return undefined;
}

/**
 * Auto-detect video dataset path by scanning HDF5 structure.
 * Async version for streaming file access.
 */
async function findVideoDatasetStreaming(
  file: StreamingH5File,
  videoIndex: number,
): Promise<string | null> {
  // Try explicit path first (video0/video, video1/video, etc.)
  const explicitPath = `video${videoIndex}/video`;
  if (await hasPath(file, explicitPath)) {
    return explicitPath;
  }

  // Scan root keys for video groups
  const rootKeys = file.keys();
  for (const key of rootKeys) {
    if (key.startsWith("video")) {
      const groupKeys = await file.getKeys(key);
      if (groupKeys.includes("video")) {
        const candidatePath = `${key}/video`;
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

/**
 * Read suggestions from suggestions_json dataset.
 */
async function readSuggestionsStreaming(
  file: StreamingH5File,
  videos: Video[],
): Promise<SuggestionFrame[]> {
  if (!file.keys().includes("suggestions_json")) return [];

  const data = await file.getDatasetValue("suggestions_json");
  const values = normalizeDatasetArray(data.value);
  const metadataList = parseSuggestions(values);

  return metadataList
    .map((meta) => {
      const video = videos[meta.video];
      if (!video) return null;
      return new SuggestionFrame({
        video,
        frameIdx: meta.frameIdx,
        metadata: meta.metadata,
      });
    })
    .filter((s): s is SuggestionFrame => s !== null);
}

/**
 * Read identities from identities_json dataset.
 */
async function readIdentitiesStreaming(
  file: StreamingH5File,
): Promise<Identity[]> {
  const keys = file.keys();
  if (keys.includes("identities_json")) {
    const data = await file.getDatasetValue("identities_json");
    const values = normalizeDatasetArray(data.value);
    const identities: Identity[] = [];
    for (const entry of values) {
      const parsed = parseJsonEntry(entry) as Record<string, unknown>;
      const { name, color, ...rest } = parsed;
      identities.push(
        new Identity({
          name: (name as string) ?? "",
          color: color as string | undefined,
          metadata: rest,
        }),
      );
    }
    return identities;
  }
  // Python-style /identity group fallback (no identities_json).
  if (keys.includes("identity")) {
    const children = await file.getKeys("identity");
    if (children.includes("name")) {
      const strings = async (name: string): Promise<string[]> =>
        children.includes(name)
          ? normalizeDatasetArray(
              (await file.getDatasetValue(`identity/${name}`)).value,
            ).map((entry) => decodeEntryText(entry) ?? "")
          : [];
      const names = await strings("name");
      const metadata: Record<string, unknown>[] = names.map(() => ({}));
      if (children.includes("meta_owner")) {
        const owners = normalizeDatasetArray(
          (await file.getDatasetValue("identity/meta_owner")).value,
        ).map(Number);
        const mkeys = await strings("meta_key");
        const mvals = await strings("meta_val");
        owners.forEach((o, i) => {
          if (metadata[o]) metadata[o][mkeys[i]] = mvals[i];
        });
      }
      return names.map((name, i) => {
        const m = metadata[i];
        const color = typeof m.color === "string" ? m.color : undefined;
        const { color: _color, ...rest } = m;
        return new Identity({ name, color, metadata: rest });
      });
    }
  }
  return [];
}

/**
 * Read a per-detection links table, for every owner type, with the same table rules as
 * the eager reader. Empty when the file doesn't have it.
 */
async function readLinksTableStreaming(
  file: StreamingH5File,
  table: "identity/links" | "categories/links",
  idxField: string,
  scoreField: string,
  labelsPath: string,
  nCatalog: number,
): Promise<Map<number, Map<number, [number, number | null]>>> {
  const read = await readTableStreaming(file, table, labelsPath);
  if (!read) return new Map();
  if (
    read.columnar &&
    nCatalog > 0 &&
    indexOutOfRange(read.rows[idxField] ?? [], nCatalog, false)
  ) {
    reportTableProblems(labelsPath, [
      `/${table} ${idxField} is not one of the ${nCatalog} catalog entries`,
    ]);
  }
  return groupLinks(read.rows, idxField, scoreField);
}

/** Read one `/embeddings` triple, for every owner type (see {@link groupEmbeddings}). */
async function readEmbeddingTripleStreaming(
  file: StreamingH5File,
  vecName: string,
  otName: string,
  oidName: string,
): Promise<Map<number, Map<number, Embedding>>> {
  if (!(await hasPath(file, `embeddings/${vecName}`))) return new Map();
  const vectors = await file.getDatasetValue(`embeddings/${vecName}`);
  const rows = vectors.shape?.[0] ?? 0;
  const dim = vectors.shape && vectors.shape.length >= 2 ? vectors.shape[1] : 0;
  if (!rows || !dim) return new Map();
  const column = async (name: string): Promise<ArrayLike<unknown>> =>
    (await hasPath(file, `embeddings/${name}`))
      ? ((await file.getDatasetValue(`embeddings/${name}`))
          .value as ArrayLike<unknown>)
      : [];
  return groupEmbeddings(
    vectors.value as ArrayLike<number>,
    rows,
    dim,
    await column(otName),
    await column(oidName),
  );
}

/**
 * Read all per-detection identity (SLP 2.5) + category (SLP 2.7) join maps + the
 * category catalog. Empty maps on older files.
 */
async function readPerDetectionMapsStreaming(
  file: StreamingH5File,
  labelsPath: string,
  nIdentities: number,
): Promise<PerDetectionMaps> {
  const categoryCatalog = (await hasPath(file, "categories/name"))
    ? normalizeDatasetArray(
        (await file.getDatasetValue("categories/name")).value,
      ).map((n) => decodeEntryText(n) ?? "")
    : [];
  return {
    idLinks: await readLinksTableStreaming(
      file,
      "identity/links",
      "identity_idx",
      "identity_score",
      labelsPath,
      nIdentities,
    ),
    idEmbs: await readEmbeddingTripleStreaming(
      file,
      "vectors",
      "owner_type",
      "owner_id",
    ),
    catLinks: await readLinksTableStreaming(
      file,
      "categories/links",
      "category_idx",
      "category_score",
      labelsPath,
      categoryCatalog.length,
    ),
    catEmbs: await readEmbeddingTripleStreaming(
      file,
      "category_vectors",
      "category_owner_type",
      "category_owner_id",
    ),
    categoryCatalog,
  };
}

/**
 * Read every annotation table, in table order. Mirrors `readAnnotations` in `read.ts`.
 *
 * Every table and buffer is optional, and a file without one pays nothing for it: the
 * root keys are already known, and only what is present is read. What is present is
 * read in full, and a failed read is an error: unlike a missing table, it would drop
 * annotations from the project, and the next save would lose them for good.
 */
async function readAnnotationsStreaming(
  file: StreamingH5File,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
  instances?: Array<Instance | PredictedInstance>,
): Promise<AnnotationTuples> {
  const { rois, migratedBboxes } = await readRoisWithMigrationStreaming(
    file,
    videos,
    tracks,
    context,
    instances,
  );
  const bboxes = await readBboxesStreaming(file, tracks, context.labelsPath);
  return {
    rois,
    bboxes:
      bboxes.length === 0 && migratedBboxes.length > 0
        ? migratedBboxes
        : bboxes,
    masks: await readMasksStreaming(file, videos, tracks, context),
    centroids: await readCentroidsStreaming(file, tracks),
    labelImages: await readLabelImagesStreaming(
      file,
      videos,
      tracks,
      context,
      instances,
    ),
  };
}

async function readRoisWithMigrationStreaming(
  file: StreamingH5File,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
  instances?: Array<Instance | PredictedInstance>,
): Promise<{
  rois: Routed<ROI>[];
  migratedBboxes: Routed<BoundingBox>[];
}> {
  const table = await readTableStreaming(file, "rois", context.labelsPath);
  if (!table) return { rois: [], migratedBboxes: [] };
  const rows = table.rows;
  if (!(rows.annotation_type ?? []).length) {
    return { rois: [], migratedBboxes: [] };
  }

  const wkb = await readTableBufferStreaming(file, "rois", table.columnar);
  // A 2.x table without its geometry buffer has never been readable.
  if (!wkb && !table.columnar) return { rois: [], migratedBboxes: [] };

  return buildRois(
    {
      rows,
      columnar: table.columnar,
      wkb: toBytes(wkb?.value),
      categories: table.columnar
        ? rows.category
        : await readStringMetadataStreaming(
            file,
            "roi_categories",
            "rois",
            "categories",
          ),
      names: table.columnar
        ? rows.name
        : await readStringMetadataStreaming(file, "roi_names", "rois", "names"),
      sources: table.columnar
        ? rows.source
        : await readStringMetadataStreaming(
            file,
            "roi_sources",
            "rois",
            "sources",
          ),
    },
    videos,
    tracks,
    context,
    instances,
  );
}

async function readBboxesStreaming(
  file: StreamingH5File,
  tracks: Track[],
  labelsPath: string,
): Promise<Routed<BoundingBox>[]> {
  if (!file.keys().includes("bboxes")) return [];
  // Python has written /bboxes as a group of columns since format 2.0; sleap-io.js
  // writes a flat matrix with sibling string datasets.
  const columnar = await isGroupStreaming(file, "bboxes");
  const rows = columnar
    ? (normalizeColumnarTable(
        labelsPath,
        "bboxes",
        await readGroupTableStreaming(file, "bboxes"),
      ) as Rows)
    : ((await readStructDatasetStrict(file, "bboxes")) as Rows);
  if (!bboxRowCount(rows)) return [];

  return buildBboxes(
    {
      rows,
      categories: columnar
        ? rows.category
        : await readStringMetadataStreaming(
            file,
            "bbox_categories",
            "bboxes",
            "categories",
          ),
      names: columnar
        ? rows.name
        : await readStringMetadataStreaming(
            file,
            "bbox_names",
            "bboxes",
            "names",
          ),
      sources: columnar
        ? rows.source
        : await readStringMetadataStreaming(
            file,
            "bbox_sources",
            "bboxes",
            "sources",
          ),
    },
    tracks,
  );
}

async function readMasksStreaming(
  file: StreamingH5File,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
): Promise<Routed<SegmentationMask>[]> {
  const table = await readTableStreaming(file, "masks", context.labelsPath);
  if (!table) return [];
  const rows = table.rows;
  if (!(rows.height ?? []).length) return [];

  const rle = await readTableBufferStreaming(file, "masks", table.columnar);
  // A 2.x table without its RLE buffer has never been readable.
  if (!rle && !table.columnar) return [];

  return buildMasks(
    {
      rows,
      columnar: table.columnar,
      rle: toBytes(rle?.value),
      categories: table.columnar
        ? rows.category
        : await readStringMetadataStreaming(
            file,
            "mask_categories",
            "masks",
            "categories",
          ),
      names: table.columnar
        ? rows.name
        : await readStringMetadataStreaming(
            file,
            "mask_names",
            "masks",
            "names",
          ),
      sources: table.columnar
        ? rows.source
        : await readStringMetadataStreaming(
            file,
            "mask_sources",
            "masks",
            "sources",
          ),
      scoreMaps: await readScoreMapsStreaming(
        file,
        "masks/score_maps",
        context.labelsPath,
      ),
    },
    videos,
    tracks,
    context,
  );
}

async function readLabelImagesStreaming(
  file: StreamingH5File,
  videos: Video[],
  tracks: Track[],
  context: TableContext,
  instances?: Array<Instance | PredictedInstance>,
): Promise<Routed<LabelImage>[]> {
  const { labelsPath } = context;
  const table = await readTableStreaming(file, "label_images", labelsPath);
  if (!table) return [];
  const rows = table.rows;
  if (!(rows.video ?? []).length) return [];

  const sources = table.columnar
    ? rows.source
    : await readStringMetadataStreaming(
        file,
        "label_image_sources",
        "label_images",
        "sources",
      );
  const dataPath = tableBufferPath("label_images", table.columnar) as string;
  const data = (await hasPath(file, dataPath))
    ? await readPixelDataStreaming(file, dataPath)
    : null;
  const base = { rows, columnar: table.columnar, sources };
  if (!data) {
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
  const objTable = await readTableStreaming(
    file,
    "label_images/objects",
    labelsPath,
  );
  if (objTable) {
    objects = {
      rows: objTable.rows,
      categories: objTable.columnar
        ? objTable.rows.category
        : await readStringMetadataStreaming(
            file,
            "label_image_obj_categories",
            "label_image_objects",
            "categories",
          ),
      names: objTable.columnar
        ? objTable.rows.name
        : await readStringMetadataStreaming(
            file,
            "label_image_obj_names",
            "label_image_objects",
            "names",
          ),
    };
  }

  return buildLabelImages(
    {
      ...base,
      data,
      objects,
      scoreMaps: await readScoreMapsStreaming(
        file,
        "label_images/score_maps",
        labelsPath,
      ),
    },
    videos,
    tracks,
    context,
    instances,
  );
}

/**
 * Read `/centroids` (SLP first-class centroid annotations), in either layout: a group
 * of columns, or the flat matrix older JS versions wrote. Mirrors `readCentroids` in
 * `read.ts`. Returns `[centroid, videoIdx, frameIdx]` routing tuples; `[]` when absent.
 */
async function readCentroidsStreaming(
  file: StreamingH5File,
  tracks: Track[],
): Promise<Routed<Centroid>[]> {
  if (!file.keys().includes("centroids")) return [];
  const children = await file.getKeys("centroids");
  if (!children.includes("x")) {
    return buildCentroidTuples(
      (await readStructDatasetStrict(file, "centroids")) as Record<
        string,
        ArrayLike<number>
      >,
      await readStringMetadataStreaming(
        file,
        "centroid_categories",
        "centroids",
        "categories",
      ),
      await readStringMetadataStreaming(
        file,
        "centroid_names",
        "centroids",
        "names",
      ),
      await readStringMetadataStreaming(
        file,
        "centroid_sources",
        "centroids",
        "sources",
      ),
      tracks,
    );
  }

  const numCol = async (name: string): Promise<ArrayLike<number>> =>
    children.includes(name)
      ? ((await file.getDatasetValue(`centroids/${name}`))
          .value as ArrayLike<number>)
      : [];
  const data: Record<string, ArrayLike<number>> = {
    x: await numCol("x"),
    y: await numCol("y"),
    z: await numCol("z"),
    video: await numCol("video"),
    frame_idx: await numCol("frame_idx"),
    track: await numCol("track"),
    instance: await numCol("instance"),
    is_predicted: await numCol("is_predicted"),
    score: await numCol("score"),
    tracking_score: await numCol("tracking_score"),
  };
  const strCol = async (name: string): Promise<string[]> => {
    if (!children.includes(name)) return [];
    const d = await file.getDatasetValue(`centroids/${name}`);
    return normalizeDatasetArray(d.value).map((v) =>
      typeof v === "string"
        ? v
        : v instanceof Uint8Array
          ? new TextDecoder().decode(v)
          : String(v ?? ""),
    );
  };
  const categories = await strCol("category");
  const names = await strCol("name");
  const sources = await strCol("source");
  return buildCentroidTuples(data, categories, names, sources, tracks);
}

/**
 * Read recording sessions from sessions_json dataset.
 */
// Exported for a Worker-free unit test of the streaming session-read wiring:
// the full streaming reader (`readSlpStreaming`) is browser/Worker-gated and
// unreachable in the Node test runner, so tests drive this directly with a fake
// `StreamingH5File`. Not re-exported from the package barrel (see src/index.ts).
export async function readSessionsStreaming(
  file: StreamingH5File,
  videos: Video[],
  skeletons: Skeleton[],
  identities?: Identity[],
  captureRaw: boolean = false,
  context: { labelsPath: string; nFrames: number | null } = {
    labelsPath: "This file",
    nFrames: null,
  },
): Promise<RecordingSession[]> {
  try {
    const keys = file.keys();
    if (!keys.includes("sessions_json")) return [];

    const data = await file.getDatasetValue("sessions_json");
    const values = normalizeDatasetArray(data.value);
    // A present-but-empty dataset is a legitimate "no sessions" state (return []).
    // A present dataset whose entries read back blank/unparseable is NOT — it is
    // the h5wasm vlen read ceiling (sleap-io.js#220), handled by sessionsReadError.
    if (values.length === 0) return [];

    // SLP 2.8 columnar /session_data (read once for all sessions), or null for
    // legacy ≤2.7 files.
    const sessionData = await readSessionDataStreaming(
      file,
      context.labelsPath,
      identities?.length ?? 0,
      context.nFrames,
    );

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
      const calibration = (parsed.calibration ?? {}) as Record<string, unknown>;

      const cameraGroup = new CameraGroup();
      const cameraMap = new Map<string, Camera>();

      for (const [key, data] of Object.entries(calibration)) {
        if (key === "metadata") continue;
        const cameraData = data as Record<string, unknown>;
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
        metadata:
          (parsed.metadata as Record<string, unknown> | undefined) ?? {},
      });
      // Optionally retain the verbatim parsed sessions_json dict (deprecated,
      // opt-in via `rawSessions`). Deep-cloned so it is an INDEPENDENT snapshot
      // (the object model reuses `parsed`'s nested metadata/calibration objects
      // by reference). Never re-written to disk; see RecordingSession.rawJson.
      // Mirrors readSessions in read.ts.
      if (captureRaw) session.rawJson = structuredClone(parsed);

      const map = (parsed.camcorder_to_video_idx_map ?? {}) as Record<
        string,
        unknown
      >;
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

      // SLP 2.8 columnar path: reconstruct from /session_data via the session's
      // fg_start/fg_end range (shared with the eager reader). Dispatched on both
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

      // Legacy (≤2.7) inline path.
      const frameGroups = Array.isArray(parsed.frame_group_dicts)
        ? parsed.frame_group_dicts
        : [];
      for (const group of frameGroups) {
        const groupRecord = group as Record<string, unknown>;
        const frameIdx =
          (groupRecord.frame_idx as number | undefined) ??
          (groupRecord.frameIdx as number | undefined) ??
          0;
        const instanceGroups: InstanceGroup[] = [];
        const instanceGroupList = Array.isArray(groupRecord.instance_groups)
          ? groupRecord.instance_groups
          : [];
        for (const instanceGroup of instanceGroupList) {
          const instanceGroupRecord = instanceGroup as Record<string, unknown>;
          // Concrete instances only for the JS-inline format; the Python/
          // camcorder format is stored as index refs and resolved lazily via the
          // injected frame resolver (no frame materialization at read time).
          let instanceByCamera: Map<Camera, Instance> | undefined;
          const instancesRecord = (instanceGroupRecord.instances ??
            {}) as Record<string, unknown>;
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
            if (!instanceByCamera)
              instanceByCamera = new Map<Camera, Instance>();
            instanceByCamera.set(
              camera,
              new Instance({
                points: points as Record<string, number[]>,
                skeleton,
              }),
            );
          }

          // Capture verbatim index refs (camera -> [lfIdx, instIdx]) as NUMBERS.
          let instanceRefsByCamera: Map<Camera, [number, number]> | undefined;
          const lfInstMap =
            (instanceGroupRecord.camcorder_to_lf_and_inst_idx_map ??
              {}) as Record<string, unknown>;
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
            instanceRefsByCamera.set(camera, [
              Number(pair[0]),
              Number(pair[1]),
            ]);
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

        // Capture labeled-frame index refs (camera -> lfIdx) verbatim.
        let labeledFrameRefsByCamera: Map<Camera, number> | undefined;
        const labeledFrameMap = (groupRecord.labeled_frame_by_camera ??
          {}) as Record<string, unknown>;
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
            const igRecord = instanceGroup as Record<string, unknown>;
            const lfInstMap = (igRecord.camcorder_to_lf_and_inst_idx_map ??
              {}) as Record<string, unknown>;
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
              (groupRecord.metadata as Record<string, unknown> | undefined) ??
              {},
          }),
        );
      }
      sessions.push(session);
    }
    return sessions;
  } catch (err) {
    // Fail loud (sleap-io.js#220): a present-but-unreadable sessions_json must
    // surface, not silently drop calibration + grouping + 3D as an empty list
    // (which the next save would then overwrite 2D-only). Absent/empty datasets
    // already returned [] above without reaching here.
    if (err instanceof Error) throw err;
    throw new Error(`Failed to read sessions_json: ${String(err)}`);
  }
}

/**
 * Read the columnar `/session_data` group (SLP 2.8) over the streaming/worker API
 * into a backend-agnostic {@link SessionData} for {@link reconstructColumnarFrameGroups},
 * or `null` when the group is absent (legacy ≤2.7) or missing a struct table.
 *
 * `file.keys()` is a root-level snapshot, so the group's child datasets are listed
 * via `getKeys("session_data")`; the struct tables are read through
 * `readRawTableStreaming` (with `assumePresent`, since presence is already
 * confirmed), and `points_3d`/`pred_points_3d` + the vlen meta arrays via
 * `getDatasetValue` (whole-read — the tables are small; range-reads can come later).
 */
async function readSessionDataStreaming(
  file: StreamingH5File,
  labelsPath: string,
  nIdentities: number,
  nFrames: number | null,
): Promise<SessionData | null> {
  if (!file.keys().includes("session_data")) return null;
  const children = new Set(await file.getKeys("session_data"));

  let allColumnar = true;
  const struct = async (name: string): Promise<Columns | null> => {
    if (!children.has(name)) return null;
    const table = `session_data/${name}` as ColumnarTableName;
    const raw = await readRawTableStreaming(file, table, true);
    allColumnar &&= raw.columnar;
    if (raw.columnar) return normalizeColumnarTable(labelsPath, table, raw);
    return Object.keys(raw.columns).length ? raw.columns : null;
  };
  const frameGroups = await struct("frame_groups");
  const instanceGroups = await struct("instance_groups");
  const members = await struct("instance_group_members");
  if (!frameGroups || !instanceGroups || !members) return null;

  const matrix = async (
    name: string,
    ncolsDefault: number,
  ): Promise<SessionPointMatrix | null> => {
    if (!children.has(name)) return null;
    const d = await file.getDatasetValue(`session_data/${name}`);
    const ncols = d.shape && d.shape.length >= 2 ? d.shape[1] : ncolsDefault;
    return { flat: d.value as ArrayLike<number>, ncols };
  };
  const meta = async (name: string): Promise<unknown[] | null> => {
    if (!children.has(name)) return null;
    const d = await file.getDatasetValue(`session_data/${name}`);
    return normalizeDatasetArray(d.value);
  };

  const sessionData: SessionData = {
    frameGroups,
    instanceGroups,
    members,
    points3d: await matrix("points_3d", 3),
    predPoints3d: await matrix("pred_points_3d", 4),
    frameGroupMeta: await meta("frame_group_meta"),
    instanceGroupMeta: await meta("instance_group_meta"),
  };
  if (allColumnar) {
    reportTableProblems(
      labelsPath,
      sessionProblems(sessionData, { nIdentities, nFrames }),
    );
  }
  return sessionData;
}

/** Whether a node is a group of columns: the only thing with no shape. */
async function isGroupStreaming(
  file: StreamingH5File,
  path: string,
): Promise<boolean> {
  return (await file.getDatasetMeta(path)).shape === undefined;
}

/** Read a table's group of columns (SLP 3.0), leaving out its buffer and sub-tables. */
async function readGroupTableStreaming(
  file: StreamingH5File,
  table: ColumnarTableName,
): Promise<RawTable> {
  const columns = (
    await file.getDatasetValue(table, undefined, tableMembers(table))
  ).value as Columns;
  const members = await file.getKeys(table);
  return {
    columns,
    otherMembers: members.filter((name) => !(name in columns)),
    columnar: true,
  };
}

/**
 * Read one table into columns, whichever layout it is in. The worker reads a group
 * of columns (SLP 3.0) as it reads a compound table, and a group is the only thing
 * with no shape. A table the file doesn't have reads as no columns, but a failed read
 * is thrown: it would otherwise open the file with rows missing, for the next save to
 * make permanent.
 *
 * `file.keys()` lists only the root, so a caller that reads a nested table and has
 * already found it passes `assumePresent`.
 */
async function readRawTableStreaming(
  file: StreamingH5File,
  table: ColumnarTableName,
  assumePresent = false,
): Promise<RawTable> {
  if (!assumePresent && !file.keys().includes(table)) {
    return { columns: {}, columnar: false };
  }
  if (await isGroupStreaming(file, table)) {
    return readGroupTableStreaming(file, table);
  }
  return {
    columns: await readStructDatasetStrict(file, table),
    columnar: false,
  };
}

/**
 * Read one table, whichever layout it is in, as columns the rest of this file can index
 * (`columnar` says it was a group of them), or `null` if the file doesn't have it. Any
 * failed read is thrown. Mirrors `readTable` in `read.ts`.
 */
async function readTableStreaming(
  file: StreamingH5File,
  table: ColumnarTableName,
  labelsPath: string,
): Promise<{ rows: Rows; columnar: boolean } | null> {
  if ((await hasPath(file, table)) && (await isGroupStreaming(file, table))) {
    return {
      rows: normalizeColumnarTable(
        labelsPath,
        table,
        await readGroupTableStreaming(file, table),
      ) as Rows,
      columnar: true,
    };
  }
  const legacyPath = legacyTablePath(table);
  if (!(await hasPath(file, legacyPath))) return null;
  return {
    rows: (await readStructDatasetStrict(file, legacyPath)) as Rows,
    columnar: false,
  };
}

/** The byte buffer a table's rows index into, or `null` if the file doesn't have it. */
async function readTableBufferStreaming(
  file: StreamingH5File,
  table: ColumnarTableName,
  columnar: boolean,
): Promise<{ value: unknown; shape: number[] } | null> {
  const path = tableBufferPath(table, columnar) as string;
  return (await hasPath(file, path)) ? file.getDatasetValue(path) : null;
}

/**
 * The most label image pixels to read from the worker at once. A dataset read as one
 * piece needs that much of the worker's WebAssembly memory, which stops at 2 GiB and
 * never shrinks back, and 3-D pixel data reaches that at about 2,000 frames of 512x512.
 */
const PIXEL_BLOCK_BYTES = 256 * 1024 * 1024;

/**
 * Read label image pixel data: a blob (1-D), or frames (3-D), which take a block of
 * frames at a time once they are too large for one read.
 *
 * Exported for testing the block reads without a file of that size; not part of the
 * public API.
 */
export async function readPixelDataStreaming(
  file: StreamingH5File,
  path: string,
  blockBytes: number = PIXEL_BLOCK_BYTES,
): Promise<LabelImagePixels> {
  const meta = await file.getDatasetMeta(path);
  const shape = meta.shape ?? [];
  const frameBytes =
    shape[1] * shape[2] * ((meta.metadata?.size as number) ?? 4);
  if (shape.length !== 3 || shape[0] * frameBytes <= blockBytes) {
    return file.getDatasetValue(path);
  }
  // The worker checks this on whole reads only, so a frame of video doesn't pay for it.
  checkDatasetFilters(meta, path);
  const framesPerBlock = Math.max(1, Math.floor(blockBytes / frameBytes));
  const blocks: unknown[] = [];
  for (let start = 0; start < shape[0]; start += framesPerBlock) {
    const end = Math.min(start + framesPerBlock, shape[0]);
    blocks.push(
      (await file.getDatasetValue(path, [[start, end], [], []])).value,
    );
  }
  return { value: blocks, shape, framesPerBlock };
}

/** Read a score-map index and the bytes it indexes into, if the file has them. */
async function readScoreMapsStreaming(
  file: StreamingH5File,
  table: "masks/score_maps" | "label_images/score_maps",
  labelsPath: string,
): Promise<ScoreMapTable | null> {
  const index = await readTableStreaming(file, table, labelsPath);
  if (!index) return null;
  const data = await readTableBufferStreaming(file, table, index.columnar);
  if (!data && !index.columnar) return null;
  return {
    rows: index.rows,
    columnar: index.columnar,
    data: toBytes(data?.value),
  };
}

/**
 * Read string metadata from a root-level dataset or fall back to a JSON attribute on
 * the table. v1.9+ writes string datasets; older files use JSON-encoded attrs. Mirrors
 * `readStringMetadata` in `read.ts`.
 */
async function readStringMetadataStreaming(
  file: StreamingH5File,
  datasetPath: string,
  tablePath: string,
  attrName: string,
): Promise<string[]> {
  if (file.keys().includes(datasetPath)) {
    // Try the "json" attribute on the string dataset first
    const fromJson = parseStringListAttr(
      (await file.getAttrs(datasetPath)).json,
    );
    if (fromJson.length > 0) return fromJson;
    // Fall back to raw value
    const strings = stringsFromDataset(
      (await file.getDatasetValue(datasetPath)).value,
    );
    if (strings) return strings;
  }
  return parseStringListAttr((await file.getAttrs(tablePath))[attrName]);
}

/** The value of `/provenance_json`, if the file has one (see {@link parseProvenance}). */
async function readProvenanceStreaming(
  file: StreamingH5File,
): Promise<unknown> {
  if (!file.keys().includes("provenance_json")) return undefined;
  return (await file.getDatasetValue("provenance_json")).value;
}

/**
 * Read a structured dataset that is known to exist, as columns. A failed read is
 * thrown.
 */
async function readStructDatasetStrict(
  file: StreamingH5File,
  path: string,
): Promise<Record<string, unknown[]>> {
  const meta = await file.getDatasetMeta(path);
  const data = await file.getDatasetValue(path);

  // Get field names: try dtype metadata first, then HDF5 dataset attributes.
  // Matches main-thread reader's getFieldNames (read.ts:1356-1372).
  let fieldNames = getFieldNamesFromMeta(meta);
  if (fieldNames.length === 0) {
    const attrs = await file.getAttrs(path);
    const fnAttr = attrs.field_names ?? attrs.fieldNames;
    if (fnAttr) {
      // Accept an array, a `{value}` wrapper (worker-serialized), or a bare
      // string/bytes (some backends return the attribute value unwrapped).
      let raw: unknown = Array.isArray(fnAttr)
        ? fnAttr
        : ((fnAttr as { value?: unknown })?.value ?? fnAttr);
      if (typeof raw === "string") {
        try {
          raw = JSON.parse(raw);
        } catch {
          /* not JSON */
        }
      }
      if (raw instanceof Uint8Array) {
        try {
          raw = JSON.parse(new TextDecoder().decode(raw));
        } catch {
          /* not JSON */
        }
      }
      if (Array.isArray(raw)) {
        fieldNames = raw.map(String);
      }
    }
  }

  return normalizeStructData(data.value, data.shape, fieldNames);
}

/**
 * Extract field names from dataset metadata.
 */
export function getFieldNamesFromMeta(meta: {
  shape: number[];
  dtype: string;
}): string[] {
  // dtype might be a string like "{'names':['x','y','visible','complete'],...}"
  // or an object with compound type info
  const dtype = meta.dtype;

  if (typeof dtype === "string") {
    // Try to parse compound type from string representation
    const namesMatch = dtype.match(/'names':\s*\[([^\]]+)\]/);
    if (namesMatch) {
      const namesStr = namesMatch[1];
      const names = namesStr.match(/'([^']+)'/g);
      if (names) {
        return names.map((n) => n.replace(/'/g, ""));
      }
    }
  }

  if (Array.isArray(dtype)) {
    return dtype.map((pair: [string, string]) => pair[0]);
  }

  if (typeof dtype === "object" && dtype !== null) {
    const dtypeObj = dtype as Record<string, unknown>;
    if (dtypeObj.compound_type && typeof dtypeObj.compound_type === "object") {
      const compound = dtypeObj.compound_type as {
        members?: Array<{ name?: string }>;
      };
      if (compound.members) {
        return compound.members
          .map((m) => m.name)
          .filter((n): n is string => !!n);
      }
    }
  }

  return [];
}

/**
 * Normalize dataset value to column-oriented format.
 */
function normalizeStructData(
  value: unknown,
  shape: number[],
  fieldNames: string[],
): Record<string, unknown[]> {
  if (!value) return {};

  // If value is already an object with arrays (column format)
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    !ArrayBuffer.isView(value)
  ) {
    const obj = value as Record<string, unknown>;
    // Check if it looks like column data: a `{ field: array }` record. Columns
    // may be plain arrays or TypedArrays — the streaming worker now returns
    // Float64Array columns (transferred, not cloned), which the downstream
    // builder consumes by index just like plain arrays.
    const firstKey = Object.keys(obj)[0];
    const firstCol = firstKey ? obj[firstKey] : undefined;
    if (firstKey && (Array.isArray(firstCol) || ArrayBuffer.isView(firstCol))) {
      return obj as Record<string, unknown[]>;
    }
  }

  // If value is a typed array with 2D shape, convert to columns
  if (ArrayBuffer.isView(value) && shape.length === 2) {
    const [rowCount, colCount] = shape;
    const arr = value as unknown as ArrayLike<number>;

    if (fieldNames.length > 0) {
      // As the eager reader's normalizeStructDataset: a field past the columns reads
      // as undefined, and a column past the fields is dropped.
      const result: Record<string, unknown[]> = {};
      for (let col = 0; col < fieldNames.length; col++) {
        const colData: unknown[] = new Array(rowCount);
        if (col < colCount) {
          for (let row = 0; row < rowCount; row++) {
            colData[row] = arr[row * colCount + col];
          }
        }
        result[fieldNames[col]] = colData;
      }
      return result;
    }
  }

  // If value is an array of arrays (row format)
  if (Array.isArray(value) && value.length > 0 && Array.isArray(value[0])) {
    const rows = value as unknown[][];
    if (fieldNames.length) {
      const result: Record<string, unknown[]> = {};
      fieldNames.forEach((field, colIdx) => {
        result[field] = rows.map((row) => row[colIdx]);
      });
      return result;
    }
  }

  return {};
}

/**
 * Normalize a dataset value to an array.
 */
function normalizeDatasetArray(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (ArrayBuffer.isView(value)) {
    // For typed arrays of strings or buffers, convert to array
    return Array.from(value as unknown as ArrayLike<unknown>);
  }
  return [];
}

/**
 * Build LabeledFrame objects from normalized data.
 * (Adapted from read.ts)
 */
function buildLabeledFrames(options: {
  framesData: Record<string, unknown[]>;
  instancesData: Record<string, unknown[]>;
  pointsData: Record<string, unknown[]>;
  predPointsData: Record<string, unknown[]>;
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
  const frameIds = (framesData.frame_id ?? []) as number[];
  const videoIdToIndex = buildVideoIdMap(framesData, videos);
  const instanceById = new Map<number, Instance | PredictedInstance>();
  const fromPredictedPairs: Array<[number, number]> = [];

  for (let frameIdx = 0; frameIdx < frameIds.length; frameIdx += 1) {
    const rawVideoId = Number((framesData.video as number[])?.[frameIdx] ?? 0);
    const videoIndex = videoIdToIndex.get(rawVideoId) ?? rawVideoId;
    const frameIndex = Number(
      (framesData.frame_idx as number[])?.[frameIdx] ?? 0,
    );
    const instStart = Number(
      (framesData.instance_id_start as number[])?.[frameIdx] ?? 0,
    );
    const instEnd = Number(
      (framesData.instance_id_end as number[])?.[frameIdx] ?? 0,
    );
    const video = videos[videoIndex];
    if (!video) continue;

    const instances: Array<Instance | PredictedInstance> = [];
    for (let instIdx = instStart; instIdx < instEnd; instIdx += 1) {
      const instanceType = Number(
        (instancesData.instance_type as number[])?.[instIdx] ?? 0,
      );
      // Rows of an unknown type are skipped, as in Python.
      if (instanceType !== 0 && instanceType !== 1) continue;
      const skeletonId = Number(
        (instancesData.skeleton as number[])?.[instIdx] ?? 0,
      );
      const trackId = Number(
        (instancesData.track as number[])?.[instIdx] ?? -1,
      );
      const pointStart = Number(
        (instancesData.point_id_start as number[])?.[instIdx] ?? 0,
      );
      const pointEnd = Number(
        (instancesData.point_id_end as number[])?.[instIdx] ?? 0,
      );
      const score = Number((instancesData.score as number[])?.[instIdx] ?? 0);
      const rawTrackingScore =
        formatId < 1.2
          ? 0
          : Number((instancesData.tracking_score as number[])?.[instIdx] ?? 0);
      const trackingScore = Number.isNaN(rawTrackingScore)
        ? 0
        : rawTrackingScore;
      const fromPredicted = Number(
        (instancesData.from_predicted as number[])?.[instIdx] ?? -1,
      );
      const skeleton =
        skeletons[skeletonId] ?? skeletons[0] ?? new Skeleton({ nodes: [] });
      const track = trackId >= 0 ? tracks[trackId] : null;

      let instance: Instance | PredictedInstance;
      if (instanceType === 0) {
        // Build straight from the point columns — no intermediate Point[].
        instance = Instance._fromColumns({
          columns: pointsData as PointColumns,
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
          columns: predPointsData as PointColumns,
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
