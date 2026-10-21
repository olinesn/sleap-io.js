import {
  openH5File,
  type OpenH5Options,
  type SlpSource,
  getH5EmscriptenModule,
} from "./h5.js";
import { readCompoundColumnsManual } from "./h5-compound.js";
import {
  attrToNumber,
  attrToString,
  datasetValueToString,
  parseMetadataJson,
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
import { Embedding } from "../../model/embedding.js";
import { LazyDataStore, LazyFrameList } from "../../model/lazy.js";
import { buildVideoIdMap } from "../../model/video-id-map.js";
import {
  markNegativeFrames,
  negativeFrameKeys,
} from "../../model/negative-frames.js";
import {
  type ROI,
  UserROI,
  PredictedROI,
  AnnotationType,
  decodeWkb,
} from "../../model/roi.js";
import {
  type SegmentationMask,
  UserSegmentationMask,
  PredictedSegmentationMask,
} from "../../model/mask.js";
import {
  type BoundingBox,
  UserBoundingBox,
  PredictedBoundingBox,
} from "../../model/bbox.js";
import { type Centroid } from "../../model/centroid.js";
import { buildCentroidTuples } from "./centroids.js";
import {
  type LabelImage,
  UserLabelImage,
  PredictedLabelImage,
} from "../../model/label-image.js";
import type { LabelImageObjectInfo } from "../../model/label-image.js";
import { inflate } from "pako";

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
    const formatId = Number(
      metadataAttrs["format_id"]?.value ?? metadataAttrs["format_id"] ?? 1.0,
    );
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
    const framesData = normalizeStructDataset(file.get("frames"), emscripten);
    const instancesData = normalizeStructDataset(
      file.get("instances"),
      emscripten,
    );
    const pointsData = normalizeStructDataset(file.get("points"), emscripten);
    const predPointsData = normalizeStructDataset(
      file.get("pred_points"),
      emscripten,
    );

    const labeledFrames = buildLabeledFrames({
      framesData,
      instancesData,
      pointsData,
      predPointsData,
      skeletons,
      tracks,
      videos,
      formatId,
    });

    // Read negative frames (video ids resolved like /frames', see helper)
    const negativeFramesDs = file.get("negative_frames");
    if (negativeFramesDs) {
      markNegativeFrames(
        labeledFrames,
        negativeFrameKeys(normalizeStructDataset(negativeFramesDs)),
        framesData,
        videos,
      );
    }

    report(5); // Reading identities
    const identities = readIdentityCatalog(file);
    report(6); // Reading sessions
    const sessions = readSessions(
      file.get("sessions_json"),
      videos,
      skeletons,
      identities,
      options?.rawSessions ?? false,
      readSessionDataEager(file, emscripten),
    );
    report(7); // Reading annotations
    const allInstances = labeledFrames.flatMap((f) => f.instances);
    const { rois: roiTuples, bboxes: bboxTuples } = readRoisAndBboxes(
      file,
      videos,
      tracks,
      allInstances,
    );
    const maskTuples = readMasks(file, videos, tracks);
    const centroidTuples = readCentroids(file, videos, tracks);
    const labelImageTuples = readLabelImages(
      file,
      videos,
      tracks,
      allInstances,
    );

    // Per-detection re-ID identity (SLP 2.5) + category (SLP 2.7): attach onto each
    // modality's ordered detection list (index == owner_id). No-op on older files.
    const pdMaps = readPerDetectionMaps(file, emscripten);
    attachOwnerType(pdMaps, 0, allInstances, identities);
    attachOwnerType(
      pdMaps,
      2,
      centroidTuples.map((t) => t[0]),
      identities,
    );
    attachOwnerType(
      pdMaps,
      3,
      maskTuples.map((t) => t[0]),
      identities,
    );
    attachOwnerType(
      pdMaps,
      4,
      bboxTuples.map((t) => t[0]),
      identities,
    );
    attachOwnerType(
      pdMaps,
      5,
      roiTuples.map((t) => t[0]),
      identities,
    );

    // Distribute annotations into LabeledFrames using routing tuples
    const frameMap = new Map<string, LabeledFrame>();
    for (const lf of labeledFrames) {
      const vidIdx = videos.indexOf(lf.video);
      frameMap.set(`${vidIdx}:${lf.frameIdx}`, lf);
    }
    const getOrCreateFrame = (
      vidIdx: number,
      frameIdx: number,
    ): LabeledFrame => {
      const key = `${vidIdx}:${frameIdx}`;
      let lf = frameMap.get(key);
      if (!lf) {
        lf = new LabeledFrame({ video: videos[vidIdx], frameIdx });
        frameMap.set(key, lf);
        labeledFrames.push(lf);
      }
      return lf;
    };

    const staticRois: ROI[] = [];
    const distributeTuples = <T>(
      tuples: [T, number, number][],
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
    // in readRoisWithMigration since it receives `instances`).
    const allInstancesFlat = labeledFrames.flatMap((lf) => lf.instances);
    const resolveInstanceRef = (ann: {
      _instanceIdx: number | null;
      instance: Instance | null;
    }): void => {
      if (
        ann._instanceIdx !== null &&
        ann._instanceIdx >= 0 &&
        ann._instanceIdx < allInstancesFlat.length
      ) {
        ann.instance = allInstancesFlat[ann._instanceIdx];
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
            if (obj && instIdx >= 0 && instIdx < allInstancesFlat.length) {
              obj.instance = allInstancesFlat[instIdx];
            }
          }
          li._objectInstanceIdxs = null;
        }
      }
    }

    onProgress?.(total, total, "Finalizing");
    const labels = new Labels({
      labeledFrames,
      videos,
      skeletons,
      tracks,
      suggestions,
      sessions,
      identities,
      provenance: (metadataJson?.provenance as Record<string, unknown>) ?? {},
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
    const formatId = Number(
      metadataAttrs["format_id"]?.value ?? metadataAttrs["format_id"] ?? 1.0,
    );
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
    const framesData = normalizeStructDataset(file.get("frames"), emscripten);
    const instancesData = normalizeStructDataset(
      file.get("instances"),
      emscripten,
    );
    const pointsData = normalizeStructDataset(file.get("points"), emscripten);
    const predPointsData = normalizeStructDataset(
      file.get("pred_points"),
      emscripten,
    );

    // Read negative frames (raw video ids; the store resolves them per frame)
    const negativeFramesDs = file.get("negative_frames");
    const negativeFrames = negativeFramesDs
      ? negativeFrameKeys(normalizeStructDataset(negativeFramesDs))
      : new Set<string>();

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
      readSessionDataEager(file, emscripten),
    );
    report(7); // Reading annotations
    const { rois: roiTuples, bboxes: bboxTuples } = readRoisAndBboxes(
      file,
      videos,
      tracks,
    );
    const maskTuples = readMasks(file, videos, tracks);
    const centroidTuples = readCentroids(file, videos, tracks);
    const labelImageTuples = readLabelImages(file, videos, tracks);

    // Per-detection identity (SLP 2.5) + category (SLP 2.7). Instances materialize
    // lazily, so the OWNER_INSTANCE maps + catalogs are handed to the store and
    // attached in materializeFrame; the other modalities are concrete and attached
    // now.
    const pdMaps = readPerDetectionMaps(file, emscripten);
    store.identities = identities;
    store._instanceIdentityLinks = pdMaps.idLinks.get(0);
    store._instanceEmbeddings = pdMaps.idEmbs.get(0);
    store.categoryCatalog = pdMaps.categoryCatalog;
    store._instanceCategoryLinks = pdMaps.catLinks.get(0);
    store._instanceCategoryEmbeddings = pdMaps.catEmbs.get(0);
    attachOwnerType(
      pdMaps,
      2,
      centroidTuples.map((t) => t[0]),
      identities,
    );
    attachOwnerType(
      pdMaps,
      3,
      maskTuples.map((t) => t[0]),
      identities,
    );
    attachOwnerType(
      pdMaps,
      4,
      bboxTuples.map((t) => t[0]),
      identities,
    );
    attachOwnerType(
      pdMaps,
      5,
      roiTuples.map((t) => t[0]),
      identities,
    );

    // Build per-frame annotation dicts for lazy materialization
    const buildAnnByFrame = <T>(
      tuples: [T, number, number][],
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

    const cResult = buildAnnByFrame(centroidTuples);
    const bResult = buildAnnByFrame(bboxTuples);
    const mResult = buildAnnByFrame(maskTuples);
    const rResult = buildAnnByFrame(roiTuples);
    const liResult = buildAnnByFrame(labelImageTuples);

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

    // Check for annotation-only frames not in /frames (e.g., TrackMate SLPs)
    const frameKeys = new Set<string>();
    const frameVideoIds = framesData.video ?? [];
    const frameFrameIdxs = framesData.frame_idx ?? [];
    for (let i = 0; i < (framesData.frame_id ?? []).length; i++) {
      frameKeys.add(`${Number(frameVideoIds[i])}:${Number(frameFrameIdxs[i])}`);
    }

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

    // Don't pass annotations to Labels — they live on the store
    const labels = new Labels({
      videos,
      skeletons,
      tracks,
      suggestions,
      sessions,
      identities,
      provenance: (metadataJson?.provenance as Record<string, unknown>) ?? {},
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
 * Read a per-detection links table — `/identity/links` (SLP 2.5) or
 * `/categories/links` (SLP 2.7) — a genuine compound (Python) or a
 * flat-2-D+`field_names` (JS) `(owner_type, owner_id, <idxField>, <scoreField>)`
 * table via `normalizeStructDataset` (handles both). Returns
 * `owner_type → owner_id → [idx, score|null]` (NaN score → null); empty when the
 * dataset is absent. `i8`/`u8` columns are `Number()`-coerced.
 */
function readLinksTable(
  file: any,
  path: string,
  idxField: string,
  scoreField: string,
  emscripten?: unknown,
): Map<number, Map<number, [number, number | null]>> {
  const result = new Map<number, Map<number, [number, number | null]>>();
  const ds = file.get(path);
  if (!ds) return result;
  const cols = normalizeStructDataset(ds, emscripten);
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

const readIdentityLinks = (file: any, emscripten?: unknown) =>
  readLinksTable(
    file,
    "identity/links",
    "identity_idx",
    "identity_score",
    emscripten,
  );

const readCategoryLinks = (file: any, emscripten?: unknown) =>
  readLinksTable(
    file,
    "categories/links",
    "category_idx",
    "category_score",
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
  const result = new Map<number, Map<number, Embedding>>();
  const vecDs = file.get(`embeddings/${vecName}`);
  if (!vecDs) return result;
  const flat = vecDs.value as ArrayLike<number>;
  const shape = vecDs.shape as number[] | undefined;
  const rows = shape?.[0] ?? 0;
  const dim = shape && shape.length >= 2 ? shape[1] : 0;
  if (!rows || !dim) return result;
  const ot = file.get(`embeddings/${otName}`)?.value ?? [];
  const oid = file.get(`embeddings/${oidName}`)?.value ?? [];
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

const readEmbeddings = (file: any) =>
  readEmbeddingTriple(file, "vectors", "owner_type", "owner_id");

const readCategoryEmbeddings = (file: any) =>
  readEmbeddingTriple(
    file,
    "category_vectors",
    "category_owner_type",
    "category_owner_id",
  );

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
interface PerDetectionMaps {
  idLinks: Map<number, Map<number, [number, number | null]>>;
  idEmbs: Map<number, Map<number, Embedding>>;
  catLinks: Map<number, Map<number, [number, number | null]>>;
  catEmbs: Map<number, Map<number, Embedding>>;
  categoryCatalog: string[];
}

function readPerDetectionMaps(
  file: any,
  emscripten?: unknown,
): PerDetectionMaps {
  return {
    idLinks: readIdentityLinks(file, emscripten),
    idEmbs: readEmbeddings(file),
    catLinks: readCategoryLinks(file, emscripten),
    catEmbs: readCategoryEmbeddings(file),
    categoryCatalog: readCategoryCatalog(file),
  };
}

/** Attach identity + category for one owner type onto its ordered detection list. */
function attachOwnerType(
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
 * Read the columnar `/session_data` group (SLP 2.8) into an in-memory
 * {@link SessionData} for the eager (h5wasm) reader, or `null` when the group is
 * absent (legacy ≤2.7 files) or missing a required struct table. Struct tables go
 * through `normalizeStructDataset` (so both a JS-written flat-2D+`field_names` matrix
 * and a Python-written compound resolve to column records); the `points_3d` /
 * `pred_points_3d` float matrices are read raw (flat buffer + column count).
 */
function readSessionDataEager(
  file: any,
  emscripten?: unknown,
): SessionData | null {
  if (!file.get("session_data")) return null;
  const struct = (name: string): Record<string, any[]> | null => {
    const ds = file.get(`session_data/${name}`);
    return ds ? normalizeStructDataset(ds, emscripten) : null;
  };
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

  return {
    frameGroups,
    instanceGroups,
    members,
    points3d: matrix("points_3d", 3),
    predPoints3d: matrix("pred_points_3d", 4),
    frameGroupMeta: meta("frame_group_meta"),
    instanceGroupMeta: meta("instance_group_meta"),
  };
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
  const raw = attrs[name];
  if (!raw) return [];
  const value = raw.value ?? raw;
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

function readRoisAndBboxes(
  file: any,
  videos: Video[],
  tracks: Track[],
  instances?: Array<Instance | PredictedInstance>,
): { rois: [ROI, number, number][]; bboxes: [BoundingBox, number, number][] } {
  const { rois, migratedBboxes } = readRoisWithMigration(
    file,
    videos,
    tracks,
    instances,
  );
  let bboxes = readBboxes(file, videos, tracks);
  if (bboxes.length === 0 && migratedBboxes.length > 0) {
    bboxes = migratedBboxes;
  }
  return { rois, bboxes };
}

function readRoisWithMigration(
  file: any,
  videos: Video[],
  tracks: Track[],
  instances?: Array<Instance | PredictedInstance>,
): {
  rois: [ROI, number, number][];
  migratedBboxes: [BoundingBox, number, number][];
} {
  const roisDs = file.get("rois");
  if (!roisDs) return { rois: [], migratedBboxes: [] };
  const roisData = normalizeStructDataset(roisDs);
  const annotationTypes = roisData.annotation_type ?? [];
  if (!annotationTypes.length) return { rois: [], migratedBboxes: [] };

  const wkbDs = file.get("roi_wkb");
  if (!wkbDs) return { rois: [], migratedBboxes: [] };
  const wkbFlat: Uint8Array =
    wkbDs.value instanceof Uint8Array
      ? wkbDs.value
      : new Uint8Array(wkbDs.value ?? []);

  // v1.9+: string datasets; fallback to JSON attrs
  const categories = readStringMetadata(
    file,
    "roi_categories",
    roisDs,
    "categories",
  );
  const names = readStringMetadata(file, "roi_names", roisDs, "names");
  const sources = readStringMetadata(file, "roi_sources", roisDs, "sources");

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

  const rois: [ROI, number, number][] = [];
  const migratedBboxes: [BoundingBox, number, number][] = [];

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
        if (instances && instIdx >= 0 && instIdx < instances.length) {
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
        if (instances && instIdx >= 0 && instIdx < instances.length) {
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

function readBboxes(
  file: any,
  _videos: Video[],
  tracks: Track[],
): [BoundingBox, number, number][] {
  const bboxesDs = file.get("bboxes");
  if (!bboxesDs) return [];
  const bboxesData = normalizeStructDataset(bboxesDs);

  // Legacy format: center-based columns (x_center, y_center, width, height)
  const xCenters = bboxesData.x_center ?? [];
  const isLegacy = xCenters.length > 0;

  // New format: corner-based columns (x1, y1, x2, y2)
  const x1s = bboxesData.x1 ?? [];
  const count = isLegacy ? xCenters.length : x1s.length;
  if (!count) return [];

  // v1.9+: string datasets at root level; fallback to JSON attrs on dataset
  const categories = readStringMetadata(
    file,
    "bbox_categories",
    bboxesDs,
    "categories",
  );
  const names = readStringMetadata(file, "bbox_names", bboxesDs, "names");
  const sources = readStringMetadata(file, "bbox_sources", bboxesDs, "sources");

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

  const bboxes: [BoundingBox, number, number][] = [];
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

    let bbox: BoundingBox;
    if (Number.isNaN(scoreVal)) {
      bbox = new UserBoundingBox(options);
    } else {
      bbox = new PredictedBoundingBox({ ...options, score: scoreVal });
    }

    if (instanceIdx >= 0) {
      bbox._instanceIdx = instanceIdx;
    }

    bboxes.push([bbox, videoIdx, frameIdxVal]);
  }
  return bboxes;
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
    const val = ds.value;
    if (Array.isArray(val)) {
      return val.map((v: any) => (typeof v === "string" ? v : String(v ?? "")));
    }
  }
  return readAttrString(dataset, attrName);
}

/**
 * Read score maps from index + data datasets.
 * Returns a Map from annotation index to { scoreMap, height, width }.
 */
function readScoreMaps(
  file: any,
  indexPath: string,
  dataPath: string,
): Map<number, { scoreMap: Float32Array; height: number; width: number }> {
  const result = new Map<
    number,
    { scoreMap: Float32Array; height: number; width: number }
  >();
  const indexDs = file.get(indexPath);
  const dataDs = file.get(dataPath);
  if (!indexDs || !dataDs) return result;

  const indexData = normalizeStructDataset(indexDs);
  const idxCol = indexData.mask_idx ?? indexData.li_idx ?? [];
  const starts = indexData.data_start ?? [];
  const ends = indexData.data_end ?? [];
  const smHeights = indexData.height ?? [];
  const smWidths = indexData.width ?? [];

  const dataFlat: Uint8Array =
    dataDs.value instanceof Uint8Array
      ? dataDs.value
      : new Uint8Array(dataDs.value ?? []);

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

function readMasks(
  file: any,
  _videos: Video[],
  tracks: Track[],
): [SegmentationMask, number, number][] {
  const masksDs = file.get("masks");
  if (!masksDs) return [];
  const masksData = normalizeStructDataset(masksDs);
  const heights = masksData.height ?? [];
  if (!heights.length) return [];

  const rleDs = file.get("mask_rle");
  if (!rleDs) return [];
  const rleFlat: Uint8Array =
    rleDs.value instanceof Uint8Array
      ? rleDs.value
      : new Uint8Array(rleDs.value ?? []);

  // v1.9+: string datasets at root level; fallback to JSON attrs on mask dataset
  const categories = readStringMetadata(
    file,
    "mask_categories",
    masksDs,
    "categories",
  );
  const names = readStringMetadata(file, "mask_names", masksDs, "names");
  const sources = readStringMetadata(file, "mask_sources", masksDs, "sources");

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

  // Read score maps if present
  const scoreMaps = readScoreMaps(
    file,
    "mask_score_map_index",
    "mask_score_maps",
  );

  const masks: [SegmentationMask, number, number][] = [];
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

// Pixel data is eagerly materialized into a standalone Int32Array per
// LabelImage; the h5wasm File is closed by readSlp's finally block before
// Labels is returned. label-image .data therefore has no dependency on the
// source file lifetime — the Python issue addressed by PR #419 (h5py
// Dataset closure invalidated by Labels.__del__) does not apply here.
function readLabelImages(
  file: any,
  _videos: Video[],
  tracks: Track[],
  instances?: Array<Instance | PredictedInstance>,
): [LabelImage, number, number][] {
  const liDs = file.get("label_images");
  if (!liDs) return [];
  const liData = normalizeStructDataset(liDs);
  const videoIndices = liData.video ?? [];
  if (!videoIndices.length) return [];

  const frameIndices = liData.frame_idx ?? [];
  const heights = liData.height ?? [];
  const widths = liData.width ?? [];
  const nObjectsList = liData.n_objects ?? [];
  const objectsStarts = liData.objects_start ?? [];
  const dataStarts = liData.data_start ?? [];
  const dataEnds = liData.data_end ?? [];

  // v1.9+: string datasets; fallback to JSON attrs
  const sources = readStringMetadata(
    file,
    "label_image_sources",
    liDs,
    "sources",
  );

  // v1.9+ columns (may not exist in older files)
  const isPredictedCol = liData.is_predicted ?? [];
  const liScoreCol = liData.score ?? [];

  // v2.1+: spatial metadata columns (default to 1.0/0.0 for old files)
  const liScaleXCol = liData.scale_x ?? [];
  const liScaleYCol = liData.scale_y ?? [];
  const liOffsetXCol = liData.offset_x ?? [];
  const liOffsetYCol = liData.offset_y ?? [];

  // Read pixel data: either blob (1D, v1.8-v2.1) or chunked (3D, v2.2+)
  const dataDs = file.get("label_image_data");
  if (!dataDs) return [];
  const dataShape = dataDs.shape ?? [];
  const isChunked = dataShape.length === 3;
  // For blob format: flat Uint8Array of concatenated zlib-compressed frames
  // For chunked format: 3D Int32Array [T, H, W] (decompression handled by HDF5 lib)
  let dataFlat: Uint8Array = new Uint8Array(0);
  let dataChunked: any = null;
  if (isChunked) {
    dataChunked = dataDs.value; // 3D array or flat typed array with shape metadata
  } else {
    dataFlat =
      dataDs.value instanceof Uint8Array
        ? dataDs.value
        : new Uint8Array(dataDs.value ?? []);
  }

  // Read objects table (may not exist if all label images have 0 objects)
  let objLabelIds: any[] = [];
  let objTrackIndices: any[] = [];
  let objInstanceIndices: any[] = [];
  let objCategories: string[] = [];
  let objNames: string[] = [];
  let objScoreCol: any[] = [];
  let objTrackingScoreCol: any[] = [];

  const objDs = file.get("label_image_objects");
  if (objDs) {
    const objData = normalizeStructDataset(objDs);
    objLabelIds = objData.label_id ?? [];
    objTrackIndices = objData.track ?? [];
    objInstanceIndices = objData.instance ?? [];
    // v1.9+: string datasets at root level
    objCategories = readStringMetadata(
      file,
      "label_image_obj_categories",
      objDs,
      "categories",
    );
    objNames = readStringMetadata(
      file,
      "label_image_obj_names",
      objDs,
      "names",
    );
    objScoreCol = objData.score ?? [];
    objTrackingScoreCol = objData.tracking_score ?? [];
  }

  // Read score maps if present
  const liScoreMaps = readScoreMaps(
    file,
    "label_image_score_map_index",
    "label_image_score_maps",
  );

  const labelImages: [LabelImage, number, number][] = [];
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

      if (instances && instIdx >= 0 && instIdx < instances.length) {
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

function normalizeStructDataset(
  dataset: any,
  module?: unknown,
): Record<string, any[]> {
  if (!dataset) return {};

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

  if (file.get("centroids/x")) {
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
    const centroidsDs = file.get("centroids");
    if (!centroidsDs) return [];
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
