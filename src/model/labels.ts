import {
  LabeledFrame,
  _relinkFromPredicted,
  _resolveMergedIsNegative,
} from "./labeled-frame.js";
import { Instance, PredictedInstance, Track, clonePoint } from "./instance.js";
import type {
  Point,
  PointsArray,
  PredictedPoint,
  PredictedPointsArray,
} from "./instance.js";
import { Skeleton, Node, Edge, Symmetry } from "./skeleton.js";
import { SuggestionFrame } from "./suggestions.js";
import { Video } from "./video.js";
import {
  type RecordingSession,
  cloneRecordingSession,
  injectSessionFrameResolver,
} from "./camera.js";
import type { Identity } from "./identity.js";
import { toDict } from "../codecs/dictionary.js";
import { labelsFromNumpy } from "../codecs/numpy.js";
import { type LazyDataStore, LazyFrameList } from "./lazy.js";
import { LabelsSet } from "./labels-set.js";
import {
  SkeletonMatcher,
  VideoMatcher,
  TrackMatcher,
  InstanceMatcher,
  SkeletonMatchMethod,
  VideoMatchMethod,
  TrackMatchMethod,
  InstanceMatchMethod,
  ErrorMode,
  MergeResult,
  MatchResult,
  ConflictResolution,
  MergeError,
  SkeletonMismatchError,
  isSameFile,
  toSkeletonMatchMethod,
  toVideoMatchMethod,
  toTrackMatchMethod,
  toInstanceMatchMethod,
  toErrorMode,
} from "./matching.js";
import type { ROI } from "./roi.js";
import type { SegmentationMask } from "./mask.js";
import type { BoundingBox } from "./bbox.js";
import type { Centroid } from "./centroid.js";
import type { LabelImage } from "./label-image.js";

/** Package version recorded in merge provenance (mirrors `sleap_io.__version__`). */
const SLEAP_IO_VERSION = "0.3.1";

/** Basename: final path component, splitting on BOTH "/" and "\" (Python `Path(f).name`). */
function pathBasename(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1];
}

/**
 * Coerce a skeleton-matcher argument (mirrors the merge/match coercion blocks):
 * `null`/`undefined` -> default `SkeletonMatcher(STRUCTURE)`; a string -> a matcher
 * built from the validated enum (THROWS on an unknown string); else used as-is.
 */
function coerceSkeletonMatcher(
  x: string | SkeletonMatcher | null | undefined,
): SkeletonMatcher {
  if (x == null) {
    return new SkeletonMatcher(SkeletonMatchMethod.STRUCTURE);
  }
  if (typeof x === "string") {
    return new SkeletonMatcher(toSkeletonMatchMethod(x));
  }
  return x;
}

/** Coerce a video-matcher argument: `null` -> default `VideoMatcher()` (AUTO). */
function coerceVideoMatcher(
  x: string | VideoMatcher | null | undefined,
): VideoMatcher {
  if (x == null) {
    return new VideoMatcher();
  }
  if (typeof x === "string") {
    return new VideoMatcher(toVideoMatchMethod(x));
  }
  return x;
}

/**
 * Coerce a track-matcher argument: `null` -> default `TrackMatcher()`, whose
 * constructor default is now IDENTITY (matches only the same Track object;
 * correctness-first). Pass `"name"` to opt in to name-based matching.
 */
function coerceTrackMatcher(
  x: string | TrackMatcher | null | undefined,
): TrackMatcher {
  if (x == null) {
    return new TrackMatcher();
  }
  if (typeof x === "string") {
    return new TrackMatcher(toTrackMatchMethod(x));
  }
  return x;
}

/** Coerce an instance-matcher argument: `null` -> default `InstanceMatcher()` (SPATIAL/5.0). */
function coerceInstanceMatcher(
  x: string | InstanceMatcher | null | undefined,
): InstanceMatcher {
  if (x == null) {
    return new InstanceMatcher();
  }
  if (typeof x === "string") {
    return new InstanceMatcher(toInstanceMatchMethod(x));
  }
  return x;
}

/**
 * Produce a naive local-time ISO-8601 string WITHOUT a trailing `Z`, mirroring
 * Python `datetime.now().isoformat()` (local wall-clock, no timezone suffix,
 * with sub-second precision). The workflow-script ban on time APIs does NOT
 * apply to this runtime code.
 */
function localIsoStringWithoutZ(): string {
  const d = new Date();
  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}` +
    `.${pad(d.getMilliseconds(), 3)}000`
  );
}

/** Python `repr()` of a filename (string) or filename-list (ImageVideo). */
function filenameRepr(filename: string | string[]): string {
  const reprStr = (s: string): string => {
    // Mirror Python str repr: prefer single quotes; escape backslashes and the
    // chosen quote.
    const escaped = s.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
    return `'${escaped}'`;
  };
  if (Array.isArray(filename)) {
    return `[${filename.map(reprStr).join(", ")}]`;
  }
  return reprStr(filename);
}

/** Shared shape for annotations that support deferred instance resolution. */
interface DeferredInstanceRef {
  _instanceIdx: number | null;
  instance: Instance | null;
}

export class Labels {
  labeledFrames: LabeledFrame[];
  videos: Video[];
  skeletons: Skeleton[];
  tracks: Track[];
  suggestions: SuggestionFrame[];
  sessions: RecordingSession[];
  provenance: Record<string, unknown>;
  identities: Identity[];

  // Static ROIs: not tied to any specific frame (e.g., arena boundaries).
  _staticRois: ROI[];

  /** @internal Lazy frame list for on-demand materialization. */
  _lazyFrameList: LazyFrameList | null = null;
  /** @internal Lazy data store holding raw HDF5 data. */
  _lazyDataStore: LazyDataStore | null = null;

  // Index caches (excluded from serialization, rebuilt on demand)
  private _frameIndex: Map<Video, Map<number, LabeledFrame>> | null = null;
  private _frameIndexLen: number = -1;
  private _trackIndex: Map<
    Video,
    Map<
      Track,
      Array<
        | Centroid
        | BoundingBox
        | SegmentationMask
        | ROI
        | LabelImage
        | Instance
        | PredictedInstance
      >
    >
  > | null = null;
  private _trackIndexLen: number = -1;

  constructor(options?: {
    labeledFrames?: LabeledFrame[];
    videos?: Video[];
    skeletons?: Skeleton[];
    tracks?: Track[];
    suggestions?: SuggestionFrame[];
    sessions?: RecordingSession[];
    provenance?: Record<string, unknown>;
    rois?: ROI[];
    identities?: Identity[];
  }) {
    this.labeledFrames = options?.labeledFrames ?? [];
    this.videos = options?.videos ?? [];
    this.skeletons = options?.skeletons ?? [];
    this.tracks = options?.tracks ?? [];
    this.suggestions = options?.suggestions ?? [];
    this.sessions = options?.sessions ?? [];
    this.provenance = options?.provenance ?? {};
    this._staticRois = options?.rois ?? [];
    this.identities = options?.identities ?? [];

    if (!this.videos.length && this.labeledFrames.length) {
      const uniqueVideos = new Map<string | Video, Video>();
      for (const frame of this.labeledFrames) {
        uniqueVideos.set(frame.video, frame.video);
      }
      this.videos = Array.from(uniqueVideos.values());
    }

    // Register instance skeletons and tracks (Python `update()` loop). Skeletons
    // are deduplicated structurally via `_registerSkeleton`, so two
    // structurally-equal but distinct `Skeleton` objects collapse to a single
    // canonical entry (Python #447). Explicitly-provided `skeletons` are kept
    // (by identity) and instance skeletons dedup against them.
    // One membership set shared across both passes and every frame — rebuilding
    // `new Set(this.tracks)` per frame was O(frames × tracks) and dominated load
    // time on many-track projects (see _collectAnnotationTracks).
    const existingTracks = new Set(this.tracks);
    if (!this._lazyFrameList) {
      for (const frame of this.labeledFrames) {
        for (const instance of frame.instances) {
          this._registerSkeleton(instance);
          if (instance.track && !existingTracks.has(instance.track)) {
            existingTracks.add(instance.track);
            this.tracks.push(instance.track);
          }
        }
      }
      // Collect tracks from annotations already on frames (reuse the set).
      for (const lf of this.labeledFrames) {
        this._collectAnnotationTracks(lf, existingTracks);
      }
    }
    // Collect tracks from static ROIs
    for (const roi of this._staticRois) {
      if (roi.track && !existingTracks.has(roi.track)) {
        existingTracks.add(roi.track);
        this.tracks.push(roi.track);
      }
    }
  }

  /**
   * @deprecated Transitional. Only populated when a read entrypoint is called
   * with `{ rawSessions: true }`; `undefined` per entry otherwise. Prefer the
   * faithful typed session model. See `RecordingSession.rawJson`.
   *
   * The verbatim, as-read `sessions_json` dicts, index-aligned with `sessions`.
   *
   * Derived (no stored field) so it cannot desync in length from `sessions`.
   * `rawSessionsJson[i]` corresponds to `sessions[i]`, and is `undefined` for
   * sessions constructed in-memory (or read without `rawSessions`). Each entry
   * is the untouched `JSON.parse` result of that session's `sessions_json`
   * entry — a read-time snapshot, never itself re-written to disk. See
   * `RecordingSession.rawJson`.
   */
  get rawSessionsJson(): Array<Record<string, unknown> | undefined> {
    return this.sessions.map((s) => s.rawJson);
  }

  /**
   * Get the labeled frame at global index `i`, materializing only that frame
   * under the lazy reader. Backs the session grouping's lazy frame resolver
   * (`injectSessionFrameResolver`): resolving one cross-camera ref materializes
   * a single frame via `LazyFrameList.at(i)` rather than the whole table, so an
   * untouched session round-trips with zero frame materialization.
   */
  frameAt(i: number): LabeledFrame | undefined {
    return this._lazyFrameList
      ? this._lazyFrameList.at(i)
      : this.labeledFrames[i];
  }

  /**
   * Drop the cached (materialized) frame at `i` from a lazy `Labels`, so a
   * windowed read→transform→write sweep over a huge file stays memory-bounded
   * without reaching into internals. No-op on an eager `Labels`. See #207.
   */
  releaseFrame(i: number): void {
    this._lazyFrameList?.release(i);
  }

  /** Drop cached lazy frames in the half-open range `[start, end)`. No-op if eager. */
  releaseFrameWindow(start: number, end: number): void {
    this._lazyFrameList?.releaseWindow(start, end);
  }

  /**
   * Bound how many materialized frames a lazy `Labels` keeps cached (0 =
   * unbounded). When exceeded, oldest-first eviction keeps peak memory flat
   * across a sequential sweep. No-op / returns 0 on an eager `Labels`.
   */
  get frameCacheLimit(): number {
    return this._lazyFrameList?.cacheLimit ?? 0;
  }

  set frameCacheLimit(n: number) {
    if (this._lazyFrameList) this._lazyFrameList.cacheLimit = Math.max(0, n);
  }

  /**
   * Collect tracks from annotations on a frame into this.tracks.
   *
   * `seen` lets a caller iterating many frames share one membership set instead
   * of rebuilding `new Set(this.tracks)` per frame — the difference between
   * O(frames) and O(frames × tracks) on files with many tracks (e.g. a 21.8k-
   * frame / 7k-track project spent seconds here rebuilding the set). When
   * omitted (single-frame `append`), the set is built from the current tracks.
   * The set must stay in sync with `this.tracks`: every push here also adds to
   * it, so a later frame sees tracks an earlier one contributed.
   */
  private _collectAnnotationTracks(lf: LabeledFrame, seen?: Set<Track>): void {
    const existing = seen ?? new Set(this.tracks);
    const add = (track: Track | null | undefined) => {
      if (track && !existing.has(track)) {
        existing.add(track);
        this.tracks.push(track);
      }
    };
    for (const c of lf.centroids) add(c.track);
    for (const b of lf.bboxes) add(b.track);
    for (const m of lf.masks) add(m.track);
    for (const r of lf.rois) add(r.track);
    for (const li of lf.labelImages) {
      for (const info of li.objects.values()) add(info.track);
    }
  }

  /** Raise if Labels is lazy-loaded. */
  private _checkNotLazy(operation: string): void {
    if (this.isLazy) {
      throw new Error(
        `Cannot ${operation} on lazy-loaded Labels.\n\n` +
          `To use, first materialize:\n` +
          `    labels.materialize();\n` +
          `    labels.${operation}(...);`,
      );
    }
  }

  /** Clear all cached indices so they rebuild on next access. */
  private _invalidateIndices(): void {
    this._frameIndex = null;
    this._frameIndexLen = -1;
    this._trackIndex = null;
    this._trackIndexLen = -1;
  }

  /** Build or return the frame index, rebuilding if stale. */
  private _ensureFrameIndex(): Map<Video, Map<number, LabeledFrame>> {
    if (this._lazyFrameList) this.materialize();
    const n = this.labeledFrames.length;
    if (this._frameIndex !== null && this._frameIndexLen === n) {
      return this._frameIndex;
    }
    this._frameIndex = new Map();
    for (const lf of this.labeledFrames) {
      let videoMap = this._frameIndex.get(lf.video);
      if (!videoMap) {
        videoMap = new Map();
        this._frameIndex.set(lf.video, videoMap);
      }
      if (videoMap.has(lf.frameIdx)) {
        console.warn(
          `Duplicate LabeledFrame for video=${lf.video}, frame_idx=${lf.frameIdx}. Using last occurrence.`,
        );
      }
      videoMap.set(lf.frameIdx, lf);
    }
    this._frameIndexLen = n;
    return this._frameIndex;
  }

  /** Build or return the track index, rebuilding if stale. */
  private _ensureTrackIndex(): Map<
    Video,
    Map<
      Track,
      Array<
        | Centroid
        | BoundingBox
        | SegmentationMask
        | ROI
        | LabelImage
        | Instance
        | PredictedInstance
      >
    >
  > {
    if (this._lazyFrameList) this.materialize();
    const n = this.labeledFrames.length;
    if (this._trackIndex !== null && this._trackIndexLen === n) {
      return this._trackIndex;
    }
    this._trackIndex = new Map();
    for (const lf of this.labeledFrames) {
      let videoMap = this._trackIndex.get(lf.video);
      if (!videoMap) {
        videoMap = new Map();
        this._trackIndex.set(lf.video, videoMap);
      }
      for (const ann of [
        ...lf.centroids,
        ...lf.bboxes,
        ...lf.masks,
        ...lf.rois,
        ...lf.instances,
      ]) {
        const track = (ann as { track?: Track | null }).track;
        if (track) {
          let list = videoMap.get(track);
          if (!list) {
            list = [];
            videoMap.set(track, list);
          }
          list.push(ann);
        }
      }
      for (const li of lf.labelImages) {
        for (const info of li.objects.values()) {
          if (info.track) {
            let list = videoMap.get(info.track);
            if (!list) {
              list = [];
              videoMap.set(info.track, list);
            }
            list.push(li);
          }
        }
      }
    }
    // Build annotation -> frameIdx map for sorting
    const annFrameIdx = new Map<unknown, number>();
    for (const lf of this.labeledFrames) {
      for (const ann of [
        ...lf.centroids,
        ...lf.bboxes,
        ...lf.masks,
        ...lf.rois,
        ...lf.instances,
      ]) {
        annFrameIdx.set(ann, lf.frameIdx);
      }
      for (const li of lf.labelImages) {
        annFrameIdx.set(li, lf.frameIdx);
      }
    }
    // Sort each list by frameIdx
    for (const videoMap of this._trackIndex.values()) {
      for (const list of videoMap.values()) {
        list.sort(
          (a, b) => (annFrameIdx.get(a) ?? 0) - (annFrameIdx.get(b) ?? 0),
        );
      }
    }
    this._trackIndexLen = n;
    return this._trackIndex;
  }

  /**
   * O(1) lookup of a LabeledFrame by video and frame index.
   *
   * The index is rebuilt lazily. If you mutate frames directly (e.g.,
   * `lf.frameIdx = newIdx`) without calling `reindex()`, the lookup may
   * return stale results.
   */
  getFrame(video: Video, frameIdx: number): LabeledFrame | null {
    this._checkNotLazy("getFrame");
    return this._ensureFrameIndex().get(video)?.get(frameIdx) ?? null;
  }

  /**
   * O(1) lookup of all annotations for a track in a video, sorted by frameIdx.
   *
   * The index is rebuilt lazily. If you mutate frames directly (e.g.,
   * `lf.frameIdx = newIdx`) without calling `reindex()`, the lookup may
   * return stale results.
   */
  getTrackAnnotations(
    video: Video,
    track: Track,
  ): Array<
    | Centroid
    | BoundingBox
    | SegmentationMask
    | ROI
    | LabelImage
    | Instance
    | PredictedInstance
  > {
    this._checkNotLazy("getTrackAnnotations");
    return this._ensureTrackIndex().get(video)?.get(track) ?? [];
  }

  /** Force rebuild of all indices on next access. */
  reindex(): void {
    this._invalidateIndices();
  }

  /**
   * Remove all predicted instances and predicted annotations from all frames.
   *
   * Mirrors Python `Labels.remove_predictions` (labels.py:1684-1710).
   *
   * @param clean - If `true` (the default), also prune empty frames and unused
   *   skeletons/tracks via {@link clean} with `frames`, `skeletons`, `tracks`
   *   enabled and `emptyInstances`/`videos` disabled. Does NOT remove videos
   *   with no labeled frames, nor instances with no visible points.
   */
  removePredictions(clean = true): void {
    if (this._lazyFrameList) this.materialize();
    for (const lf of this.labeledFrames) {
      lf.removePredictions();
    }
    this._invalidateIndices();

    if (clean) {
      this.clean({
        frames: true,
        emptyInstances: false,
        skeletons: true,
        tracks: true,
        videos: false,
      });
    }
  }

  /**
   * Collapse structurally-equal skeletons into a single canonical entry.
   *
   * Skeletons are partitioned via {@link Skeleton.matches} called with
   * `requireSameOrder: true` (same node count, same node names IN THE SAME
   * ORDER, same edge set, and same symmetry set). The first member of each
   * equivalence class is kept as canonical; the rest are removed from
   * `this.skeletons` and every instance referencing a non-canonical skeleton is
   * reassigned to the canonical via direct property assignment. Points are
   * positional and are NOT remapped, so order-identical matching is required to
   * keep reassignment safe.
   *
   * Note: skeleton `name` is not part of `matches()` — the canonical's name wins.
   *
   * Note: skeletons that share node names but differ in node ORDER are treated
   * as distinct here (they are not collapsed), since collapsing them would
   * misalign instance points.
   *
   * Legacy `.slp` files often carry content-duplicate skeletons (a pre-1.5 Python
   * sleap quirk). Call this method after `loadSlp` if you want them collapsed —
   * it is not run automatically on load.
   *
   * In lazy mode this forces full materialization, consistent with other Labels
   * mutators.
   *
   * @returns Number of duplicate skeletons collapsed (0 if none).
   */
  dedupSkeletons(): { canonicalized: number } {
    if (this._lazyFrameList) this.materialize();
    if (this.skeletons.length <= 1) return { canonicalized: 0 };

    const canonicals: Skeleton[] = [];
    const canonicalFor = new Map<Skeleton, Skeleton>();
    for (const skel of this.skeletons) {
      // requireSameOrder: true — dedup reassigns instances to the canonical
      // skeleton without remapping point positions, so only skeletons whose
      // node ORDER also matches can be safely collapsed (points are positional).
      const existing = canonicals.find((c) =>
        skel.matches(c, { requireSameOrder: true }),
      );
      if (existing) {
        canonicalFor.set(skel, existing);
      } else {
        canonicals.push(skel);
        canonicalFor.set(skel, skel);
      }
    }

    const canonicalized = this.skeletons.length - canonicals.length;
    if (canonicalized === 0) return { canonicalized: 0 };

    this.skeletons = canonicals;
    for (const frame of this.labeledFrames) {
      for (const inst of frame.instances) {
        const canon = canonicalFor.get(inst.skeleton);
        if (canon && inst.skeleton !== canon) inst.skeleton = canon;
      }
    }

    return { canonicalized };
  }

  /** Flat view of all centroids across all frames. */
  get centroids(): Centroid[] {
    if (this._lazyFrameList && this._lazyDataStore) {
      const byFrame = this._lazyDataStore._centroidByFrame;
      const undist = this._lazyDataStore._undistributedCentroids;
      return [...undist, ...[...byFrame.values()].flat()];
    }
    return this.labeledFrames.flatMap((lf) => lf.centroids);
  }

  /** Flat view of all bounding boxes across all frames. */
  get bboxes(): BoundingBox[] {
    if (this._lazyFrameList && this._lazyDataStore) {
      const byFrame = this._lazyDataStore._bboxByFrame;
      const undist = this._lazyDataStore._undistributedBboxes;
      return [...undist, ...[...byFrame.values()].flat()];
    }
    return this.labeledFrames.flatMap((lf) => lf.bboxes);
  }

  /** Flat view of all segmentation masks across all frames. */
  get masks(): SegmentationMask[] {
    if (this._lazyFrameList && this._lazyDataStore) {
      const byFrame = this._lazyDataStore._maskByFrame;
      const undist = this._lazyDataStore._undistributedMasks;
      return [...undist, ...[...byFrame.values()].flat()];
    }
    return this.labeledFrames.flatMap((lf) => lf.masks);
  }

  /** Flat view of all label images across all frames. */
  get labelImages(): LabelImage[] {
    if (this._lazyFrameList && this._lazyDataStore) {
      const byFrame = this._lazyDataStore._labelImageByFrame;
      const undist = this._lazyDataStore._undistributedLabelImages;
      return [...undist, ...[...byFrame.values()].flat()];
    }
    return this.labeledFrames.flatMap((lf) => lf.labelImages);
  }

  /** Flat view of all ROIs across all frames and static ROIs. */
  get rois(): ROI[] {
    if (this._lazyFrameList && this._lazyDataStore) {
      const byFrame = this._lazyDataStore._roiByFrame;
      const undist = this._lazyDataStore._undistributedRois;
      return [...undist, ...[...byFrame.values()].flat()];
    }
    return [
      ...this._staticRois,
      ...this.labeledFrames.flatMap((lf) => lf.rois),
    ];
  }

  /** Whether this Labels instance is in lazy mode. */
  get isLazy(): boolean {
    return this._lazyFrameList !== null;
  }

  /**
   * Materialize all lazy frames, converting to eager mode.
   * No-op if already eager.
   */
  materialize(): void {
    if (!this._lazyFrameList) return;
    const store = this._lazyDataStore;
    this.labeledFrames = this._lazyFrameList.toArray();
    this._lazyFrameList = null;
    this._lazyDataStore = null;

    // Resolve deferred instance references on per-frame annotations
    const allInstances = this.labeledFrames.flatMap((f) => f.instances);
    for (const lf of this.labeledFrames) {
      for (const ann of [
        ...lf.centroids,
        ...lf.bboxes,
        ...lf.masks,
        ...lf.rois,
      ] as DeferredInstanceRef[]) {
        if (
          ann._instanceIdx !== null &&
          ann._instanceIdx >= 0 &&
          ann._instanceIdx < allInstances.length
        ) {
          ann.instance = allInstances[ann._instanceIdx];
          ann._instanceIdx = null;
        }
      }
      for (const li of lf.labelImages) {
        if (li._objectInstanceIdxs) {
          for (const [labelId, instIdx] of li._objectInstanceIdxs) {
            const obj = li.objects.get(labelId);
            if (obj && instIdx >= 0 && instIdx < allInstances.length) {
              obj.instance = allInstances[instIdx];
            }
          }
          li._objectInstanceIdxs = null;
        }
      }
    }

    // Keep undistributed ROIs as static ROIs
    if (store) {
      this._staticRois = store._undistributedRois;
    }
  }

  /**
   * Frames with any user labeling: user instances, a negative (background)
   * flag, or user centroids / boxes / masks / ROIs / label images. sleap-io's
   * `user_labeled_frames` — what `embed: "user"` embeds.
   */
  get userLabeledFrames(): LabeledFrame[] {
    if (this._lazyFrameList) this.materialize();
    return this.labeledFrames.filter((f) => f.isUserLabeled);
  }

  get negativeFrames(): LabeledFrame[] {
    if (this._lazyFrameList) this.materialize();
    return this.labeledFrames.filter((f) => f.isNegative);
  }

  get video(): Video {
    if (!this.videos.length) {
      throw new Error("No videos available on Labels.");
    }
    return this.videos[0];
  }

  get length(): number {
    if (this._lazyFrameList) return this._lazyFrameList.length;
    return this.labeledFrames.length;
  }

  [Symbol.iterator](): Iterator<LabeledFrame> {
    if (this._lazyFrameList) return this._lazyFrameList[Symbol.iterator]();
    return this.labeledFrames[Symbol.iterator]();
  }

  get instances(): Array<Instance | PredictedInstance> {
    if (this._lazyFrameList) this.materialize();
    return this.labeledFrames.flatMap((frame) => frame.instances);
  }

  /**
   * Search for labeled frames given video and/or frame index.
   *
   * A foreign `Video` instance or filename (`string`/`URL`) is resolved to the
   * matching `Video` in `this.videos` via {@link _resolveVideo} (SYNC; see its
   * documented divergence from `matchVideo`), so an object created independently
   * still works. When the video does not resolve to a project video the foreign
   * reference is used as-is, so identity-based lookups yield no results.
   */
  find(options: {
    video?: Video | string | URL;
    frameIdx?: number;
  }): LabeledFrame[] {
    if (this._lazyFrameList) this.materialize();
    // Canonicalize a foreign Video / filename to the matching project Video.
    const resolved =
      options.video !== undefined
        ? (this._resolveVideo(options.video) ?? undefined)
        : undefined;
    // Fast path: O(1) lookup when both video and frameIdx are specified
    if (resolved !== undefined && options.frameIdx !== undefined) {
      const frame = this.getFrame(resolved, options.frameIdx);
      return frame ? [frame] : [];
    }
    return this.labeledFrames.filter((frame) => {
      if (resolved && frame.video !== resolved) {
        return false;
      }
      if (
        options.frameIdx !== undefined &&
        frame.frameIdx !== options.frameIdx
      ) {
        return false;
      }
      return true;
    });
  }

  addVideo(video: Video): void {
    if (!this.videos.includes(video)) {
      this.videos.push(video);
    }
  }

  /**
   * Append a labeled frame to the labels.
   *
   * Registers the frame's video, the skeletons and tracks of its instances, and
   * the tracks of its nested annotations. Skeletons are registered via
   * {@link _registerSkeleton}, so an instance referencing a structurally-equal
   * but distinct `Skeleton` is rebound to the existing canonical one instead of
   * growing `this.skeletons` (Python #447).
   *
   * Mirrors Python `Labels.append` (PR #447).
   */
  append(frame: LabeledFrame): void {
    if (this._lazyFrameList) this.materialize();
    this.labeledFrames.push(frame);
    this._invalidateIndices();
    this.addVideo(frame.video);
    for (const inst of frame.instances) {
      this._registerSkeleton(inst);
      if (inst.track != null && !this.tracks.includes(inst.track)) {
        this.tracks.push(inst.track);
      }
    }
    this._collectAnnotationTracks(frame);
  }

  /**
   * Append multiple labeled frames to the labels.
   *
   * Like calling {@link append} on each frame in `frames`: registers each
   * frame's video, the skeletons and tracks of its instances (deduplicating
   * structurally-equal skeletons via {@link _registerSkeleton}, Python #447),
   * and the tracks of its nested annotations.
   *
   * Mirrors Python `Labels.extend` (PR #447).
   */
  extend(frames: LabeledFrame[]): void {
    if (this._lazyFrameList) this.materialize();
    // Shared track set: `this.tracks.includes` per instance was O(n × tracks).
    const seenTracks = new Set(this.tracks);
    for (const frame of frames) {
      this.labeledFrames.push(frame);
      this.addVideo(frame.video);
      for (const inst of frame.instances) {
        this._registerSkeleton(inst);
        if (inst.track != null && !seenTracks.has(inst.track)) {
          seenTracks.add(inst.track);
          this.tracks.push(inst.track);
        }
      }
      this._collectAnnotationTracks(frame, seenTracks);
    }
    this._invalidateIndices();
  }

  /**
   * Add a static ROI (not tied to any specific frame, e.g., an arena boundary).
   *
   * Registers the ROI's track (if any) on `this.tracks`. Use
   * `lf.append(roi)` on a `LabeledFrame` to add a frame-bound ROI instead.
   */
  addStaticRoi(roi: ROI): void {
    this._staticRois.push(roi);
    if (roi.track && !this.tracks.includes(roi.track)) {
      this.tracks.push(roi.track);
    }
  }

  toDict(options?: { video?: Video | number; skipEmptyFrames?: boolean }) {
    if (this._lazyFrameList) this.materialize();
    return toDict(this, options);
  }

  /** Static ROIs (not attached to any LabeledFrame). */
  get staticRois(): ROI[] {
    return [...this._staticRois];
  }

  /** Frame-bound ROIs (attached to LabeledFrames). */
  get temporalRois(): ROI[] {
    return this.labeledFrames.flatMap((lf) => lf.rois);
  }

  /**
   * Filter ROIs across the Labels object.
   *
   * Filtering rule (matches sibling getters like `getMasks`/`getBboxes`):
   *   - Frame-aware filters (`video` or `frameIdx`) walk only `labeledFrames`.
   *     Static ROIs are excluded from these results.
   *   - Otherwise (no filter, or only `category`/`track`/`instance`/`predicted`)
   *     the search runs over `this.rois` — the union of static + frame-bound.
   *
   * To access static ROIs directly, use `staticRois`. To access only frame-bound
   * ROIs across all frames, use `temporalRois`.
   */
  getRois(filters?: {
    video?: Video | string | URL;
    frameIdx?: number;
    category?: string;
    track?: Track;
    instance?: Instance | PredictedInstance;
    predicted?: boolean;
  }): ROI[] {
    if (!filters) return [...this.rois];
    // Canonicalize a foreign Video / filename via the SYNC resolver (see
    // `_resolveVideo` for its documented divergence from `matchVideo`).
    const video =
      filters.video !== undefined
        ? (this._resolveVideo(filters.video) ?? undefined)
        : undefined;
    let results: ROI[];
    if (video !== undefined && filters.frameIdx !== undefined) {
      const lf = this.getFrame(video, filters.frameIdx);
      results = lf ? lf.rois : [];
    } else if (video !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.video === video) results.push(...lf.rois);
      }
    } else if (filters.frameIdx !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.frameIdx === filters.frameIdx) results.push(...lf.rois);
      }
    } else {
      results = this.rois;
    }
    if (filters.category !== undefined) {
      results = results.filter((r) => r.category === filters.category);
    }
    if (filters.track !== undefined) {
      results = results.filter((r) => r.track === filters.track);
    }
    if (filters.instance !== undefined) {
      results = results.filter((r) => r.instance === filters.instance);
    }
    if (filters.predicted !== undefined) {
      results = results.filter((r) => r.isPredicted === filters.predicted);
    }
    return results;
  }

  getMasks(filters?: {
    video?: Video | string | URL;
    frameIdx?: number;
    category?: string;
    track?: Track;
    instance?: Instance | PredictedInstance;
    predicted?: boolean;
  }): SegmentationMask[] {
    if (!filters) return [...this.masks];
    // Canonicalize a foreign Video / filename via the SYNC resolver (see
    // `_resolveVideo` for its documented divergence from `matchVideo`).
    const video =
      filters.video !== undefined
        ? (this._resolveVideo(filters.video) ?? undefined)
        : undefined;
    let results: SegmentationMask[];
    if (video !== undefined && filters.frameIdx !== undefined) {
      const lf = this.getFrame(video, filters.frameIdx);
      results = lf ? lf.masks : [];
    } else if (video !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.video === video) results.push(...lf.masks);
      }
    } else if (filters.frameIdx !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.frameIdx === filters.frameIdx) results.push(...lf.masks);
      }
    } else {
      results = this.masks;
    }
    if (filters.category !== undefined) {
      results = results.filter((m) => m.category === filters.category);
    }
    if (filters.track !== undefined) {
      results = results.filter((m) => m.track === filters.track);
    }
    if (filters.instance !== undefined) {
      results = results.filter((m) => m.instance === filters.instance);
    }
    if (filters.predicted !== undefined) {
      results = results.filter((m) => m.isPredicted === filters.predicted);
    }
    return results;
  }

  getBboxes(filters?: {
    video?: Video | string | URL;
    frameIdx?: number;
    category?: string;
    track?: Track;
    instance?: Instance;
    predicted?: boolean;
  }): BoundingBox[] {
    if (!filters) return [...this.bboxes];
    // Canonicalize a foreign Video / filename via the SYNC resolver (see
    // `_resolveVideo` for its documented divergence from `matchVideo`).
    const video =
      filters.video !== undefined
        ? (this._resolveVideo(filters.video) ?? undefined)
        : undefined;
    let results: BoundingBox[];
    if (video !== undefined && filters.frameIdx !== undefined) {
      const lf = this.getFrame(video, filters.frameIdx);
      results = lf ? lf.bboxes : [];
    } else if (video !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.video === video) results.push(...lf.bboxes);
      }
    } else if (filters.frameIdx !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.frameIdx === filters.frameIdx) results.push(...lf.bboxes);
      }
    } else {
      results = this.bboxes;
    }
    if (filters.category !== undefined) {
      results = results.filter((b) => b.category === filters.category);
    }
    if (filters.track !== undefined) {
      results = results.filter((b) => b.track === filters.track);
    }
    if (filters.instance !== undefined) {
      results = results.filter((b) => b.instance === filters.instance);
    }
    if (filters.predicted !== undefined) {
      results = results.filter((b) => b.isPredicted === filters.predicted);
    }
    return results;
  }

  getCentroids(filters?: {
    video?: Video | string | URL;
    frameIdx?: number;
    category?: string;
    track?: Track;
    instance?: Instance | PredictedInstance;
    predicted?: boolean;
  }): Centroid[] {
    if (!filters) return [...this.centroids];
    // Canonicalize a foreign Video / filename via the SYNC resolver (see
    // `_resolveVideo` for its documented divergence from `matchVideo`).
    const video =
      filters.video !== undefined
        ? (this._resolveVideo(filters.video) ?? undefined)
        : undefined;
    let results: Centroid[];
    if (video !== undefined && filters.frameIdx !== undefined) {
      const lf = this.getFrame(video, filters.frameIdx);
      results = lf ? lf.centroids : [];
    } else if (video !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.video === video) results.push(...lf.centroids);
      }
    } else if (filters.frameIdx !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.frameIdx === filters.frameIdx) results.push(...lf.centroids);
      }
    } else {
      results = this.centroids;
    }
    if (filters.category !== undefined) {
      results = results.filter((c) => c.category === filters.category);
    }
    if (filters.track !== undefined) {
      results = results.filter((c) => c.track === filters.track);
    }
    if (filters.instance !== undefined) {
      results = results.filter((c) => c.instance === filters.instance);
    }
    if (filters.predicted !== undefined) {
      results = results.filter((c) => c.isPredicted === filters.predicted);
    }
    return results;
  }

  getLabelImages(filters?: {
    video?: Video | string | URL;
    frameIdx?: number;
    track?: Track;
    category?: string;
    predicted?: boolean;
  }): LabelImage[] {
    if (!filters) return [...this.labelImages];
    // Canonicalize a foreign Video / filename via the SYNC resolver (see
    // `_resolveVideo` for its documented divergence from `matchVideo`).
    const video =
      filters.video !== undefined
        ? (this._resolveVideo(filters.video) ?? undefined)
        : undefined;
    let results: LabelImage[];
    if (video !== undefined && filters.frameIdx !== undefined) {
      const lf = this.getFrame(video, filters.frameIdx);
      results = lf ? lf.labelImages : [];
    } else if (video !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.video === video) results.push(...lf.labelImages);
      }
    } else if (filters.frameIdx !== undefined) {
      results = [];
      for (const lf of this.labeledFrames) {
        if (lf.frameIdx === filters.frameIdx) results.push(...lf.labelImages);
      }
    } else {
      results = this.labelImages;
    }
    if (filters.track !== undefined) {
      results = results.filter((li) =>
        Array.from(li.objects.values()).some(
          (info) => info.track === filters.track,
        ),
      );
    }
    if (filters.category !== undefined) {
      results = results.filter((li) =>
        Array.from(li.objects.values()).some(
          (info) => info.category === filters.category,
        ),
      );
    }
    if (filters.predicted !== undefined) {
      results = results.filter((li) => li.isPredicted === filters.predicted);
    }
    return results;
  }

  /**
   * Replace videos and update all references across the Labels object.
   *
   * Provide either `oldVideos`/`newVideos` arrays or a `videoMap`.
   * If only `newVideos` is provided and its length matches `this.videos`,
   * the current videos are used as `oldVideos`.
   */
  replaceVideos(options: {
    oldVideos?: Video[];
    newVideos?: Video[];
    videoMap?: Map<Video, Video>;
  }): void {
    if (this._lazyFrameList) this.materialize();

    let { oldVideos, newVideos, videoMap } = options;

    if (!oldVideos && newVideos && newVideos.length === this.videos.length) {
      oldVideos = this.videos;
    }

    if (!videoMap) {
      if (!oldVideos || !newVideos) {
        throw new Error("Must provide oldVideos/newVideos or videoMap.");
      }
      videoMap = new Map<Video, Video>();
      for (let i = 0; i < oldVideos.length; i++) {
        videoMap.set(oldVideos[i], newVideos[i]);
      }
    }

    for (const frame of this.labeledFrames) {
      const mapped = videoMap.get(frame.video);
      if (mapped) frame.video = mapped;

      // Update ROIs that have video references
      for (const r of frame.rois) {
        if (r.video && videoMap.has(r.video)) r.video = videoMap.get(r.video)!;
      }
    }

    for (const suggestion of this.suggestions) {
      const mapped = videoMap.get(suggestion.video);
      if (mapped) suggestion.video = mapped;
    }

    // Update static ROIs
    for (const roi of this._staticRois) {
      if (roi.video && videoMap.has(roi.video))
        roi.video = videoMap.get(roi.video)!;
    }

    this.videos = this.videos.map((v) => videoMap!.get(v) ?? v);

    // Frame index is keyed by video identity, so must be rebuilt
    this._invalidateIndices();
  }

  /**
   * Remove one or more videos and every reference to them — the "drop" analog
   * of {@link replaceVideos}. Labeled frames and suggestions belonging to a
   * removed video are dropped (a frame's ROIs go with it), as are static ROIs
   * that reference it. Videos not present are ignored (a no-op, matching
   * {@link addVideo}'s lenient convention).
   *
   * Tracks and skeletons are intentionally left untouched — cleaning those up
   * is a separate concern (see {@link clean}).
   */
  removeVideos(videos: Video[]): void {
    if (videos.length === 0) return;
    if (this._lazyFrameList) this.materialize();

    const toRemove = new Set(videos);

    this.labeledFrames = this.labeledFrames.filter(
      (lf) => !toRemove.has(lf.video),
    );
    this.suggestions = this.suggestions.filter((s) => !toRemove.has(s.video));
    this._staticRois = this._staticRois.filter(
      (roi) => !roi.video || !toRemove.has(roi.video),
    );
    this.videos = this.videos.filter((v) => !toRemove.has(v));

    // Frame/track indices are keyed by video identity, so must be rebuilt.
    this._invalidateIndices();
  }

  /** Remove a single video and all references to it (see {@link removeVideos}). */
  removeVideo(video: Video): void {
    this.removeVideos([video]);
  }

  /**
   * Create a deep copy of this Labels object.
   *
   * @param options.openVideos - Controls video backend behavior in the copy:
   *   - `undefined` (default): Preserve each video's current `openBackend` setting.
   *   - `true`: Enable auto-opening for all videos.
   *   - `false`: Disable auto-opening and close any open backends.
   * @returns A new Labels with deep-copied data. Video backends (file handles)
   *   are not copied — they will be re-opened on demand if `openBackend` is true.
   */
  copy(options?: { openVideos?: boolean }): Labels {
    // 1. Clone videos (without backends — file handles can't be copied)
    const videoMap = new Map<Video, Video>();
    const newVideos = this.videos.map((v) => {
      const nv = new Video({
        filename: Array.isArray(v.filename) ? [...v.filename] : v.filename,
        backendMetadata: { ...v.backendMetadata },
        openBackend: v.openBackend,
        embedded: v.hasEmbeddedImages,
      });
      nv.shape = v.shape;
      nv.fps = v.fps;
      videoMap.set(v, nv);
      return nv;
    });

    // 2. Clone skeletons (rebuild from constructors so internal maps are correct)
    const skeletonMap = new Map<Skeleton, Skeleton>();
    const newSkeletons = this.skeletons.map((s) => {
      const nodeMap = new Map<Node, Node>();
      const newNodes = s.nodes.map((n) => {
        const nn = new Node(n.name);
        nodeMap.set(n, nn);
        return nn;
      });
      const newEdges = s.edges.map(
        (e) => new Edge(nodeMap.get(e.source)!, nodeMap.get(e.destination)!),
      );
      const newSymmetries = s.symmetries.map((sym) => {
        const nodes = [...sym.nodes];
        return new Symmetry([nodeMap.get(nodes[0])!, nodeMap.get(nodes[1])!]);
      });
      const ns = new Skeleton({
        nodes: newNodes,
        edges: newEdges,
        symmetries: newSymmetries,
        name: s.name,
      });
      skeletonMap.set(s, ns);
      return ns;
    });

    // 3. Clone tracks
    const trackMap = new Map<Track, Track>();
    const newTracks = this.tracks.map((t) => {
      const nt = new Track(t.name);
      trackMap.set(t, nt);
      return nt;
    });

    // Helper: clone an instance with remapped skeleton/track
    const cloneInstance = (
      inst: Instance | PredictedInstance,
    ): Instance | PredictedInstance => {
      const newPoints = inst.points.map((p) => clonePoint(p));
      const newSkeleton = skeletonMap.get(inst.skeleton) ?? inst.skeleton;
      const newTrack = inst.track
        ? (trackMap.get(inst.track) ?? inst.track)
        : null;
      if (inst instanceof PredictedInstance) {
        return new PredictedInstance({
          points: newPoints as any,
          skeleton: newSkeleton,
          track: newTrack,
          score: inst.score,
          trackingScore: inst.trackingScore,
        });
      }
      const ni = new Instance({
        points: newPoints,
        skeleton: newSkeleton,
        track: newTrack,
        trackingScore: inst.trackingScore,
      });
      // fromPredicted can't be fully remapped (would need global instance identity),
      // so we leave it null on the copy.
      return ni;
    };

    // Helper: clone ancillary items by stripping object refs, structuredClone, then remap
    const cloneAncillary = <T extends object>(items: T[]): T[] =>
      items.map((item) => {
        const saved: [string, any][] = [];
        for (const key of ["video", "track", "instance"] as const) {
          if (key in item && (item as any)[key] != null) {
            saved.push([key, (item as any)[key]]);
            (item as any)[key] = null;
          }
        }
        // For LabelImage: also strip track/instance refs inside objects Map
        let objectRefs: Map<number, [any, any]> | null = null;
        if ("objects" in item && (item as any).objects instanceof Map) {
          objectRefs = new Map();
          for (const [id, info] of (item as any).objects as Map<number, any>) {
            if (info.track || info.instance) {
              objectRefs.set(id, [info.track, info.instance]);
              info.track = null;
              info.instance = null;
            }
          }
        }

        const clone = structuredClone(item);
        Object.setPrototypeOf(clone, Object.getPrototypeOf(item));

        // Restore original
        for (const [key, val] of saved) (item as any)[key] = val;
        if (objectRefs) {
          for (const [id, [track, inst]] of objectRefs) {
            const info = (item as any).objects.get(id);
            if (info) {
              info.track = track;
              info.instance = inst;
            }
          }
        }

        // Remap refs on clone
        for (const [key, val] of saved) {
          if (key === "video") (clone as any).video = videoMap.get(val) ?? val;
          else if (key === "track")
            (clone as any).track = trackMap.get(val) ?? val;
          else if (key === "instance") (clone as any).instance = null;
        }
        if (objectRefs) {
          for (const [id, [track]] of objectRefs) {
            const info = (clone as any).objects.get(id);
            if (info) {
              info.track = track ? (trackMap.get(track) ?? track) : null;
              info.instance = null;
            }
          }
        }
        return clone;
      });

    let labelsCopy: Labels;

    if (this.isLazy) {
      // Lazy-aware copy: deep copy the store with independent arrays
      // Annotations are stored on the lazy store's per-frame dicts
      // and will be attached to frames when they are materialized.
      const newStore = this._lazyDataStore!.copy();
      newStore.videos = newVideos;
      newStore.skeletons = newSkeletons;
      newStore.tracks = newTracks;

      const newLazyFrames = new LazyFrameList(newStore);

      // Copy supplementary frames (annotation-only, non-lazy)
      if (this._lazyFrameList?._supplementary.length) {
        newLazyFrames._supplementary = this._lazyFrameList._supplementary.map(
          (lf) => {
            return new LabeledFrame({
              video: videoMap.get(lf.video) ?? lf.video,
              frameIdx: lf.frameIdx,
              instances: lf.instances.map(cloneInstance),
              isNegative: lf.isNegative,
              centroids: cloneAncillary(lf.centroids) as Centroid[],
              bboxes: cloneAncillary(lf.bboxes) as BoundingBox[],
              masks: cloneAncillary(lf.masks) as SegmentationMask[],
              labelImages: cloneAncillary(lf.labelImages) as LabelImage[],
              rois: cloneAncillary(lf.rois) as ROI[],
            });
          },
        );
      }

      labelsCopy = new Labels({
        videos: newVideos,
        skeletons: newSkeletons,
        tracks: newTracks,
        suggestions: this.suggestions.map((s) => {
          const newVideo = videoMap.get(s.video) ?? s.video;
          return new SuggestionFrame({
            video: newVideo,
            frameIdx: s.frameIdx,
            group: s.group,
            metadata: { ...s.metadata },
          });
        }),
        // Rebuild sessions via constructors (NOT structuredClone, which strips
        // class prototypes/getters and the non-enumerable frame resolver). In
        // lazy mode frames are not materialized here, so ref-backed grouping is
        // carried as refs and re-resolves against the copy's lazy frame list
        // once injectSessionFrameResolver runs below.
        sessions: this.sessions.map((s) =>
          cloneRecordingSession(s, { videoMap }),
        ),
        provenance: { ...this.provenance },
        identities: structuredClone(this.identities),
      });

      labelsCopy._lazyDataStore = newStore;
      labelsCopy._lazyFrameList = newLazyFrames;
      injectSessionFrameResolver(labelsCopy);
    } else {
      // Eager deep copy: rebuild from constructors, including per-frame annotations
      const frameMap = new Map<LabeledFrame, LabeledFrame>();
      const instanceMap = new Map<Instance, Instance>();
      const newFrames = this.labeledFrames.map((f) => {
        const newInstances = f.instances.map((inst) => {
          const ni = cloneInstance(inst);
          instanceMap.set(inst, ni);
          return ni;
        });
        const nf = new LabeledFrame({
          video: videoMap.get(f.video) ?? f.video,
          frameIdx: f.frameIdx,
          instances: newInstances,
          isNegative: f.isNegative,
          centroids: cloneAncillary(f.centroids) as Centroid[],
          bboxes: cloneAncillary(f.bboxes) as BoundingBox[],
          masks: cloneAncillary(f.masks) as SegmentationMask[],
          labelImages: cloneAncillary(f.labelImages) as LabelImage[],
          rois: cloneAncillary(f.rois) as ROI[],
        });
        frameMap.set(f, nf);
        return nf;
      });

      labelsCopy = new Labels({
        labeledFrames: newFrames,
        videos: newVideos,
        skeletons: newSkeletons,
        tracks: newTracks,
        suggestions: this.suggestions.map((s) => {
          const newVideo = videoMap.get(s.video) ?? s.video;
          return new SuggestionFrame({
            video: newVideo,
            frameIdx: s.frameIdx,
            group: s.group,
            metadata: { ...s.metadata },
          });
        }),
        // Rebuild sessions via constructors, remapping Camera keys and
        // Instance/LabeledFrame references onto the copied objects; ref-backed
        // grouping re-resolves via the resolver injected below.
        sessions: this.sessions.map((s) =>
          cloneRecordingSession(s, { videoMap, frameMap, instanceMap }),
        ),
        provenance: { ...this.provenance },
        rois: cloneAncillary(this._staticRois),
        identities: structuredClone(this.identities),
      });
      injectSessionFrameResolver(labelsCopy);
    }

    if (options?.openVideos !== undefined) {
      for (const video of labelsCopy.videos) {
        video.openBackend = options.openVideos;
        if (!options.openVideos) video.close();
      }
    }

    return labelsCopy;
  }

  static fromNumpy(
    data: number[][][][],
    options: {
      videos?: Video[];
      video?: Video;
      skeletons?: Skeleton[] | Skeleton;
      skeleton?: Skeleton;
      trackNames?: string[];
      firstFrame?: number;
      returnConfidence?: boolean;
    },
  ): Labels {
    const video = options.video ?? options.videos?.[0];
    if (!video) throw new Error("fromNumpy requires a video.");
    if (options.video && options.videos) {
      throw new Error("Cannot specify both video and videos.");
    }
    const skeletons = Array.isArray(options.skeletons)
      ? options.skeletons
      : options.skeletons
        ? [options.skeletons]
        : options.skeleton
          ? [options.skeleton]
          : [];
    if (!skeletons.length) throw new Error("fromNumpy requires a skeleton.");
    return labelsFromNumpy(data, {
      video,
      skeleton: skeletons[0],
      trackNames: options.trackNames,
      firstFrame: options.firstFrame,
      returnConfidence: options.returnConfidence,
    });
  }

  /**
   * Convert labels to a dense `[frames, tracks, nodes, coords]` array.
   *
   * @param options.numFrames Optional explicit length of the output's frame
   *   dimension. Takes precedence over `video.shape[0]` (the inferred fallback).
   *   Useful when `video.shape` is null — for example, Mp4Box-backed browser
   *   videos — and you still want a video-length-sized array. If smaller than
   *   `maxLabeledFrame + 1`, it is clamped up so no labeled frames are dropped.
   *   Non-finite, non-positive, or fractional values are sanitized via
   *   `Math.floor` and ignored when `<= 0`.
   */
  /**
   * Build a dense `(frames, tracks, nodes, channels)` array from instance points.
   *
   * A foreign `Video` instance or filename (`string`/`URL`) is resolved to the
   * matching project `Video` via {@link _resolveVideo} (SYNC; see its documented
   * divergence from `matchVideo`). When `options.video` is absent, defaults to
   * `this.video` (the first video).
   */
  numpy(options?: {
    video?: Video | string | URL;
    returnConfidence?: boolean;
    numFrames?: number;
  }): number[][][][] {
    // Canonicalize a foreign Video / filename to the matching project Video,
    // defaulting to the first video when absent.
    const targetVideo = this._resolveVideo(options?.video) ?? this.video;
    if (this._lazyDataStore) {
      return this._lazyDataStore.toNumpy({ ...options, video: targetVideo });
    }
    const frames = this.labeledFrames.filter((frame) =>
      frame.video.matchesPath(targetVideo, true),
    );
    if (!frames.length) return [];

    let maxFrame = Math.max(...frames.map((frame) => frame.frameIdx));
    const rawOverride = options?.numFrames;
    const override =
      Number.isFinite(rawOverride) && (rawOverride as number) > 0
        ? Math.floor(rawOverride as number)
        : 0;
    const effectiveLength =
      override > 0 ? override : (targetVideo.shape?.[0] ?? 0);
    if (effectiveLength > 0) {
      maxFrame = Math.max(maxFrame, effectiveLength - 1);
    }
    const tracks = this.tracks.length
      ? this.tracks.length
      : Math.max(1, ...frames.map((frame) => frame.instances.length));
    const nodes = this.skeletons[0]?.nodes.length ?? 0;
    const channelCount = options?.returnConfidence ? 3 : 2;

    const videoArray: number[][][][] = Array.from(
      { length: maxFrame + 1 },
      () =>
        Array.from({ length: tracks }, () =>
          Array.from({ length: nodes }, () =>
            Array.from({ length: channelCount }, () => Number.NaN),
          ),
        ),
    );

    for (const frame of frames) {
      const frameSlot = videoArray[frame.frameIdx];
      if (!frameSlot) continue;
      frame.instances.forEach((inst, idx) => {
        const trackIndex = inst.track ? this.tracks.indexOf(inst.track) : idx;
        const resolvedTrack = trackIndex >= 0 ? trackIndex : idx;
        const trackSlot = frameSlot[resolvedTrack];
        if (!trackSlot) return;
        inst.points.forEach((point, nodeIdx) => {
          if (!trackSlot[nodeIdx]) return;
          const row = [point.xy[0], point.xy[1]];
          if (options?.returnConfidence) {
            const score =
              "score" in point
                ? (point as { score: number }).score
                : Number.NaN;
            row.push(score);
          }
          trackSlot[nodeIdx] = row;
        });
      });
    }

    return videoArray;
  }

  /**
   * Register an instance's skeleton, deduplicating structurally-equal ones.
   *
   * If a skeleton with the same structure *and* the same node order already
   * exists in `this.skeletons`, the instance is rebound to that canonical object
   * instead of leaking a duplicate. If no match exists, the instance's skeleton
   * is appended as a new canonical skeleton.
   *
   * A skeleton that is already registered (by object identity) is left
   * untouched. This deliberately preserves distinct-but-compatible skeletons a
   * caller added explicitly (e.g. via `new Labels({ skeletons: [...] })`), so
   * workflows that reason about them separately keep working; only
   * newly-discovered duplicates are canonicalized.
   *
   * Matching uses `Skeleton.matches(..., { requireSameOrder: true })` — the same
   * strictness as {@link dedupSkeletons} — so a newly-seen skeleton is only
   * treated as a duplicate when its node names, edges, symmetries, *and* node
   * order all match an existing skeleton. Because the node order is identical,
   * the instance's positional points array is already aligned to the canonical
   * skeleton, so rebinding `inst.skeleton` never moves any point data. Two
   * structurally-equal skeletons with *different* node order are intentionally
   * kept distinct, since their positional point semantics genuinely differ.
   *
   * Mirrors Python `Labels._register_skeleton` (PR #447).
   *
   * @param inst - Instance whose skeleton should be registered. Both `Instance`
   *   and `PredictedInstance` are supported.
   */
  private _registerSkeleton(inst: Instance | PredictedInstance): void {
    // Already registered (identity check) -> keep as-is.
    if (this.skeletons.includes(inst.skeleton)) {
      return;
    }
    // Newly-seen skeleton: canonicalize to a structurally-equal, same-order one
    // already registered, otherwise register it as a new skeleton.
    const canonical = this.skeletons.find((s) =>
      s.matches(inst.skeleton, { requireSameOrder: true }),
    );
    if (canonical === undefined) {
      this.skeletons.push(inst.skeleton);
    } else {
      inst.skeleton = canonical;
    }
  }

  /**
   * Update data structures based on contents.
   *
   * Repopulates `videos`, `skeletons`, and `tracks` from the labeled frames,
   * their instances and nested annotations, and the suggestions. Existing
   * entries are preserved (in order); only missing ones are appended.
   *
   * Skeletons are registered via {@link _registerSkeleton}, so two
   * structurally-equal but distinct `Skeleton` objects collapse to a single
   * canonical entry (Python #447) instead of leaking a duplicate.
   *
   * Mirrors Python `Labels.update` (labels.py:435-457).
   */
  update(): void {
    if (this._lazyFrameList) this.materialize();
    // Shared membership sets: `includes` per frame/instance was O(n × videos)
    // and O(n × tracks). Sets keep this O(n) on large many-track projects.
    const seenVideos = new Set(this.videos);
    const seenTracks = new Set(this.tracks);
    for (const lf of this.labeledFrames) {
      if (!seenVideos.has(lf.video)) {
        seenVideos.add(lf.video);
        this.videos.push(lf.video);
      }
      for (const inst of lf.instances) {
        this._registerSkeleton(inst);
        if (inst.track != null && !seenTracks.has(inst.track)) {
          seenTracks.add(inst.track);
          this.tracks.push(inst.track);
        }
      }
      // Collect tracks from nested annotations (reuse the set).
      this._collectAnnotationTracks(lf, seenTracks);
    }
    for (const sf of this.suggestions) {
      if (!seenVideos.has(sf.video)) {
        seenVideos.add(sf.video);
        this.videos.push(sf.video);
      }
    }
  }

  /**
   * Remap video and track references on a frame's annotations in place.
   *
   * Mirrors Python `Labels._remap_frame_annotations` (labels.py:3621-3648).
   * Centroids/bboxes/masks: only `.track` is remapped. ROIs: both `.video` and
   * `.track`. Label-image objects: nested `info.track` only. Membership is by
   * reference via `Map.has`/`Map.get` (never a `?? default`), so a track/video
   * absent from the map is left untouched.
   *
   * @param frame - LabeledFrame whose annotations should be remapped.
   * @param videoMap - Map from old videos to new videos.
   * @param trackMap - Map from old tracks to new tracks.
   */
  static _remapFrameAnnotations(
    frame: LabeledFrame,
    videoMap: Map<Video, Video>,
    trackMap: Map<Track, Track>,
  ): void {
    for (const ann of [
      ...frame.centroids,
      ...frame.bboxes,
      ...frame.masks,
    ] as Array<Centroid | BoundingBox | SegmentationMask>) {
      if (ann.track != null && trackMap.has(ann.track)) {
        ann.track = trackMap.get(ann.track)!;
      }
    }
    for (const r of frame.rois) {
      if (r.video != null && videoMap.has(r.video)) {
        r.video = videoMap.get(r.video)!;
      }
      if (r.track != null && trackMap.has(r.track)) {
        r.track = trackMap.get(r.track)!;
      }
    }
    for (const li of frame.labelImages) {
      for (const info of li.objects.values()) {
        if (info.track != null && trackMap.has(info.track)) {
          info.track = trackMap.get(info.track)!;
        }
      }
    }
  }

  /**
   * Map an instance to use mapped skeleton and track, returning a NEW instance.
   *
   * Mirrors Python `Labels._map_instance` (labels.py:3953-4020). The source
   * instance is never mutated: its points are deep-copied and the returned
   * instance is a fresh object of the SAME exact type (`Instance` vs
   * `PredictedInstance`, dispatched via `constructor ===`). Skeleton/track are
   * resolved through the maps with `?? original` fallback.
   *
   * When the source instance's node order differs from the mapped skeleton's
   * node order (e.g. the default structure matcher matched `[A, B, C]` with
   * `[C, B, A]`), the points are reordered by node NAME so that each node's
   * coordinates and score follow its name rather than its position (Python
   * #489). When the node orders are identical (the common case) the points are
   * copied positionally to avoid any overhead on the hot path. Nodes present in
   * the mapped skeleton but absent from the source are filled with a missing,
   * invisible point (NaN xy).
   *
   * @param instance - Instance to map.
   * @param skeletonMap - Map from old skeletons to new skeletons.
   * @param trackMap - Map from old tracks to new tracks.
   * @param memo - Optional map from the source instance to the new instance,
   *   mutated in place. Used by {@link _relinkFromPredicted} to repair
   *   `fromPredicted` links so a remapped user instance references the remapped
   *   source prediction now in the merged frame (Python #491).
   * @returns New instance with mapped skeleton and track.
   */
  _mapInstance(
    instance: Instance | PredictedInstance,
    skeletonMap: Map<Skeleton, Skeleton>,
    trackMap: Map<Track, Track>,
    memo?: Map<object, object>,
  ): Instance | PredictedInstance {
    const mappedSkeleton =
      skeletonMap.get(instance.skeleton) ?? instance.skeleton;
    const mappedTrack = instance.track
      ? (trackMap.get(instance.track) ?? instance.track)
      : null;

    const isPredicted = instance.constructor === PredictedInstance;
    const sourcePoints = instance.points;
    const mappedNames = mappedSkeleton.nodeNames;

    // Reorder points by node name when the source order differs from the mapped
    // skeleton's order, otherwise the per-node coordinates/scores would be
    // carried over positionally and silently misaligned (Python #489). When the
    // orders already match, copy positionally on the hot path.
    let mappedPoints: PointsArray | PredictedPointsArray;
    const sameOrder =
      sourcePoints.length === mappedNames.length &&
      sourcePoints.every((p, i) => p.name === mappedNames[i]);
    if (sameOrder) {
      mappedPoints = sourcePoints.map((p) => clonePoint(p)) as
        | PointsArray
        | PredictedPointsArray;
    } else {
      // Index the source points by node name so each mapped-skeleton node can
      // copy the coordinates/score that belong to its name.
      const sourceByName = new Map<string, Point | PredictedPoint>();
      for (const p of sourcePoints) {
        if (p.name !== undefined) sourceByName.set(p.name, p);
      }
      mappedPoints = mappedNames.map((name) => {
        const src = sourceByName.get(name);
        if (src !== undefined) {
          return clonePoint(src, name);
        }
        // Node absent from the source skeleton: fill a missing/invisible point
        // (NaN xy), mirroring how missing nodes are represented elsewhere.
        return isPredicted
          ? {
              xy: [Number.NaN, Number.NaN] as [number, number],
              visible: false,
              complete: false,
              score: Number.NaN,
              name,
            }
          : {
              xy: [Number.NaN, Number.NaN] as [number, number],
              visible: false,
              complete: false,
              name,
            };
      }) as PointsArray | PredictedPointsArray;
    }

    let newInstance: Instance | PredictedInstance;
    if (isPredicted) {
      const predicted = instance as PredictedInstance;
      newInstance = new PredictedInstance({
        points: mappedPoints as PredictedPointsArray,
        skeleton: mappedSkeleton,
        score: predicted.score,
        track: mappedTrack,
        trackingScore: predicted.trackingScore,
        fromPredicted: predicted.fromPredicted,
      });
    } else {
      newInstance = new Instance({
        points: mappedPoints as PointsArray,
        skeleton: mappedSkeleton,
        track: mappedTrack,
        trackingScore: instance.trackingScore,
        fromPredicted: instance.fromPredicted,
      });
    }
    if (memo !== undefined) memo.set(instance, newInstance);
    return newInstance;
  }

  /**
   * Merge another `Labels` object into this one in place.
   *
   * Faithful port of Python `Labels.merge` (labels.py:3149-3618). Runs the fixed
   * 5-step pipeline (skeletons -> videos -> tracks -> frames -> suggestions),
   * building reference-keyed maps FROM `other`'s objects TO `self`'s objects (or
   * to a newly-appended `other` object), and returns a {@link MergeResult}.
   *
   * Async (DECISIONS D8): the AUTO video cascade awaits filesystem and pixel
   * reads. Coercion of the matcher/error-mode arguments happens BEFORE the merge
   * body, so a bad method/error-mode string propagates (it is NOT collected into
   * the result).
   *
   * @param other - The `Labels` to merge into `self`.
   * @param opts.skeleton - Skeleton matcher (`null` -> STRUCTURE; string ->
   *   validated; else used as-is).
   * @param opts.video - Video matcher (`null` -> AUTO).
   * @param opts.track - Track matcher (`null` -> IDENTITY). The default matches
   *   tracks only by object identity (the same Track instance) and appends all
   *   other tracks as new — a correctness-first default that never collapses
   *   distinct tracks by their (often arbitrary, tracker-assigned) names. Pass
   *   `"name"` to match tracks by their name attribute instead, for cases where
   *   track names are semantically meaningful (e.g. user-assigned identities or
   *   identity-classification model outputs).
   * @param opts.frame - The frame merge strategy as a RAW string (default
   *   `"auto"`; NOT validated against the enum — an invalid value falls through
   *   `LabeledFrame.merge`'s strategy chain into the AUTO branch).
   * @param opts.instance - Instance matcher (`null` -> SPATIAL/5.0).
   * @param opts.validate - If `true` (default), an unmatched skeleton under
   *   STRICT raises `SkeletonMismatchError`.
   * @param opts.progressCallback - Called `(current, total, message)` per frame
   *   and once at the end.
   * @param opts.errorMode - `"continue"` (default), `"strict"`, or `"warn"`.
   */
  async merge(
    other: Labels,
    opts: {
      skeleton?: string | SkeletonMatcher | null;
      video?: string | VideoMatcher | null;
      track?: string | TrackMatcher | null;
      frame?: string;
      instance?: string | InstanceMatcher | null;
      validate?: boolean;
      progressCallback?: (
        current: number,
        total: number,
        message: string,
      ) => void;
      errorMode?: string;
    } = {},
  ): Promise<MergeResult> {
    // ----- §6.2 coercion (BEFORE the try: throws PROPAGATE) -----------------
    const skeletonMatcher = coerceSkeletonMatcher(opts.skeleton);
    const videoMatcher = coerceVideoMatcher(opts.video);
    const trackMatcher = coerceTrackMatcher(opts.track);
    const instanceMatcher = coerceInstanceMatcher(opts.instance);
    const frame = opts.frame ?? "auto"; // RAW string, NOT validated.
    const validate = opts.validate ?? true;
    const progressCallback = opts.progressCallback;
    const errorModeEnum = toErrorMode(opts.errorMode ?? "continue");

    // ----- §6.3 lazy guard + init -------------------------------------------
    // JS convention: materialize a lazy Labels rather than raising.
    if (this._lazyFrameList) this.materialize();

    const result = new MergeResult(true);

    // Ensure merge_history exists OUTSIDE the try (so even a failed merge leaves
    // an empty history if it was absent).
    if (!("merge_history" in this.provenance)) {
      this.provenance.merge_history = [];
    }
    const mergeHistory = this.provenance.merge_history as Array<
      Record<string, unknown>
    >;

    const mergeRecord: Record<string, unknown> = {
      timestamp: localIsoStringWithoutZ(),
      source_filename:
        (other.provenance.filename as string | undefined) ?? null,
      target_filename: (this.provenance.filename as string | undefined) ?? null,
      source_labels: {
        n_frames: other.labeledFrames.length,
        n_videos: other.videos.length,
        n_skeletons: other.skeletons.length,
        n_tracks: other.tracks.length,
      },
      strategy: frame,
      sleap_io_version: SLEAP_IO_VERSION,
    };

    // §6.11 / DECISIONS D11: initialize `total` so the final progressCallback is
    // always in scope even if an early non-strict error short-circuits.
    let total = 0;

    try {
      // ----- §6.4 STEP 1: skeletons ---------------------------------------
      const skeletonMap = new Map<Skeleton, Skeleton>();
      for (const otherSkel of other.skeletons) {
        let matched = false;
        for (const selfSkel of this.skeletons) {
          // arg order (self, other) — matters for SUBSET (asymmetric).
          if (skeletonMatcher.match(selfSkel, otherSkel)) {
            skeletonMap.set(otherSkel, selfSkel);
            matched = true;
            break; // first match wins
          }
        }
        if (!matched) {
          if (validate && errorModeEnum === ErrorMode.STRICT) {
            throw new SkeletonMismatchError(
              `No matching skeleton found for ${otherSkel.name}`,
              { skeleton: otherSkel },
            );
          } else if (errorModeEnum === ErrorMode.WARN) {
            console.warn(`Warning: No matching skeleton for ${otherSkel.name}`);
          }
          this.skeletons.push(otherSkel); // append foreign ref
          skeletonMap.set(otherSkel, otherSkel); // map to itself
        }
      }

      // ----- §6.5 STEP 2: videos ------------------------------------------
      const videoMap = new Map<Video, Video>();
      // frameIdxMap: (otherVideo, oldIdx) -> [newVideo, newIdx], reference-keyed.
      const frameIdxMap = new Map<Video, Map<number, [Video, number]>>();
      const setFrameIdx = (
        v: Video,
        oldIdx: number,
        newVideo: Video,
        newIdx: number,
      ): void => {
        let inner = frameIdxMap.get(v);
        if (inner == null) {
          inner = new Map();
          frameIdxMap.set(v, inner);
        }
        inner.set(oldIdx, [newVideo, newIdx]);
      };

      for (const otherVideo of other.videos) {
        let matched = false;

        if (
          videoMatcher.method === VideoMatchMethod.IMAGE_DEDUP ||
          videoMatcher.method === VideoMatchMethod.SHAPE
        ) {
          // Pairwise match() loop (first match wins).
          for (const selfVideo of this.videos) {
            // arg order (self, other); await the async matcher.
            if (await videoMatcher.match(selfVideo, otherVideo)) {
              if (videoMatcher.method === VideoMatchMethod.IMAGE_DEDUP) {
                const dedupedVideo = otherVideo.deduplicateWith(selfVideo);
                if (dedupedVideo === null) {
                  // All images were duplicates -> map to existing video.
                  videoMap.set(otherVideo, selfVideo);
                  if (
                    Array.isArray(otherVideo.filename) &&
                    Array.isArray(selfVideo.filename)
                  ) {
                    const otherBasenames =
                      otherVideo.filename.map(pathBasename);
                    const selfBasenames = selfVideo.filename.map(pathBasename);
                    otherBasenames.forEach((bn, oldIdx) => {
                      const newIdx = selfBasenames.indexOf(bn);
                      if (newIdx !== -1) {
                        setFrameIdx(otherVideo, oldIdx, selfVideo, newIdx);
                      }
                    });
                  }
                } else {
                  // New deduplicated video.
                  this.videos.push(dedupedVideo);
                  videoMap.set(otherVideo, dedupedVideo);
                  if (
                    Array.isArray(otherVideo.filename) &&
                    Array.isArray(dedupedVideo.filename)
                  ) {
                    const otherBasenames =
                      otherVideo.filename.map(pathBasename);
                    const dedupedBasenames =
                      dedupedVideo.filename.map(pathBasename);
                    const selfBasenames = Array.isArray(selfVideo.filename)
                      ? selfVideo.filename.map(pathBasename)
                      : [];
                    otherBasenames.forEach((bn, oldIdx) => {
                      const dedupIdx = dedupedBasenames.indexOf(bn);
                      if (dedupIdx !== -1) {
                        setFrameIdx(otherVideo, oldIdx, dedupedVideo, dedupIdx);
                      } else {
                        const selfIdx = selfBasenames.indexOf(bn);
                        if (selfIdx === -1) {
                          throw new Error(
                            "Unexpected basename mismatch, possible file corruption.",
                          );
                        }
                        setFrameIdx(otherVideo, oldIdx, selfVideo, selfIdx);
                      }
                    });
                  }
                }
              } else {
                // SHAPE: merge videos with the same shape.
                const mergedVideo = selfVideo.mergeWith(otherVideo);
                const selfVideoIdx = this.videos.indexOf(selfVideo);
                this.videos[selfVideoIdx] = mergedVideo; // in-place replace
                videoMap.set(otherVideo, mergedVideo);
                videoMap.set(selfVideo, mergedVideo); // ONLY self-key set
                if (
                  Array.isArray(otherVideo.filename) &&
                  Array.isArray(mergedVideo.filename)
                ) {
                  const otherBasenames = otherVideo.filename.map(pathBasename);
                  const mergedBasenames =
                    mergedVideo.filename.map(pathBasename);
                  otherBasenames.forEach((bn, oldIdx) => {
                    const newIdx = mergedBasenames.indexOf(bn);
                    if (newIdx !== -1) {
                      setFrameIdx(otherVideo, oldIdx, mergedVideo, newIdx);
                    }
                  });
                }
              }
              matched = true;
              break;
            }
          }
        } else {
          // All other methods: full find_match cascade.
          const matchedVideo = await videoMatcher.findMatch(
            otherVideo,
            this.videos,
            { labelsIncoming: other, labelsBase: this },
          );
          if (matchedVideo !== null) {
            videoMap.set(otherVideo, matchedVideo);
            matched = true;
          }
        }

        if (!matched) {
          this.videos.push(otherVideo); // append foreign ref
          videoMap.set(otherVideo, otherVideo); // map to itself
        }
      }

      // ----- §6.6 STEP 3: tracks ------------------------------------------
      const trackMap = new Map<Track, Track>();
      for (const otherTrack of other.tracks) {
        let matched = false;
        // this.tracks GROWS in-loop; the inner loop re-reads it each iteration.
        for (const selfTrack of this.tracks) {
          if (trackMatcher.match(selfTrack, otherTrack)) {
            trackMap.set(otherTrack, selfTrack);
            matched = true;
            break; // first match wins
          }
        }
        if (!matched) {
          this.tracks.push(otherTrack);
          trackMap.set(otherTrack, otherTrack);
        }
      }

      // Warn (diagnostic only) if any name-matched track pair carries instances
      // that diverge spatially on every shared frame. This does not alter
      // trackMap or any merge result.
      this._warnTrackNameDivergence(
        other,
        videoMap,
        trackMap,
        trackMatcher,
        instanceMatcher,
      );

      // ----- §6.7 STEP 4: frames ------------------------------------------
      total = other.labeledFrames.length;

      for (let idx = 0; idx < total; idx++) {
        const otherFrame = other.labeledFrames[idx];
        progressCallback?.(idx, total, `Merging frame ${idx + 1}/${total}`);

        // Resolve mapped (video, frameIdx): frameIdxMap takes precedence.
        let mappedVideo: Video;
        let mappedFrameIdx: number;
        const inner = frameIdxMap.get(otherFrame.video);
        const mapped = inner?.get(otherFrame.frameIdx);
        if (mapped != null) {
          [mappedVideo, mappedFrameIdx] = mapped;
        } else {
          mappedVideo = videoMap.get(otherFrame.video) ?? otherFrame.video;
          mappedFrameIdx = otherFrame.frameIdx;
        }

        const matching = this.find({
          video: mappedVideo,
          frameIdx: mappedFrameIdx,
        });

        if (matching.length === 0) {
          // BRANCH A: create a new frame.
          const newFrame = new LabeledFrame({
            video: mappedVideo,
            frameIdx: mappedFrameIdx,
            instances: [],
            isNegative: otherFrame.isNegative,
          });
          const instanceMemo = new Map<object, object>();
          for (const inst of otherFrame.instances) {
            newFrame.instances.push(
              this._mapInstance(inst, skeletonMap, trackMap, instanceMemo),
            );
            result.instancesAdded += 1; // per instance
          }
          // Repair `fromPredicted` links to the remapped source.
          _relinkFromPredicted(newFrame.instances, instanceMemo);
          newFrame.mergeAnnotations(otherFrame); // default "keep_both"
          Labels._remapFrameAnnotations(newFrame, videoMap, trackMap);
          this.append(newFrame);
          result.framesMerged += 1;
        } else {
          // BRANCH B: merge into the existing frame.
          const selfFrame = matching[0];
          const selfWasNegative = selfFrame.isNegative; // BEFORE merge mutates

          const [rawMerged, conflicts] = selfFrame.merge(otherFrame, {
            instance: instanceMatcher,
            frame,
          });

          // Remap skeleton/track of instances that came from `other`.
          const instanceMemo = new Map<object, object>();
          const mergedInstances = rawMerged.map((inst) =>
            skeletonMap.has(inst.skeleton)
              ? this._mapInstance(inst, skeletonMap, trackMap, instanceMemo)
              : inst,
          );
          // Repair `fromPredicted` links so a remapped user instance references
          // the remapped source prediction in this frame.
          _relinkFromPredicted(mergedInstances, instanceMemo);

          const nBefore = selfFrame.instances.length; // BEFORE reassignment
          const nAfter = mergedInstances.length;
          result.instancesAdded += Math.max(0, nAfter - nBefore);

          for (const [orig, nw, resolution] of conflicts) {
            result.conflicts.push(
              new ConflictResolution(
                selfFrame,
                "instance_conflict",
                orig,
                nw,
                resolution,
              ),
            );
          }

          const [, negativeConflict] = _resolveMergedIsNegative(
            selfWasNegative,
            otherFrame.isNegative,
            mergedInstances,
          );
          if (negativeConflict) {
            result.conflicts.push(
              new ConflictResolution(
                selfFrame,
                "negative_flag_conflict",
                selfWasNegative,
                otherFrame.isNegative,
                "dropped_for_user_pose",
              ),
            );
          }

          selfFrame.instances = mergedInstances;
          Labels._remapFrameAnnotations(selfFrame, videoMap, trackMap);
          result.framesMerged += 1;
        }
      }

      // ----- §6.8 STEP 5: suggestions -------------------------------------
      for (const otherSuggestion of other.suggestions) {
        const mappedVideo =
          videoMap.get(otherSuggestion.video) ?? otherSuggestion.video;
        let exists = false;
        for (const selfSuggestion of this.suggestions) {
          if (
            selfSuggestion.video === mappedVideo &&
            selfSuggestion.frameIdx === otherSuggestion.frameIdx
          ) {
            exists = true;
            break;
          }
        }
        if (!exists) {
          this.suggestions.push(
            new SuggestionFrame({
              video: mappedVideo,
              frameIdx: otherSuggestion.frameIdx,
            }),
          );
        }
      }

      // ----- §6.9 success tail --------------------------------------------
      mergeRecord.result = {
        frames_merged: result.framesMerged,
        instances_added: result.instancesAdded,
        conflicts: result.conflicts.length, // COUNT, not the list
      };
      mergeHistory.push(mergeRecord); // appended ONLY on success
    } catch (e) {
      // ----- §6.10 exception handlers -------------------------------------
      if (e instanceof MergeError) {
        result.successful = false;
        result.errors.push(e);
        if (errorModeEnum === ErrorMode.STRICT) throw e;
      } else {
        result.successful = false;
        const err = e as { message?: string; constructor?: { name?: string } };
        result.errors.push(
          new MergeError(String(err?.message ?? e), {
            exception: err?.constructor?.name,
          }),
        );
        if (errorModeEnum === ErrorMode.STRICT) throw e; // rethrow ORIGINAL
      }
    }

    // Mutations occurred above (videos/tracks/skeletons/frames/suggestions).
    this._invalidateIndices();

    // ----- §6.11 final tail (OUTSIDE try, always runs) --------------------
    progressCallback?.(total, total, "Merge complete");

    return result;
  }

  /**
   * Warn when name-matched tracks diverge spatially on all shared frames.
   *
   * Faithful port of Python `Labels._warn_track_name_divergence`
   * (labels.py:3672-3776, PR talmolab/sleap-io#448). Name-based track merging
   * silently coalesces tracks that share a name across two `Labels`. If those
   * tracks actually label different animals, this can glue distinct tracks
   * together. This helper emits a diagnostic `console.warn` (purely additive; it
   * never changes the merge result) when a track pair matched by name carries
   * instances on overlapping frames that do not spatially correspond under the
   * merge's instance matcher.
   *
   * The check is a no-op unless track matching is by NAME (divergence is
   * meaningless for identity/object track matching) and the instance matcher is
   * spatial (SPATIAL or IOU). A warning fires at most once per colliding
   * `(otherTrack, selfTrack)` pair, only when the pair has at least one shared
   * frame with instances on both sides and zero spatial instance matches across
   * all such frames.
   *
   * @param other - The other `Labels` being merged into `self`.
   * @param videoMap - Mapping from `other` videos to the matched `self` videos,
   *   as built in {@link merge}.
   * @param trackMap - Mapping from `other` tracks to the matched `self` tracks
   *   (or back to themselves if appended as new), as built in {@link merge}.
   * @param trackMatcher - The `TrackMatcher` used for the merge. The check is
   *   skipped unless its method is NAME.
   * @param instanceMatcher - The `InstanceMatcher` used for the merge. Reused
   *   here as the divergence primitive (no new threshold introduced). Skipped
   *   when its method is IDENTITY (see below).
   */
  private _warnTrackNameDivergence(
    other: Labels,
    videoMap: Map<Video, Video>,
    trackMap: Map<Track, Track>,
    trackMatcher: TrackMatcher,
    instanceMatcher: InstanceMatcher,
  ): void {
    // Only name-based merging can silently glue distinct tracks together.
    if (trackMatcher.method !== TrackMatchMethod.NAME) {
      return;
    }

    // Divergence is a spatial question. An IDENTITY instance matcher compares
    // track-object identity, which is always false across a name collision (the
    // tracks are distinct objects by definition), so it cannot assess spatial
    // divergence and would warn unconditionally. Skip it.
    if (instanceMatcher.method === InstanceMatchMethod.IDENTITY) {
      return;
    }

    // Select true name collisions: an otherTrack coalesced onto a distinct
    // selfTrack object with an equal name (not a track appended as new).
    const collidingPairs: Array<[Track, Track]> = [];
    for (const [otherTrack, selfTrack] of trackMap.entries()) {
      if (selfTrack !== otherTrack && selfTrack.name === otherTrack.name) {
        collidingPairs.push([otherTrack, selfTrack]);
      }
    }
    if (collidingPairs.length === 0) {
      return;
    }

    for (const [otherTrack, selfTrack] of collidingPairs) {
      let nShared = 0;
      let nMatches = 0;
      let divergentVideo: Video | null = null;

      for (const otherFrame of other.labeledFrames) {
        const mappedVideo = videoMap.get(otherFrame.video) ?? otherFrame.video;
        const matchingFrames = this.find({
          video: mappedVideo,
          frameIdx: otherFrame.frameIdx,
        });
        if (matchingFrames.length === 0) {
          continue;
        }

        const selfInsts: Instance[] = [];
        for (const frame of matchingFrames) {
          for (const inst of frame.instances) {
            if (inst.track === selfTrack) {
              selfInsts.push(inst);
            }
          }
        }
        const otherInsts = otherFrame.instances.filter(
          (inst) => inst.track === otherTrack,
        );
        if (selfInsts.length === 0 || otherInsts.length === 0) {
          continue;
        }

        nShared += 1;
        nMatches += instanceMatcher.findMatches(selfInsts, otherInsts).length;
        if (divergentVideo === null) {
          divergentVideo = mappedVideo;
        }
      }

      if (nShared >= 1 && nMatches === 0) {
        const videoRepr =
          divergentVideo != null
            ? filenameRepr(divergentVideo.filename)
            : "None";
        console.warn(
          `Track '${selfTrack.name}' was merged by name across labels that ` +
            `share video ${videoRepr} but instances on that track ` +
            `diverge spatially on all ${nShared} overlapping frame(s) (no ` +
            `instance matched under the merge's instance matcher). If these ` +
            `tracking runs label different animals, name-based merging may ` +
            `glue distinct tracks together. Review the merge or resolve tracks ` +
            `at the instance level.`,
        );
      }
    }
  }

  /**
   * Build correspondence maps between this `Labels` and another WITHOUT mutating
   * either (read-only twin of {@link merge}).
   *
   * Faithful port of Python `Labels.match` (labels.py:3020-3147). Coerces only
   * the video/skeleton/track matchers (NO instance matcher, NO error mode). No
   * lazy guard, no try/except, no provenance, no mutation. AUTO videos use the
   * full `findMatch` cascade; every other method (including IMAGE_DEDUP/SHAPE)
   * uses a simple first-match-wins loop. Unmatched -> `null`.
   *
   * Async (DECISIONS D8): the AUTO cascade awaits filesystem/pixel reads.
   *
   * @param other - The `Labels` to match against (maps `other` -> `self`).
   * @param opts.video - Video matcher (`null` -> AUTO).
   * @param opts.skeleton - Skeleton matcher (`null` -> STRUCTURE).
   * @param opts.track - Track matcher (`null` -> IDENTITY). The default matches
   *   tracks only by object identity (the same Track instance); all other tracks
   *   map to `null` — a correctness-first default that never collapses distinct
   *   tracks by their (often arbitrary, tracker-assigned) names. Pass `"name"` to
   *   match tracks by their name attribute instead, for cases where track names
   *   are semantically meaningful (e.g. user-assigned identities or
   *   identity-classification model outputs).
   */
  async match(
    other: Labels,
    opts: {
      video?: string | VideoMatcher | null;
      skeleton?: string | SkeletonMatcher | null;
      track?: string | TrackMatcher | null;
    } = {},
  ): Promise<MatchResult> {
    const skeletonMatcher = coerceSkeletonMatcher(opts.skeleton);
    const videoMatcher = coerceVideoMatcher(opts.video);
    const trackMatcher = coerceTrackMatcher(opts.track);

    const result = new MatchResult();

    // Skeletons: first self match wins, else null.
    for (const otherSkel of other.skeletons) {
      let matchedSkel: Skeleton | null = null;
      for (const selfSkel of this.skeletons) {
        if (skeletonMatcher.match(selfSkel, otherSkel)) {
          matchedSkel = selfSkel;
          break;
        }
      }
      result.skeletonMap.set(otherSkel, matchedSkel);
    }

    // Videos: AUTO uses the full cascade; others use a first-match-wins loop.
    for (const otherVideo of other.videos) {
      let matchedVideo: Video | null;
      if (videoMatcher.method === VideoMatchMethod.AUTO) {
        matchedVideo = await videoMatcher.findMatch(otherVideo, this.videos, {
          labelsIncoming: other,
          labelsBase: this,
        });
      } else {
        matchedVideo = null;
        for (const selfVideo of this.videos) {
          if (await videoMatcher.match(selfVideo, otherVideo)) {
            matchedVideo = selfVideo;
            break;
          }
        }
      }
      result.videoMap.set(otherVideo, matchedVideo);
    }

    // Tracks: first self match wins, else null.
    for (const otherTrack of other.tracks) {
      let matchedTrack: Track | null = null;
      for (const selfTrack of this.tracks) {
        if (trackMatcher.match(selfTrack, otherTrack)) {
          matchedTrack = selfTrack;
          break;
        }
      }
      result.trackMap.set(otherTrack, matchedTrack);
    }

    return result;
  }

  /**
   * Resolve a video argument to the canonical `Video` in this `Labels` (SYNC).
   *
   * Mirrors Python `Labels._resolve_video` (labels.py:1346-1374). Used internally
   * by the video-accepting query methods ({@link find}, {@link numpy},
   * {@link extract}, and the `get*` family) to canonicalize a foreign `Video`,
   * filename, or index so that identity-based lookups succeed.
   *
   * DOCUMENTED DIVERGENCE (DECISIONS-107): unlike the async {@link matchVideo},
   * this resolver is SYNCHRONOUS and therefore does NOT perform inode/pose/image
   * matching. It uses only the synchronous matching subset:
   *   1. identity (`===`),
   *   2. unique `v.matchesPath(query, true)` (strict; posix-normalized),
   *   3. unique `v.matchesPath(query, false)` (basename),
   * raising on ambiguity (>1 match at a tier) with messages mirroring
   * {@link matchVideo}. For all in-memory and non-existent-file lookups (the
   * realistic case) this is observably identical to Python's `match_video`-based
   * resolution, since strict `matchesPath` already does normalized path equality.
   *
   * @param video - A `Video`, filename (`string`/`URL`), integer index into
   *   `this.videos`, or `null`/`undefined`.
   * @returns The canonical `Video`, or `null` if `video` is `null`/`undefined`.
   *   If no video matches, the foreign `Video` is returned unchanged and a
   *   path is coerced into a new (unopened) `Video`, so identity-based lookups
   *   simply yield empty results (preserving the "no match" behavior).
   */
  private _resolveVideo(
    video: Video | string | URL | number | null | undefined,
  ): Video | null {
    if (video == null) return null;
    if (typeof video === "number") return this.videos[video];

    // Coerce the query into a Video (path -> unopened Video).
    const query =
      video instanceof Video
        ? video
        : new Video({ filename: String(video), openBackend: false });

    // Identity short-circuit.
    for (const v of this.videos) {
      if (v === query) return v;
    }

    const ambiguous = (candidates: Video[], by: string): Error => {
      const names = candidates.map((v) => filenameRepr(v.filename)).join(", ");
      return new Error(
        `Ambiguous video match for ${filenameRepr(query.filename)}: matched ` +
          `${candidates.length} videos ${by}: ${names}.`,
      );
    };

    // Tier 1: strict (posix-normalized) path match.
    const strict = this.videos.filter((v) => v.matchesPath(query, true));
    if (strict.length > 1) {
      throw ambiguous(strict, "by file identity");
    }
    if (strict.length) return strict[0];

    // Tier 2: basename match.
    const byBasename = this.videos.filter((v) => v.matchesPath(query, false));
    if (byBasename.length > 1) {
      throw ambiguous(byBasename, "by basename");
    }
    if (byBasename.length) return byBasename[0];

    // No match: return a usable Video so callers can still attach it to new
    // frames; identity-based lookups against it simply yield empty results.
    return query;
  }

  /**
   * Resolve a foreign `Video` or path to the canonical `Video` in `this.videos`.
   *
   * Faithful port of Python `Labels.match_video` (labels.py:1216-1344). Uses its
   * OWN simpler cascade (NOT `findMatch`). Method validation runs BEFORE the
   * identity short-circuit. RAISES on ambiguity (>1 candidate), unlike
   * {@link match} which silently takes the first.
   *
   * Async (DECISIONS D8): the file-identity tier awaits `isSameFile` / FS checks.
   *
   * @param videoOrPath - A `Video`, or a filename string (wrapped in an unopened
   *   `Video`).
   * @param method - `"auto"` (default), another method string, or a
   *   `VideoMatcher`. AUTO (string or matcher) uses the tiered cascade.
   * @returns The canonical `Video` from `this.videos`, or `null` if none match.
   */
  async matchVideo(
    videoOrPath: Video | string,
    method: string | VideoMatcher = "auto",
  ): Promise<Video | null> {
    // Coerce the query into a Video (path -> unopened Video).
    let query: Video;
    if (videoOrPath instanceof Video) {
      query = videoOrPath;
    } else if (typeof videoOrPath === "string") {
      query = new Video({ filename: videoOrPath, openBackend: false });
    } else {
      throw new TypeError(
        `match_video() expects a Video, str, or Path, got ${
          (videoOrPath as { constructor?: { name?: string } })?.constructor
            ?.name ?? typeof videoOrPath
        }.`,
      );
    }

    // Normalize the strategy (runs BEFORE the identity short-circuit). AUTO ->
    // null sentinel (tiered cascade); non-AUTO -> a VideoMatcher.
    let matcher: VideoMatcher | null;
    if (typeof method === "string") {
      const methodEnum = toVideoMatchMethod(method); // throws on unknown
      matcher =
        methodEnum === VideoMatchMethod.AUTO
          ? null
          : new VideoMatcher(methodEnum);
    } else if (method instanceof VideoMatcher) {
      matcher = method.method === VideoMatchMethod.AUTO ? null : method;
    } else {
      throw new TypeError(
        `match_video() expects method to be a str or VideoMatcher, got ${
          (method as { constructor?: { name?: string } })?.constructor?.name ??
          typeof method
        }.`,
      );
    }

    // Identity short-circuit.
    for (const video of this.videos) {
      if (video === query) return video;
    }

    const ambiguous = (candidates: Video[], by: string): Error => {
      const names = candidates.map((v) => filenameRepr(v.filename)).join(", ");
      return new Error(
        `Ambiguous video match for ${filenameRepr(query.filename)}: matched ` +
          `${candidates.length} videos ${by}: ${names}.`,
      );
    };

    if (matcher === null) {
      // AUTO tiered cascade.
      const definitive: Video[] = [];
      for (const v of this.videos) {
        if ((await isSameFile(v, query)) || v.matchesPath(query, true)) {
          definitive.push(v);
        }
      }
      if (definitive.length > 1) {
        throw ambiguous(definitive, "by file identity");
      }
      if (definitive.length) {
        return definitive[0];
      }

      const byBasename = this.videos.filter((v) => v.matchesPath(query, false));
      if (byBasename.length > 1) {
        throw ambiguous(byBasename, "by basename");
      }
      return byBasename.length ? byBasename[0] : null;
    }

    // Explicit (non-AUTO) strategy.
    const matches: Video[] = [];
    for (const v of this.videos) {
      if (await matcher.match(v, query)) matches.push(v);
    }
    if (matches.length > 1) {
      throw ambiguous(matches, `with method '${matcher.method}'`);
    }
    return matches.length ? matches[0] : null;
  }

  /**
   * Remove empty frames, unused skeletons, tracks and videos.
   *
   * Mirrors Python `Labels.clean` (labels.py:1577-1682). In-place, returns
   * void. This is an explicit opt-in operation (never auto-run on load).
   *
   * @param opts.frames - If `true` (default), remove empty frames. Negative
   *   frames (`isNegative === true`) and annotation-only frames are preserved.
   * @param opts.emptyInstances - If `true` (NOT default), remove instances with
   *   no visible points (before the emptiness check).
   * @param opts.skeletons - If `true` (default), remove unused skeletons.
   * @param opts.tracks - If `true` (default), remove unused tracks and the
   *   annotations/objects that reference removed tracks (track=null is always
   *   preserved).
   * @param opts.videos - If `true` (NOT default), remove videos with no labeled
   *   frames.
   */
  clean(opts?: {
    frames?: boolean;
    emptyInstances?: boolean;
    skeletons?: boolean;
    tracks?: boolean;
    videos?: boolean;
  }): void {
    // JS convention (and ARCHITECTURE §7.3): materialize lazy Labels rather than
    // raising, unlike Python's clean() which calls _check_not_lazy.
    if (this._lazyFrameList) this.materialize();

    const frames = opts?.frames ?? true;
    const emptyInstances = opts?.emptyInstances ?? false;
    const skeletons = opts?.skeletons ?? true;
    const tracks = opts?.tracks ?? true;
    const videos = opts?.videos ?? false;

    const usedSkeletons: Skeleton[] = [];
    const usedTracks: Track[] = [];
    const usedVideos: Video[] = [];
    const keptFrames: LabeledFrame[] = [];

    for (const lf of this.labeledFrames) {
      if (emptyInstances) {
        lf.removeEmptyInstances();
      }

      // A frame is non-empty if it has pose instances or any annotations.
      const hasAnnotations =
        lf.centroids.length > 0 ||
        lf.bboxes.length > 0 ||
        lf.masks.length > 0 ||
        lf.labelImages.length > 0 ||
        lf.rois.length > 0;

      // Empty-frame skip: instances here = POSE instances only.
      if (
        frames &&
        lf.instances.length === 0 &&
        !lf.isNegative &&
        !hasAnnotations
      ) {
        continue;
      }

      if (videos && !usedVideos.includes(lf.video)) {
        usedVideos.push(lf.video);
      }

      if (skeletons || tracks) {
        for (const inst of lf.instances) {
          if (skeletons && !usedSkeletons.includes(inst.skeleton)) {
            usedSkeletons.push(inst.skeleton);
          }
          if (
            tracks &&
            inst.track != null &&
            !usedTracks.includes(inst.track)
          ) {
            usedTracks.push(inst.track);
          }
        }
      }

      // Also collect tracks from annotations (centroids -> bboxes -> masks ->
      // rois, then label-image objects in insertion order).
      if (tracks) {
        for (const ann of [
          ...lf.centroids,
          ...lf.bboxes,
          ...lf.masks,
          ...lf.rois,
        ] as Array<Centroid | BoundingBox | SegmentationMask | ROI>) {
          if (ann.track != null && !usedTracks.includes(ann.track)) {
            usedTracks.push(ann.track);
          }
        }
        for (const li of lf.labelImages) {
          for (const info of li.objects.values()) {
            if (info.track != null && !usedTracks.includes(info.track)) {
              usedTracks.push(info.track);
            }
          }
        }
      }

      if (frames) {
        keptFrames.push(lf);
      }
    }

    if (videos) {
      this.videos = this.videos.filter((v) => usedVideos.includes(v));
    }

    if (skeletons) {
      this.skeletons = this.skeletons.filter((s) => usedSkeletons.includes(s));
    }

    if (tracks) {
      this.tracks = this.tracks.filter((t) => usedTracks.includes(t));

      // Remove annotations within frames that reference removed tracks.
      const validTracks = new Set(this.tracks);
      const targetFrames = frames ? keptFrames : this.labeledFrames;
      for (const lf of targetFrames) {
        if (lf.centroids.length) {
          lf.centroids = lf.centroids.filter(
            (a) => a.track == null || validTracks.has(a.track),
          );
        }
        if (lf.bboxes.length) {
          lf.bboxes = lf.bboxes.filter(
            (a) => a.track == null || validTracks.has(a.track),
          );
        }
        if (lf.masks.length) {
          lf.masks = lf.masks.filter(
            (a) => a.track == null || validTracks.has(a.track),
          );
        }
        if (lf.rois.length) {
          lf.rois = lf.rois.filter(
            (a) => a.track == null || validTracks.has(a.track),
          );
        }
        if (lf.labelImages.length) {
          for (const li of lf.labelImages) {
            if (li.objects.size) {
              const kept = new Map(li.objects);
              for (const [k, v] of li.objects) {
                if (!(v.track == null || validTracks.has(v.track))) {
                  kept.delete(k);
                }
              }
              li.objects = kept;
            }
          }
        }
      }
    }

    if (frames) {
      this.labeledFrames = keptFrames;
    }

    this._invalidateIndices();
  }

  /**
   * Extract a set of frames into a new Labels object.
   *
   * Mirrors Python `Labels.extract` (labels.py:2482-2551). Copies the selected
   * frames and their reachable graph (instances/skeletons/tracks/videos/
   * annotations) with structural sharing (each shared object copied once), keeps
   * the relative ordering of tracks/skeletons by NAME, copies/dedups suggestions
   * for the extracted videos, and records the source labels in provenance.
   *
   * @param inds - Frame selection: an array of integer indices, an array of
   *   `[Video, frameIdx]` tuples, or a single `Video` (all of its frames). A
   *   foreign `Video`/filename (`string`/`URL`) selector or tuple element is
   *   resolved to the matching project `Video` via {@link _resolveVideo} (SYNC;
   *   see its documented divergence from `matchVideo`).
   * @param copy - If `true` (default), deep-copy the frames and containing
   *   objects; otherwise share references with this Labels.
   * @returns A new `Labels` containing the selected frames.
   */
  extract(
    inds:
      | number[]
      | Array<[Video | string | URL, number]>
      | Video
      | string
      | URL,
    copy = true,
  ): Labels {
    if (this._lazyFrameList) this.materialize();

    // Canonicalize any foreign Video / filename selector or tuple element to the
    // matching project Video so `_selectFrames` receives canonical Videos.
    const resolvedInds = this._resolveExtractInds(inds);

    let lfs = this._selectFrames(resolvedInds);

    if (copy) {
      lfs = this._deepCopyFrames(lfs);
    }
    const labels = new Labels({ labeledFrames: lfs });

    // Keep the track list in the same relative order as the source by NAME.
    const trackToInd = new Map<string, number>();
    this.tracks.forEach((t, i) => {
      trackToInd.set(t.name, i);
    });
    labels.tracks = labels.tracks
      .map((t, i) => [t, i] as const)
      .sort((a, b) => {
        const ka = trackToInd.get(a[0].name) ?? 0;
        const kb = trackToInd.get(b[0].name) ?? 0;
        return ka === kb ? a[1] - b[1] : ka - kb;
      })
      .map(([t]) => t);

    // Keep the skeleton list in source order by NAME.
    const skelToInd = new Map<string, number>();
    this.skeletons.forEach((s, i) => {
      skelToInd.set(s.name ?? "", i);
    });
    labels.skeletons = labels.skeletons
      .map((s, i) => [s, i] as const)
      .sort((a, b) => {
        const ka = skelToInd.get(a[0].name ?? "") ?? 0;
        const kb = skelToInd.get(b[0].name ?? "") ?? 0;
        return ka === kb ? a[1] - b[1] : ka - kb;
      })
      .map(([s]) => s);

    // Copy suggestion frames for the extracted videos. Re-select on the ORIGINAL
    // frames to read their (un-copied) videos for membership.
    const extractedVideos = new Set<Video>(
      this._selectFrames(resolvedInds).map((lf) => lf.video),
    );
    let suggestions = this.suggestions.filter((sf) =>
      extractedVideos.has(sf.video),
    );
    if (copy) {
      suggestions = suggestions.map(
        (sf) =>
          new SuggestionFrame({
            video: sf.video,
            frameIdx: sf.frameIdx,
            group: sf.group,
            metadata: { ...sf.metadata },
          }),
      );
    }

    // De-duplicate suggestion videos against labels.videos.
    for (const sf of suggestions) {
      for (const vid of labels.videos) {
        if (vid.matchesContent(sf.video) && vid.matchesPath(sf.video)) {
          sf.video = vid;
          break;
        }
      }
    }

    labels.suggestions.push(...suggestions);
    labels.update();

    labels.provenance = { ...labels.provenance };
    labels.provenance.source_labels =
      (this.provenance.filename as string | undefined) ?? null;

    return labels;
  }

  /**
   * Canonicalize an {@link extract} selector, resolving foreign `Video` /
   * filename references to the matching project `Video` via {@link _resolveVideo}
   * (SYNC). The `number[]` index-array path is returned unchanged. Returns a
   * narrowed selector that {@link _selectFrames} can consume directly.
   */
  private _resolveExtractInds(
    inds:
      | number[]
      | Array<[Video | string | URL, number]>
      | Video
      | string
      | URL,
  ): number[] | Array<[Video, number]> | Video {
    if (inds instanceof Video) {
      return this._resolveVideo(inds) as Video;
    }
    if (typeof inds === "string" || inds instanceof URL) {
      return this._resolveVideo(inds) as Video;
    }
    if (Array.isArray(inds)) {
      if (inds.length === 0) return inds as number[];
      if (Array.isArray(inds[0])) {
        return (inds as Array<[Video | string | URL, number]>).map(
          ([video, frameIdx]) =>
            [this._resolveVideo(video) as Video, frameIdx] as [Video, number],
        );
      }
      return inds as number[];
    }
    return inds;
  }

  /**
   * Resolve an extraction selection to a list of LabeledFrame references.
   *
   * Supports the subset of Python `__getitem__` selectors needed by
   * `extract`/`split`: integer index arrays, `[Video, frameIdx]` tuple arrays,
   * and a single `Video`. Foreign `Video`/filename references are canonicalized
   * by {@link _resolveExtractInds} before reaching this method, so it receives
   * canonical project `Video` instances.
   */
  private _selectFrames(
    inds: number[] | Array<[Video, number]> | Video,
  ): LabeledFrame[] {
    if (inds instanceof Video) {
      return this.find({ video: inds });
    }
    if (Array.isArray(inds)) {
      if (inds.length === 0) return [];
      if (Array.isArray(inds[0])) {
        const tuples = inds as Array<[Video, number]>;
        const result: LabeledFrame[] = [];
        for (const [video, frameIdx] of tuples) {
          const res = this.find({ video, frameIdx });
          if (res.length === 1) {
            result.push(res[0]);
          } else if (res.length === 0) {
            throw new Error(
              `No labeled frames found for video ${video} and frame index ${frameIdx}.`,
            );
          }
        }
        return result;
      }
      return (inds as number[]).map((i) => this.labeledFrames[i]);
    }
    return [];
  }

  /**
   * Deep-copy a list of frames with structural sharing.
   *
   * Reproduces Python `deepcopy(lfs)`: shared Track/Skeleton/Video objects within
   * the selected subgraph are copied exactly once (via memo maps), so references
   * shared across frames/instances remain shared in the copy.
   */
  private _deepCopyFrames(frames: LabeledFrame[]): LabeledFrame[] {
    const videoMap = new Map<Video, Video>();
    const skeletonMap = new Map<Skeleton, Skeleton>();
    const trackMap = new Map<Track, Track>();

    const mapVideo = (v: Video): Video => {
      let nv = videoMap.get(v);
      if (!nv) {
        nv = new Video({
          filename: Array.isArray(v.filename) ? [...v.filename] : v.filename,
          backendMetadata: { ...v.backendMetadata },
          openBackend: v.openBackend,
          embedded: v.hasEmbeddedImages,
        });
        nv.shape = v.shape;
        nv.fps = v.fps;
        videoMap.set(v, nv);
      }
      return nv;
    };

    const mapSkeleton = (s: Skeleton): Skeleton => {
      let ns = skeletonMap.get(s);
      if (!ns) {
        const nodeMap = new Map<Node, Node>();
        const newNodes = s.nodes.map((n) => {
          const nn = new Node(n.name);
          nodeMap.set(n, nn);
          return nn;
        });
        const newEdges = s.edges.map(
          (e) => new Edge(nodeMap.get(e.source)!, nodeMap.get(e.destination)!),
        );
        const newSymmetries = s.symmetries.map((sym) => {
          const nodes = [...sym.nodes];
          return new Symmetry([nodeMap.get(nodes[0])!, nodeMap.get(nodes[1])!]);
        });
        ns = new Skeleton({
          nodes: newNodes,
          edges: newEdges,
          symmetries: newSymmetries,
          name: s.name,
        });
        skeletonMap.set(s, ns);
      }
      return ns;
    };

    const mapTrack = (t: Track | null | undefined): Track | null => {
      if (t == null) return null;
      let nt = trackMap.get(t);
      if (!nt) {
        nt = new Track(t.name);
        trackMap.set(t, nt);
      }
      return nt;
    };

    const cloneInstance = (
      inst: Instance | PredictedInstance,
    ): Instance | PredictedInstance => {
      const newPoints = inst.points.map((p) => clonePoint(p));
      const newSkeleton = mapSkeleton(inst.skeleton);
      const newTrack = mapTrack(inst.track);
      if (inst.constructor === PredictedInstance) {
        const predicted = inst as PredictedInstance;
        return new PredictedInstance({
          points: newPoints as unknown as PredictedPointsArray,
          skeleton: newSkeleton,
          track: newTrack,
          score: predicted.score,
          trackingScore: predicted.trackingScore,
        });
      }
      return new Instance({
        points: newPoints,
        skeleton: newSkeleton,
        track: newTrack,
        trackingScore: inst.trackingScore,
      });
    };

    // Clone an ancillary annotation: shallow-clone preserving prototype, then
    // remap its shared video/track refs (and nested label-image object refs)
    // through the memo maps so structural sharing is preserved.
    const cloneAncillary = <T extends object>(items: T[]): T[] =>
      items.map((item) => {
        const clone = Object.create(
          Object.getPrototypeOf(item),
          Object.getOwnPropertyDescriptors(item),
        ) as T;
        const anyClone = clone as Record<string, unknown>;
        if ("video" in anyClone && anyClone.video != null) {
          anyClone.video = mapVideo(anyClone.video as Video);
        }
        if ("track" in anyClone && anyClone.track != null) {
          anyClone.track = mapTrack(anyClone.track as Track);
        }
        // Drop instance back-refs (cannot be remapped without global identity).
        if ("instance" in anyClone) {
          anyClone.instance = null;
        }
        // LabelImage: copy the objects Map and remap nested track refs.
        if ("objects" in anyClone && anyClone.objects instanceof Map) {
          const oldObjects = anyClone.objects as Map<
            number,
            Record<string, unknown>
          >;
          const newObjects = new Map<number, Record<string, unknown>>();
          for (const [id, info] of oldObjects) {
            const newInfo = { ...info };
            if (newInfo.track != null) {
              newInfo.track = mapTrack(newInfo.track as Track);
            }
            newInfo.instance = null;
            newObjects.set(id, newInfo);
          }
          anyClone.objects = newObjects;
        }
        return clone;
      });

    return frames.map(
      (f) =>
        new LabeledFrame({
          video: mapVideo(f.video),
          frameIdx: f.frameIdx,
          instances: f.instances.map(cloneInstance),
          isNegative: f.isNegative,
          centroids: cloneAncillary(f.centroids) as Centroid[],
          bboxes: cloneAncillary(f.bboxes) as BoundingBox[],
          masks: cloneAncillary(f.masks) as SegmentationMask[],
          labelImages: cloneAncillary(f.labelImages) as LabelImage[],
          rois: cloneAncillary(f.rois) as ROI[],
        }),
    );
  }

  /**
   * Separate the labels into two random splits.
   *
   * Mirrors Python `Labels.split` (labels.py:2553-2607) for the count/branch
   * logic. Per DECISIONS D5, the index selection uses a deterministic seeded
   * RNG (NOT NumPy PCG64) — counts and edge cases match Python exactly, but the
   * specific frames chosen are not bit-identical to NumPy.
   *
   * @param n - Size of the first split. `>= 1` is an absolute frame count;
   *   `< 1.0` is a fraction of the total (`max(trunc(n0*n), 1)`).
   * @param seed - Optional integer seed for reproducibility within JS. When
   *   omitted/null, a fixed default seed is used.
   * @returns A `LabelsSet` with keys `"split1"` and `"split2"`.
   */
  split(n: number, seed?: number | null): LabelsSet {
    if (this._lazyFrameList) this.materialize();

    const n0 = this.labeledFrames.length;
    if (n0 === 0) {
      return new LabelsSet({ split1: this, split2: this });
    }

    let n1: number;
    if (n < 1.0) {
      n1 = Math.max(Math.trunc(n0 * n), 1);
    } else {
      n1 = Math.trunc(n);
    }
    // Python also computes n2 = max(n0 - n1, 1), but it never drives selection
    // (split2 is the sorted set-difference complement), so it is elided here.

    // Deterministic seeded RNG (mulberry32). Default seed when none provided.
    const rng = Labels._mulberry32(seed == null ? 0x9e3779b9 : seed >>> 0);

    // Choose n1 distinct indices from 0..n0-1 (partial Fisher-Yates).
    const pool = Array.from({ length: n0 }, (_, i) => i);
    const take = Math.min(n1, n0);
    for (let i = 0; i < take; i += 1) {
      const j = i + Math.floor(rng() * (n0 - i));
      const tmp = pool[i];
      pool[i] = pool[j];
      pool[j] = tmp;
    }
    const inds1 = pool.slice(0, take);

    let inds2: number[];
    if (n0 === 1) {
      inds2 = [0];
    } else {
      const inds1Set = new Set(inds1);
      inds2 = [];
      for (let i = 0; i < n0; i += 1) {
        if (!inds1Set.has(i)) inds2.push(i);
      }
    }

    const split1 = this.extract(inds1, true);
    const split2 = this.extract(inds2, true);

    return new LabelsSet({ split1, split2 });
  }

  /** Deterministic 32-bit RNG (mulberry32). Returns floats in [0, 1). */
  private static _mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
}
