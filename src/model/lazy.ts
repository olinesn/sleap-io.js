import { LabeledFrame } from "./labeled-frame.js";
import {
  Instance,
  PredictedInstance,
  Track,
  pointsFromArray,
  predictedPointsFromArray,
} from "./instance.js";
import { Skeleton } from "./skeleton.js";
import { Video } from "./video.js";
import { buildVideoIdMap } from "./video-id-map.js";
import type { Centroid } from "./centroid.js";
import type { BoundingBox } from "./bbox.js";
import type { SegmentationMask } from "./mask.js";
import type { LabelImage } from "./label-image.js";
import type { ROI } from "./roi.js";
import type { Identity } from "./identity.js";
import type { Embedding } from "./embedding.js";

/**
 * Raw data store holding HDF5 dataset arrays for lazy materialization.
 * Keeps the parsed column data from frames/instances/points datasets
 * so individual frames can be materialized on demand.
 */
export class LazyDataStore {
  framesData: Record<string, any[]>;
  instancesData: Record<string, any[]>;
  pointsData: Record<string, any[]>;
  predPointsData: Record<string, any[]>;
  skeletons: Skeleton[];
  tracks: Track[];
  videos: Video[];
  formatId: number;
  negativeFrames: Set<string>;

  /**
   * Memoized raw-`frames.video`-id -> `videos` index map (see
   * {@link buildVideoIdMap}). Lazily built on first frame access; rebuilt from
   * scratch by a copied store since `copy()` does not carry it over.
   */
  private _videoIdToIndex?: Map<number, number>;

  // Per-frame annotation lookups: "videoIdx:frameIdx" -> annotation[]
  _centroidByFrame: Map<string, Centroid[]> = new Map();
  _bboxByFrame: Map<string, BoundingBox[]> = new Map();
  _maskByFrame: Map<string, SegmentationMask[]> = new Map();
  _labelImageByFrame: Map<string, LabelImage[]> = new Map();
  _roiByFrame: Map<string, ROI[]> = new Map();

  // Undistributed annotations (video=null or frameIdx=null, e.g. static ROIs)
  _undistributedCentroids: Centroid[] = [];
  _undistributedBboxes: BoundingBox[] = [];
  _undistributedMasks: SegmentationMask[] = [];
  _undistributedLabelImages: LabelImage[] = [];
  _undistributedRois: ROI[] = [];

  constructor(options: {
    framesData: Record<string, any[]>;
    instancesData: Record<string, any[]>;
    pointsData: Record<string, any[]>;
    predPointsData: Record<string, any[]>;
    skeletons: Skeleton[];
    tracks: Track[];
    videos: Video[];
    formatId: number;
    negativeFrames?: Set<string>;
    identities?: Identity[];
    instanceIdentityLinks?: Map<number, [number, number | null]>;
    instanceEmbeddings?: Map<number, Embedding>;
    categoryCatalog?: string[];
    instanceCategoryLinks?: Map<number, [number, number | null]>;
    instanceCategoryEmbeddings?: Map<number, Embedding>;
  }) {
    this.framesData = options.framesData;
    this.instancesData = options.instancesData;
    this.pointsData = options.pointsData;
    this.predPointsData = options.predPointsData;
    this.skeletons = options.skeletons;
    this.tracks = options.tracks;
    this.videos = options.videos;
    this.formatId = options.formatId;
    this.negativeFrames = options.negativeFrames ?? new Set();
    this.identities = options.identities;
    this._instanceIdentityLinks = options.instanceIdentityLinks;
    this._instanceEmbeddings = options.instanceEmbeddings;
    this.categoryCatalog = options.categoryCatalog;
    this._instanceCategoryLinks = options.instanceCategoryLinks;
    this._instanceCategoryEmbeddings = options.instanceCategoryEmbeddings;
  }

  /** Identity catalog for resolving per-instance identity links (SLP 2.5). @internal */
  identities?: Identity[];
  /** owner_id (global instance_id) → [identity_idx, score|null], OWNER_INSTANCE. @internal */
  _instanceIdentityLinks?: Map<number, [number, number | null]>;
  /** owner_id (global instance_id) → Embedding, OWNER_INSTANCE. @internal */
  _instanceEmbeddings?: Map<number, Embedding>;
  /** Category catalog (strings) for resolving per-instance category links (SLP 2.7). @internal */
  categoryCatalog?: string[];
  /** owner_id (global instance_id) → [category_idx, score|null], OWNER_INSTANCE. @internal */
  _instanceCategoryLinks?: Map<number, [number, number | null]>;
  /** owner_id (global instance_id) → category Embedding, OWNER_INSTANCE. @internal */
  _instanceCategoryEmbeddings?: Map<number, Embedding>;

  /**
   * Create an independent copy of this store's raw column data.
   * Videos, skeletons, and tracks arrays are shared (not cloned) —
   * the caller is expected to replace them with new references.
   */
  copy(): LazyDataStore {
    const copyRecord = (rec: Record<string, any[]>): Record<string, any[]> => {
      const out: Record<string, any[]> = {};
      for (const key of Object.keys(rec)) {
        out[key] = rec[key].slice();
      }
      return out;
    };

    const copyAnnMap = <T>(map: Map<string, T[]>): Map<string, T[]> => {
      const out = new Map<string, T[]>();
      for (const [key, list] of map) {
        out.set(key, [...list]);
      }
      return out;
    };

    const newStore = new LazyDataStore({
      framesData: copyRecord(this.framesData),
      instancesData: copyRecord(this.instancesData),
      pointsData: copyRecord(this.pointsData),
      predPointsData: copyRecord(this.predPointsData),
      skeletons: this.skeletons,
      tracks: this.tracks,
      videos: this.videos,
      formatId: this.formatId,
      negativeFrames: new Set(this.negativeFrames),
      identities: this.identities,
      instanceIdentityLinks: this._instanceIdentityLinks,
      instanceEmbeddings: this._instanceEmbeddings,
      categoryCatalog: this.categoryCatalog,
      instanceCategoryLinks: this._instanceCategoryLinks,
      instanceCategoryEmbeddings: this._instanceCategoryEmbeddings,
    });

    // Copy per-frame annotation dicts
    newStore._centroidByFrame = copyAnnMap(this._centroidByFrame);
    newStore._bboxByFrame = copyAnnMap(this._bboxByFrame);
    newStore._maskByFrame = copyAnnMap(this._maskByFrame);
    newStore._labelImageByFrame = copyAnnMap(this._labelImageByFrame);
    newStore._roiByFrame = copyAnnMap(this._roiByFrame);

    // Copy undistributed annotations
    newStore._undistributedCentroids = [...this._undistributedCentroids];
    newStore._undistributedBboxes = [...this._undistributedBboxes];
    newStore._undistributedMasks = [...this._undistributedMasks];
    newStore._undistributedLabelImages = [...this._undistributedLabelImages];
    newStore._undistributedRois = [...this._undistributedRois];

    return newStore;
  }

  /** Total number of frames in the store. */
  get frameCount(): number {
    return (this.framesData.frame_id ?? []).length;
  }

  /**
   * Resolve a raw `frames.video` id to its `videos` array index, applying the
   * same remap the eager readers use so sparse / non-contiguous group ids
   * (e.g. `video0`, `video2`) resolve to the correct video. Falls back to the
   * raw id when unmapped — identical to `buildLabeledFrames`.
   */
  private videoIndexFor(rawVideoId: number): number {
    this._videoIdToIndex ??= buildVideoIdMap(this.framesData, this.videos);
    return this._videoIdToIndex.get(rawVideoId) ?? rawVideoId;
  }

  /**
   * Materialize a single LabeledFrame by index.
   */
  materializeFrame(frameIdx: number): LabeledFrame | null {
    const frameIds = this.framesData.frame_id ?? [];
    if (frameIdx < 0 || frameIdx >= frameIds.length) return null;

    const rawVideoId = Number(this.framesData.video?.[frameIdx] ?? 0);
    const videoIndex = this.videoIndexFor(rawVideoId);
    const frameIndex = Number(this.framesData.frame_idx?.[frameIdx] ?? 0);
    const instStart = Number(
      this.framesData.instance_id_start?.[frameIdx] ?? 0,
    );
    const instEnd = Number(this.framesData.instance_id_end?.[frameIdx] ?? 0);
    const video = this.videos[videoIndex];
    if (!video) return null;

    const instances: Array<Instance | PredictedInstance> = [];
    const instanceById = new Map<number, Instance | PredictedInstance>();
    const fromPredictedPairs: Array<[number, number]> = [];

    for (let instIdx = instStart; instIdx < instEnd; instIdx++) {
      const instanceType = Number(
        this.instancesData.instance_type?.[instIdx] ?? 0,
      );
      const skeletonId = Number(this.instancesData.skeleton?.[instIdx] ?? 0);
      const trackId = Number(this.instancesData.track?.[instIdx] ?? -1);
      const pointStart = Number(
        this.instancesData.point_id_start?.[instIdx] ?? 0,
      );
      const pointEnd = Number(this.instancesData.point_id_end?.[instIdx] ?? 0);
      const score = Number(this.instancesData.score?.[instIdx] ?? 0);
      const rawTrackingScore =
        this.formatId < 1.2
          ? 0
          : Number(this.instancesData.tracking_score?.[instIdx] ?? 0);
      const trackingScore = Number.isNaN(rawTrackingScore)
        ? 0
        : rawTrackingScore;
      const fromPredicted = Number(
        this.instancesData.from_predicted?.[instIdx] ?? -1,
      );
      const skeleton = this.skeletons[skeletonId] ?? this.skeletons[0];
      const track = trackId >= 0 ? this.tracks[trackId] : null;

      let instance: Instance | PredictedInstance;
      if (instanceType === 0) {
        const points = this.slicePoints(this.pointsData, pointStart, pointEnd);
        instance = new Instance({
          points: pointsFromArray(points, skeleton.nodeNames),
          skeleton,
          track,
          trackingScore,
        });
        if (this.formatId < 1.1) {
          instance.points.forEach((point) => {
            point.xy = [point.xy[0] - 0.5, point.xy[1] - 0.5];
          });
        }
        if (fromPredicted >= 0) {
          fromPredictedPairs.push([instIdx, fromPredicted]);
        }
      } else {
        const points = this.slicePoints(
          this.predPointsData,
          pointStart,
          pointEnd,
          true,
        );
        instance = new PredictedInstance({
          points: predictedPointsFromArray(points, skeleton.nodeNames),
          skeleton,
          track,
          score,
          trackingScore,
        });
        if (this.formatId < 1.1) {
          instance.points.forEach((point) => {
            point.xy = [point.xy[0] - 0.5, point.xy[1] - 0.5];
          });
        }
      }

      // Per-detection identity + embedding (SLP 2.5), by global instance_id.
      const link = this._instanceIdentityLinks?.get(instIdx);
      if (
        link &&
        this.identities &&
        link[0] >= 0 &&
        link[0] < this.identities.length
      ) {
        instance.identity = this.identities[link[0]];
        instance.identityScore = link[1];
      }
      const emb = this._instanceEmbeddings?.get(instIdx);
      if (emb) instance.identityEmbedding = emb;

      const catLink = this._instanceCategoryLinks?.get(instIdx);
      if (
        catLink &&
        this.categoryCatalog &&
        catLink[0] >= 0 &&
        catLink[0] < this.categoryCatalog.length
      ) {
        instance.category = this.categoryCatalog[catLink[0]];
        instance.categoryScore = catLink[1];
      }
      const catEmb = this._instanceCategoryEmbeddings?.get(instIdx);
      if (catEmb) instance.categoryEmbedding = catEmb;

      instanceById.set(instIdx, instance);
      instances.push(instance);
    }

    // Resolve from_predicted links
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

    // Attach per-frame annotations from eagerly-loaded dicts
    const annKey = `${videoIndex}:${frameIndex}`;
    const centroids = this._centroidByFrame.get(annKey) ?? [];
    const bboxes = this._bboxByFrame.get(annKey) ?? [];
    const masks = this._maskByFrame.get(annKey) ?? [];
    const labelImages = this._labelImageByFrame.get(annKey) ?? [];
    const rois = this._roiByFrame.get(annKey) ?? [];

    const frame = new LabeledFrame({
      video,
      frameIdx: frameIndex,
      instances,
      centroids,
      bboxes,
      masks,
      labelImages,
      rois,
    });
    // `negativeFrames` holds raw `/negative_frames` ids (copied through on a
    // lazy save), so match on the raw `/frames` id, not the resolved index.
    if (this.negativeFrames.has(`${rawVideoId}:${frameIndex}`)) {
      frame.isNegative = true;
    }
    return frame;
  }

  /**
   * Convert lazy-mode labels to a dense `[frames, tracks, nodes, coords]` array
   * directly from raw column data without materializing any LabeledFrame or
   * Instance objects. Coords is `[x, y]` or `[x, y, score]` when
   * `returnConfidence` is true.
   *
   * @param options.numFrames Optional explicit length of the output's frame
   *   dimension. Takes precedence over `video.shape[0]` (the inferred fallback).
   *   Useful when `video.shape` is null — for example, Mp4Box-backed browser
   *   videos — and you still want a video-length-sized array. If smaller than
   *   `maxLabeledFrame + 1`, it is clamped up so no labeled frames are dropped.
   *   Non-finite, non-positive, or fractional values are sanitized via
   *   `Math.floor` and ignored when `<= 0`.
   */
  toNumpy(options?: {
    video?: Video;
    returnConfidence?: boolean;
    numFrames?: number;
  }): number[][][][] {
    const targetVideo = options?.video ?? this.videos[0];
    if (!targetVideo) return [];

    const targetVideoIdx = this.videos.indexOf(targetVideo);
    if (targetVideoIdx < 0) return [];

    const frameIds = this.framesData.frame_id ?? [];
    const frameVideos = this.framesData.video ?? [];
    const frameIndices = this.framesData.frame_idx ?? [];
    const instStarts = this.framesData.instance_id_start ?? [];
    const instEnds = this.framesData.instance_id_end ?? [];

    // First pass: find max frame index and determine track count
    let maxFrameIdx = 0;
    const trackCount = this.tracks.length
      ? this.tracks.length
      : (() => {
          let maxInst = 1;
          for (let i = 0; i < frameIds.length; i++) {
            if (this.videoIndexFor(Number(frameVideos[i])) !== targetVideoIdx)
              continue;
            const count = Number(instEnds[i]) - Number(instStarts[i]);
            if (count > maxInst) maxInst = count;
          }
          return maxInst;
        })();

    // Collect matching frame indices
    const matchingFrames: number[] = [];
    for (let i = 0; i < frameIds.length; i++) {
      if (this.videoIndexFor(Number(frameVideos[i])) !== targetVideoIdx)
        continue;
      const fi = Number(frameIndices[i]);
      if (fi > maxFrameIdx) maxFrameIdx = fi;
      matchingFrames.push(i);
    }
    if (!matchingFrames.length) return [];

    const rawOverride = options?.numFrames;
    const override =
      Number.isFinite(rawOverride) && (rawOverride as number) > 0
        ? Math.floor(rawOverride as number)
        : 0;
    const effectiveLength =
      override > 0 ? override : (targetVideo.shape?.[0] ?? 0);
    if (effectiveLength > 0) {
      maxFrameIdx = Math.max(maxFrameIdx, effectiveLength - 1);
    }

    const nodeCount = this.skeletons[0]?.nodes.length ?? 0;
    const channelCount = options?.returnConfidence ? 3 : 2;

    // Allocate NaN-filled output
    const output: number[][][][] = Array.from({ length: maxFrameIdx + 1 }, () =>
      Array.from({ length: trackCount }, () =>
        Array.from({ length: nodeCount }, () =>
          Array.from({ length: channelCount }, () => Number.NaN),
        ),
      ),
    );

    // Instance column data
    const instTypes = this.instancesData.instance_type ?? [];
    const instTracks = this.instancesData.track ?? [];
    const instPointStarts = this.instancesData.point_id_start ?? [];
    const instPointEnds = this.instancesData.point_id_end ?? [];
    const instScores = this.instancesData.score ?? [];

    // Point column data
    const px = this.pointsData.x ?? [];
    const py = this.pointsData.y ?? [];
    const ppx = this.predPointsData.x ?? [];
    const ppy = this.predPointsData.y ?? [];
    const ppScores = this.predPointsData.score ?? [];

    const coordOffset = this.formatId < 1.1 ? -0.5 : 0;

    for (const fi of matchingFrames) {
      const frameSlotIdx = Number(frameIndices[fi]);
      const frameSlot = output[frameSlotIdx];
      if (!frameSlot) continue;

      const iStart = Number(instStarts[fi]);
      const iEnd = Number(instEnds[fi]);
      let localIdx = 0;

      for (let instIdx = iStart; instIdx < iEnd; instIdx++) {
        const isPredicted = Number(instTypes[instIdx]) === 1;
        const trackId = Number(instTracks[instIdx]);
        const trackIndex =
          trackId >= 0 && this.tracks.length ? trackId : localIdx;
        localIdx++;

        const trackSlot = frameSlot[trackIndex];
        if (!trackSlot) continue;

        const pStart = Number(instPointStarts[instIdx]);
        const pEnd = Number(instPointEnds[instIdx]);
        const pointCount = Math.min(pEnd - pStart, nodeCount);

        if (isPredicted) {
          for (let p = 0; p < pointCount; p++) {
            const row = trackSlot[p];
            if (!row) continue;
            row[0] = Number(ppx[pStart + p]) + coordOffset;
            row[1] = Number(ppy[pStart + p]) + coordOffset;
            if (channelCount === 3) {
              row[2] = Number(ppScores[pStart + p] ?? Number.NaN);
            }
          }
        } else {
          for (let p = 0; p < pointCount; p++) {
            const row = trackSlot[p];
            if (!row) continue;
            row[0] = Number(px[pStart + p]) + coordOffset;
            row[1] = Number(py[pStart + p]) + coordOffset;
            if (channelCount === 3) {
              row[2] = Number.NaN;
            }
          }
        }
      }
    }

    return output;
  }

  /** Materialize all frames at once. */
  materializeAll(): LabeledFrame[] {
    const frames: LabeledFrame[] = [];
    for (let i = 0; i < this.frameCount; i++) {
      const frame = this.materializeFrame(i);
      if (frame) frames.push(frame);
    }
    return frames;
  }

  private slicePoints(
    data: Record<string, any[]>,
    start: number,
    end: number,
    predicted = false,
  ): number[][] {
    const xs = data.x ?? [];
    const ys = data.y ?? [];
    const visible = data.visible ?? [];
    const complete = data.complete ?? [];
    const scores = data.score ?? [];
    const points: number[][] = [];
    for (let i = start; i < end; i++) {
      if (predicted) {
        points.push([xs[i], ys[i], scores[i], visible[i], complete[i]]);
      } else {
        points.push([xs[i], ys[i], visible[i], complete[i]]);
      }
    }
    return points;
  }
}

/**
 * A lazy array-like container for LabeledFrames.
 * Frames are materialized from the LazyDataStore only when accessed.
 * Supports indexing, iteration, length, and conversion to a real array.
 */
export class LazyFrameList {
  private store: LazyDataStore;
  private cache: Map<number, LabeledFrame>;
  _supplementary: LabeledFrame[] = [];
  /**
   * Max materialized frames to keep cached (0 = unbounded, the default). When
   * exceeded, the oldest-inserted entries are evicted (FIFO ≈ LRU for a
   * sequential sweep) so a read→transform→write pass over a huge lazy file stays
   * memory-bounded without the caller touching internals. Combine with (or use
   * instead of) explicit {@link release}/{@link releaseWindow}. See #207.
   */
  cacheLimit = 0;

  constructor(store: LazyDataStore) {
    this.store = store;
    this.cache = new Map();
  }

  get length(): number {
    return this.store.frameCount + this._supplementary.length;
  }

  /** Get a frame by index, materializing it if needed. */
  at(index: number): LabeledFrame | undefined {
    const n = this.length;
    const nStore = this.store.frameCount;

    // Handle negative indexing
    if (index < 0) index += n;
    if (index < 0 || index >= n) return undefined;

    // Supplementary frames
    if (index >= nStore) {
      return this._supplementary[index - nStore];
    }

    if (this.cache.has(index)) return this.cache.get(index)!;
    const frame = this.store.materializeFrame(index);
    if (frame) {
      this.cache.set(index, frame);
      if (this.cacheLimit > 0 && this.cache.size > this.cacheLimit) {
        for (const k of this.cache.keys()) {
          if (this.cache.size <= this.cacheLimit) break;
          if (k !== index) this.cache.delete(k);
        }
      }
    }
    return frame ?? undefined;
  }

  /** Drop the cached (materialized) frame at `index`, if any. */
  release(index: number): void {
    this.cache.delete(index);
  }

  /** Drop cached frames in the half-open range `[start, end)`. */
  releaseWindow(start: number, end: number): void {
    for (let i = start; i < end; i++) this.cache.delete(i);
  }

  /** Drop all cached frames (keeps the underlying store). */
  clearCache(): void {
    this.cache.clear();
  }

  /** Materialize all frames and return as a regular array. */
  toArray(): LabeledFrame[] {
    const result: LabeledFrame[] = [];
    for (let i = 0; i < this.length; i++) {
      const frame = this.at(i);
      if (frame) result.push(frame);
    }
    return result;
  }

  /** Iterator support. Skips null frames instead of stopping early. */
  [Symbol.iterator](): Iterator<LabeledFrame> {
    let index = 0;
    const self = this;
    return {
      next(): IteratorResult<LabeledFrame> {
        while (index < self.length) {
          const frame = self.at(index++);
          if (frame) return { value: frame, done: false };
        }
        return { value: undefined as any, done: true };
      },
    };
  }

  /** Number of frames that have been materialized. */
  get materializedCount(): number {
    return this.cache.size;
  }
}
