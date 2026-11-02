import type { Track, Instance } from "./instance.js";
import {
  type ROI,
  UserROI,
  PredictedROI,
  _registerMaskFactory,
} from "./roi.js";
import type { Geometry } from "./roi.js";
import type { Identity } from "./identity.js";
import type { Embedding } from "./embedding.js";
import {
  type BoundingBox,
  UserBoundingBox,
  PredictedBoundingBox,
} from "./bbox.js";

export function encodeRle(
  mask: Uint8Array,
  height: number,
  width: number,
): Uint32Array {
  const total = height * width;
  if (total === 0) return new Uint32Array(0);

  const runs: number[] = [];
  let currentVal = 0;
  let count = 0;

  for (let i = 0; i < total; i++) {
    const val = mask[i] ? 1 : 0;
    if (val === currentVal) {
      count++;
    } else {
      runs.push(count);
      currentVal = val;
      count = 1;
    }
  }
  runs.push(count);

  return new Uint32Array(runs);
}

export function decodeRle(
  rleCounts: Uint32Array,
  height: number,
  width: number,
): Uint8Array {
  const total = height * width;
  if (rleCounts.length === 0) return new Uint8Array(total);

  const flat = new Uint8Array(total);
  let pos = 0;
  for (let i = 0; i < rleCounts.length; i++) {
    const val = i % 2 === 0 ? 0 : 1;
    const count = rleCounts[i];
    if (val === 1) {
      for (let j = 0; j < count && pos + j < total; j++) {
        flat[pos + j] = 1;
      }
    }
    pos += count;
  }

  return flat;
}

/**
 * Resize a typed array using nearest-neighbor interpolation.
 * The input is a flat (H*W) array and the output is a flat (dstH*dstW) array.
 */
export function resizeNearest<T extends Uint8Array | Int32Array | Float32Array>(
  data: T,
  srcH: number,
  srcW: number,
  dstH: number,
  dstW: number,
): T {
  const Ctor = data.constructor as new (len: number) => T;
  const result = new Ctor(dstH * dstW);
  for (let r = 0; r < dstH; r++) {
    const srcR = Math.min(Math.floor((r * srcH) / dstH), srcH - 1);
    for (let c = 0; c < dstW; c++) {
      const srcC = Math.min(Math.floor((c * srcW) / dstW), srcW - 1);
      result[r * dstW + c] = data[srcR * srcW + srcC];
    }
  }
  return result;
}

/**
 * Trace the boundary contours of a binary raster as closed polygon rings.
 *
 * Uses pixel-edge boundary tracing: every foreground pixel contributes its four
 * cell-border edges with a fixed winding, and edges shared by two foreground
 * pixels cancel (they are emitted in opposite directions). What remains is the
 * exact region boundary, which chains into closed loops — one ring per outer
 * boundary or hole, for any number of disjoint components. Collinear runs are
 * collapsed, so an axis-aligned block yields four corners.
 *
 * Coordinates are pixel-corner integers in mask space: a foreground block
 * spanning columns `[c0, c1)` and rows `[r0, r1)` traces to the rectangle
 * `(c0, r0) → (c1, r0) → (c1, r1) → (c0, r1) → (c0, r0)`. Each ring is closed
 * (last point equals first). Returns `[]` for an all-background raster.
 *
 * Outer boundaries and holes get opposite winding (via the shoelace sign), so
 * {@link groupRingsIntoPolygons} can nest holes inside their containing outer.
 */
export function traceMaskContours(
  raster: Uint8Array,
  height: number,
  width: number,
): number[][][] {
  // Boundary half-edges keyed "ax,ay>bx,by". Adding an edge whose reverse is
  // already present cancels both (a shared interior edge); otherwise it is kept.
  const edges = new Map<string, [number, number, number, number]>();
  const addEdge = (ax: number, ay: number, bx: number, by: number): void => {
    const rev = `${bx},${by}>${ax},${ay}`;
    if (edges.has(rev)) {
      edges.delete(rev);
    } else {
      edges.set(`${ax},${ay}>${bx},${by}`, [ax, ay, bx, by]);
    }
  };

  for (let r = 0; r < height; r++) {
    const rowBase = r * width;
    for (let c = 0; c < width; c++) {
      if (!raster[rowBase + c]) continue;
      const x0 = c;
      const y0 = r;
      const x1 = c + 1;
      const y1 = r + 1;
      // Fixed winding around the pixel (clockwise in image y-down coords):
      // top L→R, right T→B, bottom R→L, left B→T. Adjacent foreground pixels
      // share an edge with the opposite winding, so those edges cancel.
      addEdge(x0, y0, x1, y0);
      addEdge(x1, y0, x1, y1);
      addEdge(x1, y1, x0, y1);
      addEdge(x0, y1, x0, y0);
    }
  }
  if (edges.size === 0) return [];

  // Build out-adjacency lists. A grid corner has out-degree 1 except where the
  // foreground touches only diagonally (a "saddle", cells [1,0;0,1] or
  // [0,1;1,0]), where it has out-degree 2. The walk below resolves those.
  const outAdj = new Map<string, string[]>();
  for (const [, [ax, ay, bx, by]] of edges) {
    const from = `${ax},${ay}`;
    const to = `${bx},${by}`;
    const list = outAdj.get(from);
    if (list) list.push(to);
    else outAdj.set(from, [to]);
  }

  const parseKey = (k: string): [number, number] => {
    const i = k.indexOf(",");
    return [Number(k.slice(0, i)), Number(k.slice(i + 1))];
  };

  const rings: number[][][] = [];
  for (const startKey of outAdj.keys()) {
    let avail = outAdj.get(startKey);
    while (avail?.length) {
      const ring: number[][] = [];
      let curKey = startKey;
      let curXY = parseKey(curKey);
      let prevXY: [number, number] | null = null;
      // Follow the boundary until we return to this loop's start. At a saddle
      // (>1 outgoing edge) pick the outgoing edge that turns most clockwise
      // relative to the incoming direction (max cross product in y-down image
      // coords). That keeps each 4-connected component's boundary together
      // instead of splicing two diagonal pixels into one self-touching ring.
      while (true) {
        const outs = outAdj.get(curKey);
        if (!outs || outs.length === 0) break;
        let pick = outs.length - 1;
        if (outs.length > 1 && prevXY) {
          const dinX = curXY[0] - prevXY[0];
          const dinY = curXY[1] - prevXY[1];
          let best = Number.NEGATIVE_INFINITY;
          for (let i = 0; i < outs.length; i++) {
            const nXY = parseKey(outs[i]);
            const cross =
              dinX * (nXY[1] - curXY[1]) - dinY * (nXY[0] - curXY[0]);
            if (cross > best) {
              best = cross;
              pick = i;
            }
          }
        }
        const nextKey = outs.splice(pick, 1)[0];
        ring.push([curXY[0], curXY[1]]);
        if (nextKey === startKey) break;
        prevXY = curXY;
        curKey = nextKey;
        curXY = parseKey(nextKey);
      }
      if (ring.length >= 3) {
        ring.push([ring[0][0], ring[0][1]]);
        rings.push(simplifyCollinear(ring));
      }
      avail = outAdj.get(startKey);
    }
  }
  return rings;
}

/** Drop vertices that lie on a straight run, keeping the closed ring closed. */
function simplifyCollinear(ring: number[][]): number[][] {
  const pts = ring.slice(0, -1); // strip the duplicated closing vertex
  const n = pts.length;
  if (n <= 2) return ring;
  const keep: number[][] = [];
  for (let i = 0; i < n; i++) {
    const a = pts[(i - 1 + n) % n];
    const b = pts[i];
    const c = pts[(i + 1) % n];
    const cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (cross !== 0) keep.push(b);
  }
  if (keep.length < 3) return ring;
  keep.push([keep[0][0], keep[0][1]]);
  return keep;
}

/** Signed area of a closed ring (shoelace; sign encodes winding). */
function ringSignedArea(ring: number[][]): number {
  let a = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    a += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
  }
  return a / 2;
}

/** Even-odd ray cast: is `[x, y]` inside the closed `ring`? */
function pointInRing(point: number[], ring: number[][]): boolean {
  const [x, y] = point;
  let inside = false;
  for (let i = 0, j = ring.length - 2; i < ring.length - 1; j = i++) {
    const xi = ring[i][0];
    const yi = ring[i][1];
    const xj = ring[j][0];
    const yj = ring[j][1];
    const intersects =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / (yj - yi || Number.EPSILON) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

/**
 * Group traced contour rings into GeoJSON-style polygons (`[outer, ...holes]`).
 *
 * Outer boundaries share the winding of the largest ring; the opposite winding
 * marks holes, each assigned to the smallest containing outer. The result feeds
 * a `Polygon` (one outer) or `MultiPolygon` (several).
 */
export function groupRingsIntoPolygons(rings: number[][][]): number[][][][] {
  if (rings.length === 0) return [];
  if (rings.length === 1) return [[rings[0]]];

  const areas = rings.map(ringSignedArea);
  let maxI = 0;
  for (let i = 1; i < areas.length; i++) {
    if (Math.abs(areas[i]) > Math.abs(areas[maxI])) maxI = i;
  }
  const outerSign = Math.sign(areas[maxI]) || 1;

  const polys: Array<{ rings: number[][][]; area: number }> = [];
  const holes: Array<{ ring: number[][]; area: number }> = [];
  rings.forEach((ring, i) => {
    const absArea = Math.abs(areas[i]);
    if (Math.sign(areas[i]) === outerSign) {
      polys.push({ rings: [ring], area: absArea });
    } else {
      holes.push({ ring, area: absArea });
    }
  });

  for (const hole of holes) {
    let best = -1;
    let bestArea = Infinity;
    for (let i = 0; i < polys.length; i++) {
      if (
        polys[i].area < bestArea &&
        pointInRing(hole.ring[0], polys[i].rings[0])
      ) {
        best = i;
        bestArea = polys[i].area;
      }
    }
    if (best >= 0) polys[best].rings.push(hole.ring);
    // An uncontained hole (orphan) is dropped rather than treated as fill.
  }

  return polys.map((p) => p.rings);
}

export interface SegmentationMaskOptions {
  rleCounts: Uint32Array;
  height: number;
  width: number;
  name?: string;
  category?: string;
  source?: string;
  track?: Track | null;
  trackingScore?: number | null;
  instance?: Instance | null;
  scale?: [number, number];
  offset?: [number, number];
}

export class SegmentationMask {
  rleCounts: Uint32Array;
  height: number;
  width: number;
  name: string;
  category: string;
  source: string;
  track: Track | null;
  trackingScore: number | null = null;
  instance: Instance | null;
  /** Per-detection re-ID identity (SLP 2.5+); attached from /identity/links. */
  identity?: Identity | null = null;
  identityScore?: number | null = null;
  identityEmbedding?: Embedding | null = null;
  /** Category confidence + appearance embedding (SLP 2.7+). */
  categoryScore?: number | null = null;
  categoryEmbedding?: Embedding | null = null;
  /** Spatial scale factor: image_coord = mask_coord / scale + offset. Default [1, 1]. */
  scale: [number, number];
  /** Spatial offset: image_coord = mask_coord / scale + offset. Default [0, 0]. */
  offset: [number, number];
  /** @internal Deferred instance index for lazy resolution. */
  _instanceIdx: number | null = null;
  /**
   * @internal Memoized decoded raster and the `rleCounts` it was decoded from.
   * `get data()` returns the cached buffer when `rleCounts` is unchanged, so
   * repeated reads (rendering, contour tracing, bbox) decode the RLE once. The
   * returned buffer is shared — treat it as read-only.
   */
  private _dataCache: Uint8Array | null = null;
  private _dataCacheKey: Uint32Array | null = null;

  constructor(options: SegmentationMaskOptions) {
    if (new.target === SegmentationMask) {
      throw new TypeError(
        "SegmentationMask is abstract. Use UserSegmentationMask or PredictedSegmentationMask.",
      );
    }
    const scale = options.scale ?? [1, 1];
    if (scale[0] <= 0 || scale[1] <= 0) {
      throw new Error(
        `Scale must be positive, got [${scale[0]}, ${scale[1]}].`,
      );
    }
    this.rleCounts = options.rleCounts;
    this.height = options.height;
    this.width = options.width;
    this.name = options.name ?? "";
    this.category = options.category ?? "";
    this.source = options.source ?? "";
    this.track = options.track ?? null;
    this.trackingScore = options.trackingScore ?? null;
    this.instance = options.instance ?? null;
    this.scale = scale;
    this.offset = options.offset ?? [0, 0];
  }

  static fromArray(
    mask: Uint8Array | boolean[][],
    height: number,
    width: number,
    options?: Omit<
      SegmentationMaskOptions,
      "rleCounts" | "height" | "width"
    > & { stride?: number },
  ): UserSegmentationMask {
    let flat: Uint8Array;
    if (mask instanceof Uint8Array) {
      // Multi-class guard (parity with Python sleap-io PR #421): refuse
      // arrays with more than one distinct non-zero value to avoid silently
      // collapsing class labels. Single-non-zero-value arrays like [0, 5, 5]
      // are allowed and binarized by encodeRle's truthy check. boolean[][]
      // inputs are inherently binary and bypass this guard, matching
      // Python's array.astype(bool) opt-in path.
      const distinct = new Set<number>();
      let hasMore = false;
      for (let i = 0; i < mask.length; i++) {
        const v = mask[i];
        if (v === 0 || distinct.has(v)) continue;
        if (distinct.size >= 3) {
          hasMore = true;
          break;
        }
        distinct.add(v);
      }
      if (distinct.size > 1) {
        const sample = Array.from(distinct).join(", ");
        const more = hasMore ? "+" : "";
        throw new Error(
          `SegmentationMask is binary (one object per mask) but got an ` +
            `array with ${distinct.size}${more} distinct non-zero values ` +
            `(e.g. [${sample}]). Use UserLabelImage.fromArray(array) to ` +
            `keep all classes in one dense array, or ` +
            `UserLabelImage.fromBinaryMasks([...]) to split per-class ` +
            `binaries. To opt in to binarization explicitly, pre-binarize ` +
            `with Uint8Array.from(arr, v => v ? 1 : 0).`,
        );
      }
      flat = mask;
    } else {
      flat = new Uint8Array(height * width);
      for (let r = 0; r < height; r++) {
        for (let c = 0; c < width; c++) {
          flat[r * width + c] = mask[r][c] ? 1 : 0;
        }
      }
    }
    const rleCounts = encodeRle(flat, height, width);
    // If stride is specified, derive scale as 1/stride. Explicit scale takes precedence.
    const stride = options?.stride;
    const scaleFromStride: [number, number] | undefined =
      stride != null ? [1 / stride, 1 / stride] : undefined;
    return new UserSegmentationMask({
      rleCounts,
      height,
      width,
      ...options,
      scale: options?.scale ?? scaleFromStride,
    });
  }

  get data(): Uint8Array {
    if (this._dataCache !== null && this._dataCacheKey === this.rleCounts) {
      return this._dataCache;
    }
    const decoded = decodeRle(this.rleCounts, this.height, this.width);
    this._dataCache = decoded;
    this._dataCacheKey = this.rleCounts;
    return decoded;
  }

  get area(): number {
    let total = 0;
    for (let i = 1; i < this.rleCounts.length; i += 2) {
      total += this.rleCounts[i];
    }
    return total;
  }

  /** Whether scale != [1,1] or offset != [0,0]. */
  get hasSpatialTransform(): boolean {
    return (
      this.scale[0] !== 1 ||
      this.scale[1] !== 1 ||
      this.offset[0] !== 0 ||
      this.offset[1] !== 0
    );
  }

  /** The image-space extent of this mask (accounting for scale). */
  get imageExtent(): { height: number; width: number } {
    return {
      height: Math.floor(this.height / this.scale[1]),
      width: Math.floor(this.width / this.scale[0]),
    };
  }

  get isPredicted(): boolean {
    return false;
  }

  /**
   * Create a resampled copy of this mask at the target dimensions.
   * The returned mask has scale=[1,1] and offset=[0,0].
   */
  resampled(targetHeight: number, targetWidth: number): SegmentationMask {
    const srcData = this.data;
    const resized = resizeNearest(
      srcData,
      this.height,
      this.width,
      targetHeight,
      targetWidth,
    );
    const rleCounts = encodeRle(resized, targetHeight, targetWidth);

    const baseOpts: SegmentationMaskOptions = {
      rleCounts,
      height: targetHeight,
      width: targetWidth,
      name: this.name,
      category: this.category,
      source: this.source,
      track: this.track,
      trackingScore: this.trackingScore,
      instance: this.instance,
      scale: [1, 1],
      offset: [0, 0],
    };

    let resampled: SegmentationMask;
    if (this instanceof PredictedSegmentationMask) {
      const pm = this as PredictedSegmentationMask;
      let resampledScoreMap: Float32Array | null = null;
      if (pm.scoreMap) {
        resampledScoreMap = resizeNearest(
          pm.scoreMap,
          this.height,
          this.width,
          targetHeight,
          targetWidth,
        );
      }
      resampled = new PredictedSegmentationMask({
        ...baseOpts,
        score: pm.score,
        scoreMap: resampledScoreMap,
      });
    } else {
      resampled = new UserSegmentationMask({
        ...baseOpts,
        fromPredicted:
          this instanceof UserSegmentationMask ? this.fromPredicted : null,
      });
    }
    resampled._instanceIdx = this._instanceIdx;
    return resampled;
  }

  get bbox(): { x: number; y: number; width: number; height: number } {
    const flat = this.data;
    let minR = this.height,
      maxR = -1,
      minC = this.width,
      maxC = -1;

    for (let r = 0; r < this.height; r++) {
      for (let c = 0; c < this.width; c++) {
        if (flat[r * this.width + c]) {
          if (r < minR) minR = r;
          if (r > maxR) maxR = r;
          if (c < minC) minC = c;
          if (c > maxC) maxC = c;
        }
      }
    }

    if (maxR === -1) return { x: 0, y: 0, width: 0, height: 0 };

    const [sx, sy] = this.scale;
    const [ox, oy] = this.offset;
    return {
      x: minC / sx + ox,
      y: minR / sy + oy,
      width: (maxC - minC + 1) / sx,
      height: (maxR - minR + 1) / sy,
    };
  }

  /** Convert to a `BoundingBox` object with metadata.
   *
   * Returns a `UserBoundingBox` or `PredictedBoundingBox` depending on whether
   * this mask is predicted. Coordinates are in image space (respecting
   * scale/offset).
   */
  toBbox(): BoundingBox {
    const { x, y, width, height } = this.bbox;
    const opts = {
      x1: x,
      y1: y,
      x2: x + width,
      y2: y + height,
      track: this.track,
      trackingScore: this.trackingScore,
      instance: this.instance,
      category: this.category,
      name: this.name,
      source: this.source,
    };
    const bb =
      this instanceof PredictedSegmentationMask
        ? new PredictedBoundingBox({ ...opts, score: this.score })
        : new UserBoundingBox(opts);
    bb._instanceIdx = this._instanceIdx;
    return bb;
  }

  /**
   * Trace the mask's boundary as closed polygon rings in image space.
   *
   * Returns an array of rings (`[x, y]` vertices, each closed so the last point
   * equals the first), honoring the mask's `scale`/`offset`. Disjoint blobs and
   * holes each produce their own ring; an empty mask returns `[]`. The outlines
   * are exact, axis-aligned ("staircase") boundaries — consumers wanting smooth
   * curves can post-process (e.g. Chaikin subdivision). For a GeoJSON polygon
   * with holes nested as interior rings, use {@link toPolygon}.
   *
   * Browser-safe (pure data, no canvas), enabling interactive UIs to draw real
   * mask outlines instead of just the bounding box.
   */
  contours(): number[][][] {
    const rings = traceMaskContours(this.data, this.height, this.width);
    const [sx, sy] = this.scale;
    const [ox, oy] = this.offset;
    if (sx === 1 && sy === 1 && ox === 0 && oy === 0) return rings;
    return rings.map((ring) =>
      ring.map(([x, y]) => [x / sx + ox, y / sy + oy]),
    );
  }

  /**
   * Convert the mask to a polygon ROI tracing its actual boundary.
   *
   * Builds a `Polygon` (single blob, holes nested as interior rings) or a
   * `MultiPolygon` (disjoint blobs) from {@link contours}. An empty mask yields
   * an empty `Polygon`. Returns a `PredictedROI` (carrying `score`) for a
   * predicted mask, else a `UserROI`. Metadata (`name`, `category`, `source`,
   * `track`, `instance`) is carried over.
   *
   * Use {@link toBbox} or the `bbox` getter for the axis-aligned bounding box.
   */
  toPolygon(): ROI {
    const rings = this.contours();
    let geometry: Geometry;
    if (rings.length === 0) {
      geometry = { type: "Polygon", coordinates: [[]] };
    } else {
      const polys = groupRingsIntoPolygons(rings);
      geometry =
        polys.length === 1
          ? { type: "Polygon", coordinates: polys[0] }
          : { type: "MultiPolygon", coordinates: polys };
    }

    const options = {
      geometry,
      name: this.name,
      category: this.category,
      source: this.source,
      track: this.track,
      trackingScore: this.trackingScore,
      instance: this.instance,
    };
    const roi =
      this instanceof PredictedSegmentationMask
        ? new PredictedROI({ ...options, score: this.score })
        : new UserROI(options);
    roi._instanceIdx = this._instanceIdx;
    return roi;
  }
}

export interface UserSegmentationMaskOptions extends SegmentationMaskOptions {
  /**
   * Provenance link to the predicted mask this was adopted from. Persisted to
   * the SLP format as an index into the saved mask list (see
   * {@link UserSegmentationMask.fromPredicted}).
   */
  fromPredicted?: PredictedSegmentationMask | null;
}

/** User-annotated segmentation mask (no prediction score). */
export class UserSegmentationMask extends SegmentationMask {
  /**
   * Provenance link to the `PredictedSegmentationMask` this user mask was
   * adopted from, set by {@link PredictedSegmentationMask.toUser}.
   *
   * Mirroring `Instance.fromPredicted`, this link is persisted to the SLP
   * format as an index into the saved mask list. It survives a save/load
   * round-trip as long as the source prediction is also saved (in the same or
   * another frame). Files written before this column existed load it as `null`.
   */
  fromPredicted: PredictedSegmentationMask | null;

  constructor(options: UserSegmentationMaskOptions) {
    super(options);
    this.fromPredicted = options.fromPredicted ?? null;
  }
}

/** Predicted segmentation mask with a confidence score and optional score map. */
export class PredictedSegmentationMask extends SegmentationMask {
  score: number;
  scoreMap: Float32Array | null;
  /** Spatial scale for the score map. Default [1, 1]. */
  scoreMapScale: [number, number];
  /** Spatial offset for the score map. Default [0, 0]. */
  scoreMapOffset: [number, number];
  /**
   * @internal The score map's [height, width] as stored in the file it was read
   * from, which can differ from this one's own size. Saving needs it: the flat array
   * doesn't say.
   */
  _scoreMapSize: [number, number] | null;

  constructor(
    options: SegmentationMaskOptions & {
      score: number;
      scoreMap?: Float32Array | null;
      scoreMapScale?: [number, number];
      scoreMapOffset?: [number, number];
      _scoreMapSize?: [number, number] | null;
    },
  ) {
    super(options);
    this.score = options.score;
    this.scoreMap = options.scoreMap ?? null;
    this.scoreMapScale = options.scoreMapScale ?? [1, 1];
    this.scoreMapOffset = options.scoreMapOffset ?? [0, 0];
    this._scoreMapSize = options._scoreMapSize ?? null;
  }

  get isPredicted(): boolean {
    return true;
  }

  /**
   * Adopt this predicted mask as a user-annotated mask (human-in-the-loop).
   *
   * Returns a NEW {@link UserSegmentationMask} that carries an independent
   * COPY of the RLE raster (via `rleCounts.slice()`) plus the metadata
   * (`name`, `category`, `source`, `track`, `trackingScore`, `instance`,
   * `scale`, `offset`). Prediction-only fields (`score`, `scoreMap`,
   * `scoreMapScale`, `scoreMapOffset`) are dropped. The internal
   * `_instanceIdx` is carried over.
   *
   * The `track` and `instance` references are SHARED (not deep-copied), while
   * the RLE raster and the `scale`/`offset` tuples are copied so the user mask
   * owns independent buffers.
   *
   * Mirrors `Instance.fromPredicted` semantics: the resulting `fromPredicted`
   * link is persisted to the SLP format as an index into the saved mask list,
   * and survives a save/load round-trip as long as the source prediction is
   * also saved. Files written before this column existed load it as `null`.
   *
   * @param link - When `true` (default), set the returned mask's
   *   `fromPredicted` to this predicted mask. When `false`, leave it `null`.
   */
  toUser(link = true): UserSegmentationMask {
    const user = new UserSegmentationMask({
      rleCounts: this.rleCounts.slice(),
      height: this.height,
      width: this.width,
      name: this.name,
      category: this.category,
      source: this.source,
      track: this.track,
      trackingScore: this.trackingScore,
      instance: this.instance,
      scale: [this.scale[0], this.scale[1]],
      offset: [this.offset[0], this.offset[1]],
      fromPredicted: link ? this : null,
    });
    user._instanceIdx = this._instanceIdx;
    return user;
  }
}

// Register mask factory for ROI.toMask() to use
_registerMaskFactory(
  (
    mask: Uint8Array,
    height: number,
    width: number,
    options: Record<string, unknown> & { score?: number },
  ) => {
    const { score, ...rest } = options as { score?: number } & Omit<
      SegmentationMaskOptions,
      "rleCounts" | "height" | "width"
    >;
    // Bypass fromArray's multi-class guard: rasterizeGeometry always yields a
    // binary Uint8Array, so encoding directly is safe.
    const rleCounts = encodeRle(mask, height, width);
    if (score !== undefined) {
      return new PredictedSegmentationMask({
        rleCounts,
        height,
        width,
        ...rest,
        score,
      });
    }
    return new UserSegmentationMask({ rleCounts, height, width, ...rest });
  },
);
