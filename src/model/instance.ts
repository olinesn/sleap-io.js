import type { Skeleton, Node } from "./skeleton.js";
import type { Identity } from "./identity.js";
import type { Embedding } from "./embedding.js";

// Late-binding factory to avoid circular imports with centroid.ts.
// Set by centroid.ts when it is imported.
type CentroidFactory = (
  instance: Instance | PredictedInstance,
  options?: { method?: string; node?: string | number },
) => any;
let _centroidFactory: CentroidFactory | null = null;
export function _registerCentroidFactory(factory: CentroidFactory): void {
  _centroidFactory = factory;
}

export class Track {
  name: string;

  constructor(name = "") {
    this.name = name;
  }

  matches(other: Track, method = "name"): boolean {
    if (method === "name") {
      return this.name === other.name;
    }
    if (method === "identity") {
      return this === other;
    }
    throw new Error("Unknown matching method: " + method);
  }
}

export type Point = {
  xy: [number, number];
  visible: boolean;
  complete: boolean;
  score?: number;
  name?: string;
};

export type PredictedPoint = Point & { score: number };

export type PointsArray = Point[];
export type PredictedPointsArray = PredictedPoint[];

/**
 * The SLP readers' parsed point columns (`x`/`y`/`visible`/`complete`[/`score`]),
 * as plain `number[]` (eager reader) or `Float64Array` (streaming worker). Fed to
 * {@link Instance._fromColumns} to build an instance without a `Point[]`.
 */
export interface PointColumns {
  x?: ArrayLike<number>;
  y?: ArrayLike<number>;
  visible?: ArrayLike<number>;
  complete?: ArrayLike<number>;
  score?: ArrayLike<number>;
}

// Shared empty backing arrays for a freshly-declared Instance before _ingest.
const EMPTY_F64 = new Float64Array(0);
const EMPTY_U8 = new Uint8Array(0);

export function pointsEmpty(length: number, names?: string[]): PointsArray {
  const pts: PointsArray = [];
  for (let i = 0; i < length; i += 1) {
    pts.push({
      xy: [Number.NaN, Number.NaN],
      visible: false,
      complete: false,
      name: names?.[i],
    });
  }
  return pts;
}

export function predictedPointsEmpty(
  length: number,
  names?: string[],
): PredictedPointsArray {
  const pts: PredictedPointsArray = [];
  for (let i = 0; i < length; i += 1) {
    pts.push({
      xy: [Number.NaN, Number.NaN],
      visible: false,
      complete: false,
      score: Number.NaN,
      name: names?.[i],
    });
  }
  return pts;
}

/**
 * Deep-copy a point into a fresh plain literal, optionally under a new node
 * name. Use this instead of `{ ...point }`: `instance.points[i]` returns a
 * {@link PointView} flyweight whose fields are accessors (not own enumerable
 * properties), so a spread would silently drop `visible`/`complete`/`score`.
 * The `score` field is copied only when present (predicted points).
 */
export function clonePoint(p: Point, name?: string): Point {
  const xy = p.xy;
  const out: Point = {
    xy: [xy[0], xy[1]],
    visible: p.visible,
    complete: p.complete,
    name: name ?? p.name,
  };
  const score = (p as PredictedPoint).score;
  if (typeof score === "number") (out as PredictedPoint).score = score;
  return out;
}

export function pointsFromArray(
  array: number[][],
  names?: string[],
): PointsArray {
  const pts: PointsArray = [];
  for (let i = 0; i < array.length; i += 1) {
    const row = array[i] ?? [Number.NaN, Number.NaN];
    const visible = row.length > 2 ? Boolean(row[2]) : !Number.isNaN(row[0]);
    const complete = row.length > 3 ? Boolean(row[3]) : false;
    pts.push({
      xy: [row[0] ?? Number.NaN, row[1] ?? Number.NaN],
      visible,
      complete,
      name: names?.[i],
    });
  }
  return pts;
}

export function predictedPointsFromArray(
  array: number[][],
  names?: string[],
): PredictedPointsArray {
  const pts: PredictedPointsArray = [];
  for (let i = 0; i < array.length; i += 1) {
    const row = array[i] ?? [Number.NaN, Number.NaN, Number.NaN];
    const visible = row.length > 3 ? Boolean(row[3]) : !Number.isNaN(row[0]);
    const complete = row.length > 4 ? Boolean(row[4]) : false;
    pts.push({
      xy: [row[0] ?? Number.NaN, row[1] ?? Number.NaN],
      score: row[2] ?? Number.NaN,
      visible,
      complete,
      name: names?.[i],
    });
  }
  return pts;
}

// Readers build millions of instances at once. Their columns are views of shared
// chunks: an array buffer per column would be four per instance, tens of millions on a
// long multi-camera project, for the garbage collector to walk on every collection.
const POOL_CHUNK_ELEMENTS = 1 << 20;
let poolFloat64 = new Float64Array(0);
let poolFloat64Used = 0;
let poolUint8 = new Uint8Array(0);
let poolUint8Used = 0;

function pooledFloat64(n: number): Float64Array {
  if (poolFloat64Used + n > poolFloat64.length) {
    poolFloat64 = new Float64Array(Math.max(POOL_CHUNK_ELEMENTS, n));
    poolFloat64Used = 0;
  }
  const out = poolFloat64.subarray(poolFloat64Used, poolFloat64Used + n);
  poolFloat64Used += n;
  return out;
}

function pooledUint8(n: number): Uint8Array {
  if (poolUint8Used + n > poolUint8.length) {
    poolUint8 = new Uint8Array(Math.max(POOL_CHUNK_ELEMENTS, n));
    poolUint8Used = 0;
  }
  const out = poolUint8.subarray(poolUint8Used, poolUint8Used + n);
  poolUint8Used += n;
  return out;
}

/**
 * A live view of one keypoint over an {@link Instance}'s columnar storage.
 *
 * `instance.points[i]` returns one of these instead of a stored `{xy,...}`
 * object, so a project's keypoints live in a few packed typed arrays per
 * instance (~a few bytes/point) rather than an object graph (~150 B/point). It
 * satisfies the structural `Point` type: reads go straight to the columns, and
 * writes (`point.xy = [...]`, `point.visible = ...`) write back through. `xy`
 * getter returns a fresh `[x, y]` copy — no code mutates `point.xy[0]` in place
 * (verified), and returning a copy keeps callers that stash `point.xy` by
 * reference (e.g. centroid math) reading a stable snapshot.
 */
export class PointView {
  // True-private backing (ECMAScript #fields) so a view carries no enumerable
  // own properties: `{ ...point }` and `JSON.stringify(point)` see nothing but
  // the getters, and there is no circular `_owner` leak. The `Point` fields are
  // exposed as accessors below.
  readonly #owner: Instance;
  readonly #i: number;

  constructor(owner: Instance, i: number) {
    this.#owner = owner;
    this.#i = i;
  }

  get xy(): [number, number] {
    const xy = this.#owner._xy;
    const j = this.#i << 1;
    return [xy[j], xy[j + 1]];
  }
  set xy(v: ArrayLike<number>) {
    const xy = this.#owner._xy;
    const j = this.#i << 1;
    xy[j] = v[0];
    xy[j + 1] = v[1];
  }

  get visible(): boolean {
    return this.#owner._visible[this.#i] !== 0;
  }
  set visible(v: boolean) {
    this.#owner._visible[this.#i] = v ? 1 : 0;
  }

  get complete(): boolean {
    return this.#owner._complete[this.#i] !== 0;
  }
  set complete(v: boolean) {
    this.#owner._complete[this.#i] = v ? 1 : 0;
  }

  get score(): number | undefined {
    const s = this.#owner._score;
    return s ? s[this.#i] : undefined;
  }
  set score(v: number | undefined) {
    this.#owner._scoreColumn()[this.#i] = v ?? Number.NaN;
  }

  get name(): string | undefined {
    return this.#owner._pointName(this.#i);
  }
  set name(v: string | undefined) {
    this.#owner._setPointName(this.#i, v);
  }
}

export class Instance {
  skeleton: Skeleton;
  track?: Track | null;
  fromPredicted?: PredictedInstance | null;
  trackingScore: number;

  /**
   * Persistent cross-video re-ID identity of this detection (SLP 2.5+), distinct
   * from the ephemeral within-video {@link Track}. Persisted in the `/identity`
   * catalog + `/identity/links`; attached after read, defaults to null.
   */
  identity?: Identity | null = null;
  /** Confidence of the {@link identity} assignment (SLP 2.5+); null if unrecorded. */
  identityScore?: number | null = null;
  /** Per-detection re-ID appearance embedding (SLP 2.5+); persisted in `/embeddings`. */
  identityEmbedding?: Embedding | null = null;
  /** Class/category membership (SLP 2.7+); persisted via the `/categories` catalog + links. */
  category?: string | null = null;
  /** Confidence of the {@link category} assignment (SLP 2.7+); null if unrecorded. */
  categoryScore?: number | null = null;
  /** Per-detection category appearance embedding (SLP 2.7+); persisted in `/embeddings`. */
  categoryEmbedding?: Embedding | null = null;

  // Columnar keypoint storage (retained). Built once at construction from the
  // transient `Point[]`/dict, which is then discarded. `points` reads/writes go
  // through lightweight PointView flyweights over these. See {@link PointView}.
  _xy: Float64Array = EMPTY_F64; // interleaved [x0,y0,x1,y1,...], length 2n
  _visible: Uint8Array = EMPTY_U8; // n
  _complete: Uint8Array = EMPTY_U8; // n
  _score: Float64Array | null = null; // n (predicted) or null (user)
  _names: (string | undefined)[] | null = null; // null ⇒ derive from skeleton
  _n = 0;

  constructor(options: {
    points: PointsArray | Record<string, number[]>;
    skeleton: Skeleton;
    track?: Track | null;
    fromPredicted?: PredictedInstance | null;
    trackingScore?: number;
  }) {
    this.skeleton = options.skeleton;
    this.track = options.track ?? null;
    this.fromPredicted = options.fromPredicted ?? null;
    this.trackingScore = options.trackingScore ?? 0;
    let pts: PointsArray;
    if (Array.isArray(options.points)) {
      const arr = options.points as PointsArray | number[][];
      // Accept a raw `number[][]` coordinate array too (each row `[x, y, ...]`):
      // the previous constructor stored such input verbatim, leaving malformed
      // points; convert it like `fromArray` so `points` are always well-formed.
      pts =
        arr.length > 0 && Array.isArray(arr[0])
          ? pointsFromArray(arr as number[][], options.skeleton.nodeNames)
          : (arr as PointsArray);
    } else {
      pts = pointsFromDict(options.points, options.skeleton);
    }
    this._ingest(pts);
  }

  /** Pack a transient `Point[]` into the columnar typed-array storage. */
  _ingest(pts: PointsArray): void {
    const n = pts.length;
    const xy = new Float64Array(2 * n);
    const visible = new Uint8Array(n);
    const complete = new Uint8Array(n);
    // Predicted points carry a numeric score (possibly NaN); user points don't.
    const hasScore = n > 0 && typeof (pts[0] as Point).score === "number";
    const score = hasScore ? new Float64Array(n) : null;
    const nodeNames = this.skeleton.nodeNames;
    let names: (string | undefined)[] | null = null;
    for (let i = 0; i < n; i += 1) {
      const p = pts[i];
      const j = i << 1;
      xy[j] = p.xy[0];
      xy[j + 1] = p.xy[1];
      visible[i] = p.visible ? 1 : 0;
      complete[i] = p.complete ? 1 : 0;
      if (score) score[i] = p.score ?? Number.NaN;
      // Only materialize a names array if some point name differs from the
      // skeleton node order (the readers always match, so this stays null).
      if (p.name !== nodeNames[i]) {
        if (!names) {
          names = new Array(n);
          for (let k = 0; k < n; k += 1) names[k] = nodeNames[k];
        }
        names[i] = p.name;
      }
    }
    this._xy = xy;
    this._visible = visible;
    this._complete = complete;
    this._score = score;
    this._names = names;
    this._n = n;
  }

  /**
   * Fill the columnar storage directly from the SLP readers' parsed point
   * columns over `[start, end)`, skipping the intermediate `Point[]` literals
   * (the slicePoints → pointsFromArray → `_ingest` path allocates ~3 throwaway
   * objects per point). Values match that path exactly: `x ?? NaN`, `y ?? NaN`,
   * `Boolean(visible)`, `Boolean(complete)`, and (predicted) `score ?? NaN`;
   * names derive from the skeleton. Used by {@link Instance._fromColumns}.
   */
  _fillFromColumns(
    columns: PointColumns,
    start: number,
    end: number,
    predicted: boolean,
  ): void {
    const n = Math.max(0, end - start);
    const xy = pooledFloat64(2 * n);
    const visible = pooledUint8(n);
    const complete = pooledUint8(n);
    const score = predicted ? pooledFloat64(n) : null;
    const cx = columns.x;
    const cy = columns.y;
    const cv = columns.visible;
    const cc = columns.complete;
    const cs = columns.score;
    for (let i = 0; i < n; i += 1) {
      const src = start + i;
      const j = i << 1;
      // `!= null` (null OR undefined) → NaN, matching `row[k] ?? NaN`; an
      // in-range numeric 0 is kept. Float64Array would coerce null to 0, so the
      // guard is load-bearing for out-of-range/missing columns.
      xy[j] = cx && cx[src] != null ? (cx[src] as number) : Number.NaN;
      xy[j + 1] = cy && cy[src] != null ? (cy[src] as number) : Number.NaN;
      visible[i] = cv && cv[src] ? 1 : 0;
      complete[i] = cc && cc[src] ? 1 : 0;
      if (score)
        score[i] = cs && cs[src] != null ? (cs[src] as number) : Number.NaN;
    }
    this._xy = xy;
    this._visible = visible;
    this._complete = complete;
    this._score = score;
    this._names = null; // derived from the skeleton (readers keep node order)
    this._n = n;
  }

  /**
   * Build an Instance directly from reader point columns over `[start, end)`,
   * without materializing a `Point[]`. Internal fast path for buildLabeledFrames;
   * equivalent to `new Instance({ points: pointsFromArray(slicePoints(...)) })`.
   */
  static _fromColumns(opts: {
    columns: PointColumns;
    start: number;
    end: number;
    skeleton: Skeleton;
    track?: Track | null;
    fromPredicted?: PredictedInstance | null;
    trackingScore?: number;
  }): Instance {
    const inst = Object.create(Instance.prototype) as Instance;
    inst.skeleton = opts.skeleton;
    inst.track = opts.track ?? null;
    inst.fromPredicted = opts.fromPredicted ?? null;
    inst.trackingScore = opts.trackingScore ?? 0;
    inst._fillFromColumns(opts.columns, opts.start, opts.end, false);
    return inst;
  }

  /** Lazily allocate the score column (for a user instance gaining scores). */
  _scoreColumn(): Float64Array {
    if (!this._score) this._score = new Float64Array(this._n).fill(Number.NaN);
    return this._score;
  }

  /** Node name for point `i` — derived from the skeleton unless overridden. */
  _pointName(i: number): string | undefined {
    return this._names ? this._names[i] : this.skeleton.nodeNames[i];
  }
  _setPointName(i: number, v: string | undefined): void {
    if (!this._names) {
      const nn = this.skeleton.nodeNames;
      this._names = new Array(this._n);
      for (let k = 0; k < this._n; k += 1) this._names[k] = nn[k];
    }
    this._names[i] = v;
  }

  /** The keypoints as an array of live {@link PointView}s (built on demand). */
  get points(): PointsArray {
    const n = this._n;
    const out = new Array<Point>(n);
    for (let i = 0; i < n; i += 1) out[i] = new PointView(this, i);
    return out;
  }
  set points(pts: PointsArray) {
    this._ingest(pts);
  }

  static fromArray(points: number[][], skeleton: Skeleton): Instance {
    return new Instance({
      points: pointsFromArray(points, skeleton.nodeNames),
      skeleton,
    });
  }

  static fromNumpy(options: {
    pointsData: number[][];
    skeleton: Skeleton;
    track?: Track | null;
    fromPredicted?: PredictedInstance | null;
    trackingScore?: number;
  }): Instance {
    return new Instance({
      points: pointsFromArray(options.pointsData, options.skeleton.nodeNames),
      skeleton: options.skeleton,
      track: options.track ?? null,
      fromPredicted: options.fromPredicted ?? null,
      trackingScore: options.trackingScore,
    });
  }

  static empty(options: { skeleton: Skeleton }): Instance {
    return new Instance({
      points: pointsEmpty(
        options.skeleton.nodeNames.length,
        options.skeleton.nodeNames,
      ),
      skeleton: options.skeleton,
    });
  }

  get length(): number {
    return this._n;
  }

  get nVisible(): number {
    let count = 0;
    for (let i = 0; i < this._n; i += 1) if (this._visible[i]) count += 1;
    return count;
  }

  getPoint(target: number | string | Node): Point {
    let index: number;
    if (typeof target === "number") {
      if (target < 0 || target >= this._n)
        throw new Error("Point index out of range.");
      index = target;
    } else if (typeof target === "string") {
      index = this.skeleton.index(target);
    } else {
      index = this.skeleton.index(target.name);
    }
    return new PointView(this, index);
  }

  numpy(options?: { invisibleAsNaN?: boolean }): number[][] {
    const invisibleAsNaN = options?.invisibleAsNaN ?? true;
    const xy = this._xy;
    const out = new Array<number[]>(this._n);
    for (let i = 0; i < this._n; i += 1) {
      if (invisibleAsNaN && !this._visible[i]) {
        out[i] = [Number.NaN, Number.NaN];
      } else {
        const j = i << 1;
        out[i] = [xy[j], xy[j + 1]];
      }
    }
    return out;
  }

  toString(): string {
    const trackName = this.track ? `"${this.track.name}"` : "None";
    return `Instance(points=${JSON.stringify(this.numpy({ invisibleAsNaN: false }))}, track=${trackName})`;
  }

  /** Mean of visible point coordinates as `[x, y]`, or `null` if no points visible. */
  get centroidXy(): [number, number] | null {
    let sumX = 0,
      sumY = 0,
      count = 0;
    const xy = this._xy;
    for (let i = 0; i < this._n; i += 1) {
      const j = i << 1;
      const x = xy[j];
      const y = xy[j + 1];
      if (this._visible[i] && !Number.isNaN(x) && !Number.isNaN(y)) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
    if (count === 0) return null;
    return [sumX / count, sumY / count];
  }

  /**
   * Create a Centroid from this instance.
   *
   * @param method - Computation method. Accepts the legacy camelCase names
   *   (`"centerOfMass"` (default), `"bboxCenter"`, `"anchor"`) and Python's
   *   snake_case names (`"center_of_mass"`, `"bbox_center"`). The persisted
   *   `source` is always canonical snake_case for Python interop.
   * @param node - Node specification for "anchor" method.
   * @returns UserCentroid or PredictedCentroid depending on instance type.
   */
  toCentroid(method?: string, node?: string | number) {
    if (!_centroidFactory) {
      throw new Error(
        "Centroid not available. Import centroid.ts before calling toCentroid().",
      );
    }
    return _centroidFactory(this, { method, node });
  }

  get isEmpty(): boolean {
    for (let i = 0; i < this._n; i += 1) {
      if (this._visible[i] && !Number.isNaN(this._xy[i << 1])) return false;
    }
    return true;
  }

  /**
   * Check if this instance has the same pose as another instance.
   *
   * Mirrors Python `Instance.same_pose_as` (instance.py:699-753).
   *
   * @param other - Another instance to compare with.
   * @param tolerance - Maximum distance (in pixels) between corresponding points
   *   for them to be considered the same. If `null`/`undefined`, uses exact
   *   comparison including NaN==NaN handling.
   * @returns `true` if the instances have the same pose within tolerance.
   */
  samePoseAs(other: Instance, tolerance?: number | null): boolean {
    // Check skeleton compatibility (default requireSameOrder=false). Short-circuit
    // before any point comparison.
    if (!this.skeleton.matches(other.skeleton)) return false;

    const a = this.numpy();
    const b = other.numpy();

    if (tolerance == null) {
      // Exact comparison with NaN treated as equal to NaN.
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i += 1) {
        for (let j = 0; j < 2; j += 1) {
          const av = a[i][j];
          const bv = b[i][j];
          const aNaN = Number.isNaN(av);
          const bNaN = Number.isNaN(bv);
          if (aNaN !== bNaN) return false;
          if (!aNaN && av !== bv) return false;
        }
      }
      return true;
    }

    // Tolerance-based comparison with NaN-pattern check.
    if (a.length !== b.length) return false;
    // First, check that the NaN patterns match exactly over the full (n, 2).
    for (let i = 0; i < a.length; i += 1) {
      for (let j = 0; j < 2; j += 1) {
        if (Number.isNaN(a[i][j]) !== Number.isNaN(b[i][j])) return false;
      }
    }

    // Gather the non-NaN values row-major from both arrays.
    const aVals: number[] = [];
    const bVals: number[] = [];
    for (let i = 0; i < a.length; i += 1) {
      for (let j = 0; j < 2; j += 1) {
        if (!Number.isNaN(a[i][j])) {
          aVals.push(a[i][j]);
          bVals.push(b[i][j]);
        }
      }
    }

    // If all values are NaN, they are considered equal.
    if (aVals.length === 0) return true;

    // Pair into (k, 2) and compare per-pair Euclidean distance.
    for (let k = 0; k < aVals.length; k += 2) {
      const dx = aVals[k] - bVals[k];
      const dy = aVals[k + 1] - bVals[k + 1];
      const distance = Math.hypot(dx, dy);
      if (!(distance <= tolerance)) return false;
    }
    return true;
  }

  /**
   * Check if this instance has the same identity (track) as another instance.
   *
   * Mirrors Python `Instance.same_identity_as` (instance.py:755-770). Compares
   * tracks by reference identity, not by name.
   *
   * @param other - Another instance to compare with.
   * @returns `true` if both instances share the same `Track` object.
   */
  sameIdentityAs(other: Instance): boolean {
    if (this.track == null || other.track == null) return false;
    return this.track === other.track;
  }

  /**
   * Check if this instance overlaps with another by bounding-box IoU.
   *
   * Mirrors Python `Instance.overlaps_with` (instance.py:772-830). Bounding
   * boxes are computed over VISIBLE points; if either has none, returns false.
   * If the boxes do not STRICTLY intersect on both axes (touching edges count
   * as no overlap), returns false regardless of `iouThreshold` — this matches
   * Python's `np.any(intersection_min >= intersection_max) -> False`
   * short-circuit, which runs before the threshold comparison.
   *
   * @param other - Another instance to compare with.
   * @param iouThreshold - Minimum IoU to count as overlapping (inclusive `>=`).
   */
  overlapsWith(other: Instance, iouThreshold = 0.5): boolean {
    const boxA = this.boundingBox();
    const boxB = other.boundingBox();
    if (!boxA || !boxB) return false;

    // box[0] = [minX, minY] (mins), box[1] = [maxX, maxY] (maxs).
    const interMinX = Math.max(boxA[0][0], boxB[0][0]);
    const interMinY = Math.max(boxA[0][1], boxB[0][1]);
    const interMaxX = Math.min(boxA[1][0], boxB[1][0]);
    const interMaxY = Math.min(boxA[1][1], boxB[1][1]);

    // No strict intersection on either axis -> not overlapping, independent of
    // the threshold (Python returns False here before the `>= threshold` check).
    if (interMinX >= interMaxX || interMinY >= interMaxY) return false;

    const interArea = (interMaxX - interMinX) * (interMaxY - interMinY);
    const areaA = (boxA[1][0] - boxA[0][0]) * (boxA[1][1] - boxA[0][1]);
    const areaB = (boxB[1][0] - boxB[0][0]) * (boxB[1][1] - boxB[0][1]);
    const union = areaA + areaB - interArea;
    const iou = union > 0 ? interArea / union : 0;
    return iou >= iouThreshold;
  }

  /**
   * Get the bounding box of visible points.
   *
   * Mirrors Python `Instance.bounding_box` (instance.py:832-849).
   *
   * @returns `[[minX, minY], [maxX, maxY]]` over visible points, or `null` if
   *   there are no visible points.
   */
  boundingBox(): [[number, number], [number, number]] | null {
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    let any = false;
    const xy = this._xy;
    for (let i = 0; i < this._n; i += 1) {
      if (!this._visible[i]) continue;
      const x = xy[i << 1];
      const y = xy[(i << 1) + 1];
      any = true;
      // Math.min/max (not `<`) so a visible NaN coordinate propagates to a NaN
      // bound, matching the previous `Math.min(...xs)` behavior exactly.
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    if (!any) return null;
    return [
      [minX, minY],
      [maxX, maxY],
    ];
  }
}

export class PredictedInstance extends Instance {
  score: number;

  constructor(options: {
    points: PredictedPointsArray | Record<string, number[]>;
    skeleton: Skeleton;
    track?: Track | null;
    score?: number;
    trackingScore?: number;
    fromPredicted?: PredictedInstance | null;
  }) {
    const { score = 0, ...rest } = options;
    const pts = Array.isArray(rest.points)
      ? rest.points
      : predictedPointsFromDict(
          rest.points as Record<string, number[]>,
          rest.skeleton,
        );
    super({
      points: pts as PointsArray,
      skeleton: rest.skeleton,
      track: rest.track,
      trackingScore: rest.trackingScore,
      fromPredicted: rest.fromPredicted,
    });
    this.score = score;
  }

  static fromArray(
    points: number[][],
    skeleton: Skeleton,
    score?: number,
  ): PredictedInstance {
    return new PredictedInstance({
      points: predictedPointsFromArray(points, skeleton.nodeNames),
      skeleton,
      score,
    });
  }

  static fromNumpy(options: {
    pointsData: number[][];
    skeleton: Skeleton;
    track?: Track | null;
    score?: number;
    trackingScore?: number;
  }): PredictedInstance {
    return new PredictedInstance({
      points: predictedPointsFromArray(
        options.pointsData,
        options.skeleton.nodeNames,
      ),
      skeleton: options.skeleton,
      track: options.track ?? null,
      score: options.score,
      trackingScore: options.trackingScore,
    });
  }

  static empty(options: { skeleton: Skeleton }): PredictedInstance {
    return new PredictedInstance({
      points: predictedPointsEmpty(
        options.skeleton.nodeNames.length,
        options.skeleton.nodeNames,
      ),
      skeleton: options.skeleton,
    });
  }

  /**
   * Build a PredictedInstance directly from reader point columns over
   * `[start, end)`, without materializing a `Point[]`. Internal fast path for
   * buildLabeledFrames; equivalent to `new PredictedInstance({ points:
   * predictedPointsFromArray(slicePoints(...)) })`.
   */
  static _fromColumns(opts: {
    columns: PointColumns;
    start: number;
    end: number;
    skeleton: Skeleton;
    track?: Track | null;
    score?: number;
    trackingScore?: number;
    fromPredicted?: PredictedInstance | null;
  }): PredictedInstance {
    const inst = Object.create(
      PredictedInstance.prototype,
    ) as PredictedInstance;
    inst.skeleton = opts.skeleton;
    inst.track = opts.track ?? null;
    inst.fromPredicted = opts.fromPredicted ?? null;
    inst.trackingScore = opts.trackingScore ?? 0;
    inst.score = opts.score ?? 0;
    inst._fillFromColumns(opts.columns, opts.start, opts.end, true);
    return inst;
  }

  numpy(options?: { scores?: boolean; invisibleAsNaN?: boolean }): number[][] {
    const invisibleAsNaN = options?.invisibleAsNaN ?? true;
    const withScores = options?.scores ?? false;
    const xy = this._xy;
    const score = this._score;
    const out = new Array<number[]>(this._n);
    for (let i = 0; i < this._n; i += 1) {
      const hidden = invisibleAsNaN && !this._visible[i];
      const x = hidden ? Number.NaN : xy[i << 1];
      const y = hidden ? Number.NaN : xy[(i << 1) + 1];
      out[i] = withScores ? [x, y, score ? score[i] : 0] : [x, y];
    }
    return out;
  }

  toString(): string {
    const trackName = this.track ? `"${this.track.name}"` : "None";
    return `PredictedInstance(points=${JSON.stringify(this.numpy({ invisibleAsNaN: false }))}, track=${trackName}, score=${this.score.toFixed(2)}, tracking_score=${this.trackingScore ?? "None"})`;
  }
}

export function pointsFromDict(
  pointsDict: Record<string, number[]>,
  skeleton: Skeleton,
): PointsArray {
  const points = pointsEmpty(skeleton.nodeNames.length, skeleton.nodeNames);
  for (const [nodeName, data] of Object.entries(pointsDict)) {
    const index = skeleton.index(nodeName);
    points[index] = {
      xy: [data[0] ?? Number.NaN, data[1] ?? Number.NaN],
      visible: data.length > 2 ? Boolean(data[2]) : !Number.isNaN(data[0]),
      complete: data.length > 3 ? Boolean(data[3]) : false,
      name: nodeName,
    };
  }
  return points;
}

export function predictedPointsFromDict(
  pointsDict: Record<string, number[]>,
  skeleton: Skeleton,
): PredictedPointsArray {
  const points = predictedPointsEmpty(
    skeleton.nodeNames.length,
    skeleton.nodeNames,
  );
  for (const [nodeName, data] of Object.entries(pointsDict)) {
    const index = skeleton.index(nodeName);
    points[index] = {
      xy: [data[0] ?? Number.NaN, data[1] ?? Number.NaN],
      score: data[2] ?? Number.NaN,
      visible: data.length > 3 ? Boolean(data[3]) : !Number.isNaN(data[0]),
      complete: data.length > 4 ? Boolean(data[4]) : false,
      name: nodeName,
    };
  }
  return points;
}
