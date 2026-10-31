/**
 * SLEAP Analysis HDF5 format I/O.
 *
 * A portable format for exporting pose-tracking predictions as dense numpy
 * arrays. This is a 1:1 TypeScript port of `sleap_io/io/analysis_h5.py`
 * (`read_labels` / `write_labels` / `is_analysis_h5_file`) together with the
 * occupancy/location array builder `to_analysis_arrays` from
 * `sleap_io/codecs/numpy.py`.
 *
 * Format features:
 * - Configurable axis ordering via presets ("matlab" default, "standard") or
 *   explicit dimension positions.
 * - Gzip-compressed storage.
 * - Self-documenting: dimension names stored as the dataset `dims` attribute
 *   (a JSON array string, matching Python's `json.dumps`).
 * - Optional extended metadata (skeleton symmetries, video backend metadata)
 *   for full round-trip.
 *
 * Canonical internal shape for `tracks` is `(frame, track, node, xy)`; other
 * arrays drop trailing dims: `point_scores` `(frame, track, node)`,
 * `instance_scores`/`tracking_scores`/`track_occupancy` `(frame, track)`.
 *
 * Reading works in both Node and the browser (via `openH5File`). Writing is
 * browser-safe too: `writeLabelsToBytes` builds the file in an h5wasm in-memory
 * virtual FS and returns the bytes. `writeLabels` wraps it to write those bytes
 * to disk through the Node filesystem ops registered by `h5-node.ts` — the only
 * Node-only step — so this module stays free of Node-only imports and the
 * browser bundle stays clean.
 */

import { Labels } from "../model/labels.js";
import { PredictedInstance, Track } from "../model/instance.js";
import { LabeledFrame } from "../model/labeled-frame.js";
import { Skeleton } from "../model/skeleton.js";
import { Video } from "../model/video.js";
import { maskUrlCredentials, maskUrlCredentialsDeep } from "./remote.js";
import {
  openH5File,
  getH5Module,
  getH5FileSystem,
  ensureH5StagingDir,
  nodeWriteFile,
  nodeFileExists,
  nodeReadPackageVersion,
} from "../codecs/slp/h5.js";
import {
  getDs,
  decodeStringElement,
  decodeStringArray,
  decodeScalarString,
  unwrapAttr,
  readStringAttr,
  type H5ReadFile,
} from "./h5-read-utils.js";

// =============================================================================
// Preset definitions (port of PRESETS / _get_axis_order / _get_transpose_axes /
// _get_dims_tuple)
// =============================================================================

/** Axis ordering as a map of dimension name -> position. */
export type AxisOrder = Record<string, number>;

/**
 * Dimension-position presets for the 4D `tracks` array.
 *
 * @internal Exported for unit testing.
 */
export const PRESETS: Record<string, AxisOrder> = {
  // Standard: (frame, track, node, xy) - intuitive Python indexing.
  standard: { frame: 0, track: 1, node: 2, xy: 3 },
  // MATLAB: (track, xy, node, frame) - SLEAP-compatible column-major.
  matlab: { frame: 3, track: 0, node: 2, xy: 1 },
};

/** Ordered dimension names for a given ndim (4/3/2). */
function dimsForNdim(ndim: number): string[] {
  if (ndim === 4) return ["frame", "track", "node", "xy"];
  if (ndim === 3) return ["frame", "track", "node"];
  return ["frame", "track"];
}

/**
 * Resolve axis ordering from a preset or explicit dimension positions.
 *
 * Port of `_get_axis_order`. Preset and explicit dims are mutually exclusive.
 * Explicit dims require all four and must be a permutation of [0, 1, 2, 3].
 * Defaults to the "matlab" preset.
 *
 * @internal Exported for unit testing.
 */
export function getAxisOrder(
  preset: string | undefined,
  frameDim: number | undefined,
  trackDim: number | undefined,
  nodeDim: number | undefined,
  xyDim: number | undefined,
): { axisOrder: AxisOrder; presetName: string } {
  const explicitDims = [frameDim, trackDim, nodeDim, xyDim];
  const hasExplicit = explicitDims.some((d) => d !== undefined);

  if (preset !== undefined && hasExplicit) {
    throw new Error(
      "Cannot specify both 'preset' and explicit dimension positions " +
        "(frame_dim, track_dim, node_dim, xy_dim). Use one or the other.",
    );
  }

  if (hasExplicit) {
    if (!explicitDims.every((d) => d !== undefined)) {
      throw new Error(
        "When using explicit dimensions, all four must be specified: " +
          "frame_dim, track_dim, node_dim, xy_dim",
      );
    }
    const sorted = [...(explicitDims as number[])].sort((a, b) => a - b);
    const isPermutation =
      sorted.length === 4 &&
      sorted[0] === 0 &&
      sorted[1] === 1 &&
      sorted[2] === 2 &&
      sorted[3] === 3;
    if (!isPermutation) {
      throw new Error(
        "Dimension positions must be a permutation of [0, 1, 2, 3]. " +
          `Got: frame_dim=${frameDim}, track_dim=${trackDim}, ` +
          `node_dim=${nodeDim}, xy_dim=${xyDim}`,
      );
    }
    return {
      axisOrder: {
        frame: frameDim as number,
        track: trackDim as number,
        node: nodeDim as number,
        xy: xyDim as number,
      },
      presetName: "custom",
    };
  }

  // Use preset (default to "matlab" for backwards compatibility).
  const presetName = preset ?? "matlab";
  if (!(presetName in PRESETS)) {
    throw new Error(
      `Unknown preset '${presetName}'. Available: ${JSON.stringify(Object.keys(PRESETS))}`,
    );
  }
  return { axisOrder: PRESETS[presetName], presetName };
}

/**
 * Compute the transpose axes that convert `fromOrder` into `toOrder`.
 *
 * Port of `_get_transpose_axes`. The returned tuple `axes[targetPos]` gives the
 * source position whose dimension lands at `targetPos`.
 *
 * @internal Exported for unit testing.
 */
export function getTransposeAxes(
  fromOrder: AxisOrder,
  toOrder: AxisOrder,
  ndim: number,
): number[] {
  const dims = dimsForNdim(ndim);
  const axes: number[] = [];
  for (let targetPos = 0; targetPos < ndim; targetPos++) {
    for (const dim of dims) {
      if (toOrder[dim] === targetPos) {
        axes.push(fromOrder[dim]);
        break;
      }
    }
  }
  return axes;
}

/**
 * Dimension-name tuple in stored order for a given axis ordering.
 *
 * Port of `_get_dims_tuple`.
 *
 * @internal Exported for unit testing.
 */
export function getDimsTuple(axisOrder: AxisOrder, ndim: number): string[] {
  const dims = dimsForNdim(ndim);
  const result: string[] = new Array(ndim).fill("");
  for (const dim of dims) {
    if (dim in axisOrder) {
      result[axisOrder[dim]] = dim;
    }
  }
  return result;
}

// =============================================================================
// Flat n-D array transpose helpers
// =============================================================================

/** Row-major strides for a shape. */
function rowMajorStrides(shape: number[]): number[] {
  const strides = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

/**
 * Transpose a flat row-major array according to `axes` (numpy `np.transpose`
 * semantics): output axis `i` is input axis `axes[i]`.
 *
 * Returns the transposed flat array plus its new shape.
 *
 * @internal Exported for unit testing.
 */
export function transposeFlat(
  data: ArrayLike<number>,
  shape: ArrayLike<number | bigint>,
  axes: number[],
): { data: Float64Array; shape: number[] } {
  // h5wasm reports dataset shapes as plain numbers OR BigInts (int64 dims);
  // coerce to plain numbers so all index math stays numeric.
  const numShape: number[] = Array.from(shape, (s) => Number(s));
  const ndim = numShape.length;
  const inStrides = rowMajorStrides(numShape);
  const outShape = axes.map((a) => numShape[a]);
  const total = outShape.reduce((a, b) => a * b, 1);
  const out = new Float64Array(total);

  // Stride in the input that corresponds to each output axis.
  const outStridesInInput = axes.map((a) => inStrides[a]);

  const idx = new Array(ndim).fill(0);
  for (let flat = 0; flat < total; flat++) {
    // Compute the source offset from current multi-index over the output shape.
    let src = 0;
    for (let d = 0; d < ndim; d++) {
      src += idx[d] * outStridesInInput[d];
    }
    out[flat] = data[src] as number;

    // Increment the multi-index (row-major over the output shape).
    for (let d = ndim - 1; d >= 0; d--) {
      idx[d]++;
      if (idx[d] < outShape[d]) break;
      idx[d] = 0;
    }
  }

  return { data: out, shape: outShape };
}

/** Drop dims not in `keep`, then renumber positions by sorted rank (0-based). */
function renumberOrder(order: AxisOrder, keep: string[]): AxisOrder {
  const filtered: Array<[string, number]> = [];
  for (const k of keep) {
    if (k in order) filtered.push([k, order[k]]);
  }
  const positions = filtered.map(([, v]) => v).sort((a, b) => a - b);
  const result: AxisOrder = {};
  for (const [k, v] of filtered) {
    result[k] = positions.indexOf(v);
  }
  return result;
}

// =============================================================================
// HDF5 read helpers (string / scalar / attr decoding)
// =============================================================================

// The string / scalar / attribute decode primitives and the `getDs` dataset
// fetch are shared with the other HDF5-backed readers; they are imported from
// ./h5-read-utils.js at the top of this module. A module-local TextDecoder
// remains for readDimsAttr's JSON-string decode path below.
const textDecoder = new TextDecoder();

/**
 * Read a `dims` attribute as a string[].
 *
 * The Python writer stores `dims` as `json.dumps(dim_names)` — a JSON-array
 * STRING — and the fixtures read back that way through h5wasm. A native
 * string-array attribute form is also accepted for robustness. Returns
 * undefined if absent.
 */
function readDimsAttr(
  attrs: Record<string, unknown> | undefined,
): string[] | undefined {
  if (!attrs || !("dims" in attrs)) return undefined;
  const raw = unwrapAttr(attrs["dims"]);
  if (raw == null) return undefined;
  if (Array.isArray(raw)) {
    return (raw as unknown[]).map(decodeStringElement);
  }
  // String / bytes form: a JSON array string.
  let s: string;
  if (typeof raw === "string") s = raw;
  else if (raw instanceof Uint8Array) s = textDecoder.decode(raw);
  else s = String(raw);
  try {
    const parsed = JSON.parse(s);
    if (Array.isArray(parsed)) return parsed.map((x) => String(x));
  } catch {
    // not JSON
  }
  return undefined;
}

// =============================================================================
// Format detection (port of is_analysis_h5_file)
// =============================================================================

/**
 * Check whether a file is a SLEAP Analysis HDF5 file.
 *
 * True iff the file opens as HDF5 and contains a `track_occupancy` dataset.
 * Returns false on any error. This distinguishes Analysis HDF5 files from JABS
 * HDF5 files (which have a `poseest` group instead).
 */
export async function isAnalysisH5File(
  source: string | ArrayBuffer | Uint8Array,
): Promise<boolean> {
  try {
    // For string paths in Node, fail fast if the file does not exist. The
    // h5wasm node provider opens a File handle lazily and may NOT throw
    // synchronously for a missing path (it can resolve to a stale in-memory
    // handle), which would otherwise yield a false positive. h5py raises for a
    // missing file, so we mirror that. The existence check is routed through the
    // Node provider (registered by h5-node.ts) so this module stays free of
    // Node-only imports; it resolves to null in the browser, where we skip it.
    if (typeof source === "string") {
      const exists = await nodeFileExists(source);
      if (exists === false) return false;
    }

    const { file, close } = await openH5File(source);
    try {
      // Prefer the file's key listing: the h5wasm node provider opens a File
      // handle lazily and does NOT throw synchronously for a missing or
      // non-HDF5 (garbage) path, and `file.get(name)` can then return a truthy
      // placeholder (e.g. the empty string "") rather than null. `file.keys()`,
      // by contrast, forces the root group to be read and THROWS for broken /
      // missing files, while returning the real member list for valid ones.
      const f = file as unknown as {
        keys?: () => string[];
        get(name: string): unknown;
      };
      if (typeof f.keys === "function") {
        const keys = f.keys();
        return Array.isArray(keys) && keys.includes("track_occupancy");
      }
      // Fallback for providers without `keys()`: require a real dataset object.
      const item = f.get("track_occupancy");
      return typeof item === "object" && item !== null;
    } finally {
      close();
    }
  } catch {
    return false;
  }
}

// =============================================================================
// Read (port of read_labels)
// =============================================================================

/** Options for {@link readLabels}. */
export interface ReadLabelsOptions {
  /** Video to associate with the data. Video object or path string. */
  video?: Video | string;
}

/**
 * Load a SLEAP Analysis HDF5 file into a {@link Labels} object.
 *
 * The axis ordering is detected from the stored `dims` attributes (falling back
 * to the legacy `transpose` file attribute). Extended metadata is used to
 * reconstruct skeleton symmetries and video backend metadata when present.
 *
 * @param filename - Path/URL/bytes accepted by `openH5File`.
 * @param options - Optional `video` override.
 */
export async function readLabels(
  filename: string,
  options?: ReadLabelsOptions,
): Promise<Labels> {
  const { file: rawFile, close } = await openH5File(filename);
  const file = rawFile as unknown as H5ReadFile;
  try {
    const fileAttrs = (file.attrs ?? {}) as Record<string, unknown>;

    // --- Determine stored axis order from the tracks `dims` attr or legacy. ---
    const tracksDs = getDs(file, "tracks");
    if (tracksDs == null) {
      throw new Error("Analysis HDF5 file is missing the 'tracks' dataset.");
    }
    const tracksAttrs = (tracksDs.attrs ?? {}) as Record<string, unknown>;

    let storedOrder: AxisOrder;
    const tracksDims = readDimsAttr(tracksAttrs);
    if (tracksDims) {
      storedOrder = {};
      tracksDims.forEach((dim, i) => {
        storedOrder[dim] = i;
      });
    } else {
      // Legacy file: check the transpose attribute (default true).
      const transposeRaw = unwrapAttr(fileAttrs["transpose"]);
      const wasTransposed =
        transposeRaw === undefined ? true : Boolean(transposeRaw);
      if (wasTransposed) {
        storedOrder = PRESETS["matlab"];
      } else {
        storedOrder = { frame: 0, node: 1, xy: 2, track: 3 };
      }
    }

    const canonicalOrder4d: AxisOrder = { frame: 0, track: 1, node: 2, xy: 3 };
    const canonicalOrder3d: AxisOrder = { frame: 0, track: 1, node: 2 };
    const canonicalOrder2d: AxisOrder = { frame: 0, track: 1 };

    // --- Read + reorder tracks (4D). ---
    const tracksShape = (tracksDs.shape ?? []) as ArrayLike<number | bigint>;
    const axes4d = getTransposeAxes(storedOrder, canonicalOrder4d, 4);
    const tracksT = transposeFlat(
      tracksDs.value as ArrayLike<number>,
      tracksShape,
      axes4d,
    );
    const [nFrames, nTracks, nNodes] = tracksT.shape;
    const tracksData = tracksT.data; // canonical (frame, track, node, xy)

    // --- Build 3D / 2D stored orders by dropping dims and renumbering. ---
    const storedOrder3d = renumberOrder(storedOrder, [
      "frame",
      "track",
      "node",
    ]);
    const storedOrder2d = renumberOrder(storedOrder, ["frame", "track"]);

    const axes3d = getTransposeAxes(storedOrder3d, canonicalOrder3d, 3);
    const axes2d = getTransposeAxes(storedOrder2d, canonicalOrder2d, 2);

    const pointScoresDs = getDs(file, "point_scores");
    const pointScoresData = pointScoresDs
      ? transposeFlat(
          pointScoresDs.value as ArrayLike<number>,
          (pointScoresDs.shape ?? []) as ArrayLike<number | bigint>,
          axes3d,
        ).data
      : new Float64Array(nFrames * nTracks * nNodes).fill(NaN);

    const instanceScoresDs = getDs(file, "instance_scores");
    const instanceScoresData = instanceScoresDs
      ? transposeFlat(
          instanceScoresDs.value as ArrayLike<number>,
          (instanceScoresDs.shape ?? []) as ArrayLike<number | bigint>,
          axes2d,
        ).data
      : new Float64Array(nFrames * nTracks).fill(NaN);

    const trackingScoresDs = getDs(file, "tracking_scores");
    const trackingScoresData = trackingScoresDs
      ? transposeFlat(
          trackingScoresDs.value as ArrayLike<number>,
          (trackingScoresDs.shape ?? []) as ArrayLike<number | bigint>,
          axes2d,
        ).data
      : new Float64Array(nFrames * nTracks).fill(NaN);

    // --- String arrays. ---
    const trackNamesDs = getDs(file, "track_names");
    const trackNames = trackNamesDs
      ? decodeStringArray(trackNamesDs.value)
      : [];

    const nodeNamesDs = getDs(file, "node_names");
    const nodeNames = nodeNamesDs ? decodeStringArray(nodeNamesDs.value) : [];

    // Edges: stored with shape [N, 2]; reconstruct string pairs. h5wasm may
    // return a flat array of 2N strings (with shape [N,2]) or a nested array of
    // [src, dst] pairs — handle both.
    const edgeNames: Array<[string, string]> = [];
    const edgeNamesDs = getDs(file, "edge_names");
    if (edgeNamesDs && edgeNamesDs.value != null) {
      const raw = edgeNamesDs.value;
      if (Array.isArray(raw) && raw.length > 0 && Array.isArray(raw[0])) {
        for (const pair of raw as unknown[][]) {
          if (pair.length >= 2) {
            edgeNames.push([
              decodeStringElement(pair[0]),
              decodeStringElement(pair[1]),
            ]);
          }
        }
      } else {
        const flat = decodeStringArray(raw);
        for (let i = 0; i + 1 < flat.length; i += 2) {
          edgeNames.push([flat[i], flat[i + 1]]);
        }
      }
    }

    // --- Metadata. ---
    let videoPath = "";
    const videoPathDs = getDs(file, "video_path");
    if (videoPathDs) videoPath = decodeScalarString(videoPathDs.value);

    let provenance: Record<string, unknown> = {};
    const provenanceDs = getDs(file, "provenance");
    if (provenanceDs) {
      const raw = decodeScalarString(provenanceDs.value);
      if (raw) {
        try {
          provenance = JSON.parse(raw) as Record<string, unknown>;
        } catch {
          provenance = {};
        }
      }
    }

    // --- Extended metadata from file attributes. ---
    const skeletonName = readStringAttr(fileAttrs, "skeleton_name") ?? "";

    let skeletonSymmetries: Array<[string, string]> = [];
    const symRaw = readStringAttr(fileAttrs, "skeleton_symmetries");
    if (symRaw) {
      try {
        skeletonSymmetries = JSON.parse(symRaw) as Array<[string, string]>;
      } catch {
        skeletonSymmetries = [];
      }
    }

    let videoBackendMetadata: Record<string, unknown> = {};
    const vbmRaw = readStringAttr(fileAttrs, "video_backend_metadata");
    if (vbmRaw) {
      try {
        videoBackendMetadata = JSON.parse(vbmRaw) as Record<string, unknown>;
      } catch {
        videoBackendMetadata = {};
      }
    }

    // --- Build video. ---
    let video: Video;
    if (options?.video !== undefined) {
      if (typeof options.video === "string") {
        video = new Video({ filename: options.video });
      } else {
        video = options.video;
      }
    } else {
      video = new Video({ filename: videoPath });
      video.backendMetadata = videoBackendMetadata;
    }

    // --- Build skeleton with edges + symmetries. ---
    // Drop any edge whose endpoints are not real nodes. The Python reference
    // (`Skeleton(nodes=node_names, edges=edge_names)`) silently discards edges
    // referencing unknown node names rather than creating phantom nodes; some
    // fixtures store placeholder edge_names whose endpoints differ from the
    // real nodes, so this filter is required for parity.
    const nodeNameSet = new Set(nodeNames);
    const validEdges = edgeNames.filter(
      ([s, d]) => nodeNameSet.has(s) && nodeNameSet.has(d),
    );
    const skeleton = new Skeleton({
      nodes: nodeNames,
      edges: validEdges,
      name: skeletonName ? skeletonName : undefined,
    });
    for (const pair of skeletonSymmetries) {
      try {
        skeleton.addSymmetry(pair[0], pair[1]);
      } catch {
        // Skip invalid symmetries.
      }
    }

    // --- Build tracks. ---
    const tracks: Array<Track | null> = trackNames.length
      ? trackNames.map((name) => new Track(name))
      : [null];

    // --- Build labeled frames. ---
    const labeledFrames: LabeledFrame[] = [];
    for (let frameIdx = 0; frameIdx < nFrames; frameIdx++) {
      const instances: PredictedInstance[] = [];
      for (let trackIdx = 0; trackIdx < nTracks; trackIdx++) {
        // points: tracksData[frame, track, :, :] -> (nodes, 2)
        const base4d = (frameIdx * nTracks + trackIdx) * nNodes * 2;

        // Skip if all points are NaN.
        let allNaN = true;
        for (let n = 0; n < nNodes * 2; n++) {
          if (!Number.isNaN(tracksData[base4d + n])) {
            allNaN = false;
            break;
          }
        }
        if (allNaN) continue;

        const base3d = (frameIdx * nTracks + trackIdx) * nNodes;
        const base2d = frameIdx * nTracks + trackIdx;

        const instanceScore = instanceScoresData[base2d];
        const trackingScore = trackingScoresData[base2d];

        // Column-stack [x, y, score] into (nodes, 3).
        const pointsData: number[][] = [];
        for (let n = 0; n < nNodes; n++) {
          const x = tracksData[base4d + n * 2];
          const y = tracksData[base4d + n * 2 + 1];
          const score = pointScoresData[base3d + n];
          pointsData.push([x, y, score]);
        }

        const inst = PredictedInstance.fromNumpy({
          pointsData,
          skeleton,
          score: Number.isNaN(instanceScore) ? 0.0 : instanceScore,
          track: tracks[trackIdx] ?? undefined,
          trackingScore: Number.isNaN(trackingScore)
            ? undefined
            : trackingScore,
        });
        instances.push(inst);
      }

      if (instances.length > 0) {
        labeledFrames.push(new LabeledFrame({ video, frameIdx, instances }));
      }
    }

    // --- Assemble Labels. ---
    const realTracks = tracks.filter((t): t is Track => t != null);
    const labels = new Labels({
      labeledFrames,
      videos: [video],
      skeletons: [skeleton],
      tracks: realTracks,
      provenance,
    });
    labels.provenance["filename"] = String(filename);

    return labels;
  } finally {
    close();
  }
}

// =============================================================================
// to_analysis_arrays port (canonical-shape array builder)
// =============================================================================

/** Result of {@link toAnalysisArrays}, all arrays in canonical frame-first order. */
export interface AnalysisArrays {
  /** (frame, track) occupancy as Float64 0/1, flat row-major. */
  occupancy: Float64Array;
  /** (frame, track, node, 2) locations, NaN-filled, flat row-major. */
  locations: Float64Array;
  /** (frame, track, node) point scores, NaN-filled, flat row-major. */
  pointScores: Float64Array;
  /** (frame, track) instance scores, NaN-filled, flat row-major. */
  instanceScores: Float64Array;
  /** (frame, track) tracking scores, NaN-filled, flat row-major. */
  trackingScores: Float64Array;
  trackNames: string[];
  firstFrame: number;
  nFrames: number;
  nTracks: number;
  nNodes: number;
}

/** Minimal structural type covering both Instance and PredictedInstance. */
interface InstanceLike {
  track: Track | null;
  trackingScore?: number | null;
  score?: number;
  numpy(opts?: { scores?: boolean }): number[][];
}

/** Max instances per frame (port of `_max_instances_per_frame`, both types on). */
function maxInstancesPerFrame(lfs: LabeledFrame[]): number {
  let nInstances = 0;
  for (const lf of lfs) {
    const nUser = (lf.userInstances ?? []).length;
    const nPredicted = (lf.predictedInstances ?? []).length;
    // Both instance types are included, so per-frame count is max, not sum.
    const nFrameInstances = Math.max(nUser, nPredicted);
    nInstances = Math.max(nInstances, nFrameInstances);
  }
  return nInstances;
}

/** Frame instances without track identity (port of `_untracked_frame_instances`). */
function untrackedFrameInstances(
  lf: LabeledFrame,
  isSingleInstance: boolean,
): InstanceLike[] {
  const include: InstanceLike[] = [];
  const userInsts = (lf.userInstances ?? []) as unknown as InstanceLike[];
  const predInsts = (lf.predictedInstances ?? []) as unknown as InstanceLike[];
  const hasUser = userInsts.length > 0;

  if (hasUser) {
    for (const inst of userInsts) include.push(inst);

    if (isSingleInstance && include.length > 0) {
      return include;
    }

    for (const inst of predInsts) {
      let skip = false;
      for (const userInst of userInsts) {
        if (
          (userInst as { fromPredicted?: unknown }).fromPredicted !==
            undefined &&
          (userInst as { fromPredicted?: unknown }).fromPredicted === inst
        ) {
          skip = true;
          break;
        }
        if (
          userInst.track != null &&
          inst.track != null &&
          userInst.track === inst.track
        ) {
          skip = true;
          break;
        }
      }
      if (!skip) include.push(inst);
    }
  } else {
    for (const inst of predInsts) include.push(inst);
  }

  return include;
}

/** One instance per track (port of `_tracked_frame_instances`). */
function trackedFrameInstances(lf: LabeledFrame): Map<Track, InstanceLike> {
  const trackToInstance = new Map<Track, InstanceLike>();
  for (const inst of (lf.predictedInstances ??
    []) as unknown as InstanceLike[]) {
    if (inst.track != null) trackToInstance.set(inst.track, inst);
  }
  for (const inst of (lf.userInstances ?? []) as unknown as InstanceLike[]) {
    if (inst.track != null) trackToInstance.set(inst.track, inst);
  }
  return trackToInstance;
}

/** Best-effort video frame count (matches Python `len(video)`; 0 if unknown). */
function videoFrameCount(video: Video): number {
  const shape = video.shape;
  if (shape && shape.length > 0 && typeof shape[0] === "number") {
    return shape[0];
  }
  return 0;
}

/**
 * Build occupancy and point-data matrices for an analysis HDF5 export.
 *
 * Port of `to_analysis_arrays`. All returned arrays are in canonical
 * frame-first order. Throws if there are no labeled frames for the video.
 *
 * @internal Exported for unit testing; it is the canonical array builder.
 */
export function toAnalysisArrays(
  labels: Labels,
  video: Video,
  allFrames: boolean,
  minOccupancy: number,
): AnalysisArrays {
  const lfs = labels.find({ video });
  if (!lfs.length) {
    throw new Error(
      `No labeled frames in video: ${maskUrlCredentialsDeep(video.filename)}`,
    );
  }

  const frameIdxs = lfs.map((lf) => lf.frameIdx).sort((a, b) => a - b);
  const firstFrame = allFrames ? 0 : frameIdxs[0];

  let lastFrame = frameIdxs[frameIdxs.length - 1];
  const videoLength = videoFrameCount(video);
  if (videoLength > 0) {
    lastFrame = Math.max(lastFrame, videoLength - 1);
  }

  const nFrames = lastFrame - firstFrame + 1;

  const skeleton = labels.skeletons[0];
  const nodeCount = skeleton.nodeNames.length;

  // Size the track axis. With no tracks, fall back to untracked slotting.
  const untracked = labels.tracks.length === 0;
  let nTracks: number;
  let isSingleInstance = false;
  let trackToSlot: Map<Track, number> | null = null;
  if (untracked) {
    const nInstances = maxInstancesPerFrame(lfs);
    isSingleInstance = nInstances === 1;
    nTracks = nInstances;
  } else {
    nTracks = labels.tracks.length;
    trackToSlot = new Map<Track, number>();
    labels.tracks.forEach((track, i) => {
      trackToSlot!.set(track, i);
    });
  }

  // Canonical-shape matrices.
  const occupancy = new Float64Array(nFrames * nTracks); // 0-filled
  const locations = new Float64Array(nFrames * nTracks * nodeCount * 2).fill(
    NaN,
  );
  const pointScores = new Float64Array(nFrames * nTracks * nodeCount).fill(NaN);
  const instanceScores = new Float64Array(nFrames * nTracks).fill(NaN);
  const trackingScores = new Float64Array(nFrames * nTracks).fill(NaN);

  const lfMap = new Map<number, LabeledFrame>();
  for (const lf of lfs) lfMap.set(lf.frameIdx, lf);

  for (let frameIdx = firstFrame; frameIdx <= lastFrame; frameIdx++) {
    const frameI = frameIdx - firstFrame;
    const lf = lfMap.get(frameIdx);
    if (!lf) continue;

    // (trackSlot, instance) pairs.
    const slotted: Array<[number, InstanceLike]> = [];
    if (untracked) {
      const insts = untrackedFrameInstances(lf, isSingleInstance);
      insts.forEach((inst, i) => {
        slotted.push([i, inst]);
      });
    } else {
      for (const [track, inst] of trackedFrameInstances(lf)) {
        const slot = trackToSlot!.get(track);
        if (slot !== undefined) slotted.push([slot, inst]);
      }
    }

    for (const [trackI, inst] of slotted) {
      if (trackI >= nTracks) continue;

      occupancy[frameI * nTracks + trackI] = 1;

      // locations[frame, track, :, :] = inst.numpy() (nodes x 2)
      const xy = inst.numpy();
      const locBase = (frameI * nTracks + trackI) * nodeCount * 2;
      for (let n = 0; n < nodeCount; n++) {
        const row = xy[n] ?? [NaN, NaN];
        locations[locBase + n * 2] = row[0];
        locations[locBase + n * 2 + 1] = row[1];
      }

      const ts = inst.trackingScore;
      if (ts !== undefined && ts !== null) {
        trackingScores[frameI * nTracks + trackI] = ts;
      }

      if (inst instanceof PredictedInstance) {
        // point scores from numpy({scores:true}) -> [[x,y,score],...]
        const withScores = inst.numpy({ scores: true });
        const psBase = (frameI * nTracks + trackI) * nodeCount;
        for (let n = 0; n < nodeCount; n++) {
          const row = withScores[n];
          pointScores[psBase + n] = row ? row[2] : NaN;
        }
        if (inst.score !== undefined && inst.score !== null) {
          instanceScores[frameI * nTracks + trackI] = inst.score;
        }
      }
    }
  }

  // Filter empty / low-occupancy tracks.
  const occupiedFrames = new Array<number>(nTracks).fill(0);
  for (let f = 0; f < nFrames; f++) {
    for (let t = 0; t < nTracks; t++) {
      occupiedFrames[t] += occupancy[f * nTracks + t];
    }
  }
  const keepMask: boolean[] = occupiedFrames.map(
    (count) => count > 0 && count / nFrames >= minOccupancy,
  );
  const keepIdxs: number[] = [];
  keepMask.forEach((keep, i) => {
    if (keep) keepIdxs.push(i);
  });

  let finalNTracks = nTracks;
  let finalOccupancy = occupancy;
  let finalLocations = locations;
  let finalPointScores = pointScores;
  let finalInstanceScores = instanceScores;
  let finalTrackingScores = trackingScores;

  if (keepIdxs.length !== nTracks) {
    finalNTracks = keepIdxs.length;
    finalOccupancy = new Float64Array(nFrames * finalNTracks);
    finalLocations = new Float64Array(nFrames * finalNTracks * nodeCount * 2);
    finalPointScores = new Float64Array(nFrames * finalNTracks * nodeCount);
    finalInstanceScores = new Float64Array(nFrames * finalNTracks);
    finalTrackingScores = new Float64Array(nFrames * finalNTracks);

    for (let f = 0; f < nFrames; f++) {
      for (let newT = 0; newT < finalNTracks; newT++) {
        const oldT = keepIdxs[newT];
        finalOccupancy[f * finalNTracks + newT] = occupancy[f * nTracks + oldT];
        finalInstanceScores[f * finalNTracks + newT] =
          instanceScores[f * nTracks + oldT];
        finalTrackingScores[f * finalNTracks + newT] =
          trackingScores[f * nTracks + oldT];
        for (let n = 0; n < nodeCount; n++) {
          finalPointScores[(f * finalNTracks + newT) * nodeCount + n] =
            pointScores[(f * nTracks + oldT) * nodeCount + n];
          const newBase = ((f * finalNTracks + newT) * nodeCount + n) * 2;
          const oldBase = ((f * nTracks + oldT) * nodeCount + n) * 2;
          finalLocations[newBase] = locations[oldBase];
          finalLocations[newBase + 1] = locations[oldBase + 1];
        }
      }
    }
  }

  // Track names sized to surviving tracks.
  let trackNames: string[];
  if (untracked) {
    trackNames = Array.from({ length: finalNTracks }, (_, i) => `track_${i}`);
  } else {
    trackNames = labels.tracks.filter((_, i) => keepMask[i]).map((t) => t.name);
  }

  return {
    occupancy: finalOccupancy,
    locations: finalLocations,
    pointScores: finalPointScores,
    instanceScores: finalInstanceScores,
    trackingScores: finalTrackingScores,
    trackNames,
    firstFrame,
    nFrames,
    nTracks: finalNTracks,
    nNodes: nodeCount,
  };
}

// =============================================================================
// Write (port of write_labels)
// =============================================================================

/** Options for {@link writeLabels}. */
export interface WriteLabelsOptions {
  /** Video to export. Video object or integer index. Defaults to videos[0]. */
  video?: Video | number;
  /** Source labels path (stored as metadata). */
  labelsPath?: string;
  /** Include all frames from 0 to the last labeled frame. Default true. */
  allFrames?: boolean;
  /** Minimum track occupancy ratio (0-1) to keep. Default 0. */
  minOccupancy?: number;
  /** Axis ordering preset ("matlab" default, "standard"). */
  preset?: string;
  /** Explicit frame dimension position (mutually exclusive with preset). */
  frameDim?: number;
  /** Explicit track dimension position (mutually exclusive with preset). */
  trackDim?: number;
  /** Explicit node dimension position (mutually exclusive with preset). */
  nodeDim?: number;
  /** Explicit xy dimension position (mutually exclusive with preset). */
  xyDim?: number;
  /** Store extended metadata for full round-trip. Default true. */
  saveMetadata?: boolean;
}

/** Minimal h5wasm write-file type surface used here. */
interface H5WriteFile {
  create_dataset(opts: {
    name: string;
    data: unknown;
    shape?: number[];
    dtype?: string;
    chunks?: number[];
    compression?: string | number;
    compression_opts?: number | number[];
  }): void;
  create_attribute(
    name: string,
    value: unknown,
    shape?: unknown,
    dtype?: unknown,
  ): void;
  get(
    name: string,
  ): { create_attribute: (n: string, v: unknown) => void } | null;
  close(): void;
}

/** Read the JS package version (not hardcoded), via the Node provider. */
async function readPackageVersion(): Promise<string> {
  try {
    const version = await nodeReadPackageVersion();
    if (version) return version;
  } catch {
    // ignore — fall back to the sentinel below.
  }
  return "0.0.0";
}

/** Set a root file-level string attribute, falling back to get("/"). */
function setRootAttr(f: H5WriteFile, name: string, value: string): void {
  try {
    f.create_attribute(name, value);
    return;
  } catch {
    // fall through
  }
  const root = f.get("/");
  if (root) {
    root.create_attribute(name, value);
  }
}

/** Stringify a video filename (first element when it's a list). */
function videoFilenameString(video: Video): string {
  const fn = video.filename;
  if (Array.isArray(fn)) return fn.length ? fn[0] : "";
  return fn ?? "";
}

/**
 * Build the SLEAP Analysis HDF5 bytes for a {@link Labels} object.
 *
 * Browser-safe: the file is assembled entirely in an h5wasm in-memory virtual FS
 * and returned as bytes — no Node filesystem access — so it works in the browser
 * (and the Tauri WebView) as well as Node. {@link writeLabels} wraps this to
 * write the bytes to disk. Mirrors the `saveSlpToBytes` / `writeLabels` split.
 *
 * @param labels - Labels to export.
 * @param options - Export options (see {@link WriteLabelsOptions}).
 * @returns The `.h5` file contents.
 */
export async function writeLabelsToBytes(
  labels: Labels,
  options?: WriteLabelsOptions,
): Promise<Uint8Array> {
  const allFrames = options?.allFrames ?? true;
  const minOccupancy = options?.minOccupancy ?? 0.0;
  const saveMetadata = options?.saveMetadata ?? true;

  // Resolve axis ordering.
  const { axisOrder, presetName } = getAxisOrder(
    options?.preset,
    options?.frameDim,
    options?.trackDim,
    options?.nodeDim,
    options?.xyDim,
  );

  // Resolve video.
  let video: Video;
  if (options?.video === undefined) {
    video = labels.videos[0];
  } else if (typeof options.video === "number") {
    video = labels.videos[options.video];
  } else {
    video = options.video;
  }

  // Canonical-shape arrays (throws "No labeled frames..." when empty).
  const arrays = toAnalysisArrays(labels, video, allFrames, minOccupancy);
  const { nFrames, nTracks, nNodes } = arrays;

  // Canonical orders.
  const canonicalOrder4d: AxisOrder = { frame: 0, track: 1, node: 2, xy: 3 };
  const canonicalOrder3d: AxisOrder = { frame: 0, track: 1, node: 2 };
  const canonicalOrder2d: AxisOrder = { frame: 0, track: 1 };

  // Target orders (renumbered) for 3D / 2D.
  const targetOrder3d = renumberOrder(axisOrder, ["frame", "track", "node"]);
  const targetOrder2d = renumberOrder(axisOrder, ["frame", "track"]);

  // matlab quirk: track_occupancy stays canonical (frame, track).
  const targetOrderOccupancy: AxisOrder =
    presetName === "matlab" ? { frame: 0, track: 1 } : targetOrder2d;

  // Transpose axes.
  const axes4d = getTransposeAxes(canonicalOrder4d, axisOrder, 4);
  const axes3d = getTransposeAxes(canonicalOrder3d, targetOrder3d, 3);
  const axes2d = getTransposeAxes(canonicalOrder2d, targetOrder2d, 2);
  const axesOccupancy = getTransposeAxes(
    canonicalOrder2d,
    targetOrderOccupancy,
    2,
  );

  // Reorder arrays.
  const locationsT = transposeFlat(
    arrays.locations,
    [nFrames, nTracks, nNodes, 2],
    axes4d,
  );
  const pointScoresT = transposeFlat(
    arrays.pointScores,
    [nFrames, nTracks, nNodes],
    axes3d,
  );
  const instanceScoresT = transposeFlat(
    arrays.instanceScores,
    [nFrames, nTracks],
    axes2d,
  );
  const trackingScoresT = transposeFlat(
    arrays.trackingScores,
    [nFrames, nTracks],
    axes2d,
  );
  const occupancyT = transposeFlat(
    arrays.occupancy,
    [nFrames, nTracks],
    axesOccupancy,
  );

  // Dimension-name attributes.
  const dims4d = getDimsTuple(axisOrder, 4);
  const dims3d = getDimsTuple(targetOrder3d, 3);
  const dims2d = getDimsTuple(targetOrder2d, 2);
  const dimsOccupancy = getDimsTuple(targetOrderOccupancy, 2);

  // Skeleton info.
  const skeleton = labels.skeletons[0];
  const nodeNames = skeleton.nodeNames;
  const edgeNames = skeleton.edges.map(
    (e) => [e.source.name, e.destination.name] as [string, string],
  );
  const edgeInds = skeleton.edgeIndices;

  const version = await readPackageVersion();

  // Build the file in the h5wasm virtual FS.
  const module = await getH5Module();
  ensureH5StagingDir(module);
  const memPath = `/tmp/analysis_${Date.now()}_${Math.random().toString(16).slice(2)}.h5`;
  const f = new (
    module as unknown as { File: new (p: string, m: string) => H5WriteFile }
  ).File(memPath, "w");

  try {
    // Numeric dataset writer with a JSON-array-string `dims` attribute.
    const writeNumeric = (
      name: string,
      data: Float64Array,
      shape: number[],
      dimNames: string[],
    ): void => {
      // h5wasm (HDF5) requires a chunk layout whenever a compression filter is
      // applied; an unchunked (contiguous) dataset cannot be gzip-compressed.
      // Use the full dataset shape as a single chunk. A dataset with any
      // zero-length axis cannot be chunked, so fall back to uncompressed.
      const canCompress = shape.length > 0 && shape.every((d) => d > 0);
      if (canCompress) {
        f.create_dataset({
          name,
          data,
          shape,
          dtype: "<f8",
          chunks: shape,
          compression: "gzip",
          compression_opts: 9,
        });
      } else {
        f.create_dataset({ name, data, shape, dtype: "<f8" });
      }
      const ds = f.get(name);
      // Write `dims` as a JSON-array string attribute, byte-matching Python's
      // `json.dumps(dim_names)` (the fixtures store dims this way; h5wasm and
      // h5py read it back as a string). The reader accepts both this string
      // form and a native string-array form.
      if (ds) ds.create_attribute("dims", JSON.stringify(dimNames));
    };

    // Core matrices.
    writeNumeric("tracks", locationsT.data, locationsT.shape, dims4d);
    writeNumeric(
      "track_occupancy",
      occupancyT.data,
      occupancyT.shape,
      dimsOccupancy,
    );
    writeNumeric("point_scores", pointScoresT.data, pointScoresT.shape, dims3d);
    writeNumeric(
      "instance_scores",
      instanceScoresT.data,
      instanceScoresT.shape,
      dims2d,
    );
    writeNumeric(
      "tracking_scores",
      trackingScoresT.data,
      trackingScoresT.shape,
      dims2d,
    );

    // String datasets (h5py-native).
    f.create_dataset({ name: "track_names", data: arrays.trackNames });
    f.create_dataset({ name: "node_names", data: nodeNames });
    // edge_names as 2D [N, 2] string dataset (flattened to 2N strings).
    const edgeFlat: string[] = [];
    for (const [s, d] of edgeNames) {
      edgeFlat.push(s, d);
    }
    f.create_dataset({
      name: "edge_names",
      data: edgeFlat,
      shape: [edgeNames.length, 2],
    });

    // edge_inds as int32 [N, 2]. Edge indices are small; int32 is plenty and
    // h5py reads them as plain ints regardless of width. We deliberately avoid
    // int64 (BigInt) here: h5wasm throws "Cannot mix BigInt and other types"
    // when an <i8 dataset is written AFTER a variable-length string dataset
    // (edge_names) in the same file.
    f.create_dataset({
      name: "edge_inds",
      data: Int32Array.from(edgeInds.flat()),
      shape: [edgeInds.length, 2],
      dtype: "<i4",
    });

    // Scalar metadata datasets.
    f.create_dataset({
      name: "labels_path",
      data: options?.labelsPath
        ? maskUrlCredentials(String(options.labelsPath))
        : "",
    });
    f.create_dataset({
      name: "video_path",
      data: maskUrlCredentials(videoFilenameString(video) || ""),
    });
    f.create_dataset({
      name: "video_ind",
      data: Int32Array.from([labels.videos.indexOf(video)]),
      shape: [],
      dtype: "<i4",
    });
    f.create_dataset({
      name: "provenance",
      data: JSON.stringify(maskUrlCredentialsDeep(labels.provenance ?? {})),
    });

    // File-level attributes (root). Always written for format identification.
    setRootAttr(f, "preset", presetName);
    setRootAttr(f, "format", "analysis");
    setRootAttr(f, "sleap_io_version", version);

    if (saveMetadata) {
      const symmetries = skeleton.symmetryNames;
      setRootAttr(f, "skeleton_name", skeleton.name ?? "");
      setRootAttr(f, "skeleton_symmetries", JSON.stringify(symmetries));
      setRootAttr(
        f,
        "video_backend_metadata",
        JSON.stringify(maskUrlCredentialsDeep(video.backendMetadata ?? {})),
      );
    }
  } finally {
    f.close();
  }

  const fsModule = getH5FileSystem(module);
  const bytes = fsModule.readFile!(memPath);
  fsModule.unlink!(memPath);
  return bytes;
}

/**
 * Save a {@link Labels} object to a SLEAP Analysis HDF5 file.
 *
 * Node-only: builds the bytes via {@link writeLabelsToBytes} (an h5wasm in-memory
 * virtual FS) and writes them to disk through the Node filesystem ops registered
 * by `h5-node.ts`.
 *
 * @param labels - Labels to export.
 * @param filename - Output file path.
 * @param options - Export options (see {@link WriteLabelsOptions}).
 */
export async function writeLabels(
  labels: Labels,
  filename: string,
  options?: WriteLabelsOptions,
): Promise<void> {
  const bytes = await writeLabelsToBytes(labels, options);
  // Write to disk via the registered Node file writer (Node-only). Routing
  // through the provider keeps this module free of Node-only imports.
  await nodeWriteFile(filename, bytes);
}
