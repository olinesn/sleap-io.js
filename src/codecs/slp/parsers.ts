/**
 * Shared parsing functions for SLP file metadata.
 * Used by both the full h5wasm-based reader and the lite jsfive-based reader.
 */

import { Skeleton, Node } from "../../model/skeleton.js";
import { Track } from "../../model/instance.js";
import { Camera, FrameGroup, InstanceGroup } from "../../model/camera.js";
import type { Identity } from "../../model/identity.js";
import { Instance3D, PredictedInstance3D } from "../../model/instance3d.js";
import { isUrl, redactUrl } from "../../io/remote.js";

const textDecoder = new TextDecoder();

/**
 * Parse JSON from HDF5 attribute.
 * Handles string, Uint8Array, and buffer variations.
 */
export function parseJsonAttr(attr: unknown): unknown {
  if (!attr) return null;
  const value = (attr as { value?: unknown }).value ?? attr;
  if (typeof value === "string") return JSON.parse(value);
  if (value instanceof Uint8Array) return JSON.parse(textDecoder.decode(value));
  if (value && typeof value === "object" && "buffer" in value) {
    return JSON.parse(
      textDecoder.decode(
        new Uint8Array((value as { buffer: ArrayBuffer }).buffer),
      ),
    );
  }
  // If value is already a parsed object (e.g., from streaming worker), return as-is
  if (value && typeof value === "object") {
    return value;
  }
  return JSON.parse(String(value));
}

/**
 * Determine whether a `/metadata` `json` HDF5 attribute holds usable content.
 *
 * Mirrors the precondition for which Python's `read_metadata` raises (an absent
 * `json` attribute surfaces as a `KeyError`). An attribute is considered absent
 * when it is `undefined`/`null`, an empty string (including a `{ value }`
 * wrapper around one), or an empty byte buffer. A non-empty value -- even one
 * that is malformed JSON -- is treated as present so that downstream parsing can
 * surface the real decode error rather than masking it as corruption (see
 * Python #446's `test_read_metadata_malformed_json_not_remasked`).
 */
function hasMetadataJsonAttr(attr: unknown): boolean {
  if (attr === undefined || attr === null) return false;
  const value = (attr as { value?: unknown }).value ?? attr;
  if (typeof value === "string") return value.length > 0;
  if (value instanceof Uint8Array) return value.length > 0;
  if (value && typeof value === "object" && "buffer" in value) {
    return new Uint8Array((value as { buffer: ArrayBuffer }).buffer).length > 0;
  }
  // Already-parsed objects (e.g. from the streaming worker) are usable as-is.
  return true;
}

/**
 * Build the helpful error message for a corrupt `.slp` whose `/metadata` group
 * is missing its required `json` attribute. Mirrors the wording/intent of
 * Python sleap-io PR #446 (`read_metadata` raising `ValueError`).
 */
export function missingMetadataJsonError(labelsPath: string): Error {
  return new Error(
    `The SLEAP labels file '${labelsPath}' is missing its required ` +
      "metadata JSON blob (the 'metadata' HDF5 group has no readable 'json' " +
      "attribute) and is likely corrupt. If you have a working .slp file " +
      "with the same skeleton, you can copy the attribute into a BACKUP " +
      "COPY of the corrupt file with h5py (back up first):\n" +
      "    import h5py\n" +
      "    with h5py.File('working.slp', 'r') as src, " +
      "h5py.File('corrupt_copy.slp', 'a') as dst:\n" +
      "        dst['metadata'].attrs['json'] = src['metadata'].attrs['json']\n" +
      "Only do this if the skeletons match exactly, otherwise the loaded " +
      "data will be wrong.",
  );
}

/**
 * Parse the `/metadata` group's `json` HDF5 attribute, raising a helpful error
 * when it is absent.
 *
 * The `.slp` reader needs the JSON-encoded metadata blob to recover skeletons,
 * provenance, etc. When the `metadata` group exists but the `json` attribute is
 * missing/empty (a truncated, foreign, or otherwise corrupt file), the legacy
 * code silently parsed it as `null` and failed later with an opaque error. This
 * mirrors Python sleap-io PR #446: surface a clear, actionable error naming the
 * file and the missing attribute instead. Malformed-but-present JSON still
 * surfaces as the underlying parse error.
 *
 * @param attr The raw `json` attribute value as read from the HDF5 file.
 * @param labelsPath Path/identifier of the `.slp` file (used in the message).
 * @throws {Error} If the `json` attribute is absent or empty.
 */
export function parseMetadataJson(attr: unknown, labelsPath: string): unknown {
  if (!hasMetadataJsonAttr(attr)) {
    throw missingMetadataJsonError(labelsPath);
  }
  return parseJsonAttr(attr);
}

/**
 * A `/metadata` `format_id` at or above this uses a layout this library can't read.
 *
 * Format 3.0 (the default from Python sleap-io 0.10) stores every table as a group
 * of columns. Read with the 2.x code those tables come back empty, so the file would
 * open as 0 frames with no error, and saving it would write the empty project.
 * Mirrors Python's `SLP_FORMAT_ID_LIMIT`. Raise it once the readers handle 3.0
 * (#237), and honor the `/metadata` `required_features` attribute then as Python does.
 */
const SLP_FORMAT_ID_LIMIT = 3.0;

/**
 * Thrown when a `.slp` file is newer than this library can read. Nothing has been
 * read from the file, so callers can catch this to word it for their own users.
 */
export class UnsupportedSlpFormatError extends Error {
  /** The file's `format_id`, e.g. `3`. */
  readonly formatId: number;

  /**
   * @param formatId The file's `format_id`.
   * @param labelsPath Path, URL or name of the file, if known (used in the
   *   message; credentials in a URL are masked).
   */
  constructor(formatId: number, labelsPath?: string) {
    const name =
      labelsPath && isUrl(labelsPath) ? redactUrl(labelsPath) : labelsPath;
    const version = Number.isInteger(formatId)
      ? formatId.toFixed(1)
      : String(formatId);
    super(
      `${name || "This file"} uses SLP format ${version}, which is newer than ` +
        `this version of sleap-io.js supports (max ${SLP_FORMAT_ID_LIMIT - 1}.x). ` +
        "Upgrade sleap-io.js to read it, or re-save the file from Python " +
        "sleap-io with `legacy_format=True`.",
    );
    this.name = "UnsupportedSlpFormatError";
    this.formatId = formatId;
    // Restore the prototype chain so `instanceof` holds even under older
    // transpile targets (matches UnsupportedVideoFormatError).
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Refuse a `.slp` file newer than this library can read. Mirrors Python sleap-io's
 * `_check_format_id`: a file without a `format_id` passes, as before.
 *
 * Call this before reading anything else from the file, since a newer layout may
 * not hold what the other checks look for.
 *
 * @param formatId The file's `format_id`.
 * @param labelsPath Path, URL or name of the file, if known.
 * @throws {UnsupportedSlpFormatError} If the file is too new.
 */
export function checkFormatId(formatId: number, labelsPath?: string): void {
  if (formatId >= SLP_FORMAT_ID_LIMIT) {
    throw new UnsupportedSlpFormatError(formatId, labelsPath);
  }
}

/**
 * Trim trailing nulls and whitespace from a string (for fixed-width HDF5 strings).
 */
function trimHdf5String(str: string): string {
  return str.trim().replace(/\0+$/, "");
}

/**
 * Normalize an HDF5 attribute value to a string.
 *
 * Handles the three shapes encountered across reader paths:
 * - jsfive: `{ value: string | Uint8Array }`
 * - streaming worker (serialized): `{ value: string }`
 * - already a primitive string
 */
export function attrToString(attr: unknown): string | undefined {
  if (attr === undefined || attr === null) return undefined;
  if (typeof attr === "string") return trimHdf5String(attr);
  if (attr instanceof Uint8Array)
    return trimHdf5String(textDecoder.decode(attr));
  if (typeof attr === "object" && "value" in attr) {
    const v = (attr as { value: unknown }).value;
    if (typeof v === "string") return trimHdf5String(v);
    if (v instanceof Uint8Array) return trimHdf5String(textDecoder.decode(v));
  }
  return undefined;
}

/**
 * Normalize an HDF5 numeric attribute to a finite number, or undefined.
 *
 * Handles BigInt (HDF5 int64/uint64), wrapped `{ value }` objects, and plain numbers.
 * Returns undefined for non-finite or non-numeric inputs so callers can `??`-chain
 * it with a fallback. Callers are responsible for their own range checks (e.g. > 0).
 */
export function attrToNumber(attr: unknown): number | undefined {
  if (attr === undefined || attr === null) return undefined;
  let raw: unknown = attr;
  if (typeof attr === "object" && "value" in attr) {
    raw = (attr as { value: unknown }).value;
  }
  if (
    typeof raw !== "number" &&
    typeof raw !== "bigint" &&
    typeof raw !== "string"
  ) {
    return undefined;
  }
  const num = typeof raw === "bigint" ? Number(raw) : Number(raw);
  return Number.isFinite(num) ? num : undefined;
}

/**
 * Decode the `.value` of a scalar (or length-1) HDF5 string dataset
 * (`H5T_STRING`, `|S<n>`) to a plain string, or `undefined` when it holds no
 * usable text.
 *
 * Oversized JSON that exceeds HDF5's ~64 KB attribute ceiling is spilled by
 * Python sleap-io into a `json` *dataset* written as `np.bytes_(blob)` — a
 * SCALAR (shape `()`) fixed-length string. The reader backends surface that
 * single element in several shapes, all normalized here:
 *   - h5wasm (sync and worker): a decoded JS `string`
 *   - jsfive: a length-1 `Array` whose element is the string (or `Uint8Array`)
 *   - raw bytes: `Uint8Array` / `ArrayBuffer` / another `ArrayBufferView` /
 *     a `{ buffer }` typed-array wrapper (streaming transport) / `{ value }`
 * Trailing NUL padding from fixed-length storage is trimmed. Shared by the two
 * `{group}/source_video` readers and `read_video_crops` so every scalar-`|S<n>`
 * JSON dataset decodes through one robust path. Never throws.
 */
export function datasetValueToString(value: unknown): string | undefined {
  // Trim (fixed-length NUL padding + whitespace); treat empty as "no text" so
  // callers fall back to the attribute form.
  const norm = (s: string): string | undefined => {
    const t = trimHdf5String(s);
    return t.length ? t : undefined;
  };
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string") return norm(value);
  if (value instanceof Uint8Array) return norm(textDecoder.decode(value));
  if (value instanceof ArrayBuffer) {
    return norm(textDecoder.decode(new Uint8Array(value)));
  }
  if (ArrayBuffer.isView(value)) {
    const view = value as ArrayBufferView;
    return norm(
      textDecoder.decode(
        new Uint8Array(view.buffer, view.byteOffset, view.byteLength),
      ),
    );
  }
  if (Array.isArray(value)) {
    return value.length ? datasetValueToString(value[0]) : undefined;
  }
  if (typeof value === "object") {
    const obj = value as { value?: unknown; buffer?: unknown };
    if (obj.buffer instanceof ArrayBuffer) {
      return norm(textDecoder.decode(new Uint8Array(obj.buffer)));
    }
    if ("value" in obj && obj.value !== value) {
      return datasetValueToString(obj.value);
    }
  }
  return undefined;
}

/**
 * Parse a single JSON entry from a dataset value.
 * Handles both string and Uint8Array entries.
 * Trims trailing nulls/whitespace for fixed-width HDF5 strings.
 */
export function parseJsonEntry(entry: unknown): unknown {
  if (typeof entry === "string") return JSON.parse(trimHdf5String(entry));
  if (entry instanceof Uint8Array)
    return JSON.parse(trimHdf5String(textDecoder.decode(entry)));
  if (entry && typeof entry === "object" && "buffer" in entry) {
    return JSON.parse(
      trimHdf5String(
        textDecoder.decode(
          new Uint8Array((entry as { buffer: ArrayBuffer }).buffer),
        ),
      ),
    );
  }
  return entry;
}

/**
 * Decode a variable-length HDF5 dataset entry (string / Uint8Array / buffer view)
 * to its trimmed text, without JSON-parsing. Returns `undefined` for shapes that
 * are not text-like. Used by the session readers to distinguish a genuinely
 * absent/empty dataset from an entry that read back blank (the h5wasm
 * variable-length-string read ceiling — see {@link sessionsReadError}).
 */
export function decodeEntryText(entry: unknown): string | undefined {
  if (typeof entry === "string") return trimHdf5String(entry);
  if (entry instanceof Uint8Array)
    return trimHdf5String(textDecoder.decode(entry));
  if (entry && typeof entry === "object" && "buffer" in entry)
    return trimHdf5String(
      textDecoder.decode(
        new Uint8Array((entry as { buffer: ArrayBuffer }).buffer),
      ),
    );
  return undefined;
}

/**
 * Build a descriptive error for a `sessions_json` dataset that is present but
 * cannot be recovered — either an entry read back empty (the h5wasm
 * variable-length-string read ceiling of ~0.45 GB, well under V8's ~512 MB string
 * cap; see sleap-io.js#220) or an entry that failed to JSON-parse.
 *
 * The session readers MUST throw this rather than silently return `[]`: a session
 * blob that reads empty would otherwise drop calibration + all frame grouping +
 * all 3D with no indication, and the next save re-writes the file 2D-only,
 * permanently destroying the data. Fail loud so callers can surface it.
 *
 * @param nEntries Number of entries the `sessions_json` dataset reported.
 * @param entry The offending entry (used only to detect the blank/ceiling case).
 * @param cause The underlying parse error, if any.
 */
export function sessionsReadError(
  nEntries: number,
  entry: unknown,
  cause?: unknown,
): Error {
  const text = decodeEntryText(entry);
  const blank = text === "" || text === undefined;
  const plural = nEntries === 1 ? "entry" : "entries";
  const why = blank
    ? "an entry read back empty — the variable-length string likely exceeds the " +
      "h5wasm read limit (~0.45 GB), so calibration, frame grouping, and 3D " +
      "cannot be recovered from this reader"
    : `an entry could not be parsed as JSON${cause ? ` (${String((cause as Error)?.message ?? cause)})` : ""}`;
  return new Error(
    `Failed to read sessions: sessions_json is present (${nEntries} ${plural}) but ${why}. ` +
      "Refusing to load a sessions-less (2D-only) view of a file that contains session " +
      "data, to avoid silently overwriting it on the next save.",
  );
}

/**
 * Resolve edge type from py/reduce patterns in SLEAP metadata.
 * Type 1 = regular edge, Type 2 = symmetry.
 */
export function resolveEdgeType(
  edgeType: unknown,
  cache: Map<number, number>,
  state: { nextId: number },
): number {
  if (!edgeType || typeof edgeType !== "object") return 1;
  const et = edgeType as Record<string, unknown>;

  if (et["py/reduce"]) {
    const reduce = et["py/reduce"] as unknown[];
    const tuple = (reduce[1] as Record<string, unknown>)?.["py/tuple"] as
      | number[]
      | undefined;
    const typeId = tuple?.[0] ?? 1;
    cache.set(state.nextId, typeId);
    state.nextId += 1;
    return typeId;
  }
  if (et["py/tuple"]) {
    const tuple = et["py/tuple"] as number[];
    const typeId = tuple[0] ?? 1;
    cache.set(state.nextId, typeId);
    state.nextId += 1;
    return typeId;
  }
  if (et["py/id"]) {
    const pyId = et["py/id"] as number;
    return cache.get(pyId) ?? pyId;
  }
  return 1;
}

/**
 * Parse skeletons from metadata JSON.
 * Handles py/reduce patterns for edge types and builds full Skeleton objects.
 */
export function parseSkeletons(metadataJson: unknown): Skeleton[] {
  if (!metadataJson || typeof metadataJson !== "object") return [];

  const meta = metadataJson as Record<string, unknown>;
  const nodeNames = (
    (meta.nodes as Array<{ name?: string } | string>) ?? []
  ).map((node) =>
    typeof node === "object" ? (node.name ?? "") : String(node),
  );
  const skeletonEntries =
    (meta.skeletons as Array<Record<string, unknown>>) ?? [];
  const skeletons: Skeleton[] = [];

  for (const entry of skeletonEntries) {
    const edges: Array<[number, number]> = [];
    const symmetries: Array<[number, number]> = [];
    const typeCache = new Map<number, number>();
    const typeState = { nextId: 1 };

    // Classic PyQt-SLEAP jsonpickles each skeleton and wraps its networkx
    // node-link graph under `nx_graph`; modern sleap-io writes the node-link
    // dict FLAT on the entry. Read nodes/links/graph from whichever is present —
    // otherwise a classic entry has no `entry.nodes`, and the empty-node
    // fallback below wrongly adopts the ENTIRE global node list (e.g.
    // clip.2node.slp's 2-node skeleton decoding to the parent 13-node fly).
    const graphSrc =
      entry.nx_graph && typeof entry.nx_graph === "object"
        ? (entry.nx_graph as Record<string, unknown>)
        : entry;

    const entryNodes =
      (graphSrc.nodes as Array<{ id?: number } | number>) ?? [];
    const skeletonNodeIds = entryNodes.map((node) =>
      Number(typeof node === "object" ? (node.id ?? 0) : node),
    );
    const nodeOrder = skeletonNodeIds.length
      ? skeletonNodeIds
      : nodeNames.map((_, index) => index);

    const nodes = nodeOrder
      .map((nodeId: number) => nodeNames[nodeId])
      .filter((name): name is string => name !== undefined)
      .map((name) => new Node(name));

    const nodeIndexById = new Map<number, number>();
    nodeOrder.forEach((nodeId: number, index: number) => {
      nodeIndexById.set(Number(nodeId), index);
    });

    const links =
      (graphSrc.links as Array<{
        source: number;
        target: number;
        type?: unknown;
      }>) ?? [];
    for (const link of links) {
      const source = Number(link.source);
      const target = Number(link.target);
      const edgeType = resolveEdgeType(link.type, typeCache, typeState);
      if (edgeType === 2) {
        symmetries.push([source, target]);
      } else {
        edges.push([source, target]);
      }
    }

    const remapPair = (pair: [number, number]): [number, number] | null => {
      const sourceIndex = nodeIndexById.get(pair[0]);
      const targetIndex = nodeIndexById.get(pair[1]);
      if (sourceIndex === undefined || targetIndex === undefined) return null;
      return [sourceIndex, targetIndex];
    };

    const mappedEdges = edges
      .map(remapPair)
      .filter((pair): pair is [number, number] => pair !== null);

    const seenSymmetries = new Set<string>();
    const mappedSymmetries = symmetries
      .map(remapPair)
      .filter((pair): pair is [number, number] => pair !== null)
      .filter(([a, b]) => {
        const key = a < b ? `${a}-${b}` : `${b}-${a}`;
        if (seenSymmetries.has(key)) return false;
        seenSymmetries.add(key);
        return true;
      });

    const graph = graphSrc.graph as { name?: string } | undefined;
    const skeleton = new Skeleton({
      nodes,
      edges: mappedEdges,
      symmetries: mappedSymmetries,
      name: graph?.name ?? (entry.name as string | undefined),
    });
    skeletons.push(skeleton);
  }

  return skeletons;
}

/**
 * Parse tracks from tracks_json dataset values.
 */
export function parseTracks(values: unknown[]): Track[] {
  const tracks: Track[] = [];
  for (const entry of values) {
    let parsed = entry;
    if (typeof entry === "string") {
      try {
        parsed = JSON.parse(trimHdf5String(entry));
      } catch {
        parsed = trimHdf5String(entry);
      }
    } else if (entry instanceof Uint8Array) {
      try {
        parsed = JSON.parse(trimHdf5String(textDecoder.decode(entry)));
      } catch {
        parsed = trimHdf5String(textDecoder.decode(entry));
      }
    }

    if (Array.isArray(parsed)) {
      tracks.push(new Track(String(parsed[1] ?? parsed[0])));
    } else if (parsed && typeof parsed === "object" && "name" in parsed) {
      tracks.push(new Track(String((parsed as { name: unknown }).name)));
    } else {
      tracks.push(new Track(String(parsed)));
    }
  }
  return tracks;
}

/**
 * Video metadata extracted from videos_json without creating backends.
 */
export interface VideoMetadata {
  /**
   * Original filename, or "." for embedded. For image-sequence videos
   * (Python `ImageVideo`) this is the FULL ordered list of image paths — one
   * per frame — not just the first image.
   */
  filename: string | string[];
  /** HDF5 dataset path for embedded videos */
  dataset?: string;
  /** Video format (e.g., "mp4", "hdf5") */
  format?: string;
  /** Video width in pixels */
  width?: number;
  /** Video height in pixels */
  height?: number;
  /** Number of color channels */
  channels?: number;
  /** Frames per second */
  fps?: number;
  /** Total number of frames */
  frameCount?: number;
  /** Channel order (e.g., "RGB", "BGR") */
  channelOrder?: string;
  /**
   * Persisted "force grayscale" intent (Python `MediaVideo`/`Video.grayscale`):
   * `true`/`false` force 1/3 channels regardless of the underlying decoder's
   * native channel count; `undefined`/`null` means unset (autodetect, or N/A
   * for backends where it was never recorded). Must survive the streaming
   * reader same as the eager one (read.ts) — see resolveVideos.ts's
   * `probeAndAssignBackend`, which reads this back to decide whether to wrap
   * the real playback backend in `GrayscaleVideoBackend`. Losing it here means
   * a grayscale-imported MP4's lazily-opened backend reports the raw
   * decoder's native channel count (3) instead of the intended 1.
   */
  grayscale?: boolean | null;
  /** Whether video is embedded in the SLP file */
  embedded: boolean;
  /**
   * Full serialized `source_video` lineage dict (`{ filename?, backend?,
   * source_video? }`) when present in `videos_json` — reconstructed with its
   * recorded shape via {@link buildSourceVideoFromDict}. Not the authoritative
   * source for embedded videos (that lives in the `{group}/source_video` HDF5
   * group); see the readers.
   */
  sourceVideo?: Record<string, unknown>;
}

/**
 * Resolve the stored filename(s) for a video backend. Image-sequence videos
 * (Python `ImageVideo`) serialize the FULL ordered image list under
 * `backend.filenames` (plural) and only the first image under the singular
 * `backend.filename`. Reading the singular field collapses an N-image sequence
 * to a single frame (`createVideoBackend` builds a 1-image backend whose
 * `shape[0]` is `filename.length`), so prefer the list when present; otherwise
 * fall back to the singular `backend.filename`, then the top-level `filename`.
 *
 * Shared by the streaming reader ({@link parseVideosMetadata}) and the eager
 * reader (read.ts) so both surface the whole sequence identically.
 */
export function resolveVideoFilename(
  backendMeta: Record<string, unknown>,
  parsed: Record<string, unknown>,
): string | string[] {
  const filenames = backendMeta.filenames;
  if (Array.isArray(filenames) && filenames.length > 0) {
    return filenames as string[];
  }
  return (backendMeta.filename ?? parsed.filename ?? "") as string;
}

/**
 * Parse video metadata from videos_json dataset values.
 * Returns metadata objects WITHOUT creating video backends.
 */
export function parseVideosMetadata(
  values: unknown[],
  labelsPath?: string,
): VideoMetadata[] {
  const videos: VideoMetadata[] = [];

  for (const entry of values) {
    if (!entry) continue;

    let parsed: Record<string, unknown>;
    if (typeof entry === "string") {
      // jsfive returns fixed-width strings with trailing spaces/nulls - trim before parsing
      parsed = JSON.parse(trimHdf5String(entry)) as Record<string, unknown>;
    } else if (entry instanceof Uint8Array) {
      parsed = JSON.parse(trimHdf5String(textDecoder.decode(entry))) as Record<
        string,
        unknown
      >;
    } else {
      parsed = entry as Record<string, unknown>;
    }

    const backendMeta = (parsed.backend ?? {}) as Record<string, unknown>;
    let filename = resolveVideoFilename(backendMeta, parsed);
    const dataset = (backendMeta.dataset ?? null) as string | null;
    let embedded = false;

    if (filename === ".") {
      embedded = true;
      filename = labelsPath ?? "embedded";
    }

    const shape = backendMeta.shape as number[] | undefined;

    videos.push({
      filename,
      dataset: dataset ?? undefined,
      format: backendMeta.format as string | undefined,
      width: shape?.[2],
      height: shape?.[1],
      channels: shape?.[3],
      frameCount: shape?.[0],
      fps: backendMeta.fps as number | undefined,
      channelOrder: backendMeta.channel_order as string | undefined,
      grayscale: backendMeta.grayscale as boolean | null | undefined,
      embedded,
      sourceVideo: parsed.source_video as Record<string, unknown> | undefined,
    });
  }

  return videos;
}

/**
 * Suggestion frame metadata.
 */
export interface SuggestionMetadata {
  /** Video index */
  video: number;
  /** Frame index within the video */
  frameIdx: number;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Parse suggestions from suggestions_json dataset values.
 */
export function parseSuggestions(values: unknown[]): SuggestionMetadata[] {
  const suggestions: SuggestionMetadata[] = [];
  for (const entry of values) {
    const parsed = parseJsonEntry(entry) as Record<string, unknown>;
    suggestions.push({
      video: Number(parsed.video ?? 0),
      frameIdx: (parsed.frame_idx ?? parsed.frameIdx ?? 0) as number,
      metadata: parsed,
    });
  }
  return suggestions;
}

/**
 * The frame number of a frame group in a legacy `frame_group_dicts` entry. SLEAP's own
 * multi-view files do not store one, so it comes from the labeled frames its members
 * are on (`frameNumbers`: the `frame_idx` column of `/frames`), as Python's
 * `make_frame_group` works it out. Without this every group lands on frame 0.
 */
export function frameGroupFrameIdx(
  group: Record<string, unknown>,
  frameNumbers: ArrayLike<unknown> | undefined,
): number {
  const stored = group.frame_idx ?? group.frameIdx;
  if (stored !== undefined && stored !== null) return Number(stored);
  const instanceGroups = Array.isArray(group.instance_groups)
    ? group.instance_groups
    : [];
  for (const instanceGroup of instanceGroups) {
    const members = (instanceGroup as Record<string, unknown> | null)
      ?.camcorder_to_lf_and_inst_idx_map;
    for (const pair of Object.values(members ?? {})) {
      const frameIdx = frameNumbers?.[Number((pair as unknown[])[0])];
      if (frameIdx !== undefined) return Number(frameIdx);
    }
  }
  return 0;
}

/**
 * Camera metadata from recording sessions.
 */
export interface CameraMetadata {
  /** Camera name */
  name?: string;
  /** Rotation vector (Rodrigues) */
  rvec: number[];
  /** Translation vector */
  tvec: number[];
  /** 3x3 intrinsic camera matrix */
  matrix?: number[][];
  /** Lens distortion coefficients */
  distortions?: number[];
}

/**
 * Recording session metadata.
 */
export interface SessionMetadata {
  /** Camera definitions with calibration */
  cameras: CameraMetadata[];
  /** Mapping of camera name/key to video index */
  videosByCamera: Record<string, number>;
  /** Additional session metadata */
  metadata?: Record<string, unknown>;
  /**
   * Reconstructed frame groups with 3D data (SLP 2.8+), when the columnar
   * `/session_data` group was supplied. Each `InstanceGroup` carries its
   * `instance3d` (concrete 3D points), `score`, `identity`, `metadata`, and member
   * index refs; the 2D `instanceByCamera` stays unresolved in the lite path (no
   * frames are materialized). Absent for legacy ≤2.7 files or when `/session_data`
   * was not read.
   */
  frameGroups?: FrameGroup[];
}

/**
 * Parse recording session metadata from sessions_json dataset values.
 *
 * When `sessionData` (the columnar `/session_data` group, SLP 2.8+) and, per
 * session, an `fg_start`/`fg_end` range are present, each session's `frameGroups`
 * are reconstructed with full 3D via {@link reconstructColumnarFrameGroups} — used
 * by the lite (jsfive) reader to surface 3D without materializing frames. Legacy
 * ≤2.7 files (no `fg_start`) leave `frameGroups` unset.
 */
export function parseSessionsMetadata(
  values: unknown[],
  sessionData?: SessionData | null,
  skeletons: Skeleton[] = [],
): SessionMetadata[] {
  const sessions: SessionMetadata[] = [];

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
    const cameras: CameraMetadata[] = [];
    const cameraObjs: Camera[] = [];

    for (const [key, data] of Object.entries(calibration)) {
      if (key === "metadata") continue;
      const cameraData = data as Record<string, unknown>;
      const rvec = (cameraData.rotation as number[] | undefined) ?? [0, 0, 0];
      const tvec = (cameraData.translation as number[] | undefined) ?? [
        0, 0, 0,
      ];
      const name = (cameraData.name as string | undefined) ?? key;
      const matrix = cameraData.matrix as number[][] | undefined;
      const distortions = cameraData.distortions as number[] | undefined;
      cameras.push({ name, rvec, tvec, matrix, distortions });
      cameraObjs.push(
        new Camera({
          name,
          rvec,
          tvec,
          matrix,
          distortions,
          size: normalizeCameraSize(cameraData.size),
        }),
      );
    }

    const videosByCamera: Record<string, number> = {};
    const map = (parsed.camcorder_to_video_idx_map ?? {}) as Record<
      string,
      unknown
    >;
    for (const [cameraKey, videoIdx] of Object.entries(map)) {
      videosByCamera[cameraKey] = Number(videoIdx);
    }

    // SLP 2.8 columnar path: reconstruct frame groups + 3D from /session_data.
    let frameGroups: FrameGroup[] | undefined;
    const fgStart = parsed.fg_start;
    if (fgStart != null && sessionData) {
      frameGroups = [
        ...reconstructColumnarFrameGroups(
          cameraObjs,
          skeletons,
          undefined,
          sessionData,
          Number(fgStart),
          Number(parsed.fg_end),
        ).values(),
      ];
    }

    sessions.push({
      cameras,
      videosByCamera,
      metadata: parsed.metadata as Record<string, unknown> | undefined,
      ...(frameGroups ? { frameGroups } : {}),
    });
  }

  return sessions;
}

/**
 * Normalize a camera `size` value read from calibration to a `[w, h]` tuple or
 * `undefined`. Python writes `""` for a size-less camera and `[w, h]` otherwise
 * (and older sleap-io.js omitted the key), so anything that is not a 2-element
 * array becomes `undefined`.
 */
export function normalizeCameraSize(
  raw: unknown,
): [number, number] | undefined {
  return Array.isArray(raw) && raw.length === 2
    ? [Number(raw[0]), Number(raw[1])]
    : undefined;
}

/**
 * Resolve a camera key from a map lookup, falling back to numeric index.
 * Camera keys may be camera names (JS format) or numeric indices (Python format).
 */
export function resolveCameraKey(
  cameraKey: string,
  cameraMap: Map<string, Camera>,
  cameras: Camera[],
): Camera | undefined {
  let camera = cameraMap.get(cameraKey);
  if (!camera) {
    const idx = Number(cameraKey);
    if (!isNaN(idx) && idx >= 0 && idx < cameras.length) {
      camera = cameras[idx];
    }
  }
  return camera;
}

/**
 * Reconstruct an Instance3D or PredictedInstance3D from a session record.
 */
/**
 * Map inline 3D `null` entries (missing keypoints, produced by `JSON.stringify(NaN)`
 * in ≤2.7 files and by sleap-io.js's own legacy writer) to `NaN` — a `null` row
 * becomes a full-NaN row, a `null` coord becomes `NaN`. Matches Python's
 * `_inline_3d_to_array` and `Instance3D`'s "NaN = missing keypoint" semantics.
 * Without this, `Number(null) === 0` would move a missing keypoint to the origin and
 * ragged null rows would break downstream shape assumptions (luc3d#161).
 */
function coerceInline3dPoints(raw: unknown[]): number[][] {
  const width =
    (raw.find((r) => r != null) as unknown[] | undefined)?.length ?? 3;
  return raw.map((row) =>
    row == null
      ? new Array<number>(width).fill(Number.NaN)
      : (row as unknown[]).map((v) => (v == null ? Number.NaN : Number(v))),
  );
}

/** Map a flat inline per-point score list's `null` entries to `NaN`. */
function coerceInline1d(raw: unknown[]): number[] {
  return raw.map((v) => (v == null ? Number.NaN : Number(v)));
}

export function reconstructInstance3D(
  record: Record<string, unknown>,
  skeletons: Skeleton[],
): Instance3D | undefined {
  const rawPoints = record.points;
  if (!Array.isArray(rawPoints)) return undefined;
  // Coerce legacy `null` keypoints/scores to NaN (luc3d#161).
  const pointsValue = coerceInline3dPoints(rawPoints as unknown[]);

  const skeleton = skeletons[0] ?? new Skeleton({ nodes: [] });
  const score = record.instance_3d_score as number | undefined;
  const rawScores = record.instance_3d_point_scores;
  const pointScores = Array.isArray(rawScores)
    ? coerceInline1d(rawScores as unknown[])
    : undefined;

  if (pointScores) {
    return new PredictedInstance3D({
      points: pointsValue,
      skeleton,
      score,
      pointScores,
    });
  }
  return new Instance3D({ points: pointsValue, skeleton, score });
}

/** A columnar `/session_data` point matrix read into memory: a flat float buffer +
 * its column count (row `r` = `flat[r*ncols .. r*ncols+ncols]`). */
export interface SessionPointMatrix {
  flat: ArrayLike<number>;
  ncols: number;
}

/**
 * The columnar `/session_data` group (SLP 2.8) read into memory as plain column
 * arrays, backend-agnostic so the eager (h5wasm), streaming (worker), and lite
 * (jsfive) readers can all feed {@link reconstructColumnarFrameGroups}. Struct-table
 * columns are `ArrayLike<unknown>` (numbers, or BigInt for a Python-written i8/u8
 * compound — always `Number()`-coerced on use). Meta arrays are one JSON blob per
 * row (empty string when that row had no metadata), or `null` when the
 * presence-guarded dataset was omitted.
 */
export interface SessionData {
  frameGroups: Record<string, ArrayLike<unknown>>; // frame_idx, ig_start, ig_end
  instanceGroups: Record<string, ArrayLike<unknown>>; // SESSION_INSTANCE_GROUP_FIELDS
  members: Record<string, ArrayLike<unknown>>; // camera, lf, inst
  points3d: SessionPointMatrix | null;
  predPoints3d: SessionPointMatrix | null;
  frameGroupMeta: unknown[] | null;
  instanceGroupMeta: unknown[] | null;
}

/** Decode a per-row JSON metadata blob from a `*_meta` vlen-string array. Returns
 * `{}` when the (presence-guarded) dataset was absent or the row was blank. */
function decodeMetaBlob(
  arr: unknown[] | null,
  idx: number,
): Record<string, unknown> {
  if (!arr || idx >= arr.length) return {};
  const text = decodeEntryText(arr[idx]);
  if (!text) return {};
  return JSON.parse(text) as Record<string, unknown>;
}

/**
 * Reconstruct a session's `FrameGroup`s from the columnar `/session_data` tables
 * (SLP 2.8), using its `fg_start`/`fg_end` half-open range into `frame_groups`.
 *
 * Backend-agnostic: operates purely on an in-memory {@link SessionData} + the
 * session's resolved `cameras`, so the eager, streaming, and lite readers share it.
 * Members are captured as index refs (`camera → [lf, inst]`) WITHOUT materializing
 * frames — resolved lazily later via the injected frame resolver, exactly like the
 * legacy inline reader. `Number()`-coerces every struct field so a Python-written
 * BigInt (i8/u8) compound and a JS-written `<f8` flat matrix both work.
 */
export function reconstructColumnarFrameGroups(
  cameras: Camera[],
  skeletons: Skeleton[],
  identities: Identity[] | undefined,
  sessionData: SessionData,
  fgStart: number,
  fgEnd: number,
): Map<number, FrameGroup> {
  const result = new Map<number, FrameGroup>();
  const fg = sessionData.frameGroups;
  const ig = sessionData.instanceGroups;
  const mem = sessionData.members;
  const num = (col: ArrayLike<unknown> | undefined, i: number): number =>
    Number((col as ArrayLike<unknown>)?.[i]);

  for (let f = fgStart; f < fgEnd; f++) {
    const frameIdx = num(fg.frame_idx, f);
    const igStart = num(fg.ig_start, f);
    const igEnd = num(fg.ig_end, f);

    const instanceGroups: InstanceGroup[] = [];
    let labeledFrameRefsByCamera: Map<Camera, number> | undefined;

    for (let g = igStart; g < igEnd; g++) {
      // Members → camera → [lf, inst] refs (+ the frame group's camera → lf refs).
      const memberStart = num(ig.member_start, g);
      const memberEnd = num(ig.member_end, g);
      let instanceRefsByCamera: Map<Camera, [number, number]> | undefined;
      for (let m = memberStart; m < memberEnd; m++) {
        const camera = cameras[num(mem.camera, m)];
        if (!camera) continue;
        const lf = num(mem.lf, m);
        const inst = num(mem.inst, m);
        if (!instanceRefsByCamera) instanceRefsByCamera = new Map();
        instanceRefsByCamera.set(camera, [lf, inst]);
        if (!labeledFrameRefsByCamera) labeledFrameRefsByCamera = new Map();
        labeledFrameRefsByCamera.set(camera, lf);
      }

      // 3D points from the points_3d / pred_points_3d row range (NaN = missing).
      let instance3d: Instance3D | undefined;
      const pts3dStart = num(ig.pts3d_start, g);
      if (pts3dStart >= 0) {
        const pts3dEnd = num(ig.pts3d_end, g);
        const predicted = num(ig.pts3d_predicted, g) === 1;
        const src = predicted ? sessionData.predPoints3d : sessionData.points3d;
        if (src) {
          const skeleton = skeletons[0] ?? new Skeleton({ nodes: [] });
          const i3dScore = num(ig.instance_3d_score, g);
          const score = Number.isNaN(i3dScore) ? undefined : i3dScore;
          const points: number[][] = [];
          const pointScores: number[] = [];
          const { flat, ncols } = src;
          for (let r = pts3dStart; r < pts3dEnd; r++) {
            const base = r * ncols;
            points.push([
              Number(flat[base]),
              Number(flat[base + 1]),
              Number(flat[base + 2]),
            ]);
            if (predicted) pointScores.push(Number(flat[base + 3]));
          }
          instance3d = predicted
            ? new PredictedInstance3D({ points, skeleton, score, pointScores })
            : new Instance3D({ points, skeleton, score });
        }
      }

      const scoreRaw = num(ig.score, g);
      const score = Number.isNaN(scoreRaw) ? undefined : scoreRaw;

      let identity: Identity | undefined;
      const identityIdx = num(ig.identity_idx, g);
      if (identityIdx >= 0 && identities) {
        if (identityIdx < identities.length) identity = identities[identityIdx];
        else
          console.warn(
            `identity_idx ${identityIdx} is out of bounds (${identities.length} identities available) — skipping identity for this instance group.`,
          );
      }

      instanceGroups.push(
        new InstanceGroup({
          instanceRefsByCamera,
          score,
          instance3d,
          identity,
          metadata: decodeMetaBlob(sessionData.instanceGroupMeta, g),
        }),
      );
    }

    result.set(
      frameIdx,
      new FrameGroup({
        frameIdx,
        instanceGroups,
        labeledFrameRefsByCamera,
        metadata: decodeMetaBlob(sessionData.frameGroupMeta, f),
      }),
    );
  }
  return result;
}

/**
 * Resolve an Identity from an identity_idx field in a session record.
 */
export function resolveIdentity(
  record: Record<string, unknown>,
  identities?: Identity[],
): Identity | undefined {
  const identityIdx = record.identity_idx;
  if (identityIdx == null || !identities) return undefined;

  const idx = Number(identityIdx);
  if (idx >= 0 && idx < identities.length) {
    return identities[idx];
  }
  console.warn(
    `identity_idx ${idx} is out of bounds (${identities.length} identities available) — skipping identity for this instance group.`,
  );
  return undefined;
}
