/**
 * SLP format 3.0: every table is a group with one 1-D dataset per column.
 *
 * The rest of the library works with the columns the 2.x tables have. This module
 * turns a 3.0 table into those columns, so the frame builders, the lazy store and
 * the writers don't need to know which layout a file used:
 *
 * - `/points` and `/pred_points`: `verified` is named `complete`, and a point the
 *   source didn't label reads as a hidden point with no location.
 * - `/instances` and `/frames`: the row number is the id, so the `instance_id` and
 *   `frame_id` columns are filled in.
 * - Integer columns come back as numbers, never BigInt.
 *
 * Each table is read in its own layout, whatever the file's `format_id` says, as in
 * Python's `_read_columns`. The tables and their defaults mirror `_TABLE_SPECS` and
 * `_read_pose_tables` in `sleap_io/io/slp.py`, and so do the range checks: a 3.0
 * table that fails one is refused, while a 2.x table is read as before.
 *
 * This module has no dependency on any HDF5 library, so the eager, streaming and lite
 * readers all use it.
 */

/** A table's columns by name: typed arrays, or plain arrays for strings. */
export type Columns = Record<string, ArrayLike<unknown>>;

/** One table as read from a file, before {@link normalizeColumnarTable}. */
export interface RawTable {
  columns: Columns;
  /** `true` for a group of columns (3.0), `false` for a 2.x dataset. */
  columnar: boolean;
  /** Members of a group that aren't columns, such as `metadata`. */
  otherMembers?: string[];
}

interface TableSpec {
  /** Columns a table must have. */
  required: readonly string[];
  /** Columns a table may leave out, with the value they then read as. */
  optional: Readonly<Record<string, number | string>>;
  /** Flag columns, read as 0 or 1 since any numeric dtype may hold a flag. */
  flags?: readonly string[];
  /** Members that the readers handle themselves (byte buffers and sub-tables). */
  members?: readonly string[];
  /** Where the 2.x layout keeps this table, if not at the same path. */
  legacyPath?: string;
  /** The byte buffer the rows index into: inside the group, then the 2.x path. */
  buffer?: { name: string; legacyPath: string };
}

const SPATIAL = { scale_x: 1, scale_y: 1, offset_x: 0, offset_y: 0 };
const SCORE_MAP_COLUMNS = [
  "data_start",
  "data_end",
  "height",
  "width",
] as const;

const TABLE_SPECS = {
  points: {
    required: ["x", "y"],
    optional: { visible: 0, verified: 0, labeled: 1 },
  },
  pred_points: {
    required: ["x", "y"],
    optional: { visible: 0, verified: 0, labeled: 1, score: Number.NaN },
  },
  instances: {
    required: [
      "instance_type",
      "frame_id",
      "skeleton",
      "point_id_start",
      "point_id_end",
    ],
    optional: {
      track: -1,
      from_predicted: -1,
      score: Number.NaN,
      tracking_score: Number.NaN,
    },
  },
  frames: {
    required: ["video", "frame_idx", "instance_id_start", "instance_id_end"],
    optional: { is_negative: 0 },
  },
  bboxes: {
    required: [
      "x1",
      "y1",
      "x2",
      "y2",
      "video",
      "frame_idx",
      "track",
      "instance",
      "is_predicted",
    ],
    optional: {
      angle: 0,
      score: Number.NaN,
      tracking_score: Number.NaN,
      category: "",
      name: "",
      source: "",
    },
    flags: ["is_predicted"],
  },
  rois: {
    required: ["video", "frame_idx", "track", "wkb_start", "wkb_end"],
    optional: {
      annotation_type: 0,
      instance: -1,
      is_predicted: 0,
      score: Number.NaN,
      tracking_score: Number.NaN,
      name: "",
      category: "",
      source: "",
    },
    flags: ["is_predicted"],
    members: ["wkb"],
    buffer: { name: "wkb", legacyPath: "roi_wkb" },
  },
  masks: {
    required: [
      "height",
      "width",
      "video",
      "frame_idx",
      "track",
      "rle_start",
      "rle_end",
    ],
    optional: {
      annotation_type: 2,
      instance: -1,
      from_predicted: -1,
      is_predicted: 0,
      score: Number.NaN,
      tracking_score: Number.NaN,
      ...SPATIAL,
      name: "",
      category: "",
      source: "",
    },
    flags: ["is_predicted"],
    members: ["rle", "score_maps"],
    buffer: { name: "rle", legacyPath: "mask_rle" },
  },
  "masks/score_maps": {
    required: ["mask_idx", ...SCORE_MAP_COLUMNS],
    optional: SPATIAL,
    members: ["data"],
    legacyPath: "mask_score_map_index",
    buffer: { name: "data", legacyPath: "mask_score_maps" },
  },
  label_images: {
    required: [
      "video",
      "frame_idx",
      "height",
      "width",
      "n_objects",
      "objects_start",
      "data_start",
      "data_end",
    ],
    optional: { is_predicted: 0, score: Number.NaN, ...SPATIAL, source: "" },
    flags: ["is_predicted"],
    members: ["data", "objects", "score_maps"],
    buffer: { name: "data", legacyPath: "label_image_data" },
  },
  "label_images/objects": {
    required: ["label_id", "track", "instance"],
    optional: {
      score: Number.NaN,
      tracking_score: Number.NaN,
      name: "",
      category: "",
    },
    legacyPath: "label_image_objects",
  },
  "label_images/score_maps": {
    required: ["li_idx", ...SCORE_MAP_COLUMNS],
    optional: SPATIAL,
    members: ["data"],
    legacyPath: "label_image_score_map_index",
    buffer: { name: "data", legacyPath: "label_image_score_maps" },
  },
  "identity/links": {
    required: ["owner_type", "owner_id", "identity_idx", "identity_score"],
    optional: {},
  },
  "categories/links": {
    required: ["owner_type", "owner_id", "category_idx", "category_score"],
    optional: {},
  },
  "session_data/frame_groups": {
    required: ["frame_idx", "ig_start", "ig_end"],
    optional: {},
  },
  "session_data/instance_groups": {
    required: [
      "identity_idx",
      "score",
      "instance_3d_score",
      "pts3d_start",
      "pts3d_end",
      "pts3d_predicted",
      "member_start",
      "member_end",
    ],
    optional: {},
    flags: ["pts3d_predicted"],
  },
  "session_data/instance_group_members": {
    required: ["camera", "lf", "inst"],
    optional: {},
  },
} satisfies Record<string, TableSpec>;

export type ColumnarTableName = keyof typeof TABLE_SPECS;

const specOf = (table: ColumnarTableName): TableSpec => TABLE_SPECS[table];

/** Path of a table in the 2.x layout, which is mostly the same as in 3.0. */
export function legacyTablePath(table: ColumnarTableName): string {
  return specOf(table).legacyPath ?? table;
}

/** Path of the byte buffer a table's rows index into, in the layout it was read in. */
export function tableBufferPath(
  table: ColumnarTableName,
  columnar: boolean,
): string | null {
  const buffer = specOf(table).buffer;
  if (!buffer) return null;
  return columnar ? `${table}/${buffer.name}` : buffer.legacyPath;
}

/**
 * Names in a table's group that are not columns, so the readers don't load them with
 * the rows: its byte buffer and its sub-tables.
 */
export function tableMembers(table: ColumnarTableName): readonly string[] {
  return specOf(table).members ?? [];
}

const textDecoder = new TextDecoder();

function columnString(value: unknown): string {
  const text =
    typeof value === "string"
      ? value
      : value instanceof Uint8Array
        ? textDecoder.decode(value)
        : String(value ?? "");
  // Python may store strings as fixed-length, which come back padded.
  return text.replace(/\0+$/, "");
}

/**
 * A column with 64-bit integers as doubles, which are exact below 2^53: every id and
 * index in a table is. Values go through two 32-bit views because reading a
 * `BigInt64Array` allocates one BigInt per row. Other columns are returned as they are.
 */
export function numberColumn(values: ArrayLike<unknown>): ArrayLike<unknown> {
  const signed = values instanceof BigInt64Array;
  if (!signed && !(values instanceof BigUint64Array)) return values;
  const big = values as BigInt64Array | BigUint64Array;
  const lo = new Uint32Array(big.buffer, big.byteOffset, big.length * 2);
  const hi = signed
    ? new Int32Array(big.buffer, big.byteOffset, lo.length)
    : lo;
  const out = new Float64Array(big.length);
  for (let i = 0; i < big.length; i += 1) {
    out[i] = hi[2 * i + 1] * 4294967296 + lo[2 * i];
  }
  return out;
}

/** A column for `n` rows that a table left out. */
function defaultColumn(value: number | string, n: number): ArrayLike<unknown> {
  return typeof value === "string"
    ? new Array<string>(n).fill(value)
    : new Float64Array(n).fill(value);
}

/**
 * The columns of one 3.0 table that this library reads, with the missing optional
 * ones filled in. Mirrors Python's `_rows_from_columns`.
 *
 * @throws {Error} If a required column is missing, or the columns don't all have the
 *   same length. A table this broken can't be read without guessing.
 */
export function normalizeColumnarTable(
  labelsPath: string,
  table: ColumnarTableName,
  raw: Pick<RawTable, "columns" | "otherMembers">,
): Columns {
  const spec = specOf(table);
  const columns = raw.columns;
  const missing = spec.required.filter((name) => !(name in columns));
  if (missing.length > 0) {
    throw new Error(
      `${labelsPath}: /${table} is missing column(s) ${missing.join(", ")}.`,
    );
  }
  const lengths = Object.entries(columns).map(
    ([name, col]) => [name, col.length] as const,
  );
  if (new Set(lengths.map(([, n]) => n)).size > 1) {
    const detail = lengths.map(([name, n]) => `${name}: ${n}`).join(", ");
    throw new Error(
      `${labelsPath}: /${table} columns have different lengths (${detail}).`,
    );
  }
  const n = lengths[0]?.[1] ?? 0;

  const out: Columns = {};
  const flags = spec.flags ?? [];
  const strings = Object.keys(spec.optional).filter(
    (name) => typeof spec.optional[name] === "string",
  );
  for (const name of [...spec.required, ...Object.keys(spec.optional)]) {
    const col = columns[name];
    if (col === undefined) {
      out[name] = defaultColumn(spec.optional[name], n);
    } else if (strings.includes(name)) {
      out[name] = Array.from(col, columnString);
    } else if (flags.includes(name)) {
      out[name] = Uint8Array.from(col as ArrayLike<number>, (v) => (v ? 1 : 0));
    } else {
      out[name] = numberColumn(col);
    }
  }

  warnUnreadMembers(labelsPath, table, [
    ...Object.keys(columns).filter((name) => !(name in out)),
    ...(raw.otherMembers ?? []).filter(
      (name) => !(spec.members ?? []).includes(name),
    ),
  ]);
  return out;
}

/**
 * Tell the user what a save would lose. A newer writer may add columns, and Python
 * keeps per-record `metadata/`, but this library has nowhere to put either.
 */
function warnUnreadMembers(
  labelsPath: string,
  table: ColumnarTableName,
  names: string[],
): void {
  if (names.length === 0) return;
  console.warn(
    `[sleap-io] ${labelsPath}: /${table} holds ${names.join(", ")}, which this ` +
      "version of sleap-io.js does not read. Saving this project will not keep them.",
  );
}

/**
 * A 3.0 point table, as the 2.x columns the rest of the library reads. A point that
 * isn't labeled has no location: the 2.x model can't tell it from a hidden point, so
 * that is what it becomes, exactly as Python's `legacy_format=True` writes it.
 */
function normalizePointTable(
  labelsPath: string,
  table: "points" | "pred_points",
  raw: RawTable,
): Columns {
  const out = normalizeColumnarTable(labelsPath, table, raw);
  const n = out.x.length;
  const x = floatColumn(out.x);
  const y = floatColumn(out.y);
  const visible = out.visible as ArrayLike<number>;
  const labeled = out.labeled as ArrayLike<number>;
  let unlabeled = 0;
  for (let i = 0; i < n; i += 1) {
    if (!labeled[i] && !visible[i]) {
      x[i] = Number.NaN;
      y[i] = Number.NaN;
      unlabeled += 1;
    }
  }
  if (unlabeled > 0) {
    console.warn(
      `[sleap-io] ${labelsPath}: /${table} has ${unlabeled} unlabeled point(s). ` +
        "They read as hidden points with no location, and saving this project " +
        "stores them that way.",
    );
  }
  const columns: Columns = {
    x,
    y,
    visible,
    complete: out.verified,
  };
  if (table === "pred_points") columns.score = out.score;
  return columns;
}

/** A coordinate column that can hold NaN, whatever numeric dtype it was stored in. */
function floatColumn(col: ArrayLike<unknown>): Float64Array | Float32Array {
  if (col instanceof Float64Array || col instanceof Float32Array) return col;
  return Float64Array.from(col as ArrayLike<number>);
}

/** The column of row numbers that stands in for a stored id column. */
function rowIds(n: number): Float64Array {
  const ids = new Float64Array(n);
  for (let i = 0; i < n; i += 1) ids[i] = i;
  return ids;
}

/** The four pose tables, as the 2.x columns the rest of the library reads. */
export interface PoseTables {
  framesData: Record<string, any[]>;
  instancesData: Record<string, any[]>;
  pointsData: Record<string, any[]>;
  predPointsData: Record<string, any[]>;
  /**
   * Raw `"video:frame"` keys of the negative frames of a 3.0 `/frames`, or `null` for
   * a 2.x one, which keeps them in `/negative_frames` instead.
   */
  negativeFrames: Set<string> | null;
}

/**
 * Normalize and check `/points`, `/pred_points`, `/instances` and `/frames`. Mirrors
 * Python's `_read_pose_tables`: each table goes by its own layout, and only tables
 * that are all 3.0 are held to the range checks.
 */
export function normalizePoseTables(
  labelsPath: string,
  nVideos: number,
  raw: {
    frames: RawTable;
    instances: RawTable;
    points: RawTable;
    predPoints: RawTable;
  },
): PoseTables {
  const points = raw.points.columnar
    ? normalizePointTable(labelsPath, "points", raw.points)
    : raw.points.columns;
  const predPoints = raw.predPoints.columnar
    ? normalizePointTable(labelsPath, "pred_points", raw.predPoints)
    : raw.predPoints.columns;

  let instances = raw.instances.columns;
  if (raw.instances.columnar) {
    instances = normalizeColumnarTable(labelsPath, "instances", raw.instances);
    instances.instance_id = rowIds(instances.instance_type.length);
  }

  let frames = raw.frames.columns;
  let negativeFrames: Set<string> | null = null;
  if (raw.frames.columnar) {
    frames = normalizeColumnarTable(labelsPath, "frames", raw.frames);
    frames.frame_id = rowIds(frames.video.length);
    negativeFrames = new Set();
    const isNegative = frames.is_negative as ArrayLike<number>;
    for (let i = 0; i < isNegative.length; i += 1) {
      if (isNegative[i]) {
        negativeFrames.add(`${frames.video[i]}:${frames.frame_idx[i]}`);
      }
    }
  }

  if (raw.instances.columnar && raw.frames.columnar) {
    reportTableProblems(
      labelsPath,
      poseTableProblems(instances, frames, {
        nPoints: (points.x ?? []).length,
        nPredPoints: (predPoints.x ?? []).length,
        nVideos,
      }),
    );
  }

  return {
    framesData: frames as Record<string, any[]>,
    instancesData: instances as Record<string, any[]>,
    pointsData: points as Record<string, any[]>,
    predPointsData: predPoints as Record<string, any[]>,
    negativeFrames,
  };
}

const INSTANCE_USER = 0;
const INSTANCE_PREDICTED = 1;

/** Python's `_check_pose_tables`, for tables that are all 3.0. */
function poseTableProblems(
  instances: Columns,
  frames: Columns,
  counts: { nPoints: number; nPredPoints: number; nVideos: number },
): string[] {
  const problems: string[] = [];
  const start = frames.instance_id_start as ArrayLike<number>;
  const end = frames.instance_id_end as ArrayLike<number>;
  const nFrames = start.length;
  const nInstances = instances.instance_type.length;

  let rangesOk = true;
  for (let i = 0; i < nFrames && rangesOk; i += 1) {
    if (end[i] < start[i] || start[i] < 0 || (i > 0 && start[i] < end[i - 1])) {
      rangesOk = false;
    }
  }
  if (!rangesOk) problems.push("/frames instance ranges decrease or overlap");
  const last = nFrames > 0 ? end[nFrames - 1] : 0;
  if (last !== nInstances) {
    problems.push(
      `/frames instance ranges end at ${last}, not at the instance count (${nInstances})`,
    );
  }
  if (rangesOk) {
    const frameId = instances.frame_id as ArrayLike<number>;
    let frame = 0;
    for (let row = 0; row < nInstances; row += 1) {
      while (frame < nFrames && end[frame] <= row) frame += 1;
      const owner = frame < nFrames && start[frame] <= row ? frame : -1;
      if (frameId[row] !== owner) {
        problems.push("/instances frame_id disagrees with the /frames ranges");
        break;
      }
    }
  }

  const types = instances.instance_type as ArrayLike<number>;
  const pointStart = instances.point_id_start as ArrayLike<number>;
  const pointEnd = instances.point_id_end as ArrayLike<number>;
  for (const [type, n, table] of [
    [INSTANCE_USER, counts.nPoints, "points"],
    [INSTANCE_PREDICTED, counts.nPredPoints, "pred_points"],
  ] as const) {
    for (let i = 0; i < nInstances; i += 1) {
      if (
        types[i] === type &&
        (pointStart[i] < 0 || pointEnd[i] < pointStart[i] || pointEnd[i] > n)
      ) {
        problems.push(
          `/instances point ranges fall outside /${table} (${n} rows)`,
        );
        break;
      }
    }
  }
  if (indexOutOfRange(instances.from_predicted, nInstances)) {
    problems.push("/instances from_predicted is not an instance row");
  }
  if (indexOutOfRange(frames.video, counts.nVideos, false)) {
    problems.push(`/frames video is not one of the ${counts.nVideos} videos`);
  }
  return problems;
}

/** Whether a column holds anything but a row of a table with `n` rows (or -1). */
export function indexOutOfRange(
  values: ArrayLike<unknown>,
  n: number,
  optional = true,
): boolean {
  const low = optional ? -1 : 0;
  const col = values as ArrayLike<number>;
  for (let i = 0; i < col.length; i += 1) {
    if (col[i] < low || col[i] >= n) return true;
  }
  return false;
}

/** Whether any `[start, end)` range isn't a range inside `[0, n)`. */
export function spanOutOfRange(
  start: ArrayLike<unknown>,
  end: ArrayLike<unknown>,
  n: number,
): boolean {
  const s = start as ArrayLike<number>;
  const e = end as ArrayLike<number>;
  for (let i = 0; i < s.length; i += 1) {
    if (s[i] < 0 || e[i] < s[i] || e[i] > n) return true;
  }
  return false;
}

/**
 * The links from an annotation table to videos, tracks and instances. `nInstances` is
 * `null` when the count isn't known here, which skips that check.
 */
export function annotationProblems(
  table: string,
  rows: Columns,
  counts: { nVideos: number; nTracks: number; nInstances: number | null },
): string[] {
  const problems: string[] = [];
  if (rows.video && indexOutOfRange(rows.video, counts.nVideos)) {
    problems.push(`/${table} video is not one of the ${counts.nVideos} videos`);
  }
  if (indexOutOfRange(rows.track, counts.nTracks)) {
    problems.push(`/${table} track is not one of the ${counts.nTracks} tracks`);
  }
  if (
    counts.nInstances !== null &&
    indexOutOfRange(rows.instance, counts.nInstances)
  ) {
    problems.push(
      `/${table} instance is not one of the ${counts.nInstances} instances`,
    );
  }
  return problems;
}

/** The ranges and links of the `/session_data` tables (Python's `_session_problems`). */
export function sessionProblems(
  sessionData: {
    frameGroups: Columns;
    instanceGroups: Columns;
    members: Columns;
    points3d: { flat: ArrayLike<number>; ncols: number } | null;
    predPoints3d: { flat: ArrayLike<number>; ncols: number } | null;
  },
  counts: { nIdentities: number; nFrames: number | null },
): string[] {
  const { frameGroups, instanceGroups, members } = sessionData;
  const nGroups = instanceGroups.member_start.length;
  const nMembers = members.camera.length;
  const problems: string[] = [];
  if (spanOutOfRange(frameGroups.ig_start, frameGroups.ig_end, nGroups)) {
    problems.push(
      `/session_data/frame_groups ranges fall outside instance_groups (${nGroups} rows)`,
    );
  }
  if (
    spanOutOfRange(
      instanceGroups.member_start,
      instanceGroups.member_end,
      nMembers,
    )
  ) {
    problems.push(
      `/session_data/instance_groups ranges fall outside instance_group_members (${nMembers} rows)`,
    );
  }
  const start = instanceGroups.pts3d_start as ArrayLike<number>;
  const end = instanceGroups.pts3d_end as ArrayLike<number>;
  const predicted = instanceGroups.pts3d_predicted as ArrayLike<number>;
  for (const [isPredicted, name, matrix] of [
    [0, "points_3d", sessionData.points3d],
    [1, "pred_points_3d", sessionData.predPoints3d],
  ] as const) {
    const n = matrix ? matrix.flat.length / matrix.ncols : 0;
    for (let i = 0; i < nGroups; i += 1) {
      if (
        start[i] >= 0 &&
        predicted[i] === isPredicted &&
        (end[i] < start[i] || end[i] > n)
      ) {
        problems.push(
          `/session_data/instance_groups 3D ranges fall outside ${name} (${n} rows)`,
        );
        break;
      }
    }
  }
  if (indexOutOfRange(instanceGroups.identity_idx, counts.nIdentities)) {
    problems.push(
      `/session_data/instance_groups identity_idx is not one of the ${counts.nIdentities} identities`,
    );
  }
  if (
    counts.nFrames !== null &&
    indexOutOfRange(members.lf, counts.nFrames, false)
  ) {
    problems.push(
      `/session_data/instance_group_members lf is not one of the ${counts.nFrames} frames`,
    );
  }
  const camera = members.camera as ArrayLike<number>;
  const inst = members.inst as ArrayLike<number>;
  for (let i = 0; i < nMembers; i += 1) {
    if (camera[i] < 0 || inst[i] < 0) {
      problems.push(
        "/session_data/instance_group_members has negative indices",
      );
      break;
    }
  }
  return problems;
}

/**
 * Refuse tables that contradict each other. A 3.0 file is held to this because reading
 * on would silently attach points to the wrong animal or drop them.
 *
 * @throws {Error} If there are any problems, naming the tables.
 */
export function reportTableProblems(
  labelsPath: string,
  problems: string[],
): void {
  if (problems.length === 0) return;
  throw new Error(
    `${labelsPath} has inconsistent tables: ${problems.join("; ")}.`,
  );
}

/** HDF5's own filters (deflate, shuffle, fletcher32, szip, nbit, scaleoffset) stop here. */
const LAST_BUILTIN_FILTER_ID = 6;

/**
 * Refuse a dataset compressed with a filter this library's HDF5 build doesn't have,
 * such as `compression="lzf"` from Python. h5wasm logs the failure but still hands
 * back the buffer it allocated, so without this the file reads as garbage numbers
 * with no error.
 *
 * @throws {Error} If the dataset uses such a filter.
 */
export function checkDatasetFilters(dataset: unknown, path: string): void {
  let filters: unknown;
  try {
    filters = (dataset as { filters?: unknown }).filters;
  } catch {
    return; // an h5wasm that can't list filters; nothing to go on
  }
  if (!Array.isArray(filters)) return;
  for (const filter of filters as Array<{ id: number; name?: string }>) {
    if (filter.id > LAST_BUILTIN_FILTER_ID) {
      throw new Error(
        `/${path} is compressed with "${filter.name ?? filter.id}", which ` +
          "sleap-io.js cannot read. Re-save the file from Python sleap-io with " +
          'compression="gzip" or without compression.',
      );
    }
  }
}

/** Whether an h5wasm node is a group rather than a dataset. */
export function isH5Group(node: unknown): boolean {
  const item = node as { keys?: unknown; shape?: unknown } | null | undefined;
  return !!item && typeof item.keys === "function" && item.shape === undefined;
}

/**
 * Read the columns of a group through h5wasm: every 1-D dataset in it. The other
 * members are named so that a save can warn about them, unless `members` says the
 * caller handles them itself.
 */
export function readH5GroupColumns(
  group: any,
  path: string,
  members: readonly string[] = [],
): Pick<RawTable, "columns" | "otherMembers"> {
  const columns: Columns = {};
  const otherMembers: string[] = [];
  for (const name of group.keys() as string[]) {
    if (members.includes(name)) continue;
    const child = group.get(name);
    if (child?.shape?.length !== 1) {
      otherMembers.push(name);
      continue;
    }
    checkDatasetFilters(child, `${path}/${name}`);
    columns[name] = numberColumn(child.value);
  }
  return { columns, otherMembers };
}
