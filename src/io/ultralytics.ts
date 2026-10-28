/**
 * Ultralytics YOLO format I/O (detection + segmentation + pose).
 *
 * This is a TypeScript port of `sleap_io/io/ultralytics.py` (Python sleap-io
 * v0.7.x, PR #395), adapted to the JS/Node data model and runtime.
 *
 * Ultralytics YOLO format specification:
 * - Directory structure: `dataset_root/<split>/images/` and
 *   `dataset_root/<split>/labels/`.
 * - Configuration: a `data.yaml` file defining dataset structure.
 * - Supported tasks (auto-detected per label line by value count):
 *   - **Pose**: `class_id x_center y_center width height x1 y1 v1 ... xn yn vn`
 *     (5 + 3k values) → {@link Instance}.
 *   - **Detection**: `class_id x_center y_center width height [confidence]`
 *     (5 or 6 values) → {@link UserBoundingBox} / {@link PredictedBoundingBox}.
 *   - **Segmentation**: `class_id x1 y1 x2 y2 ... xn yn` (polygon) →
 *     {@link UserROI}.
 * - Coordinates: normalized to `[0, 1]`, origin at top-left.
 * - Visibility (pose only): `0` = not visible, `1` = visible but occluded,
 *   `2` = visible and not occluded.
 *
 * Node-only: datasets are directory trees of many files, so this module reads
 * and writes through the Node `fs`/`path` APIs (like `io/trackmate.ts`) and is
 * exported only from the Node entry point (`src/index.ts`), never the browser
 * bundle.
 *
 * ## Image I/O divergence from Python
 *
 * Python uses `imageio` to read image dimensions and to extract/encode video
 * frames. JS/Node has no equivalent always-available image codec, so:
 *
 * - **Reading**: image dimensions are obtained by parsing the image file header
 *   ({@link probeImageSize}, supporting PNG/JPEG/GIF/BMP/TIFF) rather than
 *   decoding the pixels. Falls back to the `imageSize` option when probing
 *   fails.
 * - **Writing**: when a frame is backed by an on-disk image file, the file is
 *   **copied verbatim** (preserving its encoding and extension); when a frame
 *   yields raw `ImageData`-shaped pixels (`{ data, width, height }`), it is
 *   encoded to PNG via `pako`; otherwise the frame is skipped with a warning
 *   (mirroring Python's "could not load frame → skip" behavior). The
 *   `imageFormat`/`imageQuality` options apply only to the raw-pixel PNG path.
 */

import * as fs from "fs";
import * as path from "path";
import YAML from "yaml";
import { encodePngPixels } from "../video/png-encode.js";

import { Labels } from "../model/labels.js";
import { LabelsSet } from "../model/labels-set.js";
import { LabeledFrame } from "../model/labeled-frame.js";
import { Instance, Track } from "../model/instance.js";
import { Skeleton, Node, Edge } from "../model/skeleton.js";
import { Video } from "../model/video.js";
import { ROI, UserROI } from "../model/roi.js";
import {
  BoundingBox,
  UserBoundingBox,
  PredictedBoundingBox,
} from "../model/bbox.js";

// =============================================================================
// Types
// =============================================================================

/** Image dimensions as `[height, width]` in pixels. */
export type ImageShape = [number, number];

/** Auto-detected YOLO annotation format for a single label line. */
export type LineFormat =
  | "detection"
  | "detection_conf"
  | "segmentation"
  | "pose";

/** Result of {@link parseLabelFile}: the 3-tuple of parsed annotations. */
export interface ParsedLabelFile {
  instances: Instance[];
  rois: ROI[];
  bboxes: BoundingBox[];
}

/**
 * Image extensions recognized by the dataset reader when scanning an images
 * directory. Matches Python `read_labels` exactly (note: `.tiff` but not `.tif`,
 * and no `.gif`) so the two libraries enumerate the same frames.
 */
const READ_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"];

/**
 * Image extensions the writer/prober can copy verbatim or probe. Broader than
 * {@link READ_IMAGE_EXTENSIONS} (the writer's copy path is a documented JS
 * image-I/O divergence, so it accepts a few more on-disk formats).
 */
const IMAGE_EXTENSIONS = [
  ".jpg",
  ".jpeg",
  ".png",
  ".tiff",
  ".tif",
  ".bmp",
  ".gif",
];

// =============================================================================
// data.yaml parsing / skeleton construction
// =============================================================================

/** Parse an Ultralytics `data.yaml` configuration file. */
export function parseDataYaml(yamlPath: string): Record<string, unknown> {
  const text = fs.readFileSync(yamlPath, "utf-8");
  return (YAML.parse(text) ?? {}) as Record<string, unknown>;
}

/**
 * Build a class-id → category-name map from a parsed data.yaml `names` field.
 *
 * Accepts either a YAML list (`names: [cat, dog]`) or a mapping
 * (`names: {0: cat, 1: dog}`). Keys are coerced to integers so lookups work
 * regardless of how the YAML parser represented numeric keys.
 */
export function classNamesFromConfig(
  config: Record<string, unknown>,
): Map<number, string> {
  const raw = config["names"];
  const result = new Map<number, string>();
  if (Array.isArray(raw)) {
    raw.forEach((name, i) => {
      result.set(i, String(name));
    });
  } else if (raw && typeof raw === "object") {
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      const id = Number(k);
      if (Number.isFinite(id)) result.set(id, String(v));
    }
  }
  return result;
}

/** Create a {@link Skeleton} from an Ultralytics configuration object. */
export function createSkeletonFromConfig(
  config: Record<string, unknown>,
): Skeleton {
  const kptShape = (config["kpt_shape"] as number[] | undefined) ?? [1, 3];
  const numKeypoints = kptShape[0];

  const nodeNames =
    (config["node_names"] as string[] | undefined) ??
    Array.from({ length: numKeypoints }, (_, i) => `point_${i}`);
  const nodes = nodeNames
    .slice(0, numKeypoints)
    .map((name) => new Node(String(name)));

  const edges: Edge[] = [];
  const connections =
    (config["skeleton"] as Array<[number, number]> | undefined) ?? [];
  for (const connection of connections) {
    if (Array.isArray(connection) && connection.length === 2) {
      const [srcIdx, dstIdx] = connection;
      if (
        srcIdx >= 0 &&
        srcIdx < nodes.length &&
        dstIdx >= 0 &&
        dstIdx < nodes.length
      ) {
        edges.push(new Edge(nodes[srcIdx], nodes[dstIdx]));
      }
    }
  }

  return new Skeleton({ nodes, edges, name: "ultralytics_skeleton" });
}

// =============================================================================
// Per-line format detection (port of detect_line_format)
// =============================================================================

/**
 * Detect the YOLO annotation format from a single line's parsed values.
 *
 * - **5 values** → `"detection"`
 * - **6 values** → `"detection_conf"`
 * - **5 + 3k values** → `"pose"`
 * - **even count > 5 with `(n - 1)` even** → `"segmentation"`
 * - otherwise → `"pose"`
 */
export function detectLineFormat(parts: string[]): LineFormat {
  const n = parts.length;
  if (n === 5) return "detection";
  if (n === 6) return "detection_conf";
  // Pose format: 5 + 3*k keypoint values.
  const remainder = n - 5;
  if (remainder > 0 && remainder % 3 === 0) return "pose";
  // Even number of values after class_id → segmentation polygon.
  if ((n - 1) % 2 === 0 && n > 5) return "segmentation";
  return "pose";
}

// =============================================================================
// Coordinate normalization helpers (ports of normalize/denormalize_coordinates)
// =============================================================================

/**
 * Normalize an instance's point coordinates to the `[0, 1]` range.
 *
 * @returns One `[xNorm, yNorm, visibility]` triple per point, where
 *   `visibility` is `2` for visible points and `0` for invisible/NaN points.
 */
export function normalizeCoordinates(
  instance: Instance,
  imageShape: ImageShape,
): Array<[number, number, number]> {
  const [height, width] = imageShape;
  const normalized: Array<[number, number, number]> = [];
  for (const point of instance.points) {
    const [x, y] = point.xy;
    if (point.visible && !Number.isNaN(x)) {
      normalized.push([x / width, y / height, 2]);
    } else {
      normalized.push([0.0, 0.0, 0]);
    }
  }
  return normalized;
}

/**
 * Denormalize coordinates from the `[0, 1]` range back to pixel coordinates.
 *
 * @returns One `[x, y, visible]` row per point. Invisible points (visibility
 *   `0`) become `[NaN, NaN, 0]`; visible points become `[xPx, yPx, 1]`.
 */
export function denormalizeCoordinates(
  normalizedPoints: Array<[number, number, number]>,
  imageShape: ImageShape,
): number[][] {
  const [height, width] = imageShape;
  return normalizedPoints.map(([xNorm, yNorm, visibility]) => {
    if (visibility > 0) {
      return [xNorm * width, yNorm * height, 1];
    }
    return [Number.NaN, Number.NaN, 0];
  });
}

// =============================================================================
// Label file parsing (port of parse_label_file, returning the 3-tuple)
// =============================================================================

/** Options for {@link parseLabelFile}. */
export interface ParseLabelFileOptions {
  /** Class-id → category-name mapping for category assignment. */
  classNames?: Map<number, string>;
  /** Video to associate with ROIs / bounding boxes (currently unused field). */
  video?: Video | null;
  /** Frame index for ROIs / bounding boxes. Defaults to 0. */
  frameIdx?: number;
}

/**
 * Parse a single Ultralytics label file into instances, ROIs, and bounding
 * boxes.
 *
 * The format is auto-detected per line via {@link detectLineFormat}:
 *
 * - **5 values** → {@link UserBoundingBox}
 * - **6 values** → {@link PredictedBoundingBox}
 * - **5 + 3k values** → {@link Instance} (pose)
 * - **segmentation polygon** → {@link UserROI}
 *
 * @param labelPath - Path to the `.txt` label file.
 * @param skeleton - Skeleton to use for pose instances.
 * @param imageShape - Image dimensions `[height, width]` for denormalization.
 * @param options - Optional category mapping / video / frame index.
 * @returns `{ instances, rois, bboxes }` parsed from the file.
 */
export function parseLabelFile(
  labelPath: string,
  skeleton: Skeleton,
  imageShape: ImageShape,
  options?: ParseLabelFileOptions,
): ParsedLabelFile {
  const classNames = options?.classNames;
  const instances: Instance[] = [];
  const rois: ROI[] = [];
  const bboxes: BoundingBox[] = [];

  const content = fs.readFileSync(labelPath, "utf-8");
  const lines = content.split(/\r?\n/);
  const [heightPx, widthPx] = imageShape;

  for (let lineNum = 0; lineNum < lines.length; lineNum++) {
    const line = lines[lineNum].trim();
    if (!line || line.startsWith("#")) continue;

    try {
      const parts = line.split(/\s+/);
      if (parts.length < 5) {
        console.warn(
          `Invalid line ${lineNum} in ${labelPath}: insufficient data`,
        );
        continue;
      }

      const fmt = detectLineFormat(parts);
      const classId = parseStrictInt(parts[0]);
      const category = classNames ? (classNames.get(classId) ?? "") : "";

      if (fmt === "detection" || fmt === "detection_conf") {
        const xCenterNorm = parseStrictFloat(parts[1]);
        const yCenterNorm = parseStrictFloat(parts[2]);
        const wNorm = parseStrictFloat(parts[3]);
        const hNorm = parseStrictFloat(parts[4]);

        // Denormalize to pixel coordinates.
        const xCenterPx = xCenterNorm * widthPx;
        const yCenterPx = yCenterNorm * heightPx;
        const wPx = wNorm * widthPx;
        const hPx = hNorm * heightPx;

        const x1 = xCenterPx - wPx / 2;
        const y1 = yCenterPx - hPx / 2;

        if (fmt === "detection_conf") {
          const score = parseStrictFloat(parts[5]);
          bboxes.push(
            new PredictedBoundingBox({
              x1,
              y1,
              x2: x1 + wPx,
              y2: y1 + hPx,
              category,
              score,
            }),
          );
        } else {
          bboxes.push(
            new UserBoundingBox({
              x1,
              y1,
              x2: x1 + wPx,
              y2: y1 + hPx,
              category,
            }),
          );
        }
      } else if (fmt === "segmentation") {
        // class_id x1 y1 x2 y2 ... xn yn
        const coordValues = parts.slice(1).map(parseStrictFloat);
        const coords: number[][] = [];
        for (let i = 0; i + 1 < coordValues.length; i += 2) {
          coords.push([
            coordValues[i] * widthPx,
            coordValues[i + 1] * heightPx,
          ]);
        }
        rois.push(UserROI.fromPolygon(coords, { category }));
      } else {
        // Pose format. Python parses (and discards) the four bbox columns with
        // `map(float, parts[1:5])` before reading keypoints; that call still
        // VALIDATES them, so a non-numeric bbox column raises and skips the
        // whole line. Reproduce that validation here for parity.
        parts.slice(1, 5).forEach(parseStrictFloat);
        const keypointData = parts.slice(5);
        if (keypointData.length % 3 !== 0) {
          console.warn(`Invalid keypoint data in ${labelPath} line ${lineNum}`);
          continue;
        }

        const numKeypoints = keypointData.length / 3;
        if (numKeypoints !== skeleton.nodes.length) {
          console.warn(
            `Keypoint count mismatch: expected ${skeleton.nodes.length}, ` +
              `got ${numKeypoints} in ${labelPath} line ${lineNum}`,
          );
          continue;
        }

        const points: number[][] = [];
        for (let i = 0; i < numKeypoints; i++) {
          const xNorm = parseStrictFloat(keypointData[i * 3]);
          const yNorm = parseStrictFloat(keypointData[i * 3 + 1]);
          const visibility = parseStrictInt(keypointData[i * 3 + 2]);

          if (visibility === 0) {
            points.push([Number.NaN, Number.NaN, 0]);
          } else {
            // visibility > 0 → visible (whether occluded (1) or not (2)).
            points.push([xNorm * widthPx, yNorm * heightPx, 1]);
          }
        }

        instances.push(Instance.fromNumpy({ pointsData: points, skeleton }));
      }
    } catch (e) {
      console.warn(
        `Error parsing line ${lineNum} in ${labelPath}: ${(e as Error).message}`,
      );
      continue;
    }
  }

  return { instances, rois, bboxes };
}

// =============================================================================
// Label file writing
// =============================================================================

/**
 * Write a single Ultralytics **pose** label file for a frame.
 *
 * Each instance becomes a line `class_id x_center y_center width height` (a
 * 10px-padded bounding box over visible keypoints, normalized) followed by
 * `x y v` triples per keypoint. Instances whose point count does not match the
 * skeleton, or that have no visible points, are skipped.
 */
export function writeLabelFile(
  labelPath: string,
  frame: LabeledFrame,
  skeleton: Skeleton,
  imageShape: ImageShape,
  classId = 0,
): void {
  const [heightPx, widthPx] = imageShape;
  const out: string[] = [];

  for (const instance of frame.instances) {
    if (instance.points.length !== skeleton.nodes.length) {
      console.warn(
        `Instance has ${instance.points.length} points, skeleton has ` +
          `${skeleton.nodes.length} nodes. Skipping.`,
      );
      continue;
    }

    // Bounding box from visible (and non-NaN) keypoints.
    const visibleXy: Array<[number, number]> = [];
    for (const point of instance.points) {
      if (point.visible && !Number.isNaN(point.xy[0])) {
        visibleXy.push([point.xy[0], point.xy[1]]);
      }
    }
    if (visibleXy.length === 0) continue; // No visible points → skip instance.

    const xs = visibleXy.map((p) => p[0]);
    const ys = visibleXy.map((p) => p[1]);
    const padding = 10; // pixels
    const xMin = Math.max(0, Math.min(...xs) - padding);
    const yMin = Math.max(0, Math.min(...ys) - padding);
    const xMax = Math.min(widthPx, Math.max(...xs) + padding);
    const yMax = Math.min(heightPx, Math.max(...ys) + padding);

    const xCenterNorm = (xMin + xMax) / 2 / widthPx;
    const yCenterNorm = (yMin + yMax) / 2 / heightPx;
    const widthNorm = (xMax - xMin) / widthPx;
    const heightNorm = (yMax - yMin) / heightPx;

    const lineParts: string[] = [
      String(classId),
      fmt6(xCenterNorm),
      fmt6(yCenterNorm),
      fmt6(widthNorm),
      fmt6(heightNorm),
    ];

    for (const point of instance.points) {
      const [x, y] = point.xy;
      if (point.visible && !Number.isNaN(x)) {
        lineParts.push(fmt6(x / widthPx), fmt6(y / heightPx), "2");
      } else {
        lineParts.push("0.000000", "0.000000", "0");
      }
    }

    out.push(lineParts.join(" "));
  }

  fs.writeFileSync(labelPath, out.length ? out.join("\n") + "\n" : "");
}

/**
 * Write a single Ultralytics label file for detection/segmentation ROIs.
 *
 * Multi-geometries are exploded so each polygon gets its own line. Polygon
 * ROIs are written as segmentation lines (normalized exterior vertices); ROIs
 * that are axis-aligned rectangles are written as detection bounding boxes.
 * Interior rings (holes) are dropped with a warning (YOLO segmentation has no
 * hole support).
 */
export function writeRoiLabelFile(
  labelPath: string,
  rois: ROI[],
  imageShape: ImageShape,
  nameToId: Map<string, number>,
): void {
  const [heightPx, widthPx] = imageShape;

  // Explode multi-geometries so each polygon gets its own line.
  const explodedRois: ROI[] = [];
  for (const roi of rois) explodedRois.push(...roi.explode());

  const out: string[] = [];
  for (const roi of explodedRois) {
    const classId = nameToId.get(roi.category) ?? 0;

    if (!roi.isBbox) {
      // Segmentation polygon. DIVERGENCE: Python accesses `roi.geometry.exterior`
      // unconditionally, which raises (AttributeError) and aborts the whole
      // export for a degenerate non-polygon geometry (Point/LineString/etc.).
      // We instead skip such geometries gracefully — a safer behavior for
      // malformed input that normal segmentation ROIs (Polygon/MultiPolygon)
      // never hit.
      const geom = roi.geometry;
      if (geom.type === "Polygon") {
        if (geom.coordinates.length > 1) {
          console.warn(
            `ROI polygon has ${geom.coordinates.length - 1} interior ring(s) ` +
              `(holes) that will be dropped. YOLO segmentation format does not ` +
              `support polygon holes.`,
          );
        }
        const ring = geom.coordinates[0] ?? [];
        const coords = ring.slice(0, -1); // Remove closing point.
        const lineParts = [String(classId)];
        for (const [x, y] of coords) {
          lineParts.push(fmt6(x / widthPx), fmt6(y / heightPx));
        }
        out.push(lineParts.join(" "));
      }
    } else {
      // Detection bounding box.
      const { minX, minY, maxX, maxY } = roi.bounds;
      const xCenter = (minX + maxX) / 2 / widthPx;
      const yCenter = (minY + maxY) / 2 / heightPx;
      const w = (maxX - minX) / widthPx;
      const h = (maxY - minY) / heightPx;
      out.push(
        [String(classId), fmt6(xCenter), fmt6(yCenter), fmt6(w), fmt6(h)].join(
          " ",
        ),
      );
    }
  }

  fs.writeFileSync(labelPath, out.length ? out.join("\n") + "\n" : "");
}

/**
 * Write a single Ultralytics label file for detection bounding boxes.
 *
 * {@link UserBoundingBox} → 5 values; {@link PredictedBoundingBox} → 6 values
 * (the trailing value is the confidence score).
 */
export function writeBboxLabelFile(
  labelPath: string,
  bboxes: BoundingBox[],
  imageShape: ImageShape,
  nameToId: Map<string, number>,
): void {
  const [heightPx, widthPx] = imageShape;
  const out: string[] = [];

  for (const bbox of bboxes) {
    const classId = nameToId.get(bbox.category) ?? 0;
    const lineParts = [
      String(classId),
      fmt6(bbox.xCenter / widthPx),
      fmt6(bbox.yCenter / heightPx),
      fmt6(bbox.width / widthPx),
      fmt6(bbox.height / heightPx),
    ];
    if (bbox instanceof PredictedBoundingBox) {
      lineParts.push(fmt6(bbox.score));
    }
    out.push(lineParts.join(" "));
  }

  fs.writeFileSync(labelPath, out.length ? out.join("\n") + "\n" : "");
}

// =============================================================================
// data.yaml writing (port of create_data_yaml)
// =============================================================================

/** Options for {@link createDataYaml}. */
export interface CreateDataYamlOptions {
  /** YOLO task type. One of `"pose"` (default), `"detect"`, or `"segment"`. */
  task?: string;
  /** Class-id → category-name mapping. Defaults to `{ 0: "animal" }`. */
  classNames?: Map<number, string>;
}

/**
 * Create an Ultralytics `data.yaml` configuration file.
 *
 * For pose tasks, writes `kpt_shape`, `flip_idx`, `skeleton`, and `node_names`
 * derived from the skeleton. For detection/segmentation, writes the `task` key.
 */
export function createDataYaml(
  yamlPath: string,
  skeleton: Skeleton | null,
  splitRatios: Record<string, number>,
  options?: CreateDataYamlOptions,
): void {
  const task = options?.task ?? "pose";
  const classNames =
    options?.classNames ?? new Map<number, string>([[0, "animal"]]);

  // Pass `names` as a Map so the YAML emitter writes integer keys
  // (`0: animal`) matching Ultralytics/Python output rather than quoted
  // string keys.
  const config: Record<string, unknown> = {
    path: ".",
    names: classNames,
  };

  if (task !== "pose") {
    config["task"] = task;
  } else if (skeleton !== null) {
    // Pose-specific fields.
    const connections: Array<[number, number]> = [];
    for (const edge of skeleton.edges) {
      connections.push([
        skeleton.index(edge.source),
        skeleton.index(edge.destination),
      ]);
    }
    config["kpt_shape"] = [skeleton.nodes.length, 3];
    config["flip_idx"] = Array.from(
      { length: skeleton.nodes.length },
      (_, i) => i,
    );
    config["skeleton"] = connections;
    config["node_names"] = skeleton.nodes.map((node) => node.name);
  }

  // Add split paths.
  for (const splitName of Object.keys(splitRatios)) {
    config[splitName] = `${splitName}/images`;
  }

  fs.writeFileSync(yamlPath, YAML.stringify(config));
}

// =============================================================================
// Class-name builders (ports of _build_class_names_from_rois/_bboxes)
// =============================================================================

/** Build a class-id → name map from the distinct, sorted ROI categories. */
export function buildClassNamesFromRois(rois: ROI[]): Map<number, string> {
  return buildClassNames(rois.map((roi) => roi.category));
}

/** Build a class-id → name map from the distinct, sorted bbox categories. */
export function buildClassNamesFromBboxes(
  bboxes: BoundingBox[],
): Map<number, string> {
  return buildClassNames(bboxes.map((bbox) => bbox.category));
}

function buildClassNames(categories: string[]): Map<number, string> {
  const distinct = Array.from(new Set(categories.filter((c) => c))).sort();
  const result = new Map<number, string>();
  if (distinct.length === 0) {
    result.set(0, "object");
    return result;
  }
  distinct.forEach((name, i) => {
    result.set(i, name);
  });
  return result;
}

// =============================================================================
// Dataset reading (port of read_labels / read_labels_set)
// =============================================================================

/** Options for {@link readLabels}. */
export interface ReadLabelsOptions {
  /** Dataset split to read (`"train"`, `"val"`, `"test"`, ...). Default `"train"`. */
  split?: string;
  /** Skeleton to use. If omitted, inferred from `data.yaml` (pose only). */
  skeleton?: Skeleton | null;
  /** Fallback image size `[height, width]` if header probing fails. Default `[480, 640]`. */
  imageSize?: ImageShape;
}

/**
 * Read an Ultralytics YOLO dataset into a {@link Labels} object.
 *
 * Automatically detects the annotation format (pose / detection / segmentation)
 * per label line. Pose lines become instances; detection lines become bounding
 * boxes; segmentation lines become ROIs.
 *
 * @param datasetPath - Path to the dataset root (containing `data.yaml`) or to
 *   the `data.yaml` file itself.
 * @param options - Optional split / skeleton / fallback image size.
 */
export function readLabels(
  datasetPath: string,
  options?: ReadLabelsOptions,
): Labels {
  const split = options?.split ?? "train";
  const imageSize = options?.imageSize ?? [480, 640];
  let skeleton = options?.skeleton ?? null;

  // Resolve data.yaml location and dataset root.
  let dataYamlPath: string;
  let root: string;
  if (path.basename(datasetPath) === "data.yaml") {
    dataYamlPath = datasetPath;
    root = path.dirname(datasetPath);
  } else {
    root = datasetPath;
    dataYamlPath = path.join(datasetPath, "data.yaml");
  }

  if (!fs.existsSync(dataYamlPath)) {
    throw new Error(`data.yaml not found at ${dataYamlPath}`);
  }

  const config = parseDataYaml(dataYamlPath);

  // Infer skeleton from config when one was not supplied (pose only).
  if (skeleton === null && "kpt_shape" in config) {
    skeleton = createSkeletonFromConfig(config);
  }

  const classNames = classNamesFromConfig(config);

  // Resolve split image / label directories.
  const splitPath = (config[split] as string | undefined) ?? `${split}/images`;
  const imagesDir = path.join(root, splitPath);
  // Replace ALL "/images" segments (Python `str.replace` has no count limit).
  const labelsDir = path.join(root, splitPath.replace(/\/images/g, "/labels"));

  if (!fs.existsSync(imagesDir)) {
    throw new Error(`Images directory not found: ${imagesDir}`);
  }
  if (!fs.existsSync(labelsDir)) {
    throw new Error(`Labels directory not found: ${labelsDir}`);
  }

  const labeledFrames: LabeledFrame[] = [];
  const tracks = new Map<string, Track>();

  const imageFiles = fs
    .readdirSync(imagesDir)
    .filter((name) =>
      READ_IMAGE_EXTENSIONS.includes(path.extname(name).toLowerCase()),
    )
    .sort();

  for (const imageName of imageFiles) {
    const imageFile = path.join(imagesDir, imageName);
    const stem = path.basename(imageName, path.extname(imageName));
    const labelFile = path.join(labelsDir, `${stem}.txt`);

    const video = new Video({ filename: imageFile, openBackend: false });

    let instances: Instance[] = [];
    let rois: ROI[] = [];
    let bboxes: BoundingBox[] = [];

    if (fs.existsSync(labelFile)) {
      const imgShape = probeImageSize(imageFile) ?? imageSize;
      const parseSkeleton = skeleton ?? new Skeleton({ nodes: [] });
      ({ instances, rois, bboxes } = parseLabelFile(
        labelFile,
        parseSkeleton,
        imgShape,
        {
          classNames,
          video,
          frameIdx: 0,
        },
      ));

      // Assign synthetic tracks based on per-frame instance order.
      for (let i = 0; i < instances.length; i++) {
        const trackName = `track_${i}`;
        let track = tracks.get(trackName);
        if (!track) {
          track = new Track(trackName);
          tracks.set(trackName, track);
        }
        instances[i].track = track;
      }
    }

    const frame = new LabeledFrame({ video, frameIdx: 0, instances });
    frame.rois.push(...rois);
    frame.bboxes.push(...bboxes);
    labeledFrames.push(frame);
  }

  const skeletons = skeleton !== null ? [skeleton] : [];

  return new Labels({
    labeledFrames,
    skeletons,
    tracks: Array.from(tracks.values()),
    provenance: { source: root, split },
  });
}

/** Options for {@link readLabelsSet}. */
export interface ReadLabelsSetOptions {
  /** Splits to load. If omitted, auto-detects `train`/`val`/`test`/`valid`. */
  splits?: string[];
  /** Skeleton to use. If omitted, inferred from `data.yaml`. */
  skeleton?: Skeleton | null;
  /** Fallback image size `[height, width]` if header probing fails. */
  imageSize?: ImageShape;
}

/**
 * Read multiple splits from an Ultralytics dataset as a {@link LabelsSet}.
 *
 * @param datasetPath - Path to the dataset root directory.
 * @param options - Optional splits / skeleton / fallback image size.
 */
export function readLabelsSet(
  datasetPath: string,
  options?: ReadLabelsSetOptions,
): LabelsSet {
  const imageSize = options?.imageSize;
  let skeleton = options?.skeleton ?? null;
  let splits = options?.splits;

  // Auto-detect available splits.
  if (!splits) {
    splits = [];
    for (const splitName of ["train", "val", "test", "valid"]) {
      if (fs.existsSync(path.join(datasetPath, splitName))) {
        splits.push(splitName);
      }
    }
    if (splits.length === 0) {
      throw new Error(`No splits found in dataset path: ${datasetPath}`);
    }
  }

  // Infer skeleton from data.yaml when not provided.
  if (skeleton === null) {
    const dataYamlPath = path.join(datasetPath, "data.yaml");
    if (fs.existsSync(dataYamlPath)) {
      const dataConfig = parseDataYaml(dataYamlPath);

      // Prefer custom node_names + skeleton metadata.
      if ("node_names" in dataConfig && "skeleton" in dataConfig) {
        try {
          const nodeNames = dataConfig["node_names"] as string[];
          const connections = dataConfig["skeleton"] as Array<[number, number]>;
          const nodes = nodeNames.map((name) => new Node(String(name)));
          const edges: Edge[] = [];
          for (const connection of connections) {
            if (Array.isArray(connection) && connection.length === 2) {
              const [srcIdx, dstIdx] = connection;
              if (
                srcIdx >= 0 &&
                srcIdx < nodes.length &&
                dstIdx >= 0 &&
                dstIdx < nodes.length
              ) {
                edges.push(new Edge(nodes[srcIdx], nodes[dstIdx]));
              }
            }
          }
          skeleton = new Skeleton({ nodes, edges });
        } catch {
          // Fall through to basic creation.
        }
      }

      // Fall back to a basic numbered skeleton from kpt_shape.
      if (skeleton === null && "kpt_shape" in dataConfig) {
        const kptShape = dataConfig["kpt_shape"] as number[];
        if (Array.isArray(kptShape) && kptShape.length >= 2) {
          const nKeypoints = kptShape[0];
          const nodes = Array.from(
            { length: nKeypoints },
            (_, i) => new Node(String(i)),
          );
          skeleton = new Skeleton({ nodes });
        }
      }
    }
  }

  const entries: Record<string, Labels> = {};
  for (const split of splits) {
    try {
      entries[split] = readLabels(datasetPath, { split, skeleton, imageSize });
    } catch {
      continue;
    }
  }

  if (Object.keys(entries).length === 0) {
    throw new Error(`Could not load any splits from dataset: ${datasetPath}`);
  }

  return new LabelsSet(entries);
}

// =============================================================================
// Dataset writing (port of write_labels)
// =============================================================================

/** Options for {@link writeLabels}. */
export interface WriteLabelsOptions {
  /** Split-name → ratio mapping (must sum to 1.0). Default `{ train: 0.8, val: 0.2 }`. */
  splitRatios?: Record<string, number>;
  /** Class ID to use for all pose instances. Default `0`. */
  classId?: number;
  /** Image format for raw-pixel frames (`"png"` default, lossless). */
  imageFormat?: string;
  /** PNG compression level (0–9) for raw-pixel frames. */
  imageQuality?: number | null;
  /** Show progress logging. Default `true`. */
  verbose?: boolean;
  /** YOLO task type: `"pose"` (default), `"detect"`, or `"segment"`. */
  task?: string;
}

/**
 * Write a {@link Labels} object to an Ultralytics YOLO dataset on disk.
 *
 * For `"pose"`, writes images + pose label files per labeled frame. For
 * `"detect"` and `"segment"`, writes bounding boxes / ROIs from the Labels
 * object instead of pose instances.
 *
 * See the module-level "Image I/O divergence" note for how frame images are
 * obtained (on-disk copy, raw-pixel PNG encode, or skip-with-warning).
 *
 * @param labels - Labels to export.
 * @param datasetPath - Output dataset root directory.
 * @param options - Export options (see {@link WriteLabelsOptions}).
 */
export async function writeLabels(
  labels: Labels,
  datasetPath: string,
  options?: WriteLabelsOptions,
): Promise<void> {
  const splitRatios = options?.splitRatios ?? { train: 0.8, val: 0.2 };
  const classId = options?.classId ?? 0;
  const imageFormat = options?.imageFormat ?? "png";
  const imageQuality = options?.imageQuality ?? null;
  const task = options?.task ?? "pose";

  fs.mkdirSync(datasetPath, { recursive: true });

  // Validate split ratios. Python uses `np.isclose(total, 1.0)`, whose default
  // tolerance is `atol + rtol*|1.0| = 1e-8 + 1e-5`; match it so ratios like
  // three rounded thirds (sum 0.999999) are accepted as Python accepts them.
  const totalRatio = Object.values(splitRatios).reduce((a, b) => a + b, 0);
  if (Math.abs(totalRatio - 1.0) > 1e-8 + 1e-5) {
    throw new Error(`Split ratios must sum to 1.0, got ${totalRatio}`);
  }

  let skeleton: Skeleton | null;
  if (task === "pose") {
    if (labels.skeletons.length === 0) {
      throw new Error("Labels must have at least one skeleton for pose task");
    }
    skeleton = labels.skeletons[0];
  } else {
    skeleton = null;
  }

  // Build class names from ROI/bbox categories for non-pose tasks.
  let classNames: Map<number, string>;
  if (task === "detect") {
    classNames = buildClassNamesFromBboxes(labels.bboxes);
  } else if (task === "segment") {
    classNames = buildClassNamesFromRois(labels.rois);
  } else {
    classNames = new Map<number, string>([[0, "animal"]]);
  }

  createDataYaml(path.join(datasetPath, "data.yaml"), skeleton, splitRatios, {
    task,
    classNames,
  });

  if (task === "detect") {
    writeBboxLabels(
      labels,
      datasetPath,
      splitRatios,
      classNames,
      imageFormat,
      imageQuality,
    );
    return;
  }
  if (task === "segment") {
    writeRoiLabels(
      labels,
      datasetPath,
      splitRatios,
      classNames,
      imageFormat,
      imageQuality,
    );
    return;
  }

  // Pose: split labels if multiple splits requested.
  let splitLabels: Record<string, Labels>;
  const splitNames = Object.keys(splitRatios);
  if (splitNames.length === 1) {
    splitLabels = { [splitNames[0]]: labels };
  } else {
    splitLabels = createSplitsFromLabels(labels, splitRatios);
  }

  for (const [splitName, splitData] of Object.entries(splitLabels)) {
    const imagesDir = path.join(datasetPath, splitName, "images");
    const labelsDir = path.join(datasetPath, splitName, "labels");
    fs.mkdirSync(imagesDir, { recursive: true });
    fs.mkdirSync(labelsDir, { recursive: true });

    // The filename index follows Python's `enumerate`: it advances on every
    // frame, so a skipped frame leaves a numbering gap (it is NOT renumbered
    // contiguously).
    const frames = splitData.labeledFrames;
    for (let lfIdx = 0; lfIdx < frames.length; lfIdx++) {
      const frame = frames[lfIdx];
      const written = await writeFrameImage(
        frame,
        imagesDir,
        lfIdx,
        imageFormat,
        imageQuality,
      );
      if (written === null) {
        console.warn(
          `Could not load frame ${frame.frameIdx} from video, skipping.`,
        );
        continue;
      }
      const labelPath = path.join(labelsDir, `${pad7(lfIdx)}.txt`);
      writeLabelFile(labelPath, frame, skeleton!, written.shape, classId);
    }
  }
}

/** Write detection (bbox) labels grouped by source video. */
function writeBboxLabels(
  labels: Labels,
  datasetPath: string,
  splitRatios: Record<string, number>,
  classNames: Map<number, string>,
  imageFormat: string,
  imageQuality: number | null,
): void {
  const nameToId = invertClassNames(classNames);

  // Group bboxes by source video filename.
  const itemsByVideo = new Map<string, BoundingBox[]>();
  const videoByKey = new Map<string, Video>();
  for (const lf of labels.labeledFrames) {
    if (lf.bboxes.length) {
      const key = videoKey(lf.video);
      if (!itemsByVideo.has(key)) {
        itemsByVideo.set(key, []);
        videoByKey.set(key, lf.video);
      }
      itemsByVideo.get(key)!.push(...lf.bboxes);
    }
  }

  writeGroupedLabels(
    itemsByVideo,
    videoByKey,
    datasetPath,
    splitRatios,
    imageFormat,
    imageQuality,
    (labelPath, items, shape) =>
      writeBboxLabelFile(labelPath, items, shape, nameToId),
  );
}

/** Write ROI (detection/segmentation) labels grouped by source video. */
function writeRoiLabels(
  labels: Labels,
  datasetPath: string,
  splitRatios: Record<string, number>,
  classNames: Map<number, string>,
  imageFormat: string,
  imageQuality: number | null,
): void {
  const nameToId = invertClassNames(classNames);

  const itemsByVideo = new Map<string, ROI[]>();
  const videoByKey = new Map<string, Video>();
  for (const lf of labels.labeledFrames) {
    if (lf.rois.length) {
      const key = videoKey(lf.video);
      if (!itemsByVideo.has(key)) {
        itemsByVideo.set(key, []);
        videoByKey.set(key, lf.video);
      }
      itemsByVideo.get(key)!.push(...lf.rois);
    }
  }

  writeGroupedLabels(
    itemsByVideo,
    videoByKey,
    datasetPath,
    splitRatios,
    imageFormat,
    imageQuality,
    (labelPath, items, shape) =>
      writeRoiLabelFile(labelPath, items, shape, nameToId),
  );
}

/**
 * Shared driver for the detect/segment write paths: split video keys across the
 * requested splits, copy each source image, and write its label file.
 */
function writeGroupedLabels<T>(
  itemsByVideo: Map<string, T[]>,
  videoByKey: Map<string, Video>,
  datasetPath: string,
  splitRatios: Record<string, number>,
  imageFormat: string,
  imageQuality: number | null,
  writeLabel: (labelPath: string, items: T[], shape: ImageShape) => void,
): void {
  const videoKeys = Array.from(itemsByVideo.keys()).sort();
  const nVideos = videoKeys.length;
  const splitNames = Object.keys(splitRatios);

  // Split boundaries by cumulative rounded ratio. Python uses `int(round(...))`,
  // and Python 3's `round` is round-half-to-even (banker's rounding), so a
  // boundary landing exactly on `.5` rounds differently from JS `Math.round`.
  const boundaries: number[] = [];
  let cumulative = 0;
  for (const name of splitNames) {
    cumulative += splitRatios[name];
    boundaries.push(roundHalfToEven(cumulative * nVideos));
  }

  let startIdx = 0;
  for (let splitIdx = 0; splitIdx < splitNames.length; splitIdx++) {
    const splitName = splitNames[splitIdx];
    const endIdx = boundaries[splitIdx];
    const splitVideoKeys = videoKeys.slice(startIdx, endIdx);
    startIdx = endIdx;
    if (splitVideoKeys.length === 0) continue;

    const imagesDir = path.join(datasetPath, splitName, "images");
    const labelsDir = path.join(datasetPath, splitName, "labels");
    fs.mkdirSync(imagesDir, { recursive: true });
    fs.mkdirSync(labelsDir, { recursive: true });

    // Filename index follows Python's `enumerate` (advances even when a video
    // is skipped, leaving a numbering gap).
    for (let lfIdx = 0; lfIdx < splitVideoKeys.length; lfIdx++) {
      const key = splitVideoKeys[lfIdx];
      const video = videoByKey.get(key)!;
      const items = itemsByVideo.get(key)!;
      const written = copyImageFileFrom(video, imagesDir, lfIdx);
      if (written === null) {
        console.warn(`Error processing ${key}, skipping.`);
        continue;
      }
      const labelPath = path.join(labelsDir, `${pad7(lfIdx)}.txt`);
      writeLabel(labelPath, items, written.shape);
    }
  }
}

/**
 * Build dataset splits from a Labels object.
 *
 * - **Two splits**: a single fractional {@link Labels#split} (`split1`/`split2`).
 * - **Three splits**: mirrors Python `Labels.make_training_splits` — splits in
 *   `train → test → val` order, recomputing each later fraction relative to the
 *   original total and the current remainder so the per-split counts match the
 *   Python writer. (JS Labels has no `makeTrainingSplits`, so the algorithm is
 *   inlined here; unlike Python it does not pre-clean predictions, leaving the
 *   caller's frames untouched.)
 */
export function createSplitsFromLabels(
  labels: Labels,
  splitRatios: Record<string, number>,
): Record<string, Labels> {
  const splitNames = Object.keys(splitRatios);

  if (splitNames.length === 2) {
    const ratio = splitRatios[splitNames[0]];
    const set = labels.split(ratio);
    return {
      [splitNames[0]]: set.get("split1")!,
      [splitNames[1]]: set.get("split2")!,
    };
  }

  if (splitNames.length === 3) {
    const total = labels.labeledFrames.length;
    const trainRatio = splitRatios["train"] ?? 0.6;
    const valRatio = splitRatios["val"] ?? 0.2;
    const testRatio = splitRatios["test"] ?? 0.2;

    // Train split.
    const trainSet = labels.split(trainRatio);
    const train = trainSet.get("split1")!;
    let rest = trainSet.get("split2")!;

    // Test split (fraction recomputed relative to the original total / remainder).
    let nTest = testRatio;
    if (nTest < 1) {
      nTest = (testRatio * total) / Math.max(1, rest.labeledFrames.length);
    }
    const testSet = rest.split(nTest);
    const test = testSet.get("split1")!;
    rest = testSet.get("split2")!;

    // Val split: the remainder, or a final fractional split.
    let nVal = valRatio;
    if (nVal < 1) {
      nVal = (valRatio * total) / Math.max(1, rest.labeledFrames.length);
    }
    const val = nVal === 1.0 ? rest : rest.split(nVal).get("split1")!;

    return { train, val, test };
  }

  return { [splitNames[0]]: labels };
}

// =============================================================================
// Public convenience API (mirrors Python load_ultralytics / save_ultralytics)
// =============================================================================

/**
 * Load an Ultralytics YOLO dataset into a {@link Labels} object.
 *
 * Convenience wrapper around {@link readLabels}.
 *
 * @param datasetPath - Path to the dataset root or its `data.yaml` file.
 * @param options - Optional split / skeleton / fallback image size.
 */
export function loadUltralytics(
  datasetPath: string,
  options?: ReadLabelsOptions,
): Labels {
  return readLabels(datasetPath, options);
}

/**
 * Save a {@link Labels} object to an Ultralytics YOLO dataset on disk.
 *
 * Convenience wrapper around {@link writeLabels}.
 *
 * @param labels - Labels to export.
 * @param datasetPath - Output dataset root directory.
 * @param options - Export options (see {@link WriteLabelsOptions}).
 */
export async function saveUltralytics(
  labels: Labels,
  datasetPath: string,
  options?: WriteLabelsOptions,
): Promise<void> {
  return writeLabels(labels, datasetPath, options);
}

// =============================================================================
// Image helpers (Node-only)
// =============================================================================

/** Result of writing a frame image: the on-disk path and its `[height, width]`. */
interface WrittenImage {
  imagePath: string;
  shape: ImageShape;
}

/**
 * Write a frame's image to `imagesDir` as `NNNNNNN.<ext>`.
 *
 * Strategy (see module note): copy an on-disk source image verbatim; else
 * encode raw `ImageData`-shaped pixels to PNG; else return `null` (skip).
 */
async function writeFrameImage(
  frame: LabeledFrame,
  imagesDir: string,
  lfIdx: number,
  imageFormat: string,
  imageQuality: number | null,
): Promise<WrittenImage | null> {
  // 1. On-disk image source → copy verbatim.
  const copied = copyImageFileFrom(frame.video, imagesDir, lfIdx);
  if (copied !== null) return copied;

  // 2. Raw pixel frame → encode PNG.
  try {
    const img = await frame.image;
    if (isImageData(img)) {
      const png = encodePng(img.data, img.width, img.height, imageQuality);
      const imagePath = path.join(imagesDir, `${pad7(lfIdx)}.png`);
      fs.writeFileSync(imagePath, png);
      return { imagePath, shape: [img.height, img.width] };
    }
  } catch {
    // fall through to skip.
  }

  // 3. Nothing usable.
  void imageFormat;
  return null;
}

/**
 * If `video` is backed by a readable on-disk image file, copy it into
 * `imagesDir` as `NNNNNNN.<sourceExt>` and return its probed shape; else `null`.
 */
function copyImageFileFrom(
  video: Video,
  imagesDir: string,
  lfIdx: number,
): WrittenImage | null {
  const filename = Array.isArray(video.filename)
    ? video.filename[0]
    : video.filename;
  if (typeof filename !== "string" || filename.length === 0) return null;
  const ext = path.extname(filename).toLowerCase();
  if (!IMAGE_EXTENSIONS.includes(ext)) return null;
  if (!fs.existsSync(filename)) return null;

  const shape = probeImageSize(filename);
  if (shape === null) return null;

  const imagePath = path.join(imagesDir, `${pad7(lfIdx)}${ext}`);
  fs.copyFileSync(filename, imagePath);
  return { imagePath, shape };
}

/**
 * Probe an image file's `[height, width]` from its header, without decoding
 * pixels. Supports PNG, JPEG, GIF, BMP, and TIFF. Returns `null` if the
 * dimensions cannot be determined.
 */
export function probeImageSize(filePath: string): ImageShape | null {
  let fd: number | null = null;
  try {
    fd = fs.openSync(filePath, "r");
    const head = Buffer.alloc(32);
    const n = fs.readSync(fd, head, 0, 32, 0);
    if (n < 2) return null;

    // PNG: 8-byte signature, then IHDR with width/height as big-endian uint32.
    if (
      head[0] === 0x89 &&
      head[1] === 0x50 &&
      head[2] === 0x4e &&
      head[3] === 0x47
    ) {
      const width = head.readUInt32BE(16);
      const height = head.readUInt32BE(20);
      return [height, width];
    }

    // GIF: "GIF87a"/"GIF89a", width/height as little-endian uint16 at offset 6.
    if (head[0] === 0x47 && head[1] === 0x49 && head[2] === 0x46) {
      const width = head.readUInt16LE(6);
      const height = head.readUInt16LE(8);
      return [height, width];
    }

    // BMP: "BM", width/height as little-endian int32 at offset 18/22.
    if (head[0] === 0x42 && head[1] === 0x4d) {
      const width = head.readInt32LE(18);
      const height = Math.abs(head.readInt32LE(22));
      return [height, width];
    }

    // TIFF: "II*\0" (little-endian) or "MM\0*" (big-endian).
    if (
      (head[0] === 0x49 && head[1] === 0x49 && head[2] === 0x2a) ||
      (head[0] === 0x4d && head[1] === 0x4d && head[3] === 0x2a)
    ) {
      return probeTiffSize(fd);
    }

    // JPEG: scan segment markers for SOF0..SOF15.
    if (head[0] === 0xff && head[1] === 0xd8) {
      return probeJpegSize(fd);
    }

    return null;
  } catch {
    return null;
  } finally {
    if (fd !== null) {
      try {
        fs.closeSync(fd);
      } catch {
        // ignore
      }
    }
  }
}

/** Scan a JPEG file's SOF marker to extract `[height, width]`. */
function probeJpegSize(fd: number): ImageShape | null {
  // Read the whole file (JPEGs in this codec's scope are small frames).
  const stat = fs.fstatSync(fd);
  const buf = Buffer.alloc(stat.size);
  fs.readSync(fd, buf, 0, stat.size, 0);

  let offset = 2; // skip SOI (0xFFD8)
  while (offset + 9 < buf.length) {
    if (buf[offset] !== 0xff) {
      offset++;
      continue;
    }
    const marker = buf[offset + 1];
    // Standalone markers without a length field.
    if (
      marker === 0xd8 ||
      marker === 0xd9 ||
      (marker >= 0xd0 && marker <= 0xd7)
    ) {
      offset += 2;
      continue;
    }
    const segLen = buf.readUInt16BE(offset + 2);
    // SOF markers (baseline + progressive), excluding DHT(0xC4)/DAC(0xCC)/RSTn.
    const isSof =
      marker >= 0xc0 &&
      marker <= 0xcf &&
      marker !== 0xc4 &&
      marker !== 0xc8 &&
      marker !== 0xcc;
    if (isSof) {
      const height = buf.readUInt16BE(offset + 5);
      const width = buf.readUInt16BE(offset + 7);
      return [height, width];
    }
    offset += 2 + segLen;
  }
  return null;
}

/** Read the first IFD of a TIFF file to extract `[height, width]`. */
function probeTiffSize(fd: number): ImageShape | null {
  const stat = fs.fstatSync(fd);
  const size = Math.min(stat.size, 65536);
  const buf = Buffer.alloc(size);
  fs.readSync(fd, buf, 0, size, 0);

  const le = buf[0] === 0x49;
  const readU16 = (o: number) =>
    le ? buf.readUInt16LE(o) : buf.readUInt16BE(o);
  const readU32 = (o: number) =>
    le ? buf.readUInt32LE(o) : buf.readUInt32BE(o);

  const ifdOffset = readU32(4);
  if (ifdOffset + 2 > buf.length) return null;
  const numEntries = readU16(ifdOffset);

  let width = 0;
  let height = 0;
  for (let i = 0; i < numEntries; i++) {
    const entry = ifdOffset + 2 + i * 12;
    if (entry + 12 > buf.length) break;
    const tag = readU16(entry);
    const type = readU16(entry + 2);
    // Value is short (type 3) or long (type 4); stored inline at entry+8.
    const value = type === 3 ? readU16(entry + 8) : readU32(entry + 8);
    if (tag === 256) width = value; // ImageWidth
    if (tag === 257) height = value; // ImageLength
  }

  if (width > 0 && height > 0) return [height, width];
  return null;
}

/** ImageData-shaped raw pixel buffer: `{ data, width, height }` (RGBA). */
interface RawImage {
  data: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
}

function isImageData(img: unknown): img is RawImage {
  return (
    !!img &&
    typeof img === "object" &&
    typeof (img as RawImage).width === "number" &&
    typeof (img as RawImage).height === "number" &&
    !!(img as RawImage).data &&
    typeof (img as RawImage).data.length === "number"
  );
}

/**
 * Encode RGBA pixels to a PNG byte stream using `pako` for the zlib stream.
 *
 * @param rgba - Row-major RGBA bytes (length `width * height * 4`).
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @param compressLevel - zlib compression level 0–9 (default 6).
 */
export function encodePng(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  compressLevel: number | null = null,
): Uint8Array {
  return encodePngPixels(rgba, width, height, 4, compressLevel);
}

// =============================================================================
// Small shared utilities
// =============================================================================

/**
 * Format a number with 6 fixed decimal places (matches Python `f"{x:.6f}"`).
 *
 * One negligible divergence: an exact IEEE-754 `-0.0` formats as `"0.000000"`
 * here vs `"-0.000000"` in Python (which preserves the sign bit). These write
 * paths never produce a literal `-0.0` (the bbox path clamps with `Math.max(0,
 * …)` and invisible keypoints are hardcoded), so it is unreachable in practice.
 */
function fmt6(x: number): string {
  return x.toFixed(6);
}

/** Zero-pad an index to 7 digits (matches Python `f"{i:07d}"`). */
function pad7(i: number): string {
  return String(i).padStart(7, "0");
}

/**
 * Round half to even ("banker's rounding"), matching Python 3's built-in
 * `round`. Used for split-boundary computation so a `.5` boundary rounds the
 * same way the Python writer does (`Math.round` would round half up instead).
 */
function roundHalfToEven(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Strict integer parse that throws on non-numeric input (like Python `int()`). */
function parseStrictInt(s: string): number {
  if (!/^[+-]?\d+$/.test(s.trim())) {
    throw new Error(`invalid integer: ${s}`);
  }
  return parseInt(s, 10);
}

/**
 * Strict float parse matching Python `float()` grammar (throws on invalid).
 *
 * Unlike JS `Number()`, this rejects radix-prefixed literals (`0x`/`0o`/`0b`)
 * that Python `float()` rejects, and accepts the `inf`/`infinity`/`nan` tokens
 * (case-insensitive, optional sign) that Python `float()` accepts. Used for
 * coordinate / score columns so a malformed value triggers the same
 * skip-and-warn path Python takes.
 */
function parseStrictFloat(s: string): number {
  const t = s.trim();
  if (/^[+-]?(inf|infinity)$/i.test(t)) {
    return t.startsWith("-") ? -Infinity : Infinity;
  }
  if (/^[+-]?nan$/i.test(t)) return Number.NaN;
  // Decimal float grammar (no hex/octal/binary, no JS-only forms).
  if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t)) {
    throw new Error(`invalid float: ${s}`);
  }
  return Number(t);
}

/** Source-video grouping key (first filename element for image sequences). */
function videoKey(video: Video): string {
  return Array.isArray(video.filename)
    ? String(video.filename[0])
    : String(video.filename);
}

/** Invert a class-id → name map to a name → class-id map. */
function invertClassNames(
  classNames: Map<number, string>,
): Map<string, number> {
  const result = new Map<string, number>();
  for (const [id, name] of classNames) result.set(name, id);
  return result;
}
