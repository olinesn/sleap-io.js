import type {
  VideoBackend,
  VideoFrame,
  GetFrameOptions,
} from "../video/backend.js";
import { CropVideoBackend } from "../video/crop-backend.js";
import { GrayscaleVideoBackend } from "../video/grayscale-backend.js";
import {
  cropPoints,
  uncropPoints,
  type CropRect,
  type FlatPoints,
  type PointPairs,
} from "../transform/points.js";
import type { Fill } from "../transform/frame.js";
import { headOrRangeProbe, isUrl } from "../io/remote.js";

/**
 * Default TTL (ms) for the per-instance {@link Video.exists} URL probe cache.
 * Analogous to Python's `SLEAP_IO_EXISTS_TTL`.
 */
export const EXISTS_TTL_MS = 60_000;

/**
 * Bounding-box / region specs accepted by {@link Video.crop} and
 * {@link Video.fromCrop} (in addition to an explicit `crop` rect).
 *
 * Exactly one region spec must be provided across `crop` (the positional rect),
 * `bbox`, `roi`, or the (`center`, `size`) pair. Mirrors the keyword arguments
 * of Python `Video.crop` / `_resolve_crop_rect`.
 */
export interface CropOptions {
  /** A bounding box `[x1, y1, x2, y2]`; bounds may be float (floor/ceil). */
  bbox?: [number, number, number, number];
  /**
   * An object exposing axis-aligned `.bounds` as `[minx, miny, maxx, maxy]`
   * (e.g. a shapely-like geometry). `margin` is applied symmetrically around it.
   */
  roi?: { bounds: [number, number, number, number] };
  /** Window center `[cx, cy]` (used with `size`). */
  center?: [number, number];
  /** Fixed output `[width, height]` (used with `center`). */
  size?: [number, number];
  /** Pixels added around the `roi` bounds on every side. Default `0`. */
  margin?: number;
  /** Fill value for out-of-bounds regions. Default `0`. */
  fill?: Fill;
  /**
   * If `true` (the default), reuse this video's backend as the shared inner so a
   * mosaic of tiles over one file decodes each source frame once; in that case
   * the new tile does NOT own the shared decoder. Maps to
   * `ownsInner = !shareDecode` on {@link CropVideoBackend.wrap}.
   */
  shareDecode?: boolean;
}

/**
 * Resolve a crop region spec into an integer `[x1, y1, x2, y2]` rect.
 *
 * Port of Python `_resolve_crop_rect` (video.py:24-99). Exactly one region spec
 * must be provided: an explicit `crop` rect, a `bbox`, a `roi` (its axis-aligned
 * bounds + `margin`), or a (`center`, `size`) pair for a fixed-shape centered
 * window. Float bounds are rounded OUTWARD (floor of the mins, ceil of the maxs)
 * so the integer rect always *contains* the requested region; the centered
 * window uses `round` so the output shape is exactly `size`. Truncation toward
 * zero is never used for the float paths (the explicit `crop` path matches
 * Python's `int()` truncation).
 *
 * @param crop Explicit crop region `[x1, y1, x2, y2]`, x2/y2 exclusive.
 * @param opts One of `bbox` / `roi` / (`center` + `size`), plus `margin`.
 * @returns An integer crop region `[x1, y1, x2, y2]`, possibly out of bounds.
 * @throws Error If not exactly one region spec is provided, `center`/`size` are
 *   not given together, or the resolved rect is inverted (`x2 < x1`/`y2 < y1`).
 */
export function resolveCropRect(
  crop?: CropRect | null,
  opts: CropOptions = {},
): CropRect {
  const { bbox, roi, center, size, margin = 0 } = opts;

  // `center` and `size` must be given together (both or neither).
  if ((center == null) !== (size == null)) {
    throw new Error(
      "center and size must be provided together for a centered window; " +
        `got center=${JSON.stringify(center)}, size=${JSON.stringify(size)}.`,
    );
  }
  const hasCenterSize = center != null && size != null;
  const nSpecs =
    (crop != null ? 1 : 0) +
    (bbox != null ? 1 : 0) +
    (roi != null ? 1 : 0) +
    (hasCenterSize ? 1 : 0);
  if (nSpecs !== 1) {
    throw new Error(
      "Exactly one of {crop, bbox, roi, (center, size)} must be provided " +
        `to specify a crop region, got ${nSpecs}. For a centered window, ` +
        "pass both center and size.",
    );
  }

  let x1: number;
  let y1: number;
  let x2: number;
  let y2: number;
  if (crop != null) {
    // `int()` truncation toward zero (matches Python's `int(v)`).
    x1 = Math.trunc(crop[0]);
    y1 = Math.trunc(crop[1]);
    x2 = Math.trunc(crop[2]);
    y2 = Math.trunc(crop[3]);
  } else if (bbox != null) {
    const [bx1, by1, bx2, by2] = bbox;
    x1 = Math.floor(bx1);
    y1 = Math.floor(by1);
    x2 = Math.ceil(bx2);
    y2 = Math.ceil(by2);
  } else if (roi != null) {
    const [minx, miny, maxx, maxy] = roi.bounds;
    x1 = Math.floor(minx) - margin;
    y1 = Math.floor(miny) - margin;
    x2 = Math.ceil(maxx) + margin;
    y2 = Math.ceil(maxy) + margin;
  } else {
    // Centered window with fixed output shape.
    const [cx, cy] = center as [number, number];
    const [w, h] = size as [number, number];
    x1 = Math.round(cx - w / 2);
    y1 = Math.round(cy - h / 2);
    x2 = x1 + Math.round(w);
    y2 = y1 + Math.round(h);
  }

  if (x2 < x1 || y2 < y1) {
    throw new Error(
      `Inverted crop rect: x2 (${x2}) < x1 (${x1}) or y2 (${y2}) < y1 (${y1}). ` +
        "Crop bounds must satisfy x2 >= x1 and y2 >= y1.",
    );
  }
  return [x1, y1, x2, y2];
}

/**
 * Why a video's backend could not be opened during load (the backend is then
 * left `null`). Drives the consumer's message/action — e.g. "locate image
 * folder" for an image-sequence, "unsupported format" for `.avi`/`.mpeg`.
 */
export type VideoBackendErrorKind =
  | "image-sequence"
  | "unsupported-format"
  | "decode";

export interface VideoBackendError {
  kind: VideoBackendErrorKind;
  message: string;
}

export class Video {
  /**
   * Where the video is. A URL keeps its credentials here, because {@link exists} and
   * apps go back to it. A save masks them (`maskUrlCredentials`), so a URL that needs
   * them doesn't open from a saved file.
   */
  filename: string | string[];
  backend: VideoBackend | null;
  /** Set when the backend failed to open during load (then `backend` is null). */
  backendError: VideoBackendError | null;
  backendMetadata: Record<string, unknown>;
  sourceVideo: Video | null;
  openBackend: boolean;
  private _embedded: boolean;
  private _shape: [number, number, number, number] | null = null;
  private _fps: number | null = null;
  /** Auth headers persisted from a remote `.slp` load (mirror Python `_url_headers`). */
  private _urlHeaders?: Record<string, string>;
  /** Per-instance TTL cache for {@link exists}. */
  private _existsCache?: { key: string; value: boolean; expiresAt: number };

  constructor(options: {
    filename: string | string[];
    backend?: VideoBackend | null;
    backendError?: VideoBackendError | null;
    backendMetadata?: Record<string, unknown>;
    sourceVideo?: Video | null;
    openBackend?: boolean;
    embedded?: boolean;
  }) {
    this.filename = options.filename;
    this.backend = options.backend ?? null;
    this.backendError = options.backendError ?? null;
    this.backendMetadata = options.backendMetadata ?? {};
    this.sourceVideo = options.sourceVideo ?? null;
    this.openBackend = options.openBackend ?? true;
    this._embedded = options.embedded ?? false;
  }

  get hasEmbeddedImages(): boolean {
    return this._embedded;
  }

  /**
   * Sorted, de-duplicated source frame numbers that have an available image, or
   * `null` when every frame is available (continuous video) or the set is
   * unknown / the backend is closed. For an embedded-image video (`pkg.slp`)
   * the backend exposes the stored `frame_numbers`; callers should treat `null`
   * as "no restriction" (all frames imaged).
   */
  get embeddedFrameIndices(): number[] | null {
    const nums = this.backend?.frameNumbers;
    if (!nums || nums.length === 0) return null;
    return [...new Set(nums)].sort((a, b) => a - b);
  }

  get originalVideo(): Video | null {
    if (!this.sourceVideo) return null;
    let current = this.sourceVideo;
    while (current.sourceVideo) {
      current = current.sourceVideo;
    }
    return current;
  }

  get shape(): [number, number, number, number] | null {
    return (
      this._shape ??
      this.backend?.shape ??
      (this.backendMetadata.shape as
        | [number, number, number, number]
        | undefined) ??
      null
    );
  }

  set shape(value: [number, number, number, number] | null) {
    this._shape = value;
  }

  get fps(): number | null {
    return (
      this._fps ??
      this.backend?.fps ??
      (this.backendMetadata.fps as number | undefined) ??
      null
    );
  }

  set fps(value: number | null) {
    this._fps = value;
  }

  async getFrame(
    frameIndex: number,
    opts?: GetFrameOptions,
  ): Promise<VideoFrame | null> {
    if (!this.backend) return null;
    return this.backend.getFrame(frameIndex, opts);
  }

  /**
   * Raw stored encoded blob for `frameNumber` on an embedded video, WITHOUT
   * decoding — mirrors Python `HDF5Video.get_frame_raw_bytes`. Returns null for
   * a continuous video, a closed backend, or an unstored frame.
   */
  async getFrameBuffer(frameNumber: number): Promise<Uint8Array | null> {
    if (!this.backend?.getFrameBuffer) return null;
    return this.backend.getFrameBuffer(frameNumber);
  }

  async getFrameTimes(): Promise<number[] | null> {
    if (!this.backend?.getFrameTimes) return null;
    return this.backend.getFrameTimes();
  }

  close(): void {
    this.backend?.close();
  }

  /**
   * Persist the remote auth headers from a remote `.slp` load so later
   * {@link exists} probes (and any URL-backed backend) stay authenticated.
   * Mirrors Python's `HDF5Video._url_headers`. Internal — set by the SLP reader.
   * @internal
   */
  _setUrlPersistence(persistence: { headers?: Record<string, string> }): void {
    if (persistence.headers) this._urlHeaders = persistence.headers;
  }

  /**
   * The auth headers to use for remote requests on this video, preferring the
   * value persisted on this `Video`, then any persisted on the backend. Mirrors
   * Python `Video._backend_url_headers`.
   * @internal
   */
  _backendUrlHeaders(): Record<string, string> | undefined {
    return (
      this._urlHeaders ??
      // Python-parity placeholder: no JS backend currently sets `_urlHeaders`,
      // so this fallback is inert today. Kept so a future URL-backed backend
      // that carries its own headers (as Python's HDF5Video does) is honored
      // without changing this accessor.
      (this.backend as { _urlHeaders?: Record<string, string> } | null)
        ?._urlHeaders
    );
  }

  /**
   * Non-throwing check that this video's source is reachable.
   *
   * For a URL filename, probes with HEAD (falling back to a `Range: bytes=0-0`
   * GET), forwarding any persisted auth headers, and caches the result
   * per-instance for {@link EXISTS_TTL_MS}. Never throws (a probe failure → a
   * cached `false`). For a non-URL (local/embedded) filename, this returns
   * `true` only when a backend is present (the JS port has no generic
   * filesystem stat); callers needing real local existence should use a Node
   * file check.
   *
   * Port of Python `Video.exists`.
   */
  async exists(): Promise<boolean> {
    const filename = this.filename;
    // Image sequences / non-URL paths: presence is implied by an open backend.
    if (Array.isArray(filename) || !isUrl(filename)) {
      return this.backend != null;
    }

    const dataset = (this.backendMetadata?.dataset as string | undefined) ?? "";
    const key = `${filename} ${dataset}`;
    const now = Date.now();
    if (
      this._existsCache &&
      this._existsCache.key === key &&
      this._existsCache.expiresAt > now
    ) {
      return this._existsCache.value;
    }

    const value = await headOrRangeProbe(filename, {
      headers: this._backendUrlHeaders(),
    });
    this._existsCache = { key, value, expiresAt: now + EXISTS_TTL_MS };
    return value;
  }

  /**
   * Return a virtual, on-read cropped view of this video.
   *
   * Port of Python `Video.crop` (video.py:304-389). Exactly one region spec must
   * be given: an explicit `crop` rect (the positional argument), or one of
   * `bbox` / `roi` / (`center` + `size`) via {@link CropOptions}. The returned
   * `Video` shares no pixels with this one; frames are decoded on read and
   * cropped, with out-of-bounds regions pad-filled with `fill` (never clamped),
   * so the output shape is always exactly `[y2 - y1, x2 - x1]`.
   *
   * The crop composes (FLATTENS when fills agree and the region is in-bounds)
   * with any existing crop via {@link CropVideoBackend.wrap}. `sourceVideo` is
   * set to this video for provenance, and `backendMetadata` is seeded with the
   * cropped `shape`, the uncropped `source_shape`, the composed `crop` rect, and
   * `crop_fill` so a closed re-serialize and a close/open round-trip keep the
   * crop. When `shareDecode` (the default), the new crop reuses this video's
   * backend as the shared inner (the new tile does NOT own the decoder).
   *
   * @param crop Explicit crop region `[x1, y1, x2, y2]`, x2/y2 exclusive.
   * @param opts One of `bbox` / `roi` / (`center` + `size`), plus `margin`,
   *   `fill`, and `shareDecode`.
   * @returns A new `Video` exposing the cropped view.
   * @throws Error If there is no backend to crop, or the region spec is invalid.
   */
  crop(crop?: CropRect | null, opts: CropOptions = {}): Video {
    const rect = resolveCropRect(crop, opts);
    if (this.backend == null) {
      throw new Error(
        "Cannot crop a video with no open backend. Provide a backend (the JS " +
          "port has no filesystem auto-open) before cropping.",
      );
    }
    const fill: Fill = opts.fill ?? 0;
    const shareDecode = opts.shareDecode ?? true;

    // Canonical wrap order is Grayscale(Crop(rawInner)): if this video's
    // backend is already grayscale-wrapped, unwrap it, crop the TRUE inner,
    // then re-wrap grayscale on the outside. Cropping is spatial and
    // grayscale-forcing is channel-only — the two operations commute — but a
    // single canonical order avoids ever nesting Crop(Grayscale(...)), which
    // would otherwise let the two wrappers be composed in either order
    // depending on call sequence.
    const grayscaleLayer =
      this.backend instanceof GrayscaleVideoBackend ? this.backend : null;
    const inner = grayscaleLayer ? grayscaleLayer.inner : this.backend;
    const croppedBackend = CropVideoBackend.wrap({
      inner,
      crop: rect,
      fill,
      ownsInner: !shareDecode,
    });
    const finalBackend = grayscaleLayer
      ? GrayscaleVideoBackend.wrap({
          inner: croppedBackend,
          grayscale: grayscaleLayer.grayscale,
        })
      : croppedBackend;

    const [x1, y1, x2, y2] = croppedBackend.crop;
    const srcShape = this.shape;
    const cropped = new Video({
      filename: this.filename,
      backend: finalBackend,
      sourceVideo: this,
      openBackend: this.openBackend,
    });
    cropped.backendMetadata = {
      ...this.backendMetadata,
      shape:
        srcShape != null
          ? ([srcShape[0], y2 - y1, x2 - x1, srcShape[3]] as [
              number,
              number,
              number,
              number,
            ])
          : null,
      // The uncropped source shape, so a closed re-serialize keeps videos_json
      // describing the full frame even without a live sourceVideo.
      source_shape: srcShape != null ? [...srcShape] : null,
      // COMPOSED source rect from wrap: keeps open/closed crop keys identical and
      // root-canonical, and survives close()->open().
      crop: [...croppedBackend.crop],
      crop_fill: croppedBackend.fill,
    };
    return cropped;
  }

  /**
   * Crop a `Video` and return a virtual cropped view.
   *
   * Port of Python `Video.from_crop` (video.py:391-440). Accepts the same region
   * specs as {@link crop}. Unlike Python (which can open a path via
   * `from_filename`), the JS port has no generic filesystem-backed open facade,
   * so `video` must already be a `Video` with a backend; passing a path string
   * throws.
   *
   * @param video An existing `Video` to crop.
   * @param crop Explicit crop region `[x1, y1, x2, y2]`, x2/y2 exclusive.
   * @param opts One of `bbox` / `roi` / (`center` + `size`), plus `margin`,
   *   `fill`, and `shareDecode`.
   * @returns A new `Video` exposing the cropped view.
   * @throws Error If `video` is a path string (unsupported in the JS port).
   */
  static fromCrop(
    video: Video | string,
    crop?: CropRect | null,
    opts: CropOptions = {},
  ): Video {
    if (typeof video === "string") {
      throw new Error(
        "Video.fromCrop does not support opening a path string in the JS port " +
          "(there is no filesystem auto-open). Construct a Video with a backend " +
          "first, then call Video.fromCrop(video, ...) or video.crop(...).",
      );
    }
    return video.crop(crop, opts);
  }

  /**
   * Return this video's crop rect `[x1, y1, x2, y2]` or `null`.
   *
   * Port of Python `Video._crop_tuple` (video.py:442-454). Reads `backend.crop`
   * when the backend is a {@link CropVideoBackend} (open path), else
   * `backendMetadata.crop` (closed path), else `null` (uncropped).
   */
  _cropTuple(): CropRect | null {
    if (this.backend instanceof CropVideoBackend) {
      return [...this.backend.crop] as CropRect;
    }
    const crop = this.backendMetadata.crop;
    return crop != null ? ([...(crop as number[])] as CropRect) : null;
  }

  /**
   * Return this video's crop fill value (open: backend; closed: metadata).
   *
   * Port of Python `Video._crop_fill` (video.py:456-465). Returns `0` for an
   * uncropped video.
   */
  _cropFill(): Fill {
    if (this.backend instanceof CropVideoBackend) {
      return this.backend.fill;
    }
    const fill = this.backendMetadata.crop_fill;
    return (fill as Fill | undefined) ?? 0;
  }

  /** Whether this video is a virtual crop of another video. */
  get isCropped(): boolean {
    return this._cropTuple() !== null;
  }

  /** Crop rect `[x1, y1, x2, y2]` in source coords, or `null` if uncropped. */
  get cropRect(): CropRect | null {
    return this._cropTuple();
  }

  /** The out-of-bounds fill value for this video's crop (`0` if uncropped). */
  get cropFill(): Fill {
    return this._cropFill();
  }

  /**
   * Map source-frame `(x, y)` coordinates into this video's cropped frame.
   *
   * Port of Python `Video.to_crop_coords` (video.py:482-494). If this video is
   * not cropped, a copy of `points` is returned unchanged (NaN preserved).
   * Accepts a flat interleaved buffer or an array of `[x, y]` pairs and returns
   * the same kind.
   */
  toCropCoords<T extends FlatPoints>(points: T): T;
  toCropCoords(points: PointPairs): [number, number][];
  toCropCoords(
    points: FlatPoints | PointPairs,
  ): FlatPoints | [number, number][] {
    const crop = this._cropTuple();
    if (crop === null) {
      return copyPoints(points);
    }
    return cropPoints(points as FlatPoints, crop);
  }

  /**
   * Map cropped-frame `(x, y)` coordinates back to source-frame coordinates.
   *
   * Port of Python `Video.to_source_coords` (video.py:496-510). Inverse of
   * {@link toCropCoords}. If this video is not cropped, a copy of `points` is
   * returned unchanged (NaN preserved).
   */
  toSourceCoords<T extends FlatPoints>(points: T): T;
  toSourceCoords(points: PointPairs): [number, number][];
  toSourceCoords(
    points: FlatPoints | PointPairs,
  ): FlatPoints | [number, number][] {
    const crop = this._cropTuple();
    if (crop === null) {
      return copyPoints(points);
    }
    return uncropPoints(points as FlatPoints, crop);
  }

  /**
   * Check if this video has the same path as another video.
   *
   * Port of Python `Video.matches_path` (video.py:637-715). The public default
   * is kept at `strict = true` (DECISIONS D1) because every merge/match call
   * site passes `strict` explicitly, so the default is never load-bearing for
   * parity; the LOGIC below mirrors Python exactly.
   *
   * @param other - Another video to compare with.
   * @param strict - If `true`, require an exact (posix-normalized) path match.
   *   If `false`, consider videos with the same basename as matching.
   */
  matchesPath(other: Video, strict = true): boolean {
    // HDF5 backends: prioritize source_filename matching since multiple videos
    // can share the same HDF5 file path but reference different source videos.
    const selfIsHdf5 = isHdf5Video(this);
    const otherIsHdf5 = isHdf5Video(other);

    if (selfIsHdf5 && otherIsHdf5) {
      const selfSource = hdf5SourceFilename(this);
      const otherSource = hdf5SourceFilename(other);
      const selfDataset = hdf5Dataset(this);
      const otherDataset = hdf5Dataset(other);

      // If both have datasets, they must match.
      if (selfDataset !== null && otherDataset !== null) {
        if (selfDataset !== otherDataset) {
          return false; // Different datasets = different videos
        }
      }

      // If both have source_filenames, compare them.
      if (selfSource !== null && otherSource !== null) {
        if (strict) {
          return toPosix(selfSource) === toPosix(otherSource);
        }
        return basename(selfSource) === basename(otherSource);
      }

      // If only datasets available (no source_filename), they must match.
      if (selfDataset !== null && otherDataset !== null) {
        return selfDataset === otherDataset;
      }

      // Neither source_filename nor dataset available: cannot match.
      return false;
    }

    const selfIsList = Array.isArray(this.filename);
    const otherIsList = Array.isArray(other.filename);

    if (selfIsList && otherIsList) {
      // Both are image sequences.
      const selfList = this.filename as string[];
      const otherList = other.filename as string[];
      if (strict) {
        // Exact, order-sensitive list equality.
        return arraysEqual(selfList, otherList);
      }
      // Compare basenames (order-sensitive list of basenames).
      return arraysEqual(selfList.map(basename), otherList.map(basename));
    }

    if (selfIsList || otherIsList) {
      // One is an image sequence, the other a single file.
      return false;
    }

    // Both are single files.
    const selfName = this.filename as string;
    const otherName = other.filename as string;
    if (strict) {
      // Deterministic posix-string comparison (no Node `path`, no FS access:
      // the AUTO cascade's `isSameFile` handles symlink/inode resolution).
      return toPosix(selfName) === toPosix(otherName);
    }
    return basename(selfName) === basename(otherName);
  }

  /**
   * Check if this video has the same content as another video.
   *
   * Port of Python `Video.matches_content` (video.py:717-742). Compares the
   * FULL 4-tuple shape (frames, height, width, channels) and the backend type
   * name, NOT actual frame data.
   *
   * @param other - Another video to compare with.
   * @returns `true` if the videos have the same shape and backend type.
   */
  matchesContent(other: Video): boolean {
    // Compare shapes (full tuple including frames and channels).
    if (!shapeTupleEqual(this.shape, other.shape)) {
      return false;
    }

    // Compare backend presence/type.
    if (this.backend === null && other.backend === null) {
      return true;
    }
    if (this.backend === null || other.backend === null) {
      return false;
    }

    return backendTypeName(this) === backendTypeName(other);
  }

  /**
   * Check if this video has the same shape as another video.
   *
   * Port of Python `Video.matches_shape` (video.py:744-772). Compares only
   * height, width, and channels (INCLUDING channels, EXCLUDING frames).
   *
   * @param other - Another video to compare with.
   * @returns `true` if the videos have the same height, width, and channels.
   */
  matchesShape(other: Video): boolean {
    // Prefer backendMetadata["shape"] when backend is null but the key is
    // present (real key-presence check, not truthiness).
    const selfShape =
      this.backend === null && hasOwn(this.backendMetadata, "shape")
        ? (this.backendMetadata.shape as
            | [number, number, number, number]
            | null
            | undefined)
        : this.shape;
    const otherShape =
      other.backend === null && hasOwn(other.backendMetadata, "shape")
        ? (other.backendMetadata.shape as
            | [number, number, number, number]
            | null
            | undefined)
        : other.shape;

    if (selfShape == null || otherShape == null) {
      return false;
    }

    // Compare only height, width, channels (indices 1..3).
    return (
      selfShape.length === otherShape.length &&
      selfShape[1] === otherShape[1] &&
      selfShape[2] === otherShape[2] &&
      selfShape[3] === otherShape[3]
    );
  }

  /**
   * Check if this video has overlapping images with another video.
   *
   * Port of Python `Video.has_overlapping_images` (video.py:774-799). Only
   * meaningful for image sequences (list filenames); compares basenames.
   *
   * @param other - Another video to compare with.
   * @returns `true` if both are image sequences with at least one shared
   *   image basename, `false` otherwise.
   */
  hasOverlappingImages(other: Video): boolean {
    if (!Array.isArray(this.filename) || !Array.isArray(other.filename)) {
      return false;
    }

    const selfBasenames = new Set((this.filename as string[]).map(basename));
    for (const f of other.filename as string[]) {
      if (selfBasenames.has(basename(f))) {
        return true;
      }
    }
    return false;
  }

  /**
   * Whether this video is grayscale, or `null` if unknown.
   *
   * Port of Python `Video.grayscale` getter (video.py:225-239): if the shape is
   * known, grayscale is `shape[-1] === 1`; otherwise fall back to a stored
   * `backendMetadata["grayscale"]` value (real key-presence, so a stored `null`
   * is returned as-is), else `null`. Used by `deduplicateWith` / `mergeWith` to
   * carry the grayscale hint onto the newly created video.
   */
  get grayscale(): boolean | null {
    const shape = this.shape;
    if (shape != null) {
      return shape[shape.length - 1] === 1;
    }
    if (hasOwn(this.backendMetadata, "grayscale")) {
      return (this.backendMetadata.grayscale as boolean | null) ?? null;
    }
    return null;
  }

  /**
   * Force (or release the forcing of) this video's grayscale state.
   *
   * Port of Python `Video.grayscale` setter (video.py): rebuilds the backend
   * as a {@link GrayscaleVideoBackend} wrapping the current TRUE inner (any
   * existing grayscale layer is unwrapped and replaced, never nested — see
   * {@link GrayscaleVideoBackend.wrap}), and persists the value into
   * {@link backendMetadata} so a closed/reopened video (which reads
   * `backendMetadata.shape`/`backendMetadata.grayscale` rather than a live
   * backend) still reports it — the same persistence pattern {@link crop}
   * already uses for `crop`/`crop_fill`/`shape`.
   *
   * Matches Python's setter signature (`bool`, not the tri-state
   * `bool | null` the constructor/{@link createVideoBackend} accept):
   * autodetection is a construction-time choice, not something re-requested
   * through this property.
   *
   * @throws Error If this video has no open backend (mirrors {@link crop},
   *   which has the same requirement — the JS port has no filesystem
   *   auto-open to fall back to).
   */
  set grayscale(value: boolean) {
    if (this.backend == null) {
      throw new Error(
        "Cannot set grayscale on a video with no open backend. Provide a " +
          "backend (the JS port has no filesystem auto-open) before setting " +
          "grayscale.",
      );
    }
    const inner =
      this.backend instanceof GrayscaleVideoBackend
        ? this.backend.inner
        : this.backend;
    this.backend = GrayscaleVideoBackend.wrap({ inner, grayscale: value });
    this.backendMetadata = { ...this.backendMetadata, grayscale: value };
    if (hasOwn(this.backendMetadata, "shape")) {
      const shape = this.backendMetadata.shape as
        | [number, number, number, number]
        | null;
      if (shape != null) {
        this.backendMetadata.shape = [
          shape[0],
          shape[1],
          shape[2],
          value ? 1 : (inner.shape?.[3] ?? shape[3]),
        ];
      }
    }
  }

  /**
   * Create a new video with duplicate images removed.
   *
   * Port of Python `Video.deduplicate_with` (video.py:801-840). Specific to
   * image-sequence videos (ImageVideo: `filename` is a list). Images are
   * considered duplicates when they share a basename. The returned video
   * contains only the images from THIS video whose basename is not present in
   * `other`, preserving this video's order.
   *
   * Return contract (matches Python exactly): returns `null` when ALL of this
   * video's images are duplicates (Python returns `None`); otherwise returns a
   * NEW `Video` (never `this`, never `other`) carrying the surviving image paths.
   *
   * @param other - Another image-sequence video to deduplicate against.
   * @returns A new `Video` with the non-duplicate images, or `null` if every
   *   image was a duplicate.
   * @throws Error - If either video's `filename` is not a list (ImageVideo).
   */
  deduplicateWith(other: Video): Video | null {
    if (!Array.isArray(this.filename)) {
      throw new Error("deduplicate_with only works with ImageVideo backends");
    }
    if (!Array.isArray(other.filename)) {
      throw new Error("Other video must also be ImageVideo backend");
    }

    // Basenames present in the other video.
    const otherBasenames = new Set((other.filename as string[]).map(basename));

    // Keep only this video's images whose basename is not a duplicate.
    const deduplicatedPaths = (this.filename as string[]).filter(
      (f) => !otherBasenames.has(basename(f)),
    );

    if (deduplicatedPaths.length === 0) {
      // All images were duplicates.
      return null;
    }

    return makeImageSequenceVideo(deduplicatedPaths, this.grayscale);
  }

  /**
   * Merge another video's images into this one.
   *
   * Port of Python `Video.merge_with` (video.py:842-883). Specific to
   * image-sequence videos (ImageVideo: `filename` is a list). Returns a NEW
   * `Video` containing all unique images (by basename) from both videos,
   * preserving order: every unique image from THIS video first, then any image
   * from `other` whose basename has not already been seen.
   *
   * @param other - Another image-sequence video to merge with.
   * @returns A new `Video` with the de-duplicated union of both videos' images.
   * @throws Error - If either video's `filename` is not a list (ImageVideo).
   */
  mergeWith(other: Video): Video {
    if (!Array.isArray(this.filename)) {
      throw new Error("merge_with only works with ImageVideo backends");
    }
    if (!Array.isArray(other.filename)) {
      throw new Error("Other video must also be ImageVideo backend");
    }

    // All unique images by basename, preserving order (self first, then other).
    const seenBasenames = new Set<string>();
    const mergedPaths: string[] = [];

    for (const path of this.filename as string[]) {
      const name = basename(path);
      if (!seenBasenames.has(name)) {
        mergedPaths.push(path);
        seenBasenames.add(name);
      }
    }

    for (const path of other.filename as string[]) {
      const name = basename(path);
      if (!seenBasenames.has(name)) {
        mergedPaths.push(path);
        seenBasenames.add(name);
      }
    }

    return makeImageSequenceVideo(mergedPaths, this.grayscale);
  }
}

/**
 * Construct a new image-sequence `Video` from a list of image paths, carrying a
 * grayscale hint.
 *
 * Stands in for Python `Video.from_filename(paths, grayscale=...)` (video.py:153-201)
 * as used by `deduplicate_with` / `merge_with`. Python's `from_filename` opens a
 * VideoBackend (which re-derives `filename` from the backend) and forwards the
 * grayscale flag to it. JS has no generic backend-opening path that works without
 * filesystem/image I/O, so we DEGRADE GRACEFULLY: build a plain `Video` whose
 * `filename` is the (unchanged) path list and persist the grayscale hint into
 * `backendMetadata` (where a real backend would expose it). `backend` stays `null`
 * and `openBackend` is `false` so no I/O is implied. The merge STEP-2
 * IMAGE_DEDUP / SHAPE callers only consume the resulting `filename`/basenames, so
 * this is behaviorally faithful for the matching subsystem.
 */
function makeImageSequenceVideo(
  paths: string[],
  grayscale: boolean | null,
): Video {
  const backendMetadata: Record<string, unknown> = {};
  if (grayscale != null) {
    backendMetadata.grayscale = grayscale;
  }
  return new Video({
    filename: paths,
    backend: null,
    backendMetadata,
    openBackend: false,
  });
}

/**
 * Copy a coordinate buffer/pair-array without mutating the input, preserving its
 * concrete kind (typed array subtype, plain number[], or array-of-pairs). Used
 * by {@link Video.toCropCoords}/{@link Video.toSourceCoords} for the uncropped
 * passthrough (Python `points.copy()`).
 */
function copyPoints(
  points: FlatPoints | PointPairs,
): FlatPoints | [number, number][] {
  if (Array.isArray(points)) {
    if (points.length > 0 && Array.isArray(points[0])) {
      return (points as unknown as PointPairs).map(
        ([x, y]) => [x, y] as [number, number],
      );
    }
    return (points as number[]).slice();
  }
  return (points as Float64Array | Float32Array).slice();
}

/** Final path component, splitting on BOTH "/" and "\\" (cross-platform). */
function basename(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1];
}

/**
 * Deterministic posix-string normalization (NOT Node `path`, which is
 * OS-dependent). Converts backslashes to forward slashes, collapses repeated
 * slashes, and drops a single trailing slash (but preserves a lone root "/").
 */
function toPosix(path: string): string {
  let p = path.replace(/\\/g, "/");
  p = p.replace(/\/{2,}/g, "/");
  if (p.length > 1 && p.endsWith("/")) {
    p = p.slice(0, -1);
  }
  return p;
}

/** Element-wise, order-sensitive string array equality. */
function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/** Component-wise equality of two (possibly null) 4-tuple shapes. */
function shapeTupleEqual(
  a: [number, number, number, number] | null,
  b: [number, number, number, number] | null,
): boolean {
  if (a === null || b === null) return a === b;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/** Real key-presence check (a stored `null`/`undefined` still counts as present). */
function hasOwn(obj: Record<string, unknown>, key: string): boolean {
  return Object.hasOwn(obj, key);
}

/**
 * The dataset path for an HDF5/embedded video, or `null`. Mirrors Python
 * `getattr(backend, "dataset", None)`: prefer the live backend, then the
 * persisted `backendMetadata.dataset`.
 */
function hdf5Dataset(video: Video): string | null {
  // A non-HDF5 backend (e.g. MediaVideo from a `.slp`) carries an EMPTY dataset
  // placeholder (`""`). Treat empty as "no dataset" so such videos are not
  // misclassified as HDF5 and then matched to one another on equal-empty
  // datasets (`"" === ""`) — which made any two plain `.mp4` videos "match".
  const fromBackend = video.backend?.dataset;
  if (typeof fromBackend === "string" && fromBackend !== "") return fromBackend;
  const fromMeta = video.backendMetadata.dataset;
  return typeof fromMeta === "string" && fromMeta !== "" ? fromMeta : null;
}

/**
 * JS analog of `isinstance(backend, HDF5Video)`. JS has no formal HDF5 backend
 * class; an HDF5/embedded video is signaled by the presence of a `dataset`
 * (on the backend or in backendMetadata).
 */
function isHdf5Video(video: Video): boolean {
  return hdf5Dataset(video) !== null;
}

/**
 * The source filename for an HDF5/embedded video (Python
 * `HDF5Video.source_filename`), or `null`. In JS this is the embedded source
 * video's scalar filename.
 */
function hdf5SourceFilename(video: Video): string | null {
  const fn = video.sourceVideo?.filename;
  return typeof fn === "string" ? fn : null;
}

/**
 * Backend type name for content comparison. Mirrors Python
 * `type(backend).__name__` using the JS backend class (constructor) name.
 */
function backendTypeName(video: Video): string {
  return video.backend?.constructor?.name ?? "";
}
