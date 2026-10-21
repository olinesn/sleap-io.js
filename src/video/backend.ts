import type { CropRect } from "../transform/points.js";
import type { Fill, RawFrame } from "../transform/frame.js";

export type VideoFrame = ImageData | ImageBitmap | Uint8Array | ArrayBuffer;

/** Per-call options for {@link VideoBackend.getFrame}. */
export interface GetFrameOptions {
  /**
   * When `false`, suppress this backend's read-ahead prefetch for this call.
   * Read-ahead helps sequential playback but is wasted (and saturates I/O) while
   * scrubbing, where the caller jumps past the prefetched frames. Backends with
   * no prefetch ignore it. Defaults to `true` (prefetch enabled).
   */
  prefetch?: boolean;
  /**
   * Optional cancellation signal. Backends that decode asynchronously (e.g. the
   * MP4 backend) bail early when it aborts; backends that don't ignore it.
   */
  signal?: AbortSignal;
  /**
   * Scrub mode: decode only what's needed to show THIS frame (keyframe→target),
   * skipping the forward read-ahead/lookahead window. During a seekbar drag the
   * lookahead frames are dragged past and never seen, so decoding them just slows
   * the frame the user is actually on. Backends without a lookahead ignore it.
   */
  scrub?: boolean;
  /**
   * Pinned read: never superseded. Backends that drop stale reads when a newer
   * one arrives (the mp4/AVI decoders keep only the newest read so scrubbing
   * stays responsive) neither cancel this read for a newer one nor let it
   * cancel older ones. For batch readers that need every frame, e.g. embedding
   * frames into a `.pkg.slp` while the app keeps showing other frames.
   */
  pinned?: boolean;
}

/**
 * A lazy, seekable byte source for a video file — the video counterpart of the
 * HDF5 streaming reader's range source. Lets a backend read only the byte ranges
 * it needs (the container index + the samples for the frames being viewed)
 * instead of materializing the whole file in memory.
 *
 * The canonical use is desktop (Tauri), where the WebView has no lazy disk-backed
 * `File` for a raw path: `readRange` is backed by a native `read_range` command
 * so a multi-GB video never has to be read whole into RAM. Structurally mirrors
 * the HDF5 `RangeSource` used by `readSlpStreaming`.
 */
export interface RangeSource {
  /** Total file size in bytes. */
  size: number;
  /** Read `[offset, offset + length)`; may return fewer bytes at EOF. */
  readRange: (offset: number, length: number) => Promise<Uint8Array>;
}

/**
 * True for a {@link RangeSource} — distinguishes it from a `string` URL / `File`
 * / `Blob`. A `Blob` also has a numeric `size`, so the `readRange` function is
 * the discriminator.
 */
export function isRangeSource(source: unknown): source is RangeSource {
  return (
    typeof source === "object" &&
    source !== null &&
    typeof (source as RangeSource).size === "number" &&
    typeof (source as RangeSource).readRange === "function"
  );
}

export interface VideoBackend {
  filename: string | string[];
  shape?: [number, number, number, number];
  fps?: number;
  dataset?: string | null;
  /**
   * Embedded-image (HDF5 / `pkg.slp`) backends: the source frame numbers that
   * have a stored image, in storage order. Left unset by continuous-video
   * backends (mp4 / seq / image-sequence), where every frame is decodable.
   */
  frameNumbers?: number[];
  getFrame(
    frameIndex: number,
    opts?: GetFrameOptions,
  ): Promise<VideoFrame | null>;
  getFrameTimes?(): Promise<number[] | null>;
  /**
   * Optional cheap liveness probe: resolve to `true` if the backend can read its
   * first frame's SOURCE bytes (no decode), else `false` (never throws). Used to
   * detect a backend built from stored metadata (an `ImageVideo` given a `shape`
   * skips the up-front decode) whose media is not actually reachable, so a
   * consumer can offer a locate/repair affordance instead of rendering blanks.
   * Backends that always read on `getFrame` (or verify at construction) may omit
   * this. See issue #213.
   */
  probeFirstFrame?(): Promise<boolean>;
  /**
   * Optional crop pushdown hook (Item 1 of JS issue #153, mirroring Python
   * `read_crop`, `video_reading.py:1647`).
   *
   * When a backend can read *only* the requested crop region directly from
   * storage — e.g. a raw rank-4 chunked HDF5 pixel array via an N-D hyperslab —
   * it implements this to return a `(y2-y1) x (x2-x1) x C` {@link RawFrame}
   * that is **byte-identical** to `cropFrame(fullFrame, crop, fill)` (same pad
   * value, same clamp arithmetic). Returning `null` signals "I cannot push this
   * down — fall back to a full decode + `cropFrame`". It must never throw for
   * out-of-bounds crops.
   *
   * Backends that store opaque encoded blobs (PNG/JPEG) or per-frame-indexed
   * rows (the embedded `pkg.slp` case) cannot spatially hyperslab a frame and
   * always return `null`. The JS port ships no raw rank-4 HDF5 video backend
   * today, so this is a no-op on every current fixture; the hook keeps the
   * architecture aligned with Python and leaves the fast path open.
   */
  readCrop?(
    frameIndex: number,
    crop: CropRect,
    fill: Fill,
  ): Promise<RawFrame | null>;
  /**
   * Embedded-image (pkg.slp) backends: return the RAW stored encoded bytes
   * (PNG/JPEG blob, or raw-pixel row) for source frame number `frameNumber`,
   * WITHOUT decoding — the JS analog of Python `HDF5Video.get_frame_raw_bytes`.
   * Returns null if this backend has no stored image for that frame or cannot
   * provide raw bytes. Continuous-video backends omit this method.
   */
  getFrameBuffer?(frameNumber: number): Promise<Uint8Array | null>;
  /** Encoded format of stored blobs ("png"|"jpg"|"jpeg"|raw). Embedded only. */
  readonly embeddedFormat?: string;
  /** Channel order of stored blobs ("RGB"|"BGR"). Embedded only. */
  readonly embeddedChannelOrder?: string;
  /**
   * Deferred (lazyVideoMetadata) backends: fetch per-video metadata skipped at
   * load. Idempotent; callers await it before reading frameNumbers. No-op /
   * omitted when everything is already loaded.
   */
  ensureLoaded?(): Promise<void>;
  /**
   * Optional playback helper: proactively decode a run of frames AHEAD of
   * `fromFrame` into the backend's cache so sequential playback finds cache hits
   * instead of blocking on a decode. Fire-and-forget (returns immediately);
   * no-ops when the runway is already decoded or ahead-work is already in flight.
   * A demand {@link getFrame} always preempts it. Backends without an internal
   * decode cache (or that decode synchronously) omit this. See scrub-proxy v2.
   */
  decodeAhead?(fromFrame: number, opts?: GetFrameOptions): void;
  /**
   * Optional: the frame index of the keyframe at or before `frameIndex` — the
   * cheapest frame to show near it (a lone I-frame decode, no delta chain). The
   * scrub UI snaps to this during a fast drag so the preview keeps up. Backends
   * without keyframes (image sequences, where every frame is independent) omit
   * this. See scrub-proxy v2 keyframe-preview.
   */
  nearestKeyframe?(frameIndex: number): number;
  close(): void;
}
