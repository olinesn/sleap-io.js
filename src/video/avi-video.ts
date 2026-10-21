/**
 * AVI Video Backend
 *
 * Frame-accurate `.avi` (and `.wmv`) decoding in the browser / Tauri WebView,
 * with NO external ffmpeg install. The browser has no AVI demuxer and WebCodecs
 * cannot demux containers, so this uses `web-demuxer` (a slim ffmpeg-wasm
 * demuxer) to pull the encoded stream out of the AVI, then decodes it with the
 * platform's own decoders:
 *
 *   - H.264 / H.265 / VP8 / VP9 / AV1  → WebCodecs `VideoDecoder`
 *   - MJPEG (SLEAP's NWB / JABS annotation-frame container) → `ImageDecoder`
 *     via `createImageBitmap` per frame (each frame is an independent JPEG)
 *
 * Codecs WebCodecs cannot decode (Xvid/DivX = MPEG-4 ASP, WMV3/VC-1, raw) are
 * rejected at `initialize()` with a clear "transcode to H.264" error — the
 * factory surfaces it as {@link UnsupportedVideoFormatError} for the app.
 *
 * ── Frame-accurate seeking ────────────────────────────────────────────────────
 * To read frame N, web-demuxer does a BACKWARD seek to the keyframe ≤ N's time
 * and streams the GOP forward; we decode it and map each decoded frame back to
 * an index by `round(frame.timestamp * fps)`. Frames arrive in DECODE order
 * (B-frame reorder ≠ presentation order), so we key by timestamp, not arrival
 * order (verified in the spike: seeking to frame 7 returns byte-identical pixels
 * to a sequential decode). AVI is constant-frame-rate, so `time(i) = i / fps`.
 *
 * ── WASM delivery ─────────────────────────────────────────────────────────────
 * This module ships NO wasm (mirrors {@link file://./libav-h264-decoder.ts}).
 * The embedding app vendors web-demuxer's `.wasm` and points
 * `configureWebDemuxer({ wasmFilePath })` at it before opening a video.
 */

import { WebDemuxer, AVSeekFlag } from "web-demuxer";
import type { RangeSource, VideoBackend, VideoFrame } from "./backend.js";

// ── Configuration seam (set by the app) ──────────────────────────────────────

export interface WebDemuxerConfig {
  /** Absolute URL of web-demuxer's `.wasm` (fetched inside its worker). */
  wasmFilePath: string;
}

let webDemuxerConfig: WebDemuxerConfig | null = null;

/** Point the AVI backend at web-demuxer's vendored wasm. Call before opening a video. */
export function configureWebDemuxer(config: WebDemuxerConfig): void {
  webDemuxerConfig = { wasmFilePath: config.wasmFilePath };
}

/** True once {@link configureWebDemuxer} has been called. */
export function isWebDemuxerConfigured(): boolean {
  return webDemuxerConfig !== null;
}

// ── Capability probes (call-time, so tests can install globals post-import) ───

function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof document !== "undefined";
}

function hasWebCodecs(): boolean {
  return (
    typeof VideoDecoder !== "undefined" &&
    typeof EncodedVideoChunk !== "undefined"
  );
}

/** web-demuxer `codec_name`s that MJPEG maps to (decode each frame as a JPEG). */
const MJPEG_CODECS = new Set(["mjpeg", "mjpg", "jpeg"]);

/** How many frames past the target to also decode + cache (forward scrub window). */
const FORWARD_PREFETCH = 8;

export interface AviVideoOptions {
  cacheSize?: number;
}

export class AviVideoBackend implements VideoBackend {
  filename: string | string[];
  shape?: [number, number, number, number];
  fps?: number;
  dataset?: string | null = null;

  private demuxer: WebDemuxer | null = null;
  private mode: "webcodecs" | "mjpeg" = "webcodecs";
  private decoderConfig: VideoDecoderConfig | null = null;
  private cache: Map<number, ImageBitmap> = new Map();
  private cacheSize: number;
  private frameCount = 0;
  /** Serializes decodes so we never run two `VideoDecoder`s at once. */
  private decodeQueue: Promise<unknown> = Promise.resolve();
  /** Newest requested frame — lets an in-flight decode bail during a scrub. */
  private latestRequested = -1;

  constructor(filename: string | string[], options: AviVideoOptions = {}) {
    this.filename = filename;
    this.cacheSize = options.cacheSize ?? 120;
  }

  static async fromBlob(
    blob: Blob,
    filename: string,
    options?: AviVideoOptions,
  ): Promise<AviVideoBackend> {
    const backend = new AviVideoBackend(filename, options);
    const file =
      typeof File !== "undefined" && blob instanceof File
        ? blob
        : new File([blob], filename);
    await backend.initialize(file);
    return backend;
  }

  static async fromUrl(
    url: string,
    options?: AviVideoOptions,
  ): Promise<AviVideoBackend> {
    const backend = new AviVideoBackend(url, options);
    await backend.initialize(url);
    return backend;
  }

  /**
   * Build from a lazy {@link RangeSource} (desktop). web-demuxer 4.x's `load()`
   * accepts only a `File`/URL (no custom lazy source), so we materialize the
   * bytes into a `File`. AVIs are typically modest; true byte-range streaming for
   * multi-GB AVIs is a follow-up (needs a web-demuxer source hook).
   */
  static async fromRangeSource(
    rangeSource: RangeSource,
    filename: string,
    options?: AviVideoOptions,
  ): Promise<AviVideoBackend> {
    const backend = new AviVideoBackend(filename, options);
    const bytes = await rangeSource.readRange(0, rangeSource.size);
    await backend.initialize(new File([bytes as BlobPart], filename));
    return backend;
  }

  private async initialize(source: File | string): Promise<void> {
    if (!isBrowser() || !hasWebCodecs()) {
      throw new Error(
        "AviVideoBackend requires a browser environment with WebCodecs",
      );
    }
    if (!webDemuxerConfig) {
      throw new Error(
        "AviVideoBackend used before configureWebDemuxer({ wasmFilePath }) was called",
      );
    }

    const demuxer = new WebDemuxer({
      wasmFilePath: webDemuxerConfig.wasmFilePath,
    });
    this.demuxer = demuxer;
    await demuxer.load(source);

    const stream = await demuxer.getAVStream();
    if (!stream || stream.width <= 0 || stream.height <= 0) {
      throw new Error("No decodable video stream found in AVI");
    }
    const width = stream.width;
    const height = stream.height;

    this.fps =
      parseFrameRate(stream.avg_frame_rate) ||
      parseFrameRate(stream.r_frame_rate) ||
      0;

    // AVI/ASF `nb_frames` is frequently inflated — the container index gets
    // padded, so it advertises MORE frames than actually decode (a 10-frame
    // MJPEG/Xvid AVI reports 12). This backend already treats the stream as
    // constant-frame-rate (getFrame maps index→time as i/fps), so derive the
    // count the same way: duration × fps is the reliable signal. Fall back to the
    // metadata count only when duration is unavailable (some ASF/WMV streams).
    const metaCount = Number.parseInt(stream.nb_frames, 10) || 0;
    const durationCount =
      this.fps > 0 && stream.duration > 0
        ? Math.round(stream.duration * this.fps)
        : 0;
    this.frameCount = durationCount || metaCount;

    const codec = (stream.codec_name ?? "").toLowerCase();
    if (MJPEG_CODECS.has(codec)) {
      this.mode = "mjpeg";
    } else {
      // WebCodecs path: verify the platform can actually decode this codec.
      const config = await demuxer.getDecoderConfig("video");
      const support = await VideoDecoder.isConfigSupported({
        codec: config.codec,
        codedWidth: config.codedWidth ?? width,
        codedHeight: config.codedHeight ?? height,
        description: config.description,
      });
      if (!support?.supported) {
        throw new Error(
          `AVI video codec "${codec}" is not decodable in this environment; ` +
            `transcode to H.264 (MP4) first`,
        );
      }
      this.decoderConfig = config;
      this.mode = "webcodecs";
    }

    if (!this.frameCount || this.frameCount <= 0) {
      throw new Error("Could not determine AVI frame count");
    }

    this.shape = [this.frameCount, height, width, 3];
  }

  async getFrame(
    frameIndex: number,
    opts?: { signal?: AbortSignal; pinned?: boolean },
  ): Promise<VideoFrame | null> {
    if (frameIndex < 0 || frameIndex >= this.frameCount) return null;

    const cached = this.cache.get(frameIndex);
    if (cached) {
      // LRU touch.
      this.cache.delete(frameIndex);
      this.cache.set(frameIndex, cached);
      return cached;
    }

    // A pinned read (embedding) neither supersedes the newest read nor bails
    // when a newer one arrives (`GetFrameOptions.pinned`).
    if (!opts?.pinned) this.latestRequested = frameIndex;
    if (this.mode === "mjpeg") {
      return this.enqueue(() =>
        this.decodeMjpegFrame(frameIndex, opts?.signal),
      );
    }
    return this.enqueue(() =>
      this.decodeWebCodecsFrame(frameIndex, opts?.signal, opts?.pinned),
    );
  }

  /** Serialize decodes: one `VideoDecoder`/packet-read at a time. */
  private enqueue<T>(run: () => Promise<T>): Promise<T> {
    const next = this.decodeQueue.then(run, run) as Promise<T>;
    // Keep the chain alive even if a link rejects.
    this.decodeQueue = next.catch(() => undefined);
    return next;
  }

  // ── MJPEG: each frame is an independent JPEG ────────────────────────────────
  private async decodeMjpegFrame(
    frameIndex: number,
    signal?: AbortSignal,
  ): Promise<VideoFrame | null> {
    if (!this.demuxer) throw new Error("Backend not initialized");
    if (signal?.aborted) return null;
    if (this.cache.has(frameIndex)) return this.cache.get(frameIndex) ?? null;

    // Nudge half a frame forward so the BACKWARD seek lands on frame N (whose
    // timestamp is N/fps ≤ this time), not N-1 due to float rounding.
    const time = (frameIndex + 0.5) / (this.fps || 1);
    const packet = await this.demuxer.getAVPacket(time);
    if (!packet?.data?.length) return null;

    const blob = new Blob([packet.data as BlobPart], { type: "image/jpeg" });
    const bitmap = await createImageBitmap(blob);
    this.cacheFrame(frameIndex, bitmap);
    return bitmap;
  }

  // ── WebCodecs: seek to keyframe ≤ target, decode GOP forward ─────────────────
  private async decodeWebCodecsFrame(
    target: number,
    signal?: AbortSignal,
    pinned = false,
  ): Promise<VideoFrame | null> {
    if (!this.demuxer || !this.decoderConfig) {
      throw new Error("Backend not initialized");
    }
    if (this.cache.has(target)) return this.cache.get(target) ?? null;
    if (signal?.aborted) return null;

    const fps = this.fps || 1;
    const startTime = target / fps;
    // Also decode a short forward window for smooth playback/scrubbing.
    const endIndex = Math.min(this.frameCount - 1, target + FORWARD_PREFETCH);
    const endTime = (endIndex + 0.5) / fps;

    const bitmapJobs: Promise<void>[] = [];
    const decoder = new VideoDecoder({
      output: (frame) => {
        const idx = Math.round((frame.timestamp / 1e6) * fps);
        // Only keep frames in the requested window we don't already have.
        if (idx >= target && idx <= endIndex && !this.cache.has(idx)) {
          bitmapJobs.push(
            createImageBitmap(frame)
              .then((bmp) => {
                this.cacheFrame(idx, bmp);
              })
              .finally(() => frame.close()),
          );
        } else {
          frame.close();
        }
      },
      error: () => {
        /* surfaced via the empty result below */
      },
    });
    decoder.configure(this.decoderConfig);

    try {
      const reader = this.demuxer
        .read("video", startTime, endTime, AVSeekFlag.AVSEEK_FLAG_BACKWARD)
        .getReader();
      for (;;) {
        // A newer request during a scrub supersedes this decode — bail early.
        if (signal?.aborted || (!pinned && this.latestRequested !== target))
          break;
        const { done, value } = await reader.read();
        if (done) break;
        decoder.decode(value);
      }
      await decoder.flush();
    } catch {
      // fall through — return whatever landed in the cache (likely null)
    } finally {
      try {
        decoder.close();
      } catch {
        /* already closed */
      }
    }

    await Promise.all(bitmapJobs);
    return this.cache.get(target) ?? null;
  }

  get numFrames(): number {
    return this.frameCount;
  }

  close(): void {
    this.cache.forEach((bitmap) => {
      bitmap.close();
    });
    this.cache.clear();
    this.demuxer?.destroy();
    this.demuxer = null;
    this.decoderConfig = null;
    this.frameCount = 0;
  }

  private cacheFrame(frameIndex: number, bitmap: ImageBitmap): void {
    if (this.cache.size >= this.cacheSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey !== undefined) {
        this.cache.get(oldestKey)?.close();
        this.cache.delete(oldestKey);
      }
    }
    this.cache.set(frameIndex, bitmap);
  }
}

/** Parse an ffmpeg `"num/den"` frame-rate string to fps (0 if unparseable). */
function parseFrameRate(rate: string | undefined): number {
  if (!rate) return 0;
  const [num, den] = rate.split("/");
  const n = Number(num);
  const d = den === undefined ? 1 : Number(den);
  if (!Number.isFinite(n) || !Number.isFinite(d) || d === 0) return 0;
  return n / d;
}
