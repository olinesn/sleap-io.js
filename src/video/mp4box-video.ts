import {
  type VideoBackend,
  type VideoFrame,
  type GetFrameOptions,
  type RangeSource,
  isRangeSource,
} from "./backend.js";
import {
  RemoteIOError,
  fetchRetrying,
  identityHeaders,
  statusToMessage,
} from "../io/remote.js";
import type { Mp4ParseResult } from "./mp4-decode-core.js";

// Environment/capability probes, evaluated at call time rather than captured in
// module-level consts. A module-level capture is frozen at first import, which
// breaks under a shared module registry (e.g. bare `bun test`, where all files
// share one process): if any earlier file imports this module before a test
// installs its `window`/WebCodecs globals, the captured value is wrong and can
// never refresh, because ESM caches the evaluated module body. Probing per call
// keeps backend selection correct whenever the globals change after load — the
// browser runtime never mutates them, but the tests do. See issue #159.
function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof document !== "undefined";
}
function hasWebCodecs(): boolean {
  return (
    isBrowser() &&
    typeof window.VideoDecoder !== "undefined" &&
    typeof window.EncodedVideoChunk !== "undefined"
  );
}
const MP4BOX_CDN = "https://unpkg.com/mp4box@0.5.4/dist/mp4box.all.min.js";

async function loadMp4box(): Promise<any> {
  const globalMp4box = (globalThis as { MP4Box?: any }).MP4Box;
  if (globalMp4box) return globalMp4box;

  try {
    const module = await import("mp4box");
    return module.default ?? module;
  } catch {
    if (!isBrowser()) {
      throw new Error("Failed to load mp4box");
    }
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = MP4BOX_CDN;
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Failed to load mp4box"));
      document.head.appendChild(script);
    });
    const afterLoad = (globalThis as { MP4Box?: any }).MP4Box;
    if (afterLoad) return afterLoad;
    throw new Error("Failed to load mp4box");
  }
}

const DEFAULT_CACHE_SIZE = 120;
const DEFAULT_LOOKAHEAD = 60;
const PARSE_CHUNK_SIZE = 1024 * 1024;
// Decode-ahead trigger distance: while playing, if the frame this many steps
// ahead of the playhead isn't cached yet, schedule more decode-ahead work so
// the runway never runs dry between the periodic keyframe→+lookahead decodes.
const DECODE_AHEAD_MARGIN = 30;

type Sample = {
  offset: number;
  size: number;
  timestamp: number;
  duration: number;
  isKeyframe: boolean;
  cts: number;
  decodeIndex: number;
};

export class Mp4BoxVideoBackend implements VideoBackend {
  filename: string;
  shape?: [number, number, number, number];
  fps?: number;
  dataset?: string | null;

  private ready: Promise<void>;
  private mp4box: any;
  private mp4boxFile: any;
  private videoTrack: any;
  private samples: Sample[];
  private keyframeIndices: number[];
  private cache: Map<number, ImageBitmap>;
  private cacheSize: number;
  private lookahead: number;
  private decoder: VideoDecoder | null;
  private config: VideoDecoderConfig | null;
  private fileSize: number;
  private supportsRangeRequests: boolean;
  private fileBlob: Blob | null;
  /**
   * Lazy byte source (desktop): read only the ranges we need via a native
   * `read_range` instead of materializing the whole file. Mutually exclusive
   * with `fileBlob` / URL fetching.
   */
  private rangeSource: RangeSource | null;
  private decodeQueue: Promise<void>;
  private latestRequestedFrame: number | null;
  /**
   * AbortController for the current speculative decode-ahead task. A demand
   * {@link getFrame} miss aborts it so the demand read never waits behind
   * speculative ahead-work (which bails between decode batches on the signal).
   */
  private aheadAbort: AbortController | null;
  /** True while a decode-ahead task is queued/running, to coalesce calls. */
  private aheadScheduled: boolean;
  /** Extra HTTP headers (e.g. Authorization) applied to every byte fetch. */
  private headers: Record<string, string>;

  constructor(
    source: string | File | Blob | RangeSource,
    options?: {
      cacheSize?: number;
      lookahead?: number;
      headers?: Record<string, string>;
      /** Display name when the source carries none (e.g. a {@link RangeSource}). */
      filename?: string;
    },
  ) {
    if (!hasWebCodecs()) {
      throw new Error("Mp4BoxVideoBackend requires WebCodecs support.");
    }
    if (!isBrowser()) {
      throw new Error("Mp4BoxVideoBackend requires a browser environment.");
    }
    const asRange = isRangeSource(source);
    this.filename = asRange
      ? (options?.filename ?? "")
      : source instanceof Blob
        ? ((source as File).name ?? "")
        : source;
    this.dataset = null;
    this.headers = options?.headers ?? {};
    this.samples = [];
    this.keyframeIndices = [];
    this.cache = new Map();
    this.cacheSize = options?.cacheSize ?? DEFAULT_CACHE_SIZE;
    this.lookahead = options?.lookahead ?? DEFAULT_LOOKAHEAD;
    this.decoder = null;
    this.config = null;
    this.fileSize = 0;
    this.supportsRangeRequests = false;
    this.fileBlob = null;
    this.rangeSource = null;
    this.decodeQueue = Promise.resolve();
    this.latestRequestedFrame = null;
    this.aheadAbort = null;
    this.aheadScheduled = false;

    if (asRange) {
      // Lazy on-disk source: we already know the size; readChunk pulls ranges.
      this.rangeSource = source;
      this.fileSize = source.size;
    } else if (source instanceof Blob) {
      this.fileBlob = source;
      this.fileSize = source.size;
      this.supportsRangeRequests = false;
    }

    this.ready = this.init();
  }

  async getFrame(
    frameIndex: number,
    opts?: GetFrameOptions,
  ): Promise<VideoFrame | null> {
    await this.ready;
    if (frameIndex < 0 || frameIndex >= this.samples.length) return null;

    if (this.cache.has(frameIndex)) {
      const bitmap = this.cache.get(frameIndex) ?? null;
      if (bitmap) {
        this.cache.delete(frameIndex);
        this.cache.set(frameIndex, bitmap);
      }
      return bitmap;
    }

    // A pinned read (embedding) neither supersedes the newest demand read nor
    // is dropped when a newer one arrives (`GetFrameOptions.pinned`).
    if (!opts?.pinned) this.latestRequestedFrame = frameIndex;
    // A demand read always wins over speculative decode-ahead: abort any
    // in-flight ahead-work so this frame isn't stuck behind it (the running
    // batch bails on the signal between decode batches).
    this.aheadAbort?.abort();

    this.decodeQueue = this.decodeQueue.then(async () => {
      if (!opts?.pinned && this.latestRequestedFrame !== frameIndex) return;
      if (opts?.signal?.aborted) return;

      const keyframe = this.findKeyframeBefore(frameIndex);
      // Scrub mode decodes only keyframe→target (no forward lookahead); the
      // dragged-past lookahead frames are never seen and just slow the read.
      const end = opts?.scrub
        ? frameIndex
        : Math.min(frameIndex + this.lookahead, this.samples.length - 1);
      await this.decodeRange(keyframe, end, frameIndex, {
        signal: opts?.signal,
      });
    });
    await this.decodeQueue;

    return this.cache.get(frameIndex) ?? null;
  }

  /**
   * Proactively decode a run of frames AHEAD of `fromFrame` into the cache so
   * sequential playback finds cache hits instead of blocking on the periodic
   * keyframe→+lookahead decode (the ~2 s play-freeze). Fire-and-forget: returns
   * immediately, scheduling one background {@link decodeRange} on the shared
   * decode queue. No-ops when the runway is already cached, near the end, or an
   * ahead task is already in flight (coalesced). A demand {@link getFrame} miss
   * aborts the in-flight ahead-work. The app drives this once per painted frame
   * while playing (and stops on seek/pause), so ahead-work never fights a seek.
   */
  decodeAhead(fromFrame: number, opts?: GetFrameOptions): void {
    if (!this.samples.length) return;
    if (opts?.signal?.aborted) return;
    // Coalesce: one ahead task at a time. Repeat calls during playback no-op
    // until the current one finishes (which re-checks coverage next call).
    if (this.aheadScheduled) return;
    const probe = fromFrame + DECODE_AHEAD_MARGIN;
    if (probe >= this.samples.length) return; // near the end
    if (this.cache.has(probe)) return; // runway already decoded

    this.aheadScheduled = true;
    const controller = new AbortController();
    this.aheadAbort = controller;
    const signal = controller.signal;

    this.decodeQueue = this.decodeQueue.then(async () => {
      try {
        if (signal.aborted || opts?.signal?.aborted) return;
        // Start at the first not-yet-cached frame at/after the playhead so we
        // extend the runway rather than re-decoding what's already cached.
        let start = fromFrame;
        while (start < this.samples.length && this.cache.has(start)) {
          start += 1;
        }
        if (start >= this.samples.length) return;
        const keyframe = this.findKeyframeBefore(start);
        const end = Math.min(start + this.lookahead, this.samples.length - 1);
        await this.decodeRange(keyframe, end, start, { signal });
      } finally {
        this.aheadScheduled = false;
      }
    });
  }

  async getFrameTimes(): Promise<number[] | null> {
    await this.ready;
    return this.samples.map((sample) => sample.timestamp / 1e6);
  }

  /**
   * The parse result (sample table, keyframe indices, decoder config, shape/fps)
   * from this main-thread parse, so an off-main {@link WorkerMp4BoxBackend} can
   * decode WITHOUT re-parsing or loading mp4box in the worker. Call after the
   * backend is ready. See the off-main-decode design (scrub-proxy v2 follow-up).
   */
  async getParseResult(): Promise<Mp4ParseResult> {
    await this.ready;
    if (!this.config || !this.shape) {
      throw new Error("getParseResult: backend not initialized");
    }
    const description =
      this.config.description instanceof Uint8Array
        ? this.config.description
        : this.config.description
          ? new Uint8Array(this.config.description as ArrayBuffer)
          : undefined;
    return {
      samples: this.samples,
      keyframeIndices: this.keyframeIndices,
      config: {
        codec: this.config.codec,
        codedWidth: this.config.codedWidth ?? this.shape[2],
        codedHeight: this.config.codedHeight ?? this.shape[1],
        description,
      },
      shape: this.shape,
      fps: this.fps,
      fileSize: this.fileSize,
    };
  }

  /**
   * The keyframe index at or before `frameIndex` — the cheapest frame to show
   * near it (a lone I-frame decode). The scrub UI snaps to this on a fast drag.
   * Clamps to the valid range; returns 0 before samples are parsed.
   */
  nearestKeyframe(frameIndex: number): number {
    if (!this.samples.length) return 0;
    const clamped = Math.max(0, Math.min(frameIndex, this.samples.length - 1));
    return this.findKeyframeBefore(clamped);
  }

  close(): void {
    if (this.decoder) {
      try {
        this.decoder.close();
      } catch {
        // ignore
      }
    }
    this.decoder = null;
    this.cache.forEach((bitmap) => {
      bitmap.close();
    });
    this.cache.clear();
    this.fileBlob = null;
    this.rangeSource = null;
  }

  private async init(): Promise<void> {
    // A RangeSource already knows the size and serves bytes via readChunk; only
    // a URL source needs the ranged probe in openSource().
    if (!this.fileBlob && !this.rangeSource) {
      await this.openSource();
    }

    this.mp4box = await loadMp4box();
    this.mp4boxFile = this.mp4box.createFile();
    const ready = new Promise<any>((resolve, reject) => {
      this.mp4boxFile.onError = reject;
      this.mp4boxFile.onReady = resolve;
    });

    let offset = 0;
    let resolved = false;
    ready.then(() => {
      resolved = true;
    });

    while (offset < this.fileSize && !resolved) {
      const buffer = await this.readChunk(offset, PARSE_CHUNK_SIZE);
      (buffer as any).fileStart = offset;
      const next = this.mp4boxFile.appendBuffer(buffer as any);
      offset = next === undefined ? offset + buffer.byteLength : next;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    const info = await ready;
    if (!info.videoTracks.length) throw new Error("No video tracks found");

    this.videoTrack = info.videoTracks[0];
    const trak = this.mp4boxFile.getTrackById(this.videoTrack.id);
    const description = this.getCodecDescription(trak);
    const codec = this.videoTrack.codec.startsWith("vp08")
      ? "vp8"
      : this.videoTrack.codec;
    this.config = {
      codec,
      codedWidth: this.videoTrack.video.width,
      codedHeight: this.videoTrack.video.height,
      description,
    };

    const support = await VideoDecoder.isConfigSupported(this.config);
    if (!support.supported) {
      throw new Error(`Codec ${codec} not supported`);
    }

    this.extractSamples();

    const duration = this.videoTrack.duration / this.videoTrack.timescale;
    this.fps = duration ? this.samples.length / duration : undefined;
    const frameCount = this.samples.length;
    const height = this.videoTrack.video.height;
    const width = this.videoTrack.video.width;
    this.shape = [frameCount, height, width, 3];
  }

  private async openSource(): Promise<void> {
    // Ranged probe: force Accept-Encoding: identity so a gzip transfer-encoding
    // can't corrupt byte-range semantics, and let Range win over any
    // user-supplied Range header (never let a user header override the byte
    // range we need). Retried with backoff on transient failures / 429/5xx.
    const response = await fetchRetrying(this.filename, {
      headers: { ...identityHeaders(this.headers), Range: "bytes=0-0" },
    });

    if (response.status === 206) {
      const contentRange = response.headers.get("Content-Range");
      const match = contentRange?.match(/\/(\d+)$/);
      if (match) {
        this.fileSize = Number.parseInt(match[1], 10);
        this.supportsRangeRequests = true;
        return;
      }
    }

    if (!response.ok && response.status !== 206) {
      throw new RemoteIOError({
        message: statusToMessage(response.status),
        url: this.filename,
        status: response.status,
      });
    }

    const full = await fetchRetrying(this.filename, {
      headers: { ...this.headers },
    });
    if (!full.ok) {
      throw new RemoteIOError({
        message: statusToMessage(full.status),
        url: this.filename,
        status: full.status,
      });
    }
    const blob = await full.blob();
    this.fileBlob = blob;
    this.fileSize = blob.size;
    this.supportsRangeRequests = false;
  }

  private async readChunk(offset: number, size: number): Promise<ArrayBuffer> {
    const end = Math.min(offset + size, this.fileSize);
    if (this.rangeSource) {
      // Lazy on-disk read: only the requested bytes cross into memory. Copy to a
      // standalone ArrayBuffer so mp4box owns a right-sized buffer (the returned
      // Uint8Array may be a view onto a larger backing buffer).
      const bytes = await this.rangeSource.readRange(offset, end - offset);
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ) as ArrayBuffer;
    }
    if (this.supportsRangeRequests) {
      // Ranged read: force Accept-Encoding: identity (gzip would corrupt the
      // byte range) and retry transient failures with backoff.
      const response = await fetchRetrying(this.filename, {
        headers: {
          ...identityHeaders(this.headers),
          Range: `bytes=${offset}-${end - 1}`,
        },
      });
      return await response.arrayBuffer();
    }
    if (this.fileBlob) {
      return await this.fileBlob.slice(offset, end).arrayBuffer();
    }
    throw new Error("No video source available");
  }

  private extractSamples(): void {
    const info = this.mp4boxFile.getTrackSamplesInfo(this.videoTrack.id);
    if (!info?.length) throw new Error("No samples");

    const ts = this.videoTrack.timescale;
    const samples = info.map((sample: any, index: number) => ({
      offset: sample.offset,
      size: sample.size,
      timestamp: (sample.cts * 1e6) / ts,
      duration: (sample.duration * 1e6) / ts,
      isKeyframe: sample.is_sync,
      cts: sample.cts,
      decodeIndex: index,
    }));

    this.samples = samples.sort((a: Sample, b: Sample) => {
      if (a.cts === b.cts) return a.decodeIndex - b.decodeIndex;
      return a.cts - b.cts;
    });

    this.keyframeIndices = [];
    this.samples.forEach((sample, index) => {
      if (sample.isKeyframe) this.keyframeIndices.push(index);
    });
  }

  private findKeyframeBefore(frameIndex: number): number {
    let result = 0;
    for (const keyframe of this.keyframeIndices) {
      if (keyframe <= frameIndex) result = keyframe;
      else break;
    }
    return result;
  }

  private getCodecDescription(trak: any): Uint8Array | undefined {
    const entries = trak?.mdia?.minf?.stbl?.stsd?.entries ?? [];
    const dataStream =
      (globalThis as { DataStream?: any }).DataStream ??
      this.mp4box?.DataStream;
    if (!dataStream) return undefined;

    for (const entry of entries) {
      const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
      if (!box) continue;
      const stream = new dataStream(undefined, 0, dataStream.BIG_ENDIAN);
      box.write(stream);
      return new Uint8Array(stream.buffer, 8);
    }
    return undefined;
  }

  private async readSampleDataByDecodeOrder(
    samplesToFeed: Array<{ pi: number; sample: Sample }>,
  ): Promise<Map<number, Uint8Array>> {
    const results = new Map<number, Uint8Array>();
    let i = 0;

    while (i < samplesToFeed.length) {
      const first = samplesToFeed[i];
      let regionEnd = i;
      let regionBytes = first.sample.size;

      while (regionEnd < samplesToFeed.length - 1) {
        const current = samplesToFeed[regionEnd];
        const next = samplesToFeed[regionEnd + 1];
        if (
          next.sample.offset ===
          current.sample.offset + current.sample.size
        ) {
          regionEnd += 1;
          regionBytes += next.sample.size;
        } else {
          break;
        }
      }

      const buffer = await this.readChunk(first.sample.offset, regionBytes);
      const bufferView = new Uint8Array(buffer);
      let bufferOffset = 0;

      for (let j = i; j <= regionEnd; j += 1) {
        const { sample } = samplesToFeed[j];
        results.set(
          sample.decodeIndex,
          bufferView.slice(bufferOffset, bufferOffset + sample.size),
        );
        bufferOffset += sample.size;
      }

      i = regionEnd + 1;
    }

    return results;
  }

  private async decodeRange(
    start: number,
    end: number,
    target: number,
    opts?: { signal?: AbortSignal },
  ): Promise<void> {
    if (!this.config) throw new Error("Decoder not configured");

    if (this.decoder) {
      try {
        this.decoder.close();
      } catch {
        // ignore
      }
    }

    let minDecodeIndex = Infinity;
    let maxDecodeIndex = -Infinity;
    for (let i = start; i <= end; i += 1) {
      minDecodeIndex = Math.min(minDecodeIndex, this.samples[i].decodeIndex);
      maxDecodeIndex = Math.max(maxDecodeIndex, this.samples[i].decodeIndex);
    }

    const toFeed: Array<{ pi: number; sample: Sample }> = [];
    for (let i = 0; i < this.samples.length; i += 1) {
      const sample = this.samples[i];
      if (
        sample.decodeIndex >= minDecodeIndex &&
        sample.decodeIndex <= maxDecodeIndex
      ) {
        toFeed.push({ pi: i, sample });
      }
    }
    toFeed.sort((a, b) => a.sample.decodeIndex - b.sample.decodeIndex);

    const dataMap = await this.readSampleDataByDecodeOrder(toFeed);
    const timestampMap = new Map<number, number>();
    for (const { pi, sample } of toFeed) {
      timestampMap.set(Math.round(sample.timestamp), pi);
    }

    const halfCache = Math.floor(this.cacheSize / 2);
    const cacheStart = Math.max(start, target - halfCache);
    const cacheEnd = Math.min(end, target + halfCache);

    let decodedCount = 0;
    let resolveComplete: () => void;
    let rejectComplete: (error: Error) => void;
    const completionPromise = new Promise<void>((resolve, reject) => {
      resolveComplete = resolve;
      rejectComplete = reject;
    });

    this.decoder = new VideoDecoder({
      output: (frame) => {
        const roundedTimestamp = Math.round(frame.timestamp);
        let frameIndex = timestampMap.get(roundedTimestamp);
        if (frameIndex === undefined) {
          let bestDiff = Infinity;
          for (const [ts, idx] of timestampMap) {
            const diff = Math.abs(ts - frame.timestamp);
            if (diff < bestDiff) {
              bestDiff = diff;
              frameIndex = idx;
            }
          }
        }

        const handleClose = () => {
          frame.close();
          decodedCount += 1;
          if (decodedCount >= toFeed.length) resolveComplete();
        };

        if (
          frameIndex !== undefined &&
          frameIndex >= cacheStart &&
          frameIndex <= cacheEnd
        ) {
          createImageBitmap(frame)
            .then((bitmap) => {
              this.addToCache(frameIndex as number, bitmap);
              handleClose();
            })
            .catch(handleClose);
        } else {
          handleClose();
        }
      },
      error: (error) => {
        if ((error as DOMException).name === "AbortError") {
          resolveComplete();
        } else {
          rejectComplete(error as Error);
        }
      },
    });

    this.decoder.configure(this.config);

    const BATCH_SIZE = 15;
    for (let i = 0; i < toFeed.length; i += BATCH_SIZE) {
      const batch = toFeed.slice(i, i + BATCH_SIZE);
      for (const { sample } of batch) {
        const data = dataMap.get(sample.decodeIndex);
        if (!data) continue;
        this.decoder.decode(
          new EncodedVideoChunk({
            type: sample.isKeyframe ? "key" : "delta",
            timestamp: sample.timestamp,
            duration: sample.duration,
            data,
          }),
        );
      }
      if (i + BATCH_SIZE < toFeed.length) {
        await new Promise((resolve) => setTimeout(resolve, 0));
        // Preempted mid-decode (a demand read aborted speculative ahead-work,
        // or the caller cancelled): stop feeding and abandon the rest so the
        // next queued decode — the one the user is waiting on — starts sooner.
        if (opts?.signal?.aborted) {
          try {
            this.decoder.close();
          } catch {
            // ignore
          }
          return;
        }
      }
    }

    await this.decoder.flush();
    await completionPromise;
  }

  private addToCache(frameIndex: number, bitmap: ImageBitmap): void {
    if (this.cache.size >= this.cacheSize) {
      const first = this.cache.keys().next();
      if (!first.done) {
        const evicted = this.cache.get(first.value);
        if (evicted) evicted.close();
        this.cache.delete(first.value);
      }
    }
    this.cache.set(frameIndex, bitmap);
  }
}
