/**
 * Off-main mp4 backend: the main-thread half of the split (scrub-proxy v2 →
 * off-main-thread decode).
 *
 * Implements the SAME {@link VideoBackend} interface as {@link Mp4BoxVideoBackend}
 * so it is a drop-in replacement, but does no decoding itself. The mp4 is parsed
 * once on the main thread (reusing `Mp4BoxVideoBackend.getParseResult`); this proxy
 * then forwards each read to the decode worker (`mp4box-decode-worker.ts`) and
 * receives finished frames back as Transferable `ImageBitmap`s. It owns the LRU
 * cache (so cache hits are instant, main-only) and answers `nearestKeyframe`
 * synchronously from the keyframe table it already holds. See the design doc
 * `docs/plans/2026-09-07-offmain-decode-design.md`.
 *
 * @module
 */

import type { VideoBackend, VideoFrame, GetFrameOptions } from "./backend.js";
import {
  type Mp4ParseResult,
  type Mp4Sample,
  findKeyframeBefore,
  planDecodeRange,
  computeCacheWindow,
  nextUncached,
  shouldDecodeAhead,
} from "./mp4-decode-core.js";
import {
  type ByteSourceDescriptor,
  type WorkerInMessage,
  type WorkerOutMessage,
  createDecodeWorker,
} from "./mp4box-decode-worker.js";

const DEFAULT_CACHE_SIZE = 120;
const DEFAULT_LOOKAHEAD = 60;
const DECODE_AHEAD_MARGIN = 30;

/**
 * Cheap main-thread pre-check: can a Web Worker even be constructed here? The real
 * capability gate is the worker's own self-test (VideoDecoder-in-worker + a
 * readable byte source), which surfaces as {@link WorkerMp4BoxBackend.create}
 * rejecting — the caller then keeps the on-main {@link Mp4BoxVideoBackend}.
 */
export function isWorkerDecodeAvailable(): boolean {
  return (
    typeof Worker !== "undefined" &&
    typeof Blob !== "undefined" &&
    typeof URL !== "undefined" &&
    typeof URL.createObjectURL === "function"
  );
}

/** The subset of `Worker` this backend uses — injectable so tests pass a fake. */
export interface WorkerLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onerror?: ((ev: unknown) => void) | null;
}

export interface WorkerBackendParams {
  /** Main-thread parse result (from `Mp4BoxVideoBackend.getParseResult()`). */
  parseResult: Mp4ParseResult;
  /** Serializable byte source the WORKER reconstructs into `readRange`. */
  byteSource: ByteSourceDescriptor;
  filename: string;
  cacheSize?: number;
  lookahead?: number;
  /** Injectable worker factory (defaults to the real blob worker). For tests. */
  createWorker?: () => WorkerLike;
}

interface PendingRead {
  target: number;
  resolve: (frame: VideoFrame | null) => void;
  settled: boolean;
  /** A `pinned` read: exempt from drop-stale. */
  pinned: boolean;
}

export class WorkerMp4BoxBackend implements VideoBackend {
  filename: string;
  shape?: [number, number, number, number];
  fps?: number;
  dataset?: string | null;

  private worker: WorkerLike;
  private samples: Mp4Sample[];
  private keyframeIndices: number[];
  private samplesLength: number;
  private cache: Map<number, ImageBitmap>;
  private cacheSize: number;
  private lookahead: number;
  private reqCounter: number;
  private pending: Map<number, PendingRead>;
  private aheadReqId: number | null;
  private aheadInFlight: boolean;
  private closed: boolean;

  /** Resolves once the worker reports `ready`; rejects on `unsupported`. */
  readonly ready: Promise<void>;
  private resolveReady!: () => void;
  private rejectReady!: (err: Error) => void;

  constructor(params: WorkerBackendParams) {
    this.filename = params.filename;
    this.samples = params.parseResult.samples;
    this.keyframeIndices = params.parseResult.keyframeIndices;
    this.samplesLength = this.samples.length;
    this.shape = params.parseResult.shape;
    this.fps = params.parseResult.fps;
    this.dataset = null;
    this.cacheSize = params.cacheSize ?? DEFAULT_CACHE_SIZE;
    this.lookahead = params.lookahead ?? DEFAULT_LOOKAHEAD;
    this.cache = new Map();
    this.pending = new Map();
    this.reqCounter = 0;
    this.aheadReqId = null;
    this.aheadInFlight = false;
    this.closed = false;

    this.ready = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });

    this.worker = params.createWorker
      ? params.createWorker()
      : (createDecodeWorker() as unknown as WorkerLike);
    this.worker.onmessage = (ev) =>
      this.handleMessage(ev.data as WorkerOutMessage);
    if ("onerror" in this.worker) {
      this.worker.onerror = () =>
        this.rejectReady(new Error("decode worker errored during init"));
    }

    const init: WorkerInMessage = {
      type: "init",
      samples: this.samples,
      config: params.parseResult.config,
      byteSource: params.byteSource,
    };
    this.worker.postMessage(init);
  }

  /**
   * Build a worker backend and wait for its self-test. Rejects (worker
   * `unsupported`: no WebCodecs-in-worker / unreadable source / codec) so the
   * caller can keep the on-main {@link Mp4BoxVideoBackend} fallback.
   */
  static async create(
    params: WorkerBackendParams,
  ): Promise<WorkerMp4BoxBackend> {
    const backend = new WorkerMp4BoxBackend(params);
    try {
      await backend.ready;
    } catch (err) {
      backend.close();
      throw err;
    }
    return backend;
  }

  private handleMessage(msg: WorkerOutMessage): void {
    switch (msg.type) {
      case "ready":
        this.resolveReady();
        break;
      case "unsupported":
        this.rejectReady(new Error(`decode worker unsupported: ${msg.reason}`));
        break;
      case "bitmap":
        this.onBitmap(msg.reqId, msg.frame, msg.bitmap);
        break;
      case "decodeDone":
        this.onDecodeSettled(msg.reqId);
        break;
      case "decodeError":
        this.onDecodeSettled(msg.reqId);
        break;
    }
  }

  private onBitmap(reqId: number, frame: number, bitmap: ImageBitmap): void {
    if (this.closed) {
      try {
        bitmap.close();
      } catch {
        // ignore
      }
      return;
    }
    this.addToCache(frame, bitmap);
    const pending = this.pending.get(reqId);
    if (pending && !pending.settled && frame === pending.target) {
      pending.settled = true;
      pending.resolve(this.cache.get(pending.target) ?? null);
    }
  }

  private onDecodeSettled(reqId: number): void {
    const pending = this.pending.get(reqId);
    if (pending) {
      if (!pending.settled) {
        pending.settled = true;
        pending.resolve(this.cache.get(pending.target) ?? null);
      }
      this.pending.delete(reqId);
    }
    if (reqId === this.aheadReqId) {
      this.aheadInFlight = false;
      this.aheadReqId = null;
    }
  }

  async getFrame(
    frameIndex: number,
    opts?: GetFrameOptions,
  ): Promise<VideoFrame | null> {
    await this.ready;
    if (frameIndex < 0 || frameIndex >= this.samplesLength) return null;

    // Cache hit: served entirely on main, no worker round-trip (LRU touch).
    if (this.cache.has(frameIndex)) {
      const bitmap = this.cache.get(frameIndex) ?? null;
      if (bitmap) {
        this.cache.delete(frameIndex);
        this.cache.set(frameIndex, bitmap);
      }
      return bitmap;
    }

    // A demand read preempts speculative decode-ahead (frees the worker queue).
    if (this.aheadInFlight && this.aheadReqId != null) {
      this.worker.postMessage({ type: "abort", reqId: this.aheadReqId });
    }

    // Drop-stale: a newer demand read supersedes any older in-flight demand
    // reads. Abort them in the worker so its SERIAL decode queue can't back up —
    // otherwise a fast scrub piles up hundreds of decodes that each resolve
    // seconds later against a long-moved playhead (the frame shown then mismatches
    // the labels). Settle their promises now with whatever's cached. This mirrors
    // the on-main backend's `latestRequestedFrame` drop-stale.
    // Pinned reads (embedding) are exempt both ways: a pinned read never
    // supersedes the others, and is never superseded itself.
    if (!opts?.pinned && this.pending.size > 0) {
      for (const [staleId, stale] of this.pending) {
        if (stale.pinned) continue;
        this.worker.postMessage({ type: "abort", reqId: staleId });
        if (!stale.settled) {
          stale.settled = true;
          stale.resolve(this.cache.get(stale.target) ?? null);
        }
        this.pending.delete(staleId);
      }
    }

    this.reqCounter += 1;
    const reqId = this.reqCounter;
    const { start, end } = planDecodeRange(
      this.samplesLength,
      this.keyframeIndices,
      frameIndex,
      { scrub: opts?.scrub, lookahead: this.lookahead },
    );
    const { cacheStart, cacheEnd } = computeCacheWindow(
      start,
      end,
      frameIndex,
      this.cacheSize,
    );

    if (opts?.signal?.aborted) {
      // Never issue an already-cancelled read.
      return this.cache.get(frameIndex) ?? null;
    }

    const promise = new Promise<VideoFrame | null>((resolve) => {
      this.pending.set(reqId, {
        target: frameIndex,
        resolve,
        settled: false,
        pinned: opts?.pinned === true,
      });
    });

    if (opts?.signal) {
      opts.signal.addEventListener(
        "abort",
        () => {
          this.worker.postMessage({ type: "abort", reqId });
          const pending = this.pending.get(reqId);
          if (pending && !pending.settled) {
            pending.settled = true;
            pending.resolve(this.cache.get(frameIndex) ?? null);
            this.pending.delete(reqId);
          }
        },
        { once: true },
      );
    }

    const decode: WorkerInMessage = {
      type: "decode",
      reqId,
      start,
      end,
      target: frameIndex,
      cacheStart,
      cacheEnd,
    };
    this.worker.postMessage(decode);
    return promise;
  }

  decodeAhead(fromFrame: number, opts?: GetFrameOptions): void {
    if (this.closed || !this.samplesLength) return;
    if (opts?.signal?.aborted) return;
    if (this.aheadInFlight) return; // coalesce: one ahead task at a time
    const has = (i: number) => this.cache.has(i);
    if (
      !shouldDecodeAhead(
        fromFrame,
        this.samplesLength,
        DECODE_AHEAD_MARGIN,
        has,
      )
    )
      return;

    const startFrame = nextUncached(fromFrame, this.samplesLength, has);
    if (startFrame >= this.samplesLength) return;

    const { start, end } = planDecodeRange(
      this.samplesLength,
      this.keyframeIndices,
      startFrame,
      { scrub: false, lookahead: this.lookahead },
    );
    const { cacheStart, cacheEnd } = computeCacheWindow(
      start,
      end,
      startFrame,
      this.cacheSize,
    );

    this.reqCounter += 1;
    const reqId = this.reqCounter;
    this.aheadReqId = reqId;
    this.aheadInFlight = true;
    const decode: WorkerInMessage = {
      type: "decode",
      reqId,
      start,
      end,
      target: startFrame,
      cacheStart,
      cacheEnd,
    };
    this.worker.postMessage(decode);
  }

  nearestKeyframe(frameIndex: number): number {
    if (!this.samplesLength) return 0;
    const clamped = Math.max(0, Math.min(frameIndex, this.samplesLength - 1));
    return findKeyframeBefore(this.keyframeIndices, clamped);
  }

  async getFrameTimes(): Promise<number[] | null> {
    return this.samples.map((sample) => sample.timestamp / 1e6);
  }

  private addToCache(frameIndex: number, bitmap: ImageBitmap): void {
    const existing = this.cache.get(frameIndex);
    if (existing && existing !== bitmap) {
      try {
        existing.close();
      } catch {
        // ignore
      }
      this.cache.delete(frameIndex);
    }
    if (!this.cache.has(frameIndex) && this.cache.size >= this.cacheSize) {
      const first = this.cache.keys().next();
      if (!first.done) {
        const evicted = this.cache.get(first.value);
        if (evicted) {
          try {
            evicted.close();
          } catch {
            // ignore
          }
        }
        this.cache.delete(first.value);
      }
    }
    this.cache.set(frameIndex, bitmap);
  }

  close(): void {
    this.closed = true;
    try {
      this.worker.postMessage({ type: "close" });
    } catch {
      // ignore
    }
    try {
      this.worker.terminate();
    } catch {
      // ignore
    }
    this.cache.forEach((bitmap) => {
      try {
        bitmap.close();
      } catch {
        // ignore
      }
    });
    this.cache.clear();
    this.pending.forEach((pending) => {
      if (!pending.settled) {
        pending.settled = true;
        pending.resolve(null);
      }
    });
    this.pending.clear();
  }
}
