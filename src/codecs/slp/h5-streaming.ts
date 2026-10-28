/**
 * Streaming HDF5 file access via Web Worker.
 *
 * This module provides a high-level API for accessing remote HDF5 files
 * using HTTP range requests for efficient streaming. The actual HDF5
 * operations run in a Web Worker where synchronous XHR is allowed.
 *
 * @module
 */

import {
  createH5Worker,
  type H5WorkerMessage,
  type H5WorkerResponse,
} from "./h5-worker.js";
import { RemoteIOError, isUrl, resolveUrl } from "../../io/remote.js";
import type {
  SerializableEmbedEntry,
  SerializableEncodeEntry,
  LabelTableUpdate,
  LabelTable,
} from "./write.js";
import type { EncodeFramePayload } from "./stream-embed.js";

/**
 * Options for opening a streaming HDF5 file.
 */
export interface StreamingH5Options {
  /** URL to h5wasm IIFE bundle. Defaults to CDN. */
  h5wasmUrl?: string;
  /** Filename hint for the HDF5 file. */
  filenameHint?: string;
  /**
   * Extra HTTP request headers forwarded to the worker's `openUrl`. When
   * non-empty, the worker buffer-downloads the file (authenticated) instead of
   * using header-free `createLazyFile` range streaming.
   */
  headers?: Record<string, string>;
}

/**
 * A lazy random-access byte source backed by native reads (e.g. a Tauri
 * `read_range` command). The streaming Worker pulls slices on demand through a
 * SharedArrayBuffer + `Atomics` bridge (the "B-seam" range reader), so large
 * files are never fully materialized in WASM memory. `readRange` runs on the
 * MAIN thread (the Worker cannot do the native IPC itself).
 */
export interface RangeSource {
  /** Total file size in bytes. */
  size: number;
  /** Read `[offset, offset + length)`; may return fewer bytes at EOF. */
  readRange: (offset: number, length: number) => Promise<Uint8Array>;
}

/** Type guard for {@link RangeSource}. */
export function isRangeSource(source: unknown): source is RangeSource {
  return (
    typeof source === "object" &&
    source !== null &&
    typeof (source as RangeSource).size === "number" &&
    typeof (source as RangeSource).readRange === "function"
  );
}

/**
 * Service one B-seam byte-request against the shared control/data buffers: read
 * `[offset, offset + length)` via `reader`, copy the bytes into `dataArea`
 * (clamped to BOTH the requested length and the data-area size), publish the
 * returned length in `control[1]` (RETLEN), then set `control[0]` = READY (`2`)
 * and `notify` — RETLEN is stored BEFORE the READY release so a Worker's
 * `Atomics.wait`/load (acquire) sees a consistent (length, data) pair. On a read
 * failure, publishes 0 bytes + READY so the Worker gets a clean short read / EOF
 * instead of hanging.
 *
 * Extracted from {@link StreamingH5File}'s Worker message handler so the
 * (otherwise Worker-gated) main-thread half of the protocol is unit-testable.
 * @internal
 */
export async function serviceRangeBridge(
  control: Int32Array,
  dataArea: Uint8Array,
  reader: (offset: number, length: number) => Promise<Uint8Array>,
  offset: number,
  length: number,
): Promise<void> {
  try {
    const bytes = await reader(offset, length);
    const n = Math.min(bytes.length, dataArea.length, length);
    dataArea.set(bytes.subarray(0, n), 0);
    Atomics.store(control, 1, n); // RETLEN
    Atomics.store(control, 0, 2); // STATE = READY
    Atomics.notify(control, 0);
  } catch (err) {
    // Signal 0 bytes: the Worker treats it as a short read / EOF and h5wasm
    // surfaces a clean read error rather than hanging.
    Atomics.store(control, 1, 0);
    Atomics.store(control, 0, 2);
    Atomics.notify(control, 0);
    console.error("[StreamingH5File] readRange failed:", err);
  }
}

/**
 * A random-access, disk-backed write target (e.g. a Tauri native file handle).
 * The write counterpart to {@link RangeSource}: a Worker's writable Emscripten
 * device pushes bytes through a SharedArrayBuffer + `Atomics` bridge (the
 * "write B-seam"), and `writeAt`/`truncate`/`close` run on the MAIN thread
 * (the Worker cannot do the native IPC itself). `readAt` supports read-modify
 * writes against the same backing store (e.g. re-reading a partially written
 * region) without opening a second handle.
 */
export interface RangeSink {
  /** Write `bytes` at `[offset, offset + bytes.length)`. */
  writeAt(offset: number, bytes: Uint8Array): Promise<void>;
  /** Read `[offset, offset + length)`; may return fewer bytes at EOF. */
  readAt(offset: number, length: number): Promise<Uint8Array>;
  /** Truncate (or extend) the backing store to exactly `length` bytes. */
  truncate(length: number): Promise<void>;
  /** Flush and release the underlying handle. */
  close(): Promise<void>;
}

/**
 * Service one B-seam write request against the shared control buffer and data
 * area: copy `[0, length)` out of `dataArea` (the bytes the Worker staged
 * there), write them to `sink` at `offset`, publish the outcome in
 * `control[1]` (RESULT: `0` = ok, `-1` = error), then set `control[0]` = READY
 * (`2`) and `notify` — RESULT is stored BEFORE the READY release so a
 * Worker's `Atomics.wait`/load (acquire) sees a consistent result. On a write
 * failure, publishes RESULT = -1 + READY so the Worker gets a clean error
 * instead of hanging.
 *
 * Extracted (mirroring {@link serviceRangeBridge}) so the (otherwise
 * Worker-gated) main-thread half of the write protocol is unit-testable.
 * @internal
 */
export async function serviceWriteBridge(
  control: Int32Array,
  dataArea: Uint8Array,
  sink: RangeSink,
  offset: number,
  length: number,
): Promise<void> {
  try {
    const bytes = dataArea.subarray(0, length);
    await sink.writeAt(offset, bytes);
    Atomics.store(control, 1, 0); // RESULT = ok
    Atomics.store(control, 0, 2); // STATE = READY
    Atomics.notify(control, 0);
  } catch (err) {
    Atomics.store(control, 1, -1); // RESULT = error
    Atomics.store(control, 0, 2);
    Atomics.notify(control, 0);
    console.error("[StreamingH5Writer] writeAt failed:", err);
  }
}

/**
 * Service one B-seam truncate request against the shared control buffer:
 * `truncate` the sink to `length`, publish the outcome in `control[1]`
 * (RESULT: `0` = ok, `-1` = error) BEFORE setting `control[0]` = READY (`2`)
 * and `notify`ing, mirroring {@link serviceWriteBridge}'s ordering so the
 * Worker's acquire load always sees a consistent result.
 *
 * Extracted so the main-thread half of the truncate protocol is
 * unit-testable without a Worker.
 * @internal
 */
export async function serviceTruncateBridge(
  control: Int32Array,
  sink: RangeSink,
  length: number,
): Promise<void> {
  try {
    await sink.truncate(length);
    Atomics.store(control, 1, 0); // RESULT = ok
    Atomics.store(control, 0, 2); // STATE = READY
    Atomics.notify(control, 0);
  } catch (err) {
    Atomics.store(control, 1, -1); // RESULT = error
    Atomics.store(control, 0, 2);
    Atomics.notify(control, 0);
    console.error("[StreamingH5Writer] truncate failed:", err);
  }
}

/**
 * Source types supported by the streaming HDF5 file.
 */
export type StreamingH5Source =
  | string
  | ArrayBuffer
  | Uint8Array
  | File
  | RangeSource;

/**
 * Reconstructs a TypedArray from transferred worker data.
 */
function reconstructValue(data: unknown): unknown {
  if (data && typeof data === "object" && "type" in data) {
    const typed = data as {
      type: string;
      dtype?: string;
      buffer?: ArrayBuffer;
      byteOffset?: number;
      length?: number;
      columns?: Record<string, unknown[]>;
    };

    // Fast compound read: worker already split the record blob into per-member
    // columns (see readCompoundColumnsWorker); normalizeStructData consumes this
    // { field: array } record directly.
    if (typed.type === "columns" && typed.columns) {
      return typed.columns;
    }

    if (typed.type === "typedarray" && typed.buffer) {
      const TypedArrayConstructor = getTypedArrayConstructor(
        typed.dtype || "Uint8Array",
      );
      return new TypedArrayConstructor(
        typed.buffer,
        typed.byteOffset || 0,
        typed.length,
      );
    }

    if (typed.type === "arraybuffer" && typed.buffer) {
      return typed.buffer;
    }
  }

  return data;
}

function getTypedArrayConstructor(
  name: string,
): new (
  buffer: ArrayBuffer,
  byteOffset: number,
  length?: number,
) => ArrayBufferView {
  const constructors: Record<
    string,
    new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length?: number,
    ) => ArrayBufferView
  > = {
    Int8Array,
    Uint8Array,
    Uint8ClampedArray,
    Int16Array,
    Uint16Array,
    Int32Array,
    Uint32Array,
    Float32Array,
    Float64Array,
    BigInt64Array,
    BigUint64Array,
  };
  return constructors[name] || Uint8Array;
}

/**
 * A streaming HDF5 file handle that uses a Web Worker for range request access.
 *
 * This class provides an API similar to h5wasm.File but operates via message
 * passing to a worker where createLazyFile enables HTTP range requests.
 */
export class StreamingH5File {
  private worker: Worker;
  private messageId = 0;
  private pendingMessages = new Map<
    number,
    {
      resolve: (value: H5WorkerResponse) => void;
      reject: (error: Error) => void;
    }
  >();
  private _keys: string[] = [];
  private _isOpen = false;

  // B-seam range bridge (set by openRange): the app-provided reader plus the
  // SharedArrayBuffer views the Worker blocks on. See serviceRangeRequest.
  private rangeReader?: (offset: number, length: number) => Promise<Uint8Array>;
  private rangeControl?: Int32Array;
  private rangeData?: Uint8Array;

  constructor() {
    this.worker = createH5Worker();
    this.worker.onmessage = this.handleMessage.bind(this);
    this.worker.onerror = this.handleError.bind(this);
  }

  private handleMessage(e: MessageEvent<H5WorkerResponse>): void {
    // B-seam: the Worker's custom device asks the main thread for bytes. This is
    // a Worker->main request (not an id-correlated response), so handle it first.
    const msg = e.data as unknown as {
      type?: string;
      offset?: number;
      length?: number;
    };
    if (msg && msg.type === "rangeRequest") {
      this.serviceRangeRequest(msg.offset ?? 0, msg.length ?? 0);
      return;
    }

    const { id, ...data } = e.data;
    const pending = this.pendingMessages.get(id);
    if (pending) {
      this.pendingMessages.delete(id);
      if (data.success) {
        pending.resolve(e.data);
      } else {
        // Ensure error is a string for the Error message
        let errorMessage = "Worker operation failed";
        if (typeof data.error === "string") {
          errorMessage = data.error;
        } else if (data.error && typeof data.error === "object") {
          errorMessage = JSON.stringify(data.error);
        }
        pending.reject(new Error(errorMessage));
      }
    }
  }

  private handleError(e: ErrorEvent): void {
    console.error("[StreamingH5File] Worker error:", e.message);
    // Reject all pending messages
    for (const [id, pending] of this.pendingMessages) {
      pending.reject(new Error(`Worker error: ${e.message}`));
      this.pendingMessages.delete(id);
    }
  }

  private send(
    type: H5WorkerMessage["type"],
    payload?: Record<string, unknown>,
  ): Promise<H5WorkerResponse> {
    return new Promise((resolve, reject) => {
      const id = ++this.messageId;
      this.pendingMessages.set(id, { resolve, reject });
      this.worker.postMessage({ type, payload, id });
    });
  }

  /**
   * Initialize the h5wasm module in the worker.
   */
  async init(options?: StreamingH5Options): Promise<void> {
    await this.send("init", { h5wasmUrl: options?.h5wasmUrl });
  }

  /**
   * Open a remote HDF5 file for streaming access via URL.
   *
   * The scheme is resolved on the MAIN thread (the worker cannot import the
   * scheme gate): `gs://`/`gcs://` are mapped to `storage.googleapis.com`, and
   * `s3://`/`az://`/`abfs://` fail fast with a redacted {@link RemoteIOError}.
   * Google Drive is NOT supported on the streaming worker path (Drive requires a
   * buffered, interstitial-following download, not range streaming); a Drive URL
   * throws a redacted {@link RemoteIOError} directing the caller to the
   * non-streaming reader. The worker only ever fetches an already-resolved
   * http(s) URL.
   *
   * @param url - URL to the HDF5 file (must support HTTP range requests)
   * @param options - Optional settings
   */
  async open(url: string, options?: StreamingH5Options): Promise<void> {
    // Initialize if not already done
    await this.init(options);

    // Resolve the scheme up front so gs:// works in explicit stream mode (the
    // worker can't resolve it) and unsupported schemes / Drive fail fast with a
    // redacted typed error rather than a confusing worker error.
    let resolvedUrl = url;
    if (isUrl(url)) {
      const { url: resolved, gdrive } = resolveUrl(url);
      if (gdrive) {
        throw new RemoteIOError({
          message:
            'Google Drive URLs are not supported on the streaming worker path (Drive needs a buffered download); use the non-streaming reader (stream:"download" / loadSlp)',
          url,
          status: null,
        });
      }
      resolvedUrl = resolved;
    }

    const filename =
      options?.filenameHint ||
      resolvedUrl.split("/").pop()?.split("?")[0] ||
      "data.h5";
    const result = await this.send("openUrl", {
      url: resolvedUrl,
      filename,
      headers: options?.headers,
    });
    this._keys = (result.keys as string[]) || [];
    this._isOpen = true;
  }

  /**
   * Open a local File object using WORKERFS (zero-copy).
   *
   * @param file - File object from file input or drag-and-drop
   * @param options - Optional settings
   */
  async openLocal(file: File, options?: StreamingH5Options): Promise<void> {
    // Initialize if not already done
    await this.init(options);

    const filename = options?.filenameHint || file.name || "data.h5";
    const result = await this.send("openLocal", { file, filename });
    this._keys = (result.keys as string[]) || [];
    this._isOpen = true;
  }

  /**
   * Open an HDF5 file from an ArrayBuffer or Uint8Array.
   *
   * @param buffer - ArrayBuffer or Uint8Array containing the HDF5 file data
   * @param options - Optional settings
   */
  async openBuffer(
    buffer: ArrayBuffer | Uint8Array,
    options?: StreamingH5Options,
  ): Promise<void> {
    // Initialize if not already done
    await this.init(options);

    const filename = options?.filenameHint || "data.h5";
    // Transfer the buffer to the worker for efficiency
    const data = buffer instanceof Uint8Array ? buffer.buffer : buffer;
    const result = await this.send("openBuffer", { buffer: data, filename });
    this._keys = (result.keys as string[]) || [];
    this._isOpen = true;
  }

  /**
   * Open an HDF5 file from a lazy {@link RangeSource} via the B-seam bridge.
   *
   * The Worker registers a custom Emscripten device whose synchronous `read`
   * blocks on `Atomics.wait` over a SharedArrayBuffer; this main thread services
   * each request by calling `readRange` and waking the Worker. Requires
   * cross-origin isolation (SharedArrayBuffer / COOP+COEP).
   */
  async openRange(
    source: RangeSource,
    options?: StreamingH5Options,
  ): Promise<void> {
    if (typeof SharedArrayBuffer === "undefined") {
      throw new Error(
        "RangeSource streaming requires SharedArrayBuffer (enable cross-origin isolation: COOP 'same-origin' + COEP 'require-corp')",
      );
    }
    await this.init(options);
    this.rangeReader = source.readRange;

    // SAB layout: [0..CONTROL_BYTES) = Int32 control (idx0=STATE, idx1=RETLEN),
    // then the data area (max single bridged read). The Worker loops for larger reads.
    const CONTROL_BYTES = 32;
    const MAX_CHUNK = 4 * 1024 * 1024;
    const sab = new SharedArrayBuffer(CONTROL_BYTES + MAX_CHUNK);
    this.rangeControl = new Int32Array(sab, 0, 8);
    this.rangeData = new Uint8Array(sab, CONTROL_BYTES);

    const filename = options?.filenameHint || "data.h5";
    const result = await this.send("openRange", {
      sab,
      size: source.size,
      filename,
      controlBytes: CONTROL_BYTES,
    });
    this._keys = (result.keys as string[]) || [];
    this._isOpen = true;
  }

  /**
   * Service a Worker byte-request: read via the app's `readRange`, copy the
   * bytes into the shared data area, then wake the (blocked) Worker. STATE is
   * stored last (release) so the Worker's `Atomics.wait` return (acquire) sees
   * the data + returned length.
   */
  private serviceRangeRequest(offset: number, length: number): void {
    const control = this.rangeControl;
    const dataArea = this.rangeData;
    const reader = this.rangeReader;
    if (!control || !dataArea || !reader) return;
    // Fire-and-forget: the Worker is blocked on Atomics.wait and is woken by the
    // notify inside serviceRangeBridge; the message handler must not await.
    void serviceRangeBridge(control, dataArea, reader, offset, length);
  }

  /**
   * Open an HDF5 file from any supported source.
   *
   * @param source - URL string, File, ArrayBuffer, Uint8Array, or RangeSource
   * @param options - Optional settings
   */
  async openAny(
    source: StreamingH5Source,
    options?: StreamingH5Options,
  ): Promise<void> {
    if (isRangeSource(source)) {
      return this.openRange(source, options);
    }
    if (typeof source === "string") {
      return this.open(source, options);
    }
    if (typeof File !== "undefined" && source instanceof File) {
      return this.openLocal(source, options);
    }
    if (source instanceof ArrayBuffer || source instanceof Uint8Array) {
      return this.openBuffer(source, options);
    }
    throw new Error("Unsupported source type for StreamingH5File");
  }

  /**
   * Whether a file is currently open.
   */
  get isOpen(): boolean {
    return this._isOpen;
  }

  /**
   * Get the root-level keys in the file.
   */
  keys(): string[] {
    return this._keys;
  }

  /**
   * Get the keys (children) at a given path.
   */
  async getKeys(path: string): Promise<string[]> {
    const result = await this.send("getKeys", { path });
    return (result.keys as string[]) || [];
  }

  /**
   * Get an attribute value.
   */
  async getAttr(path: string, name: string): Promise<unknown> {
    const result = await this.send("getAttr", { path, name });
    return (result.value as { value: unknown })?.value ?? result.value;
  }

  /**
   * Get all attributes at a path.
   */
  async getAttrs(path: string): Promise<Record<string, unknown>> {
    const result = await this.send("getAttrs", { path });
    return (result.attrs as Record<string, unknown>) || {};
  }

  /**
   * Get dataset metadata (shape, dtype, and the raw h5wasm `metadata`) without
   * reading values. `metadata` carries the layout details the in-place
   * writability gate needs — `compound_type.members` (per-member dtype class,
   * used to detect Python's non-writable enum/bool columns) and `chunks` (⇒
   * resizable). Pass this straight to `onDiskTableFromMeta`.
   */
  async getDatasetMeta(path: string): Promise<{
    shape: number[];
    dtype: string;
    metadata?: Record<string, unknown>;
  }> {
    const result = await this.send("getDatasetMeta", { path });
    const meta = result.meta as {
      shape: number[];
      dtype: string;
      metadata?: Record<string, unknown>;
    };
    return meta;
  }

  /**
   * Read a dataset's value.
   *
   * @param path - Path to the dataset
   * @param slice - Optional slice specification (array of [start, end] pairs)
   */
  async getDatasetValue(
    path: string,
    slice?: Array<[number, number] | []>,
  ): Promise<{ value: unknown; shape: number[]; dtype: string }> {
    const result = await this.send("getDatasetValue", { path, slice });
    const data = result.data as {
      value: unknown;
      shape: number[];
      dtype: string;
    };
    return {
      value: reconstructValue(data.value),
      shape: data.shape,
      dtype: data.dtype,
    };
  }

  /**
   * Close the file and terminate the worker.
   */
  async close(): Promise<void> {
    if (this._isOpen) {
      await this.send("close");
      this._isOpen = false;
    }
    this.worker.terminate();
    this._keys = [];
  }
}

/**
 * A streaming HDF5 file WRITER that uses a Web Worker's writable Emscripten
 * device (the write B-seam) to stream bytes to a {@link RangeSink} instead of
 * materializing the file in WASM MEMFS. Mirrors {@link StreamingH5File}'s
 * worker plumbing (message correlation, `send`, error handling), but its
 * Worker registers a writable device whose `read`/`write`/`setattr`
 * (ftruncate) block on the SAB bridge — so `handleMessage` must service THREE
 * Worker->main request types (range reads for read-modify-write, writes, and
 * truncates) ahead of the id-correlated response branch.
 */
export class StreamingH5Writer {
  private worker: Worker;
  private messageId = 0;
  private pendingMessages = new Map<
    number,
    {
      resolve: (value: H5WorkerResponse) => void;
      reject: (error: Error) => void;
    }
  >();

  // Progress callback for the in-flight appendEmbeddedVideos copy: the Worker
  // posts un-id'd `{type:'progress'}` notifications as it streams image blobs;
  // handleMessage forwards (done,total) frame counts here. Set for the duration
  // of one appendEmbeddedVideos call.
  private appendProgressCb?: (done: number, total: number) => void;

  // Write B-seam bridge (set by openAppend): the app-provided dest sink plus
  // the SharedArrayBuffer views the Worker blocks on. See handleMessage.
  // Every write/truncate, and every 'dest'-tagged range (read) request, routes
  // through destSink (the Worker tags ALL of createWriteRangeFile's bridge
  // calls 'dest').
  private destSink?: RangeSink;
  // Dual-bridge only (set by openAppend): the source file's reader. A 'source'-
  // tagged range request routes here instead of destSink.
  private sourceReader?: (
    offset: number,
    length: number,
  ) => Promise<Uint8Array>;
  private control?: Int32Array;
  private data?: Uint8Array;

  constructor() {
    this.worker = createH5Worker();
    this.worker.onmessage = this.handleMessage.bind(this);
    this.worker.onerror = this.handleError.bind(this);
  }

  private handleMessage(e: MessageEvent<H5WorkerResponse>): void {
    // Write B-seam: the Worker's writable device(s) ask the main thread to read,
    // write, or truncate. These are Worker->main requests (not id-correlated
    // responses), so handle them first. Every request carries a `target`
    // ('source' | 'dest' | undefined) stamped by the Worker's bridgeRead/
    // bridgeWrite/bridgeTruncate — writes and truncates are ALWAYS 'dest' (only
    // the destination file is ever written to), so only range (read) requests
    // branch on it.
    const msg = e.data as unknown as {
      type?: string;
      target?: string;
      offset?: number;
      length?: number;
    };
    if (msg && msg.type === "rangeRequest") {
      if (msg.target === "source") {
        if (this.control && this.data && this.sourceReader) {
          void serviceRangeBridge(
            this.control,
            this.data,
            this.sourceReader,
            msg.offset ?? 0,
            msg.length ?? 0,
          );
        }
      } else if (this.control && this.data && this.destSink) {
        void serviceRangeBridge(
          this.control,
          this.data,
          this.destSink.readAt,
          msg.offset ?? 0,
          msg.length ?? 0,
        );
      }
      return;
    }
    // Un-id'd progress notification from appendEmbeddedVideos' copy loop.
    if (msg && msg.type === "progress") {
      const p = e.data as { done?: number; total?: number };
      this.appendProgressCb?.(p.done ?? 0, p.total ?? 0);
      return;
    }
    if (msg && msg.type === "writeRequest") {
      if (this.control && this.data && this.destSink) {
        void serviceWriteBridge(
          this.control,
          this.data,
          this.destSink,
          msg.offset ?? 0,
          msg.length ?? 0,
        );
      }
      return;
    }
    if (msg && msg.type === "truncateRequest") {
      if (this.control && this.destSink) {
        void serviceTruncateBridge(
          this.control,
          this.destSink,
          msg.length ?? 0,
        );
      }
      return;
    }

    const { id, ...data } = e.data;
    const pending = this.pendingMessages.get(id);
    if (pending) {
      this.pendingMessages.delete(id);
      if (data.success) {
        pending.resolve(e.data);
      } else {
        let errorMessage = "Worker operation failed";
        if (typeof data.error === "string") {
          errorMessage = data.error;
        } else if (data.error && typeof data.error === "object") {
          errorMessage = JSON.stringify(data.error);
        }
        pending.reject(new Error(errorMessage));
      }
    }
  }

  private handleError(e: ErrorEvent): void {
    console.error("[StreamingH5Writer] Worker error:", e.message);
    for (const [id, pending] of this.pendingMessages) {
      pending.reject(new Error(`Worker error: ${e.message}`));
      this.pendingMessages.delete(id);
    }
  }

  private send(
    type: H5WorkerMessage["type"],
    payload?: Record<string, unknown>,
    transfer?: Transferable[],
  ): Promise<H5WorkerResponse> {
    return new Promise((resolve, reject) => {
      const id = ++this.messageId;
      this.pendingMessages.set(id, { resolve, reject });
      this.worker.postMessage({ type, payload, id }, transfer ?? []);
    });
  }

  /**
   * Initialize the h5wasm module in the worker.
   */
  async init(h5wasmUrl?: string): Promise<void> {
    await this.send("init", { h5wasmUrl });
  }

  /**
   * DUAL-BRIDGE FOUNDATION: open TWO h5wasm files at once in the Worker through
   * ONE SharedArrayBuffer bridge — a read-only SOURCE (backed by `source`, a
   * {@link RangeSource}) and a read+write DEST (backed by `destSink`, a
   * {@link RangeSink}, opened in h5wasm append mode `"a"`). Every bridged
   * request is tagged `target: 'source' | 'dest'` by the Worker so
   * {@link handleMessage} can route it to the right reader/sink (see there).
   *
   * The caller is responsible for having already opened `destPath` on the
   * native side in APPEND mode (no truncate — e.g. via a `write_open_append`-
   * style command) so `destSize`'s pre-existing bytes are actually present on
   * disk; this method does not create or truncate the destination file itself.
   * `destSize` seeds the Worker's writable device with the dest file's REAL
   * current size (rather than 0) so h5wasm's file-size probe can see — and
   * read — the pre-existing content instead of treating the file as empty.
   *
   * Requires cross-origin isolation (SharedArrayBuffer / COOP+COEP).
   */
  async openAppend(
    destSink: RangeSink,
    destPath: string,
    destSize: number,
    source: RangeSource,
    sourceFilename: string,
    h5wasmUrl?: string,
  ): Promise<void> {
    if (typeof SharedArrayBuffer === "undefined") {
      throw new Error(
        "RangeSink streaming requires SharedArrayBuffer (enable cross-origin isolation: COOP 'same-origin' + COEP 'require-corp')",
      );
    }
    await this.init(h5wasmUrl);
    this.destSink = destSink;
    this.sourceReader = source.readRange;

    // SAB layout: [0..CONTROL_BYTES) = Int32 control (idx0=STATE, idx1=RETLEN/
    // RESULT), then ONE shared data area. Both the source and dest devices
    // block on this same buffer (the Worker only ever has one bridge op in
    // flight at a time), distinguished purely by the `target` tag on each
    // posted request.
    const CONTROL_BYTES = 32;
    const MAX_CHUNK = 4 * 1024 * 1024;
    const sab = new SharedArrayBuffer(CONTROL_BYTES + MAX_CHUNK);
    this.control = new Int32Array(sab, 0, 8);
    this.data = new Uint8Array(sab, CONTROL_BYTES);

    await this.send("openAppend", {
      sab,
      controlBytes: CONTROL_BYTES,
      sourceFilename,
      sourceSize: source.size,
      destFilename: destPath,
      destSize,
    });
  }

  /**
   * Worker port of `writeEmbeddedVideoData` (write.ts) for the RAW path ONLY:
   * copies each {@link SerializableEmbedEntry}'s stored embedded-image blobs
   * from the open SOURCE file into a new `video{videoIndex}` group in the open
   * DEST file (both opened via {@link openAppend}), streamed in bounded byte
   * windows so peak JS memory stays ~one window rather than the whole
   * concatenated video. Mirrors the real writer's create-empty -> resize ->
   * write_slice pattern and its #213 backstop (throws rather than writing a
   * file with dropped images).
   *
   * Does NOT close either file — call {@link close} afterward (which sends
   * the Worker's `close` message, tearing down both the source and dest
   * mounts via `closeAppendFiles`).
   */
  async appendEmbeddedVideos(
    entries: SerializableEmbedEntry[],
    onProgress?: (done: number, total: number) => void,
  ): Promise<H5WorkerResponse> {
    this.appendProgressCb = onProgress;
    try {
      return await this.send("appendEmbeddedVideos", { entries });
    } finally {
      this.appendProgressCb = undefined;
    }
  }

  /**
   * Streaming new-embed (see `appendEncodedVideos`): start writing
   * `entry`'s `video{i}` group in the open DEST file. `collapseToGray` makes
   * the worker keep one channel of each frame, as a grayscale video's reads.
   */
  async beginEncodedVideo(
    entry: SerializableEncodeEntry,
    collapseToGray: boolean,
  ): Promise<void> {
    await this.send("beginEncodedVideo", { entry, collapseToGray });
  }

  /**
   * PNG-encode (in the worker) and append one frame of the video begun with
   * {@link beginEncodedVideo}. The frame's bitmap or buffer is transferred, so
   * the caller must not use it afterwards.
   */
  async appendEncodedFrame(
    frameIdx: number,
    frame: EncodeFramePayload,
  ): Promise<void> {
    const transfer: Transferable[] =
      frame.kind === "bitmap"
        ? [frame.bitmap]
        : [
            (frame.kind === "png" ? frame.bytes : frame.data)
              .buffer as ArrayBuffer,
          ];
    await this.send("appendEncodedFrame", { frameIdx, frame }, transfer);
  }

  /** Finish the begun video: its frame index datasets and attributes. */
  async endEncodedVideo(): Promise<{ framesWritten: number }> {
    const res = await this.send("endEncodedVideo");
    return { framesWritten: Number(res.framesWritten ?? 0) };
  }

  /**
   * SINGLE-file write B-seam for the in-place label save: open ONLY the DEST
   * file (`destPath`, backed by `destSink`) in h5wasm append mode `"a"` (no
   * truncate), no source file. The counterpart to {@link openAppend} that
   * {@link updateLabelsInPlace} uses to patch the small label tables of an
   * existing `.pkg.slp` without re-copying its embedded video.
   *
   * The caller must have already opened `destPath` on the native side in a
   * NON-truncating mode so its pre-existing bytes are intact; `destSize` seeds
   * the Worker's writable device with the file's real current size. Requires
   * cross-origin isolation (SharedArrayBuffer / COOP+COEP).
   */
  async openWrite(
    destSink: RangeSink,
    destPath: string,
    destSize: number,
    h5wasmUrl?: string,
  ): Promise<void> {
    if (typeof SharedArrayBuffer === "undefined") {
      throw new Error(
        "RangeSink streaming requires SharedArrayBuffer (enable cross-origin isolation: COOP 'same-origin' + COEP 'require-corp')",
      );
    }
    await this.init(h5wasmUrl);
    this.destSink = destSink;
    this.sourceReader = undefined; // single-file: all bridge ops are 'dest'

    const CONTROL_BYTES = 32;
    const MAX_CHUNK = 4 * 1024 * 1024;
    const sab = new SharedArrayBuffer(CONTROL_BYTES + MAX_CHUNK);
    this.control = new Int32Array(sab, 0, 8);
    this.data = new Uint8Array(sab, CONTROL_BYTES);

    await this.send("openWrite", {
      sab,
      controlBytes: CONTROL_BYTES,
      destFilename: destPath,
      destSize,
    });
  }

  /**
   * Patch the open DEST file's label tables in place from `update` (see
   * {@link openWrite}). Each table's `number[][]` rows are pre-flattened to a
   * row-major `Float64Array` here and TRANSFERRED to the Worker (rather than
   * structured-cloning a nested array), which packs them into the on-disk layout
   * (flat matrix or compound record) and replaces the `/metadata` json attr when
   * `update.metadataJson` is set. Never touches any `video{i}/video` group.
   */
  async updateLabelsInPlace(
    update: LabelTableUpdate,
  ): Promise<H5WorkerResponse> {
    const tables: Array<{
      name: string;
      fields: string[];
      flat: Float64Array;
      rowCount: number;
      colCount: number;
      dtype: string;
    }> = [];
    const transfer: Transferable[] = [];
    const push = (
      name: string,
      dtype: string,
      table: LabelTable | undefined,
    ): void => {
      if (!table) return;
      const rowCount = table.rows.length;
      const colCount = table.fields.length;
      const flat = new Float64Array(rowCount * colCount);
      for (let i = 0; i < rowCount; i++) {
        const row = table.rows[i];
        const off = i * colCount;
        for (let j = 0; j < colCount; j++) flat[off + j] = row[j];
      }
      tables.push({
        name,
        fields: table.fields,
        flat,
        rowCount,
        colCount,
        dtype,
      });
      transfer.push(flat.buffer);
    };
    push("frames", "<i8", update.frames);
    push("instances", "<f8", update.instances);
    push("points", "<f8", update.points);
    push("pred_points", "<f8", update.predPoints);
    push("negative_frames", "<i8", update.negativeFrames);

    return this.send(
      "updateLabelsInPlace",
      { tables, metadataJson: update.metadataJson ?? null },
      transfer,
    );
  }

  /**
   * SINGLE-file OPFS write B-seam (Scenario A — browser large save). Opens a DEST
   * file backed by an OPFS `FileSystemSyncAccessHandle` that the WORKER owns and
   * writes to synchronously. Unlike {@link openWrite}/{@link openAppend}, this
   * needs NO SharedArrayBuffer, NO Atomics bridge, and NO cross-origin isolation
   * (COOP/COEP) — the whole read/write/truncate loop runs in-thread against the
   * sync handle — so it works on plain static hosting (e.g. GitHub Pages).
   *
   * `destSize` 0 (default) creates a FRESH file; a non-zero size opens an existing
   * OPFS file non-truncating (append mode, for later re-embed). The caller drives
   * the SLP write against the opened dest file, then calls {@link close}.
   */
  async openWriteOpfs(
    opfsPath: string,
    destSize = 0,
    h5wasmUrl?: string,
  ): Promise<void> {
    await this.init(h5wasmUrl);
    // No bridge: the Worker services its own device via the OPFS sync handle, so
    // there is nothing to wire on the main thread (no destSink/control/data) and
    // handleMessage never receives write/range/truncate requests for this path.
    this.destSink = undefined;
    this.sourceReader = undefined;
    this.control = undefined;
    this.data = undefined;
    await this.send("openWriteOpfs", { opfsPath, destSize });
  }

  /**
   * DUAL OPFS append (browser re-save / Save As of an already-embedded pkg.slp).
   * Reads the SOURCE `sourceFile` (the file the user opened) on-demand via WORKERFS
   * and writes the DEST straight to the OPFS file at `destOpfsPath` via the
   * sync-handle device — both in-Worker, so NO SharedArrayBuffer / cross-origin
   * isolation. The dest is seeded with `structureBytes` (the small labels/metadata
   * structure from {@link saveSlpStructureToBytes}, WITHOUT embedded images) and
   * opened append-mode; call {@link appendEmbeddedVideos} afterwards to copy the
   * big `video{i}/video` datasets from source to dest, then {@link close}.
   */
  async openAppendOpfs(
    destOpfsPath: string,
    sourceFile: File,
    structureBytes: Uint8Array,
    sourceFilename?: string,
    h5wasmUrl?: string,
  ): Promise<void> {
    await this.init(h5wasmUrl);
    // No bridge: source is WORKERFS (sync, in-Worker), dest is the OPFS sync-handle
    // device — nothing to wire on the main thread.
    this.destSink = undefined;
    this.sourceReader = undefined;
    this.control = undefined;
    this.data = undefined;
    await this.send("openAppendOpfs", {
      destOpfsPath,
      sourceFile,
      structureBytes,
      sourceFilename,
    });
  }

  /**
   * Close the file and terminate the worker. Best-effort: if the worker-side
   * close fails, the worker is still terminated so no handle is leaked.
   */
  async close(): Promise<void> {
    try {
      await this.send("close");
    } catch (err) {
      console.error("[StreamingH5Writer] close failed:", err);
    }
    this.worker.terminate();
  }
}

/**
 * Check if streaming via Web Worker is supported in the current environment.
 */
export function isStreamingSupported(): boolean {
  return (
    typeof Worker !== "undefined" &&
    typeof Blob !== "undefined" &&
    typeof URL !== "undefined"
  );
}

/**
 * Open a remote HDF5 file with streaming support.
 *
 * @param url - URL to the HDF5 file
 * @param options - Optional settings
 * @returns A StreamingH5File instance
 */
export async function openStreamingH5(
  url: string,
  options?: StreamingH5Options,
): Promise<StreamingH5File> {
  if (!isStreamingSupported()) {
    throw new Error("Streaming HDF5 requires Web Worker support");
  }

  const file = new StreamingH5File();
  await file.open(url, options);
  return file;
}

/**
 * Open an HDF5 file from any supported source using a Web Worker.
 *
 * This is the recommended way to open HDF5 files in the browser as it
 * offloads all h5wasm operations to a Web Worker, avoiding main thread blocking.
 *
 * @param source - URL string, File object, ArrayBuffer, or Uint8Array
 * @param options - Optional settings
 * @returns A StreamingH5File instance
 *
 * @example
 * ```typescript
 * // From URL
 * const file = await openH5Worker("https://example.com/data.h5");
 *
 * // From File (file input)
 * const file = await openH5Worker(inputElement.files[0]);
 *
 * // From ArrayBuffer
 * const file = await openH5Worker(arrayBuffer);
 * ```
 */
export async function openH5Worker(
  source: StreamingH5Source,
  options?: StreamingH5Options,
): Promise<StreamingH5File> {
  if (!isStreamingSupported()) {
    throw new Error("Web Worker HDF5 access requires Worker/Blob/URL support");
  }

  const file = new StreamingH5File();
  await file.openAny(source, options);
  return file;
}
