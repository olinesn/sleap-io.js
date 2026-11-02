import {
  RemoteIOError,
  fetchRetrying,
  isUrl,
  redactUrl,
  resolveUrl,
  statusToMessage,
} from "../../io/remote.js";
import { openGdrive } from "../../io/gdrive.js";

export type H5Module = typeof import("h5wasm");
export type H5File = InstanceType<H5Module["File"]>;

export type SlpSource =
  | string
  | ArrayBuffer
  | Uint8Array
  | File
  | FileSystemFileHandle;
export type StreamMode = "auto" | "range" | "download";

export type OpenH5Options = {
  /**
   * Streaming mode for remote files:
   * - "auto": Try range requests, fall back to download
   * - "range": Use HTTP range requests (requires Worker support in browser)
   * - "download": Always download the entire file
   */
  stream?: StreamMode;
  /** Filename hint for the HDF5 file */
  filenameHint?: string;
  /**
   * Extra HTTP request headers (e.g. `{ Authorization: "Bearer …" }`) applied to
   * every remote byte fetch for this file AND persisted onto embedded-video
   * backends so later reopens/probes stay authenticated. Header NAMES are
   * case-insensitive; `"Accept-Encoding"` is always overridden to `"identity"`
   * on range requests. Ignored for Google Drive URLs (credentials are stripped).
   *
   * Limitation: `createLazyFile` (Emscripten synchronous XHR) cannot carry
   * custom headers, so authenticated remote `.slp` is downloaded in full;
   * range streaming with custom headers is not yet supported on the main thread.
   */
  headers?: Record<string, string>;
  /**
   * URL to the h5wasm IIFE bundle loaded by the streaming Worker (via
   * `importScripts`). Defaults to a CDN. Set this to a **same-origin** URL when
   * the page is cross-origin-isolated (COOP/COEP), since COEP blocks the
   * cross-origin CDN `importScripts` — e.g. a desktop app that serves its own
   * bundled `h5wasm.js`.
   */
  h5wasmUrl?: string;
};

// Re-export streaming utilities for advanced use cases
export {
  StreamingH5File,
  openStreamingH5,
  isStreamingSupported,
} from "./h5-streaming.js";

type H5FileSystem = {
  writeFile: (path: string, data: Uint8Array) => void;
  readFile?: (path: string) => Uint8Array;
  unlink?: (path: string) => void;
  mkdir?: (path: string) => void;
  rmdir?: (path: string) => void;
  mount?: (fs: unknown, opts: unknown, mountpoint: string) => void;
  unmount?: (mountpoint: string) => void;
  createLazyFile?: (
    parent: string,
    name: string,
    url: string,
    canRead: boolean,
    canWrite: boolean,
  ) => void;
  filesystems?: Record<string, unknown>;
};

// Node provider hooks — registered by h5-node.ts (imported as side-effect from Node entry point).
// When null, only browser code paths are used, keeping the browser bundle free of Node-only imports.
let _nodeGetModule: (() => Promise<H5Module>) | null = null;
let _nodeOpenFile:
  | ((
      module: H5Module,
      source: SlpSource,
      options?: OpenH5Options,
    ) => Promise<{ file: H5File; close: () => void }>)
  | null = null;

/**
 * Register Node.js-specific h5wasm provider.
 * Called as a side-effect when the Node entry point imports h5-node.ts.
 * @internal
 */
export function _registerNodeH5(
  getModule: () => Promise<H5Module>,
  openFile: (
    module: H5Module,
    source: SlpSource,
    options?: OpenH5Options,
  ) => Promise<{ file: H5File; close: () => void }>,
): void {
  _nodeGetModule = getModule;
  _nodeOpenFile = openFile;
}

// Node-only filesystem ops — registered by h5-node.ts (imported as a side-effect
// from the Node entry point). They stay null in the browser so this shared,
// browser-safe module never references Node built-ins (issue #70).
let _nodeWriteFile:
  | ((path: string, bytes: Uint8Array) => Promise<void>)
  | null = null;
let _nodeFileExists: ((path: string) => Promise<boolean>) | null = null;
let _nodeReadPackageVersion: (() => Promise<string | null>) | null = null;

/**
 * Register Node.js filesystem operations used by codecs that read/write files
 * directly (e.g. the Analysis-HDF5 writer). Called as a side-effect when the
 * Node entry point imports h5-node.ts.
 * @internal
 */
export function _registerNodeFileOps(ops: {
  writeFile: (path: string, bytes: Uint8Array) => Promise<void>;
  fileExists: (path: string) => Promise<boolean>;
  readPackageVersion: () => Promise<string | null>;
}): void {
  _nodeWriteFile = ops.writeFile;
  _nodeFileExists = ops.fileExists;
  _nodeReadPackageVersion = ops.readPackageVersion;
}

/** Write bytes to a path via the Node provider. Throws in the browser. */
export async function nodeWriteFile(
  path: string,
  bytes: Uint8Array,
): Promise<void> {
  if (!_nodeWriteFile) {
    throw new Error(
      "Writing files requires a Node.js environment. This codec's writer is Node-only.",
    );
  }
  await _nodeWriteFile(path, bytes);
}

/**
 * Whether a path exists, via the Node provider. Resolves to null when no Node
 * provider is registered (e.g. the browser), meaning "unknown".
 */
export async function nodeFileExists(path: string): Promise<boolean | null> {
  return _nodeFileExists ? _nodeFileExists(path) : null;
}

/** Read the package version via the Node provider, or null if unavailable. */
export async function nodeReadPackageVersion(): Promise<string | null> {
  return _nodeReadPackageVersion ? _nodeReadPackageVersion() : null;
}

// Browser-only module cache — unused when Node provider is registered, but kept
// at module scope so the browser code path caches across calls.
let modulePromise: Promise<H5Module> | null = null;

export async function getH5Module(): Promise<H5Module> {
  if (_nodeGetModule) {
    return _nodeGetModule();
  }
  if (!modulePromise) {
    modulePromise = (async () => {
      const module = await import("h5wasm");
      await module.ready;
      return module as H5Module;
    })();
  }
  return modulePromise;
}

/**
 * The underlying Emscripten `Module` from h5wasm — its low-level
 * `_malloc`/`_free`/`HEAPU8`/`get_dataset_data` surface — or `null` if it cannot
 * be reached (older/renamed h5wasm build).
 *
 * Exposed for the variable-length (vlen) embedded-frame read path, which must
 * bypass h5wasm's high-level `Dataset.slice()`: after a sliced read, h5wasm's
 * `reclaim_vlen_memory` reclaims the dataset's FULL dataspace over a single-
 * element buffer and corrupts the heap (intermittent "memory access out of
 * bounds" / WASM abort). See {@link readVlenElementManual} in
 * `../../video/embedded-frame.ts` and `scratch/vlen/upstream-fix.md`.
 *
 * `Module` is a real named export of both `h5wasm` (ESM) and `h5wasm/node`, and
 * `h5wasm.Module` exists on the IIFE build used by the worker, so this resolves
 * in Node, the browser, and the streaming worker alike.
 *
 * @internal
 */
export async function getH5EmscriptenModule(): Promise<unknown> {
  const ns = await getH5Module();
  return (ns as unknown as { Module?: unknown }).Module ?? null;
}

export async function openH5File(
  source: SlpSource,
  options?: OpenH5Options,
): Promise<{ file: H5File; close: () => void }> {
  const module = await getH5Module();

  if (_nodeOpenFile) {
    return _nodeOpenFile(module, source, options);
  }

  return openH5FileBrowser(module, source, options);
}

/** What to tell someone whose file HDF5 can't open. The Worker says the same. */
export function cannotOpenAsHdf5Message(name: string): string {
  return (
    `Could not open '${name}' as an HDF5 file. It may be missing, incomplete ` +
    "(cut off, or still downloading), or not a SLEAP file at all."
  );
}

/**
 * Throw if `openH5File` did not really open `source`. Opening doesn't throw for a
 * file HDF5 can't open: it hands back an invalid id, and the first read then fails
 * with "error - name not defined!". For a caller that needs the file now; a video
 * backend may be built on a file it never ends up reading.
 */
export function checkOpenedAsHdf5(
  file: H5File,
  source: SlpSource,
  options?: OpenH5Options,
): void {
  if (Number(file.file_id) >= 0) return;
  const name =
    typeof source === "string"
      ? isUrl(source)
        ? redactUrl(source)
        : source
      : (options?.filenameHint ??
        (source as { name?: string }).name ??
        "this file");
  throw new Error(cannotOpenAsHdf5Message(name));
}

function isFileHandle(value: SlpSource): value is FileSystemFileHandle {
  return typeof value === "object" && value !== null && "getFile" in value;
}

async function openH5FileBrowser(
  module: H5Module,
  source: SlpSource,
  options?: OpenH5Options,
): Promise<{ file: H5File; close: () => void }> {
  const fs = getH5FileSystem(module);

  if (typeof source === "string" && isUrl(source)) {
    return openFromUrl(module, fs, source, options);
  }

  if (isFileHandle(source)) {
    const file = await source.getFile();
    return openFromFile(module, fs, file);
  }

  if (typeof File !== "undefined" && source instanceof File) {
    return openFromFile(module, fs, source);
  }

  if (source instanceof Uint8Array || source instanceof ArrayBuffer) {
    const data = source instanceof Uint8Array ? source : new Uint8Array(source);
    const filename = "/tmp-slp.slp";
    fs.writeFile(filename, data);
    const file = new module.File(filename, "r");
    return { file, close: () => file.close() };
  }

  if (typeof source === "string") {
    return openFromUrl(module, fs, source, options);
  }

  throw new Error("Unsupported SLP source type for browser environment.");
}

async function openFromUrl(
  module: H5Module,
  fs: H5FileSystem,
  url: string,
  options?: OpenH5Options,
): Promise<{ file: H5File; close: () => void }> {
  const filename =
    options?.filenameHint ??
    url.split("/").pop()?.split("?")[0] ??
    "slp-data.slp";
  const streamMode = options?.stream ?? "auto";
  const hasHeaders = !!(
    options?.headers && Object.keys(options.headers).length > 0
  );

  // Resolve the scheme: http(s) passthrough, gs:// -> storage.googleapis.com,
  // Drive flagged for the buffered resolver, s3/az/abfs rejected (redacted).
  const { url: resolved, gdrive } = resolveUrl(url);

  // Google Drive: resolve ONCE to bytes and feed the in-memory open path. Drive
  // is always download-mode; streamMode/range are ignored.
  if (gdrive) {
    const bytes = await openGdrive(url, { headers: options?.headers });
    const localPath = "/tmp-slp.slp";
    fs.writeFile(localPath, bytes);
    const file = new module.File(localPath, "r");
    return { file, close: () => file.close() };
  }

  // createLazyFile (Emscripten synchronous XHR) cannot carry custom headers.
  // Only use it for range/auto WITHOUT headers; with headers we fall back to the
  // header-aware full-download path below so the request stays authenticated.
  if (
    !hasHeaders &&
    fs.createLazyFile &&
    (streamMode === "auto" || streamMode === "range")
  ) {
    const mountPath = `/slp-remote-${Date.now()}`;
    fs.mkdir?.(mountPath);
    try {
      fs.createLazyFile(mountPath, filename, resolved, true, false);
      const file = new module.File(`${mountPath}/${filename}`, "r");
      return {
        file,
        close: () => {
          file.close();
          fs.unlink?.(`${mountPath}/${filename}`);
          fs.rmdir?.(mountPath);
        },
      };
    } catch {
      fs.rmdir?.(mountPath);
    }
  }

  // Header-aware full download, retried with backoff on transient failures /
  // retryable statuses (429/5xx). All URL-bearing errors go through
  // RemoteIOError (redacted); the raw transport error is never re-thrown.
  const init: RequestInit = { headers: options?.headers ?? {} };
  const response = await fetchRetrying(resolved, init);
  if (!response.ok) {
    throw new RemoteIOError({
      message: statusToMessage(response.status),
      url: resolved,
      status: response.status,
    });
  }
  const buffer = new Uint8Array(await response.arrayBuffer());
  const localPath = "/tmp-slp.slp";
  fs.writeFile(localPath, buffer);
  const file = new module.File(localPath, "r");
  return { file, close: () => file.close() };
}

let localFileCount = 0;

async function openFromFile(
  module: H5Module,
  fs: H5FileSystem,
  file: File,
): Promise<{ file: H5File; close: () => void }> {
  // WORKERFS reads a File through FileReaderSync, which only Workers have. On the
  // main thread it mounts fine and then aborts h5wasm on the first read.
  if (
    fs.mount &&
    fs.filesystems &&
    fs.filesystems.WORKERFS &&
    "FileReaderSync" in globalThis
  ) {
    const mountPath = `/slp-local-${Date.now()}`;
    fs.mkdir?.(mountPath);
    // Under a name of our own, so the path can't disagree with the mount. A hint is
    // only for showing, and file.name needn't work as a path. The worker does the
    // same: mountWorkerfsFile in h5-worker.ts.
    fs.mount(
      fs.filesystems.WORKERFS,
      { blobs: [{ name: "local.slp", data: file }] },
      mountPath,
    );
    const h5file = new module.File(`${mountPath}/local.slp`, "r");
    return {
      file: h5file,
      close: () => {
        h5file.close();
        fs.unmount?.(mountPath);
        fs.rmdir?.(mountPath);
      },
    };
  }

  // Its own path, since a video backend holds its file open while another opens,
  // and removed on close, since this is a copy of the whole file.
  const buffer = new Uint8Array(await file.arrayBuffer());
  ensureH5StagingDir(module);
  const localPath = `/tmp/slp-local-${Date.now()}-${++localFileCount}.slp`;
  fs.writeFile(localPath, buffer);
  const h5file = new module.File(localPath, "r");
  return {
    file: h5file,
    close: () => {
      h5file.close();
      fs.unlink?.(localPath);
    },
  };
}

/**
 * Resolve a remote `.slp` URL and download its bytes (header-aware, redacted).
 *
 * Routes through {@link resolveUrl}: `gs://` -> `storage.googleapis.com`,
 * Google Drive -> {@link openGdrive}, `s3://`/`az://`/`abfs://` -> a redacted
 * {@link RemoteIOError}. Used by the Node provider (which cannot stream a URL
 * via h5wasm) and as a building block for the browser download path. All
 * URL-bearing errors go through {@link RemoteIOError}; the raw transport error
 * is never re-thrown.
 *
 * @internal
 */
export async function fetchRemoteSlpBytes(
  url: string,
  options?: OpenH5Options,
): Promise<Uint8Array> {
  const { url: resolved, gdrive } = resolveUrl(url);
  if (gdrive) {
    return openGdrive(url, { headers: options?.headers });
  }
  const init: RequestInit = { headers: options?.headers ?? {} };
  const response = await fetchRetrying(resolved, init);
  if (!response.ok) {
    throw new RemoteIOError({
      message: statusToMessage(response.status),
      url: resolved,
      status: response.status,
    });
  }
  return new Uint8Array(await response.arrayBuffer());
}

export function getH5FileSystem(module: H5Module): H5FileSystem {
  const fs = (module as unknown as { FS?: H5FileSystem }).FS;
  if (!fs) {
    throw new Error("h5wasm FS is not available.");
  }
  return fs;
}

/**
 * Ensure the in-memory staging directory used for writing h5/SLP output exists
 * in h5wasm's Emscripten filesystem.
 *
 * Writers stage their file at `/tmp/...` inside the wasm FS and read the bytes
 * back. Node's h5wasm build pre-creates `/tmp`, but Bun's does not — there,
 * opening a `File` at `/tmp/...` yields an invalid file id, so every subsequent
 * `create_dataset`/`file.get`/attribute call fails. Creating the directory up
 * front fixes Bun and is a no-op elsewhere: `mkdir` throws when the directory
 * already exists, which we ignore.
 */
export function ensureH5StagingDir(module: H5Module): void {
  try {
    getH5FileSystem(module).mkdir?.("/tmp");
  } catch {
    // Directory already exists (Node/browser) — nothing to do.
  }
}
