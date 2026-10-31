/**
 * The streaming reader's real worker code and real `StreamingH5File`, for tests.
 *
 * `readSlpStreaming` needs a browser Worker, so these run the worker over h5wasm/node
 * with only the message channel replaced. What a browser adds is the `postMessage`
 * boundary and the byte source.
 */
import { ready, File as H5File } from "h5wasm/node";
import { readFromStreamingFile } from "../src/codecs/slp/read-streaming.js";
import { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import { H5_WORKER_CODE } from "../src/codecs/slp/h5-worker.js";
import { getH5EmscriptenModule } from "../src/codecs/slp/h5.js";
import type { Labels } from "../src/index.js";

export async function tmpPath(): Promise<string> {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    /* already exists */
  }
  return `/tmp/slp30_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
}

/** A `StreamingH5File` talking to the worker code, over a copy of `bytes`. */
export async function workerFile(
  bytes: ArrayBuffer,
): Promise<{ file: StreamingH5File; close: () => void }> {
  const module = await ready;
  const p = await tmpPath();
  module.FS.writeFile(p, new Uint8Array(bytes));
  const h5 = new H5File(p, "r");
  const self: any = {
    postMessage: (msg: unknown) => file.handleMessage({ data: msg }),
  };
  const install = new Function(
    "self",
    `${H5_WORKER_CODE}\nreturn function(f, m) { currentFile = f; h5wasmModule = m; };`,
  )(self);
  install(h5, { Module: await getH5EmscriptenModule() });
  const file: any = Object.create(StreamingH5File.prototype);
  Object.assign(file, {
    worker: {
      postMessage: (message: unknown) => self.onmessage({ data: message }),
      terminate: () => {},
    },
    messageId: 0,
    pendingMessages: new Map(),
    _keys: h5.keys(),
    _isOpen: true,
  });
  return {
    file: file as StreamingH5File,
    close: () => {
      h5.close();
      module.FS.unlink(p);
    },
  };
}

/** Read through a real worker, recording every request the reader makes. */
export async function readRecording(
  bytes: ArrayBuffer,
  intercept?: (type: string, path: string) => Error | undefined,
  lazy = false,
): Promise<{ labels: Labels; requests: Array<[string, string]> }> {
  const { file, close } = await workerFile(bytes);
  const requests: Array<[string, string]> = [];
  const send = (file as any).send.bind(file);
  (file as any).send = (type: string, payload: any) => {
    const path = String(payload?.path ?? "");
    requests.push([type, path]);
    const failure = intercept?.(type, path);
    return failure ? Promise.reject(failure) : send(type, payload);
  };
  try {
    const labels = await readFromStreamingFile(
      file,
      "mice.slp",
      undefined,
      false,
      undefined,
      false,
      lazy,
    );
    return { labels, requests };
  } finally {
    close();
  }
}
