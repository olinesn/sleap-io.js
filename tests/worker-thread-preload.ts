/**
 * Loaded into each worker thread by `onRealWorkerThreads` (`worker-file.ts`): the one
 * thing a browser worker has that the worker code needs and a Bun worker doesn't. The
 * only script it loads is h5wasm, which a browser leaves as a global.
 */
import { readFileSync } from "node:fs";

(self as any).importScripts = (path: string) => {
  new Function(`${readFileSync(path, "utf8")}\nglobalThis.h5wasm = h5wasm;`)();
};
