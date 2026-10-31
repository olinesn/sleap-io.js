/**
 * Node.js-specific HDF5 provider.
 *
 * This module is imported as a side-effect by the Node entry point (index.ts).
 * It registers Node-specific implementations of getH5Module and openH5File
 * so that the shared h5.ts module remains free of Node-only imports
 * (h5wasm/node, node:fs, node:os, node:path).
 *
 * The browser entry point never imports this file, ensuring the browser
 * bundle contains no references to Node built-in modules.
 */
import {
  _registerNodeH5,
  _registerNodeFileOps,
  fetchRemoteSlpBytes,
  guardH5Status,
  type H5Module,
  type OpenH5Options,
  type SlpSource,
  type H5File,
} from "./h5.js";
import { isUrl } from "../../io/remote.js";
import { _registerFileWriter } from "./write.js";

let modulePromise: Promise<H5Module> | null = null;

async function getH5ModuleNode(): Promise<H5Module> {
  if (!modulePromise) {
    modulePromise = (async () => {
      const module = await import("h5wasm/node");
      await module.ready;
      guardH5Status(module as H5Module);
      return module as H5Module;
    })();
  }
  return modulePromise;
}

async function openH5FileNode(
  module: H5Module,
  source: SlpSource,
  options?: OpenH5Options,
): Promise<{ file: H5File; close: () => void }> {
  // Remote URL: h5wasm/node cannot open a URL directly, so download the bytes
  // (header-aware, scheme-resolved, redacted) and stage them to a temp file.
  if (typeof source === "string" && isUrl(source)) {
    const bytes = await fetchRemoteSlpBytes(source, options);
    return openBytesNode(module, bytes);
  }

  if (typeof source === "string") {
    const file = new module.File(source, "r");
    return { file, close: () => file.close() };
  }

  if (source instanceof Uint8Array || source instanceof ArrayBuffer) {
    const data = source instanceof Uint8Array ? source : new Uint8Array(source);
    return openBytesNode(module, data);
  }

  throw new Error(
    "Node environments only support string paths or byte buffers for SLP inputs.",
  );
}

/** Stage bytes to a temp file and open them with h5wasm/node. */
async function openBytesNode(
  module: H5Module,
  data: Uint8Array,
): Promise<{ file: H5File; close: () => void }> {
  const { writeFileSync, unlinkSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const tempPath = join(
    tmpdir(),
    `sleap-io-${Date.now()}-${Math.random().toString(16).slice(2)}.slp`,
  );
  writeFileSync(tempPath, data);
  const file = new module.File(tempPath, "r");
  return {
    file,
    close: () => {
      file.close();
      unlinkSync(tempPath);
    },
  };
}

// Register Node providers on import (side-effect)
_registerNodeH5(getH5ModuleNode, openH5FileNode);

_registerFileWriter(async (filename: string, bytes: Uint8Array) => {
  const { writeFile } = await import("node:fs/promises");
  await writeFile(filename, bytes);
});

// Node filesystem ops for direct-file codecs (e.g. Analysis-HDF5). Keeping the
// node: imports here means the shared h5.ts (and thus the browser bundle) never
// references Node built-ins.
_registerNodeFileOps({
  writeFile: async (filename: string, bytes: Uint8Array) => {
    const { writeFile } = await import("node:fs/promises");
    await writeFile(filename, bytes);
  },
  fileExists: async (path: string) => {
    const { existsSync } = await import("node:fs");
    return existsSync(path);
  },
  readPackageVersion: async () => {
    try {
      const { readFile } = await import("node:fs/promises");
      const { fileURLToPath } = await import("node:url");
      const { dirname, join } = await import("node:path");
      const here = dirname(fileURLToPath(import.meta.url));
      // h5-node lives at <root>/(src|dist)/codecs/slp/, so package.json is 3 up.
      const candidates = [
        join(here, "..", "..", "..", "package.json"),
        join(here, "..", "..", "..", "..", "package.json"),
      ];
      for (const candidate of candidates) {
        try {
          const raw = await readFile(candidate, "utf-8");
          const pkg = JSON.parse(raw) as { version?: string };
          if (pkg.version) return pkg.version;
        } catch {
          // try next candidate
        }
      }
    } catch {
      // ignore
    }
    return null;
  },
});
