/**
 * Reading a file the user picked (a `File`), when the Worker can't read it, and on
 * the main thread.
 *
 * In a browser, `loadSlp` reads through a Worker and used to retry any failure on
 * the main thread. For a `File`, that retry mounted WORKERFS, which reads through
 * `FileReaderSync`, an API only Workers have: the mount works and the first read
 * aborts h5wasm. So every failed read of a picked file, a corrupt file say, told
 * the user "Aborted(undefined)", and `readSlp(file)` on the main thread aborted on
 * any file at all.
 *
 * The Node suite has no Worker or WORKERFS, so this file mocks the module that
 * creates the Worker, gives `loadSlp` a browser-looking environment, and reaches
 * the browser's open path by unregistering the Node one. Module mocks are per
 * process and can't be undone, which is why this is its own file (see `bun-test.ts`).
 * The real thing was checked in Chromium when this was written.
 */
import { describe, it, expect, afterEach, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as actualStreaming from "../src/codecs/slp/h5-streaming.js";
import {
  _registerNodeH5,
  cannotOpenAsHdf5Message,
  getH5FileSystem,
  getH5Module,
  openH5File,
} from "../src/codecs/slp/h5.js";

let openWorker: () => Promise<unknown> = async () => {
  throw new Error("no fake Worker file set");
};
vi.mock("../src/codecs/slp/h5-streaming.js", () => ({
  ...actualStreaming,
  isStreamingSupported: () => true,
  openH5Worker: () => openWorker(),
}));

const { loadSlp } = await import("../src/io/main.js");
const { readSlp, readSlpLazy } = await import("../src/codecs/slp/read.js");
const { H5_WORKER_CODE } = await import("../src/codecs/slp/h5-worker.js");
const { workerFile } = await import("./worker-file");

const FIXTURE = fileURLToPath(
  new URL("./data/slp/minimal_instance.slp", import.meta.url),
);
const fixtureBytes = () =>
  new Uint8Array(readFileSync(FIXTURE)).buffer as ArrayBuffer;
const fixtureFile = () => new File([fixtureBytes()], "mice.slp");

/** A Worker file that opened but fails on the first read, as for a corrupt file. */
const corruptWorkerFile = () => ({
  getAttrs: async () => ({ json: { value: "{broken" } }),
  close: async () => {},
});

const versions = process.versions;
const stubBrowser = () => {
  vi.stubGlobal("window", {});
  Object.defineProperty(process, "versions", {
    value: { ...versions, node: undefined },
    configurable: true,
  });
};

/** Use the browser's open path from here on, with Node's h5wasm under it. */
const useBrowserOpenPath = async () => {
  const module = await getH5Module();
  _registerNodeH5(async () => module, null as never);
  return getH5FileSystem(module);
};

afterEach(() => {
  vi.unstubAllGlobals();
  Object.defineProperty(process, "versions", {
    value: versions,
    configurable: true,
  });
  vi.restoreAllMocks();
});

describe("loadSlp, when the Worker can't read a picked file", () => {
  it("shows what went wrong, without reading the file again on the main thread", async () => {
    stubBrowser();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    openWorker = async () => corruptWorkerFile();
    // The bytes are fine, so a retry on the main thread would succeed: only the
    // Worker's own failure can make this reject.
    await expect(loadSlp(fixtureFile())).rejects.toThrow(SyntaxError);
    expect(warn).not.toHaveBeenCalled();
  });

  it("the same when the Worker can't even start", async () => {
    stubBrowser();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    openWorker = async () => {
      throw new Error("Failed to load h5wasm in the Worker");
    };
    await expect(loadSlp(fixtureFile())).rejects.toThrow(
      "Failed to load h5wasm in the Worker",
    );
    expect(warn).not.toHaveBeenCalled();
  });

  it("the same for a FileSystemFileHandle, which is a picked file too", async () => {
    stubBrowser();
    vi.stubGlobal("FileSystemFileHandle", class {});
    openWorker = async () => {
      throw new Error("Failed to load h5wasm in the Worker");
    };
    const file = fixtureFile();
    const handle = { kind: "file", name: file.name, getFile: async () => file };
    await expect(
      loadSlp(handle as unknown as FileSystemFileHandle),
    ).rejects.toThrow("Failed to load h5wasm in the Worker");
  });
});

describe("loadSlp, for sources other than a picked file", () => {
  it("still retries bytes on the main thread", async () => {
    stubBrowser();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    openWorker = async () => corruptWorkerFile();
    const labels = await loadSlp(fixtureBytes(), { openVideos: false });
    expect(labels.labeledFrames.length).toBe(1);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("falling back");
  });

  it("does not retry when streaming was asked for by name", async () => {
    stubBrowser();
    openWorker = async () => {
      throw new Error("Failed to load h5wasm in the Worker");
    };
    await expect(
      loadSlp(fixtureBytes(), { openVideos: false, h5: { stream: "range" } }),
    ).rejects.toThrow("Failed to load h5wasm in the Worker");
  });
});

describe("loadSlp, when the Worker can't read one table of a file", () => {
  // A 2.x file, with all five tables a failed read used to leave out.
  const legacyBytes = () =>
    new Uint8Array(
      readFileSync(
        fileURLToPath(
          new URL("./data/slp/py_written_30_twin_legacy.slp", import.meta.url),
        ),
      ),
    ).buffer as ArrayBuffer;
  const poses = (labels: Awaited<ReturnType<typeof loadSlp>>) =>
    labels.labeledFrames.map((frame) => [
      frame.frameIdx,
      frame.isNegative,
      frame.instances.map((instance) => [
        instance.constructor.name,
        instance.points.map((point) => point.xy),
      ]),
    ]);

  /** A Worker file that reads the real file, apart from one table that fails. */
  const fileFailingOn = async (table: string) => {
    const { file } = await workerFile(legacyBytes());
    const send = (file as any).send.bind(file);
    (file as any).send = (type: string, payload: any) =>
      type === "getDatasetValue" && payload?.path === table
        ? Promise.reject(new Error(`could not read ${table}`))
        : send(type, payload);
    return file;
  };

  for (const table of [
    "frames",
    "instances",
    "points",
    "pred_points",
    "negative_frames",
  ]) {
    it(`retries bytes on the main thread when /${table} fails, and gets all of it`, async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const expected = poses(
        await readSlp(legacyBytes(), { openVideos: false }),
      );
      warn.mockClear();
      const file = await fileFailingOn(table);
      stubBrowser();
      openWorker = async () => file;
      const labels = await loadSlp(legacyBytes(), { openVideos: false });
      expect(warn).toHaveBeenCalledTimes(1);
      expect(String(warn.mock.calls[0][0])).toContain("falling back");
      expect(poses(labels)).toEqual(expected);
    });

    it(`gives the error for a picked file when /${table} fails`, async () => {
      const file = await fileFailingOn(table);
      stubBrowser();
      openWorker = async () => file;
      await expect(
        loadSlp(new File([legacyBytes()], "mice.slp"), {
          openVideos: false,
        }),
      ).rejects.toThrow(`could not read ${table}`);
    });
  }
});

describe("a file HDF5 can't open says so", () => {
  // A valid start with the rest missing fails the same way as not being HDF5.
  const junk = () => new Uint8Array(4096).fill(7);
  const truncated = () => new Uint8Array(fixtureBytes()).subarray(0, 2000);

  it("readSlp and readSlpLazy, instead of 'error - name not defined!'", async () => {
    for (const read of [readSlp, readSlpLazy]) {
      for (const bytes of [junk(), truncated(), new Uint8Array(0)]) {
        await expect(
          read(bytes, { openVideos: false, h5: { filenameHint: "bad.slp" } }),
        ).rejects.toThrow(cannotOpenAsHdf5Message("bad.slp"));
      }
    }
  });

  it("a missing path is named", async () => {
    const missing = "/no/such/folder/mice.slp";
    await expect(readSlp(missing, { openVideos: false })).rejects.toThrow(
      cannotOpenAsHdf5Message(missing),
    );
  });

  it("the Worker says the same words, for every way it opens a file", () => {
    expect(H5_WORKER_CODE).not.toContain("keys: currentFile.keys()");
    const start = H5_WORKER_CODE.indexOf("function keysOfOpenedFile()");
    const end = H5_WORKER_CODE.indexOf("\n}\n", start) + 3;
    const check = new Function(
      "currentFile",
      "currentFilename",
      `${H5_WORKER_CODE.slice(start, end)}; return keysOfOpenedFile();`,
    );
    expect(() => check({ file_id: -1n }, "bad.slp")).toThrow(
      cannotOpenAsHdf5Message("bad.slp"),
    );
    const good = { file_id: 7n, keys: () => ["metadata"] };
    expect(check(good, "ok.slp")).toEqual(["metadata"]);
  });
});

// Everything below uses the browser's open path, which stays registered once a test
// has switched to it, so this comes last.
describe("reading a picked file on the main thread", () => {
  /** WORKERFS as the browser build has it, with a mount that records its use. */
  const withWorkerfs = (fs: ReturnType<typeof getH5FileSystem>) => {
    const filesystems = (fs.filesystems ?? {}) as Record<string, unknown>;
    fs.filesystems = { ...filesystems, WORKERFS: { fake: true } };
    return vi.spyOn(
      fs as unknown as { mount: (...args: unknown[]) => unknown },
      "mount",
    );
  };

  it("reads the bytes instead of mounting WORKERFS, which can't read here", async () => {
    const fs = await useBrowserOpenPath();
    const mount = withWorkerfs(fs);
    expect("FileReaderSync" in globalThis).toBe(false);

    const { file, close } = await openH5File(fixtureFile());
    try {
      expect(mount).not.toHaveBeenCalled();
      expect(file.get("metadata")).toBeTruthy();
    } finally {
      close();
    }
  });

  it("frees the copy when closed", async () => {
    const fs = await useBrowserOpenPath();
    const writeFile = vi.spyOn(fs, "writeFile");
    const unlink = vi.spyOn(
      fs as unknown as { unlink: (p: string) => void },
      "unlink",
    );
    const { close } = await openH5File(fixtureFile());
    const [localPath] = writeFile.mock.calls[0];
    expect(unlink).not.toHaveBeenCalled();
    close();
    expect(unlink).toHaveBeenCalledWith(localPath);
  });

  it("two files open at once stay two files", async () => {
    // A video backend holds its file open, so a second file must not land on the
    // same path, where the first would start reading the second's bytes.
    await useBrowserOpenPath();
    const other = new Uint8Array(
      readFileSync(
        fileURLToPath(
          new URL("./data/slp/py_written_25_identity.slp", import.meta.url),
        ),
      ),
    );
    const [a, b] = await Promise.all([
      openH5File(fixtureFile()),
      openH5File(new File([other], "other.slp")),
    ]);
    try {
      const formatId = (f: typeof a.file) =>
        (
          f.get("metadata") as unknown as {
            attrs: { format_id: { value: number } };
          }
        ).attrs.format_id.value;
      expect(formatId(a.file)).toBeCloseTo(1.1, 5);
      expect(formatId(b.file)).toBeCloseTo(2.5, 5);
    } finally {
      a.close();
      b.close();
    }
  });

  it("still mounts WORKERFS where it can read: in a Worker", async () => {
    const fs = await useBrowserOpenPath();
    const mount = withWorkerfs(fs);
    mount.mockImplementation(() => {
      throw new Error("mounted");
    });
    // Node's h5wasm works on the real disk, whose root can't take the mount point.
    vi.spyOn(
      fs as unknown as { mkdir: (p: string) => void },
      "mkdir",
    ).mockImplementation(() => {});
    vi.stubGlobal("FileReaderSync", class {});
    await expect(openH5File(fixtureFile())).rejects.toThrow("mounted");
    expect(mount).toHaveBeenCalledTimes(1);
  });

  it('loadSlp with stream: "download" reads a picked file', async () => {
    stubBrowser();
    await useBrowserOpenPath();
    const labels = await loadSlp(fixtureFile(), {
      openVideos: false,
      h5: { stream: "download" },
    });
    expect(labels.labeledFrames.length).toBe(1);
  });

  it("a picked file is named, when read on the main thread", async () => {
    await useBrowserOpenPath();
    await expect(
      readSlp(new File([new Uint8Array(4096).fill(7)], "picked.slp"), {
        openVideos: false,
      }),
    ).rejects.toThrow(cannotOpenAsHdf5Message("picked.slp"));
  });
});
