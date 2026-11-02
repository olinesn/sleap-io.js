/**
 * A picked file (a `File`) opens whatever name the caller gives it.
 *
 * A browser opens a `File` by mounting it with WORKERFS. WORKERFS named it by
 * `file.name` while the path opened came from `filenameHint`, so
 * `loadSlp(new File([bytes], "a.slp"), { h5: { filenameHint: "b.slp" } })` told the
 * user "Could not open 'b.slp' as an HDF5 file", on any valid file. The file now goes
 * in under a name of the library's own, and the hint is only for showing.
 *
 * WORKERFS only exists in h5wasm's browser build, so these tests run that build and
 * the real worker code in this process. Standing in for the browser: a Worker that
 * delivers messages in-process, `FileReaderSync` (Bun can't read a Blob
 * synchronously), and one in-memory OPFS file for the append path. Replacing the Node
 * open path can't be undone, which is why this is its own file (see `bun-test.ts`).
 * The real thing was checked in Chromium when this was written.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadSlp } from "../src/io/main.js";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { readSlpStreaming } from "../src/codecs/slp/read-streaming.js";
import {
  openH5Worker,
  StreamingH5Writer,
} from "../src/codecs/slp/h5-streaming.js";
import { H5_WORKER_CODE } from "../src/codecs/slp/h5-worker.js";
import {
  _registerNodeH5,
  cannotOpenAsHdf5Message,
  openH5File,
} from "../src/codecs/slp/h5.js";

const fixture = (name: string) =>
  new Uint8Array(
    readFileSync(fileURLToPath(new URL(`./data/slp/${name}`, import.meta.url))),
  );
const GOOD = fixture("minimal_instance.slp");
const OTHER = fixture("py_written_25_identity.slp");
const JUNK = new Uint8Array(4096).fill(7);

/** Names a `File` can really have: a browser keeps all of these as they are. */
const NAMES = [
  "a.slp",
  "",
  "folder/mice.slp",
  "/rooted.slp",
  "../up.slp",
  "..",
  ".",
  "a b#1?%.slp",
  "日本語.slp",
];
const HINTS = [
  undefined,
  "",
  "a.slp",
  "b.slp",
  "/Users/me/b.slp",
  "C:\\Users\\me\\b.slp",
  "https://example.test/b.slp?token=abc",
  "../../etc/b.slp",
];

/** A `File` that WORKERFS can read here: it slices, then uses `FileReaderSync`. */
class PickedFile extends File {
  constructor(
    private readonly bytes: Uint8Array,
    name: string,
  ) {
    super([bytes as BlobPart], name);
    Object.defineProperty(this, "name", { value: name });
  }

  override slice(start?: number, end?: number): Blob {
    const part = this.bytes.subarray(start, end);
    return { size: part.length, part } as unknown as Blob;
  }
}

class FileReaderSyncStandIn {
  readAsArrayBuffer(blob: { part: Uint8Array }): ArrayBuffer {
    return blob.part.slice().buffer as ArrayBuffer;
  }
}

type BrowserH5wasm = {
  ready: Promise<unknown>;
  File: unknown;
  FS: { readdir: (path: string) => string[] };
};

const versions = process.versions;
const looksLikeBrowser = (yes: boolean) =>
  Object.defineProperty(process, "versions", {
    value: yes ? { ...versions, node: undefined } : versions,
    configurable: true,
  });

/** h5wasm's browser build, which checks at startup that it's in a Worker. */
const h5wasm: BrowserH5wasm = await (async () => {
  vi.stubGlobal("WorkerGlobalScope", class {});
  looksLikeBrowser(true);
  const source = readFileSync(
    fileURLToPath(
      new URL("../node_modules/h5wasm/dist/iife/h5wasm.js", import.meta.url),
    ),
    "utf8",
  );
  const module = new Function(`${source}\nreturn h5wasm;`)() as BrowserH5wasm;
  await module.ready;
  looksLikeBrowser(false);
  vi.unstubAllGlobals();
  return module;
})();

/** The real worker code, answering in this process. */
class InProcessWorker {
  onmessage: ((e: { data: unknown }) => void) | null = null;
  onerror: unknown = null;
  private readonly scope: {
    postMessage: (data: unknown) => void;
    onmessage?: (e: { data: unknown }) => unknown;
  } = { postMessage: (data) => this.onmessage?.({ data }) };

  constructor() {
    new Function("self", "importScripts", "h5wasm", H5_WORKER_CODE)(
      this.scope,
      () => {},
      h5wasm,
    );
  }

  postMessage(data: unknown): void {
    void this.scope.onmessage?.({ data });
  }

  addEventListener(): void {}

  /** A real Worker takes its filesystem with it. These all share one. */
  terminate(): void {
    this.postMessage({ type: "close" });
  }
}

/** One OPFS file in memory: the destination of the append path. */
const memoryOpfs = () => {
  let stored = new Uint8Array(0);
  const handle = {
    truncate(size: number) {
      const resized = new Uint8Array(size);
      resized.set(stored.subarray(0, size));
      stored = resized;
    },
    write(view: Uint8Array, { at }: { at: number }) {
      if (at + view.length > stored.length) {
        const grown = new Uint8Array(at + view.length);
        grown.set(stored);
        stored = grown;
      }
      stored.set(view, at);
      return view.length;
    },
    read(view: Uint8Array, { at }: { at: number }) {
      const part = stored.subarray(at, at + view.length);
      view.set(part);
      return part.length;
    },
    flush() {},
    close() {},
  };
  const root = {
    getFileHandle: async () => ({
      createSyncAccessHandle: async () => handle,
    }),
  };
  return { storage: { getDirectory: async () => root } };
};

/** What's left in the filesystem root of a mount made by a test. */
const leftoverMounts = () =>
  h5wasm.FS.readdir("/").filter((name) =>
    /^(local|slp-local|opfs-append-src)-/.test(name),
  );

const formatIdOf = async (file: {
  getAttrs: (path: string) => Promise<Record<string, unknown>>;
}) =>
  (
    (await file.getAttrs("metadata")) as {
      format_id: { value: number };
    }
  ).format_id.value;

beforeEach(() => {
  vi.stubGlobal("FileReaderSync", FileReaderSyncStandIn);
  vi.stubGlobal("Worker", InProcessWorker);
  vi.stubGlobal("window", {});
  vi.stubGlobal("navigator", memoryOpfs());
  vi.stubGlobal("FileSystemFileHandle", class {});
  looksLikeBrowser(true);
});

afterEach(() => {
  vi.unstubAllGlobals();
  looksLikeBrowser(false);
  vi.restoreAllMocks();
  expect(leftoverMounts()).toEqual([]);
});

describe("a picked file, through the Worker", () => {
  it("loadSlp opens it under a hint that isn't its name", async () => {
    const labels = await loadSlp(new PickedFile(GOOD, "a.slp"), {
      openVideos: false,
      h5: { filenameHint: "b.slp" },
    });
    expect(labels.labeledFrames.length).toBe(1);
  });

  it("opens whatever the file is called and whatever the hint", async () => {
    for (const name of NAMES) {
      for (const hint of HINTS) {
        const labels = await loadSlp(new PickedFile(GOOD, name), {
          openVideos: false,
          h5: hint === undefined ? {} : { filenameHint: hint },
        });
        expect(labels.labeledFrames.length).toBe(1);
      }
    }
  });

  it("reads the same project with a hint as without", async () => {
    const read = (hint?: string) =>
      loadSlp(new PickedFile(GOOD, "a.slp"), {
        openVideos: false,
        h5: hint === undefined ? {} : { filenameHint: hint },
      });
    const points = (labels: Awaited<ReturnType<typeof read>>) =>
      labels.labeledFrames.map((frame) =>
        frame.instances.map((instance) => instance.numpy()),
      );
    expect(points(await read("/Users/me/b.slp"))).toEqual(points(await read()));
  });

  it("the other ways in: lazy, streaming, and a file handle", async () => {
    const file = new PickedFile(GOOD, "a.slp");
    const opened = [
      await loadSlp(file, {
        openVideos: false,
        lazy: true,
        h5: { filenameHint: "b.slp" },
      }),
      await readSlpStreaming(file, {
        openVideos: false,
        filenameHint: "b.slp",
      }),
      await loadSlp(
        {
          kind: "file",
          name: file.name,
          getFile: async () => file,
        } as unknown as FileSystemFileHandle,
        { openVideos: false, h5: { filenameHint: "b.slp" } },
      ),
    ];
    for (const labels of opened) {
      expect(labels.length).toBe(1);
    }
  });

  it("a file HDF5 can't open is called by its hint, or else by its name", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    for (const [hint, shown] of [
      ["b.slp", "b.slp"],
      ["/Users/me/b.slp", "/Users/me/b.slp"],
      [undefined, "a.slp"],
      ["", "a.slp"],
    ] as const) {
      await expect(
        loadSlp(new PickedFile(JUNK, "a.slp"), {
          openVideos: false,
          h5: hint === undefined ? {} : { filenameHint: hint },
        }),
      ).rejects.toThrow(cannotOpenAsHdf5Message(shown));
    }
  });

  it("a second file in the same Worker is that file, not the first", async () => {
    const file = await openH5Worker(new PickedFile(GOOD, "a.slp"), {
      filenameHint: "b.slp",
    });
    expect(await formatIdOf(file)).toBeCloseTo(1.1, 5);
    await file.openLocal(new PickedFile(OTHER, "a.slp"), {
      filenameHint: "b.slp",
    });
    expect(await formatIdOf(file)).toBeCloseTo(2.5, 5);
    await file.close();
  });
});

describe("a picked file, through the append path", () => {
  it("opens whatever the file is called and whatever the hint", async () => {
    for (const name of NAMES) {
      for (const hint of [...HINTS, name]) {
        const writer = new StreamingH5Writer();
        try {
          await writer.openAppendOpfs(
            "dest.slp",
            new PickedFile(GOOD, name),
            OTHER,
            hint,
          );
        } finally {
          await writer.close();
        }
      }
    }
  });

  it("a source HDF5 can't open is called by the name it was given, or else its own", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    for (const [given, shown] of [
      ["b.slp", "b.slp"],
      [undefined, "a.slp"],
    ] as const) {
      const writer = new StreamingH5Writer();
      try {
        await expect(
          writer.openAppendOpfs(
            "dest.slp",
            new PickedFile(JUNK, "a.slp"),
            OTHER,
            given,
          ),
        ).rejects.toThrow(cannotOpenAsHdf5Message(shown));
      } finally {
        await writer.close();
      }
    }
  });
});

// The library's own open path, as an app's Worker runs it. Everything below reads
// through the browser build, which stays registered, so this comes last.
describe("a picked file, read inside an app's own Worker", () => {
  beforeEach(() => {
    _registerNodeH5(async () => h5wasm as never, null as never);
  });

  it("openH5File opens whatever the file is called and whatever the hint", async () => {
    for (const name of NAMES) {
      for (const hint of HINTS) {
        const { file, close } = await openH5File(new PickedFile(GOOD, name), {
          filenameHint: hint,
        });
        expect(Number(file.file_id)).toBeGreaterThanOrEqual(0);
        expect(file.keys()).toContain("metadata");
        close();
      }
    }
  });

  it("readSlp and readSlpLazy read it under a hint that isn't its name", async () => {
    for (const read of [readSlp, readSlpLazy]) {
      const labels = await read(new PickedFile(GOOD, "a.slp"), {
        openVideos: false,
        h5: { filenameHint: "b.slp" },
      });
      expect(labels.length).toBe(1);
    }
  });

  it("a file HDF5 can't open is called by its hint", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await expect(
      readSlp(new PickedFile(JUNK, "a.slp"), {
        openVideos: false,
        h5: { filenameHint: "b.slp" },
      }),
    ).rejects.toThrow(cannotOpenAsHdf5Message("b.slp"));
  });
});
