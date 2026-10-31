/**
 * A URL opens whatever name the caller gives it.
 *
 * A browser reads a URL through Emscripten's `createLazyFile`, and `filenameHint` was
 * the name of that file in h5wasm's filesystem, so slashes became folders that don't
 * exist. In the streaming Worker that failed with `{"name":"ErrnoError","errno":44}`.
 * `loadSlp` and the library's own open path (`openFromUrl`) caught the failure and went
 * on to download the whole file, so there the only sign was the cost. The file now goes
 * in under a name of the library's own, and the hint is only for showing.
 *
 * `createLazyFile` only works in h5wasm's browser build, so these tests run that build
 * and the real worker code in this process. Standing in for the browser: a Worker that
 * delivers messages in-process, a synchronous `XMLHttpRequest` (Bun has none) and
 * `fetch`, both answered by one server in memory that remembers what it was asked for.
 * Replacing the Node open path can't be undone, which is why this is its own file (see
 * `bun-test.ts`). The real thing was checked in Chromium when this was written.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadSlp } from "../src/io/main.js";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { readSlpStreaming } from "../src/codecs/slp/read-streaming.js";
import { openH5Worker } from "../src/codecs/slp/h5-streaming.js";
import { H5_WORKER_CODE } from "../src/codecs/slp/h5-worker.js";
import {
  _registerNodeH5,
  cannotOpenAsHdf5Message,
  openH5File,
} from "../src/codecs/slp/h5.js";
import { redactUrl } from "../src/io/remote.js";

const fixture = (name: string) =>
  new Uint8Array(
    readFileSync(fileURLToPath(new URL(`./data/${name}`, import.meta.url))),
  );

const ORIGIN = "https://files.example.test";
const GOOD = `${ORIGIN}/good.slp`;
const OTHER = `${ORIGIN}/other.slp`;
const PKG = `${ORIGIN}/pkg.slp`;
const NOT_SLEAP = `${ORIGIN}/not-sleap.h5`;
const JUNK = `${ORIGIN}/junk.slp`;
const MISSING = `${ORIGIN}/missing.slp`;

const FILES: Record<string, Uint8Array> = {
  "/good.slp": fixture("slp/minimal_instance.slp"),
  "/other.slp": fixture("slp/py_written_25_identity.slp"),
  "/pkg.slp": fixture("slp/minimal_instance.pkg.slp"),
  "/not-sleap.h5": fixture("analysis-h5/simple_standard.analysis.h5"),
  "/junk.slp": new Uint8Array(4096).fill(7),
};

const HINTS = [
  undefined,
  "",
  "a.slp",
  "b.slp",
  "/Users/me/b.slp",
  "C:\\Users\\me\\b.slp",
  "folder/mice.slp",
  "https://example.test/b.slp",
  "https://example.test/b.slp?token=SECRET",
  "gs://bucket/b.slp",
  "../../etc/b.slp",
  "../b.slp",
  "..",
  ".",
  "/",
  "a/..",
  "a b#1?%.slp",
  "日本語.slp",
];

const withHint = (hint: string | undefined) =>
  hint === undefined ? {} : { filenameHint: hint };

type Request = {
  method: string;
  url: string;
  range: string | null;
  headers?: unknown;
};

/** The one server every request goes to. */
const server = {
  requests: [] as Request[],
  refusesHead: false,

  answer(request: Request) {
    server.requests.push(request);
    const { method, url, range } = request;
    const nothing = new Uint8Array(0);
    const bytes = FILES[new URL(url).pathname.replace(/\/$/, "")];
    if (!bytes) return { status: 404, headers: {}, body: nothing };
    if (method === "HEAD" && server.refusesHead) {
      return { status: 403, headers: {}, body: nothing };
    }
    const headers = {
      "accept-ranges": "bytes",
      "content-length": String(bytes.length),
    };
    if (method === "HEAD") return { status: 200, headers, body: nothing };
    const asked = range?.match(/^bytes=(\d+)-(\d+)$/);
    if (!asked) return { status: 200, headers, body: bytes };
    return {
      status: 206,
      headers,
      body: bytes.subarray(Number(asked[1]), Number(asked[2]) + 1),
    };
  },

  /** A GET for all of a file, which is what range reads are there to avoid. */
  wholeDownloads: () =>
    server.requests.filter((r) => r.method === "GET" && r.range === null)
      .length,
  rangeReads: () => server.requests.filter((r) => r.range !== null).length,
};

/** What `createLazyFile` uses: synchronous, which Bun's fetch can't be. */
class XhrStandIn {
  status = 0;
  response: ArrayBuffer | undefined;
  responseType = "";
  private method = "";
  private url = "";
  private range: string | null = null;
  private answerHeaders: Record<string, string> = {};

  open(method: string, url: string): void {
    this.method = method;
    this.url = url;
  }

  setRequestHeader(name: string, value: string): void {
    if (name === "Range") this.range = value;
  }

  send(): void {
    const { status, headers, body } = server.answer({
      method: this.method,
      url: this.url,
      range: this.range,
    });
    this.status = status;
    this.answerHeaders = headers;
    this.response = body.slice().buffer as ArrayBuffer;
  }

  getResponseHeader(name: string): string | null {
    return this.answerHeaders[name.toLowerCase()] ?? null;
  }
}

const fetchStandIn = async (url: string, init?: RequestInit) => {
  const { status, headers, body } = server.answer({
    method: "GET",
    url: String(url),
    range: null,
    headers: init?.headers,
  });
  return new Response(body as BodyInit, { status, headers });
};

type BrowserH5wasm = {
  ready: Promise<unknown>;
  File: unknown;
  FS: {
    readdir: (path: string) => string[];
    createLazyFile: (...args: unknown[]) => unknown;
  };
};

const versions = process.versions;
const looksLikeBrowser = (yes: boolean) =>
  Object.defineProperty(process, "versions", {
    value: yes ? { ...versions, node: undefined } : versions,
    configurable: true,
  });

/** Where h5wasm's error output goes. It takes console.error once, as it starts. */
let h5wasmOutput: (...args: unknown[]) => void = console.error;

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
  const { error } = console;
  console.error = (...args) => h5wasmOutput(...args);
  const module = new Function(`${source}\nreturn h5wasm;`)() as BrowserH5wasm;
  console.error = error;
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

/** What's left in the filesystem root of a mount made by a test. */
const leftoverMounts = () =>
  h5wasm.FS.readdir("/").filter((name) =>
    /^(remote|remote-buffer|slp-remote)-/.test(name),
  );

const formatIdOf = async (file: {
  getAttrs: (path: string) => Promise<Record<string, unknown>>;
}) =>
  (
    (await file.getAttrs("metadata")) as {
      format_id: { value: number };
    }
  ).format_id.value;

/** h5wasm and HDF5 say what went wrong on the console too. */
const quietConsole = () => {
  h5wasmOutput = () => {};
  vi.spyOn(console, "warn").mockImplementation(() => {});
};

beforeEach(() => {
  server.requests.length = 0;
  server.refusesHead = false;
  vi.stubGlobal("XMLHttpRequest", XhrStandIn);
  vi.stubGlobal("fetch", fetchStandIn);
  vi.stubGlobal("Worker", InProcessWorker);
  vi.stubGlobal("window", {});
  looksLikeBrowser(true);
});

afterEach(() => {
  vi.unstubAllGlobals();
  looksLikeBrowser(false);
  vi.restoreAllMocks();
  h5wasmOutput = console.error;
  expect(leftoverMounts()).toEqual([]);
});

describe("a URL, through the Worker", () => {
  it("readSlpStreaming opens it under a hint with folders in it", async () => {
    const labels = await readSlpStreaming(GOOD, {
      openVideos: false,
      filenameHint: "/Users/me/b.slp",
    });
    expect(labels.labeledFrames.length).toBe(1);
  });

  it("opens whatever the hint, reading only ranges", async () => {
    for (const hint of HINTS) {
      server.requests.length = 0;
      const labels = await readSlpStreaming(GOOD, {
        openVideos: false,
        ...withHint(hint),
      });
      expect(labels.labeledFrames.length).toBe(1);
      expect(server.wholeDownloads()).toBe(0);
      expect(server.rangeReads()).toBeGreaterThan(0);
    }
  });

  it("loadSlp stays with the Worker whatever the hint, not a whole download", async () => {
    const warn = vi.spyOn(console, "warn");
    for (const options of [{}, { lazy: true }, { stream: "range" as const }]) {
      for (const hint of HINTS) {
        const { lazy, stream } = options as {
          lazy?: boolean;
          stream?: "range";
        };
        const labels = await loadSlp(GOOD, {
          openVideos: false,
          lazy,
          h5: { stream, ...withHint(hint) },
        });
        expect(labels.length).toBe(1);
      }
    }
    expect(server.wholeDownloads()).toBe(0);
    expect(warn).not.toHaveBeenCalled();
  });

  it("reads the same project with a hint as without", async () => {
    const read = (hint?: string) =>
      readSlpStreaming(GOOD, { openVideos: false, ...withHint(hint) });
    const points = (labels: Awaited<ReturnType<typeof read>>) =>
      labels.labeledFrames.map((frame) =>
        frame.instances.map((instance) => instance.numpy()),
      );
    expect(points(await read("/Users/me/b.slp"))).toEqual(points(await read()));
  });

  it("with headers, downloads with them whatever the hint", async () => {
    for (const hint of HINTS) {
      server.requests.length = 0;
      const labels = await readSlpStreaming(GOOD, {
        openVideos: false,
        headers: { "X-Test": "yes" },
        ...withHint(hint),
      });
      expect(labels.labeledFrames.length).toBe(1);
      expect(server.requests.length).toBe(1);
      expect(
        (server.requests[0].headers as Record<string, string>)["X-Test"],
      ).toBe("yes");
    }
  });

  it("an embedded video goes by the hint as given, or else the URL's own name", async () => {
    for (const [hint, name] of [
      ["/Users/me/b.pkg.slp", "/Users/me/b.pkg.slp"],
      ["https://example.test/b.slp", "https://example.test/b.slp"],
      [undefined, "pkg.slp"],
    ] as const) {
      const labels = await readSlpStreaming(PKG, {
        openVideos: false,
        ...withHint(hint),
      });
      expect(labels.videos[0].filename).toBe(name);
    }
  });

  it("a second URL in the same Worker is that file, not the first", async () => {
    const hint = { filenameHint: "/Users/me/b.slp" };
    const file = await openH5Worker(GOOD, hint);
    expect(await formatIdOf(file)).toBeCloseTo(1.1, 5);
    await file.open(OTHER, hint);
    expect(await formatIdOf(file)).toBeCloseTo(2.5, 5);
    await file.open(GOOD);
    expect(await formatIdOf(file)).toBeCloseTo(1.1, 5);
    await file.close();
  });

  it("a file HDF5 can't open is called by its hint, or else the URL's own name", async () => {
    quietConsole();
    for (const headers of [undefined, { "X-Test": "yes" }]) {
      for (const [hint, shown] of [
        ["b.slp", "b.slp"],
        ["/Users/me/b.slp", "/Users/me/b.slp"],
        ["https://example.test/b.slp", "https://example.test/b.slp"],
        ["..", ".."],
        [undefined, "junk.slp"],
        ["", "junk.slp"],
      ] as const) {
        await expect(
          readSlpStreaming(JUNK, {
            openVideos: false,
            headers,
            ...withHint(hint),
          }),
        ).rejects.toThrow(cannotOpenAsHdf5Message(shown));
      }
    }
  });

  it("a token in a hint stays out of the messages", async () => {
    quietConsole();
    const hint = "https://example.test/b.slp?token=SECRET";
    const masked = redactUrl(hint);
    expect(masked).not.toContain("SECRET");
    for (const headers of [undefined, { "X-Test": "yes" }]) {
      await expect(
        readSlpStreaming(JUNK, {
          openVideos: false,
          headers,
          filenameHint: hint,
        }),
      ).rejects.toThrow(cannotOpenAsHdf5Message(masked));
      await expect(
        readSlpStreaming(NOT_SLEAP, {
          openVideos: false,
          headers,
          filenameHint: hint,
        }),
      ).rejects.toThrow(`The SLEAP labels file '${masked}'`);
    }
  });

  it("a missing file fails the same under any hint", async () => {
    quietConsole();
    for (const hint of HINTS) {
      await expect(
        readSlpStreaming(MISSING, { openVideos: false, ...withHint(hint) }),
      ).rejects.toThrow("404");
    }
  });
});

// The library's own open path, as an app's Worker runs it. Everything below reads
// through the browser build, which stays registered, so this comes last.
describe("a URL, read inside an app's own Worker", () => {
  beforeEach(() => {
    _registerNodeH5(async () => h5wasm as never, null as never);
  });

  it("openH5File opens whatever the hint, reading only ranges", async () => {
    for (const hint of HINTS) {
      server.requests.length = 0;
      const { file, close } = await openH5File(GOOD, withHint(hint));
      expect(Number(file.file_id)).toBeGreaterThanOrEqual(0);
      expect(file.keys()).toContain("metadata");
      expect(server.wholeDownloads()).toBe(0);
      expect(server.rangeReads()).toBeGreaterThan(0);
      close();
    }
  });

  it("readSlp and readSlpLazy read it under a hint with folders in it", async () => {
    for (const read of [readSlp, readSlpLazy]) {
      const labels = await read(GOOD, {
        openVideos: false,
        h5: { filenameHint: "/Users/me/b.slp" },
      });
      expect(labels.length).toBe(1);
    }
    expect(server.wholeDownloads()).toBe(0);
  });

  it("a URL with no name of its own reads only ranges too", async () => {
    const { file, close } = await openH5File(`${GOOD}/?download=1`);
    expect(file.keys()).toContain("metadata");
    expect(server.wholeDownloads()).toBe(0);
    close();
  });

  it("with headers, downloads with them whatever the hint", async () => {
    for (const hint of HINTS) {
      server.requests.length = 0;
      const { file, close } = await openH5File(GOOD, {
        headers: { "X-Test": "yes" },
        ...withHint(hint),
      });
      expect(file.keys()).toContain("metadata");
      expect(server.requests.length).toBe(1);
      close();
    }
  });

  // What the main thread gets: there, createLazyFile fails before making anything.
  it("where createLazyFile can't work, downloads whatever the hint", async () => {
    vi.spyOn(h5wasm.FS, "createLazyFile").mockImplementation(() => {
      throw new Error("Cannot do synchronous binary XHRs outside webworkers");
    });
    for (const hint of HINTS) {
      server.requests.length = 0;
      const labels = await readSlp(GOOD, {
        openVideos: false,
        h5: withHint(hint),
      });
      expect(labels.length).toBe(1);
      expect(server.wholeDownloads()).toBe(1);
    }
  });

  // These two abort h5wasm in the middle of an open, as the browser does. It goes on
  // working, but they come last.
  it("a server that refuses HEAD still gets the download, whatever the hint", async () => {
    quietConsole();
    server.refusesHead = true;
    for (const hint of HINTS) {
      server.requests.length = 0;
      const labels = await readSlp(GOOD, {
        openVideos: false,
        h5: withHint(hint),
      });
      expect(labels.length).toBe(1);
      expect(server.wholeDownloads()).toBe(1);
    }
    server.refusesHead = false;
    server.requests.length = 0;
    const next = await readSlp(GOOD, { openVideos: false });
    expect(next.length).toBe(1);
    expect(server.wholeDownloads()).toBe(0);
  });

  it("a missing file says so, whatever the hint", async () => {
    quietConsole();
    for (const hint of HINTS) {
      await expect(
        readSlp(MISSING, { openVideos: false, h5: withHint(hint) }),
      ).rejects.toThrow("file not found");
    }
  });
});
