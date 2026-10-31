/**
 * A URL's credentials stay out of what h5wasm aborts with, in an app's own Worker.
 *
 * There `readSlp` and `readSlpLazy` read a URL straight through h5wasm, with no worker
 * of the library's to mask what comes back. When a read failed after the file opened,
 * h5wasm threw `Aborted(Couldn't load https://...&token=SECRET. Status: 500)` and
 * printed the same on the console. Its `onAbort` hook runs before either.
 *
 * `createLazyFile` only works in h5wasm's browser builds, so these tests run the ESM
 * build this path really imports, through the real `getH5Module`, believing it's in a
 * Worker. Standing in for the browser: a synchronous `XMLHttpRequest` and `fetch`,
 * both answered by one server in memory. Replacing the Node open path can't be undone,
 * which is why this is its own file (see `bun-test.ts`). The real thing was checked in
 * Chromium when this was written.
 */
import { describe, it, expect, beforeAll, afterEach, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import {
  _registerNodeH5,
  getH5Module,
  keepUrlsOutOfAborts,
  type H5Module,
} from "../src/codecs/slp/h5.js";
import { RemoteIOError, redactUrl } from "../src/io/remote.js";

const fixture = (name: string) =>
  new Uint8Array(
    readFileSync(fileURLToPath(new URL(`./data/slp/${name}`, import.meta.url))),
  );

const ORIGIN = "https://files.example.test";
const GOOD = `${ORIGIN}/good.slp`;
const BIG = `${ORIGIN}/big.slp`;
const MISSING = `${ORIGIN}/missing.slp`;

const FILES: Record<string, Uint8Array> = {
  "/good.slp": fixture("minimal_instance.slp"),
  "/big.slp": fixture("centered_pair_predictions.slp"),
};

type Request = { method: string; url: string; range: string | null };

/** The one server every request goes to. */
const server = {
  refusesHead: false,
  /** Range reads from here on fail, which the file's first read doesn't reach. */
  failsFrom: Number.POSITIVE_INFINITY,
  rangeReads: 0,
  wholeDownloads: 0,

  answer({ method, url, range }: Request) {
    if (method === "GET") {
      if (range) server.rangeReads++;
      else server.wholeDownloads++;
    }
    const nothing = new Uint8Array(0);
    const bytes = FILES[new URL(url).pathname];
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
    if (Number(asked[1]) >= server.failsFrom) {
      return { status: 500, headers: {}, body: nothing };
    }
    return {
      status: 206,
      headers,
      body: bytes.subarray(Number(asked[1]), Number(asked[2]) + 1),
    };
  },
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

/** What h5wasm has printed. It takes console.error once, as it starts. */
const printed: string[] = [];

let emscripten: { onAbort?: (what: unknown) => void };

const messageOf = async (reading: Promise<unknown>): Promise<Error> =>
  reading.then(
    () => {
      throw new Error("It opened, and should have failed");
    },
    (e: Error) => e,
  );

beforeAll(async () => {
  // The browser branch of getH5Module, not the Node provider.
  _registerNodeH5(null as never, null as never);
  vi.stubGlobal("WorkerGlobalScope", class {});
  vi.stubGlobal("XMLHttpRequest", XhrStandIn);
  vi.stubGlobal("fetch", async (url: string) => {
    const { status, headers, body } = server.answer({
      method: "GET",
      url: String(url),
      range: null,
    });
    return new Response(body as BodyInit, { status, headers });
  });
  const versions = process.versions;
  Object.defineProperty(process, "versions", {
    value: { ...versions, node: undefined },
    configurable: true,
  });
  const { error } = console;
  console.error = (...args) => printed.push(args.join(" "));
  const module = await getH5Module();
  console.error = error;
  Object.defineProperty(process, "versions", {
    value: versions,
    configurable: true,
  });
  emscripten = (module as unknown as { Module: typeof emscripten }).Module;
});

afterEach(() => {
  server.refusesHead = false;
  server.failsFrom = Number.POSITIVE_INFINITY;
  vi.restoreAllMocks();
});

describe("a URL read through h5wasm in an app's own Worker", () => {
  it("really does read ranges here", async () => {
    server.rangeReads = 0;
    server.wholeDownloads = 0;
    const labels = await readSlp(GOOD, { openVideos: false });
    expect(labels.length).toBe(1);
    expect(server.rangeReads).toBeGreaterThan(0);
    expect(server.wholeDownloads).toBe(0);
  });

  it("a read that fails after the file opened doesn't show the token", async () => {
    const url = `${BIG}?token=SECRET`;
    for (const read of [readSlp, readSlpLazy]) {
      server.failsFrom = 1024 * 1024;
      const error = await messageOf(read(url, { openVideos: false }));
      expect(error.message).toBe(
        `Aborted(Couldn't load ${redactUrl(url)}. Status: 500)`,
      );
    }
    expect(printed.join("\n")).not.toContain("SECRET");
    expect(printed.join("\n")).not.toContain("Couldn't load");
  });

  it("a missing file says so, with nothing printed", async () => {
    const error = await messageOf(
      readSlp(`${MISSING}?token=SECRET`, { openVideos: false }),
    );
    expect(error).toBeInstanceOf(RemoteIOError);
    expect(error.message).toContain("file not found");
    expect(error.message).not.toContain("SECRET");
    expect(printed.join("\n")).not.toContain("SECRET");
  });

  it("a server that refuses HEAD still gets the download", async () => {
    server.refusesHead = true;
    const labels = await readSlp(`${GOOD}?token=SECRET`, { openVideos: false });
    expect(labels.length).toBe(1);
    expect(printed.join("\n")).not.toContain("SECRET");
  });

  it("h5wasm can still read the next URL", async () => {
    server.failsFrom = 1024 * 1024;
    await messageOf(readSlp(`${BIG}?token=SECRET`, { openVideos: false }));
    server.failsFrom = Number.POSITIVE_INFINITY;
    const labels = await readSlp(BIG, { openVideos: false });
    expect(labels.length).toBeGreaterThan(0);
  });

  it("an abort that quotes no URL goes on as before", () => {
    expect(() => emscripten.onAbort?.("something else")).not.toThrow();
    expect(() => emscripten.onAbort?.(undefined)).not.toThrow();
  });
});

describe("keepUrlsOutOfAborts", () => {
  const withModule = (onAbort?: (what: unknown) => void) => {
    const Module: { onAbort?: (what: unknown) => void } = { onAbort };
    keepUrlsOutOfAborts({ Module } as unknown as H5Module);
    return Module;
  };

  it("masks what it throws", () => {
    const url = "https://x.test/y.slp?token=SECRET";
    expect(() => withModule().onAbort?.(`Couldn't load ${url}.`)).toThrow(
      `Aborted(Couldn't load ${redactUrl(url)}.)`,
    );
  });

  it("an app's own hook still hears of every abort, masked", () => {
    const heard: unknown[] = [];
    const Module = withModule((what) => heard.push(what));
    Module.onAbort?.("something else");
    expect(() =>
      Module.onAbort?.("Couldn't load https://x.test/y?token=SECRET"),
    ).toThrow();
    expect(heard[0]).toBe("something else");
    expect(String(heard[1])).toContain("Couldn't load https://x.test/y");
    expect(String(heard[1])).not.toContain("SECRET");
  });

  it("copes with a build of h5wasm with no Module", () => {
    expect(() => keepUrlsOutOfAborts({} as H5Module)).not.toThrow();
  });
});
