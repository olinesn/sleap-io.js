/**
 * A URL's credentials stay out of what the reader says.
 *
 * `io/remote.ts` promises that no thrown error leaks them, and three ways of opening
 * a file went around that:
 * - A URL that fails. h5wasm and the browser quote it in their own words
 *   (`Aborted(Couldn't load https://...&token=SECRET. Status: 404)`), and the Worker
 *   passed those on.
 * - A URL with no hint, whose query holds a slash. Its name was what came after the
 *   last slash, query and all.
 * - A URL as the hint for something that isn't a URL. It went to the Worker unmasked.
 *
 * Failing like that takes `createLazyFile`, which only works in h5wasm's browser
 * build, so these tests run that build and the real worker code in this process.
 * Standing in for the browser: a Worker that delivers messages in-process, a
 * synchronous `XMLHttpRequest` and `fetch` answered by one server in memory, and
 * `FileReaderSync`. Replacing the Node open path can't be undone, which is why this is
 * its own file (see `bun-test.ts`). The real thing was checked in Chromium when this
 * was written, and the stand-in's network failure is worded as Chromium words it.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadSlp } from "../src/io/main.js";
import { readSlp } from "../src/codecs/slp/read.js";
import { readSlpStreaming } from "../src/codecs/slp/read-streaming.js";
import {
  openH5Worker,
  StreamingH5File,
  StreamingH5Writer,
} from "../src/codecs/slp/h5-streaming.js";
import { H5_WORKER_CODE } from "../src/codecs/slp/h5-worker.js";
import { cannotOpenAsHdf5Message } from "../src/codecs/slp/h5.js";
import { redactUrl } from "../src/io/remote.js";
import { SIGNED_QUERIES, shownPart, signedQuery } from "./signed-url-fixtures";

const fixture = (name: string) =>
  new Uint8Array(
    readFileSync(fileURLToPath(new URL(`./data/${name}`, import.meta.url))),
  );

const JUNK_BYTES = new Uint8Array(4096).fill(7);

const ORIGIN = "https://files.example.test";
const GOOD = `${ORIGIN}/good.slp`;
const BIG = `${ORIGIN}/big.slp`;
const PKG = `${ORIGIN}/pkg.slp`;
const NOT_SLEAP = `${ORIGIN}/not-sleap.h5`;
const JUNK = `${ORIGIN}/junk.slp`;
const JUNK_NAMED_GET = `${ORIGIN}/get`;
const MISSING = `${ORIGIN}/missing.slp`;

/** A URL as the hint, for something that isn't that URL. */
const TOKEN_HINT = "https://example.test/y.slp?token=SECRET";

const FILES: Record<string, Uint8Array> = {
  "": JUNK_BYTES,
  "/good.slp": fixture("slp/minimal_instance.slp"),
  "/big.slp": fixture("slp/centered_pair_predictions.slp"),
  "/pkg.slp": fixture("slp/minimal_instance.pkg.slp"),
  "/not-sleap.h5": fixture("analysis-h5/simple_standard.analysis.h5"),
  "/junk.slp": JUNK_BYTES,
  "/get": JUNK_BYTES,
};

type Request = { method: string; url: string; range: string | null };

/** The one server every request goes to. */
const server = {
  refusesHead: false,
  /** Nothing answers: the request fails at the network level. */
  unreachable: false,
  /** Range reads from here on fail, which the file's first read doesn't reach. */
  failsFrom: Number.POSITIVE_INFINITY,
  /** Every URL asked for, as it was asked for. */
  asked: [] as string[],

  answer({ method, url, range }: Request) {
    server.asked.push(url);
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

/** Measured: as it reads the URL, not as it was given, and a long one shortened. */
const asChromiumQuotes = (url: string) => {
  const { href } = new URL(url);
  return href.length > 1024
    ? `${href.slice(0, 511)}...${href.slice(-510)}`
    : href;
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
    if (server.unreachable) {
      throw new DOMException(
        `Failed to execute 'send' on 'XMLHttpRequest': Failed to load '${asChromiumQuotes(this.url)}'.`,
        "NetworkError",
      );
    }
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

const fetchStandIn = async (url: string) => {
  const { status, headers, body } = server.answer({
    method: "GET",
    url: String(url),
    range: null,
  });
  return new Response(body as BodyInit, { status, headers });
};

/** A `File` that WORKERFS can read here: it slices, then uses `FileReaderSync`. */
class PickedFile extends File {
  constructor(
    private readonly bytes: Uint8Array,
    name: string,
  ) {
    super([bytes as BlobPart], name);
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

const versions = process.versions;
const looksLikeBrowser = (yes: boolean) =>
  Object.defineProperty(process, "versions", {
    value: yes ? { ...versions, node: undefined } : versions,
    configurable: true,
  });

/** Where h5wasm's error output goes. It takes console.error once, as it starts. */
let h5wasmOutput: (...args: unknown[]) => void = console.error;

type BrowserH5wasm = {
  ready: Promise<unknown>;
  Module: { onAbort?: (what: unknown) => void };
};

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

/** A Worker that shows what it was sent, and answers as it's told. */
class ScriptedWorker {
  static last: ScriptedWorker;
  static answer: (type: string) => Record<string, unknown> | null = () => ({
    success: true,
  });
  sent: { type: string; payload?: Record<string, unknown> }[] = [];
  onmessage: ((e: { data: unknown }) => void) | null = null;
  onerror: ((e: { message: string }) => void) | null = null;

  constructor() {
    ScriptedWorker.last = this;
  }

  postMessage(data: {
    type: string;
    id: number;
    payload?: Record<string, unknown>;
  }): void {
    this.sent.push(data);
    const answer = ScriptedWorker.answer(data.type);
    if (!answer) return;
    queueMicrotask(() =>
      this.onmessage?.({ data: { id: data.id, ...answer } }),
    );
  }

  addEventListener(): void {}
  terminate(): void {}
}

const messageOf = async (reading: Promise<unknown>): Promise<string> => {
  const error = await reading.then(
    () => {
      throw new Error("It opened, and should have failed");
    },
    (e: Error) => e,
  );
  return error.message;
};

const failureOf = (url: string, options: Record<string, unknown> = {}) =>
  messageOf(readSlpStreaming(url, { openVideos: false, ...options }));

/** h5wasm and HDF5 say what went wrong on the console too. */
const quietConsole = () => {
  h5wasmOutput = () => {};
  return vi.spyOn(console, "warn").mockImplementation(() => {});
};

beforeEach(() => {
  server.refusesHead = false;
  server.unreachable = false;
  server.failsFrom = Number.POSITIVE_INFINITY;
  server.asked = [];
  ScriptedWorker.answer = () => ({ success: true });
  vi.stubGlobal("XMLHttpRequest", XhrStandIn);
  vi.stubGlobal("fetch", fetchStandIn);
  vi.stubGlobal("Worker", InProcessWorker);
  vi.stubGlobal("FileReaderSync", FileReaderSyncStandIn);
  vi.stubGlobal("window", {});
  looksLikeBrowser(true);
});

afterEach(() => {
  vi.unstubAllGlobals();
  looksLikeBrowser(false);
  vi.restoreAllMocks();
  h5wasmOutput = console.error;
});

describe("a URL with no hint is named by its path", () => {
  it("a slash in the query doesn't start a name", async () => {
    quietConsole();
    for (const url of [
      `${JUNK_NAMED_GET}?f=a/junk.slp&token=SECRET`,
      `${JUNK_NAMED_GET}?token=SECRET&f=a/junk.slp`,
      `${JUNK_NAMED_GET}?f=a/b/c&token=SECRET/more`,
    ]) {
      expect(await failureOf(url)).toBe(cannotOpenAsHdf5Message("get"));
    }
  });

  it("nor does a slash in a token, or in a fragment", async () => {
    quietConsole();
    for (const url of [
      `${JUNK}?token=SEC/RET`,
      `${JUNK}#SEC/RET`,
      `${JUNK}?a=1#SEC/RET`,
      `${JUNK}#a?b=1/RET`,
    ]) {
      expect(await failureOf(url)).toBe(cannotOpenAsHdf5Message("junk.slp"));
    }
  });

  it("a password before the path is no part of the name", async () => {
    quietConsole();
    const withPassword = "https://me:SECRET@files.example.test";
    for (const [url, name] of [
      [withPassword, "data.h5"],
      [`${withPassword}?a=1`, "data.h5"],
      [`${withPassword}/junk.slp`, "junk.slp"],
      [`${withPassword}/get?f=a/junk.slp`, "get"],
    ]) {
      expect(await failureOf(url)).toBe(cannotOpenAsHdf5Message(name));
    }
  });

  it("the metadata error goes by that name", async () => {
    quietConsole();
    const message = await failureOf(
      `${NOT_SLEAP}?f=a/not-sleap.h5&token=SECRET`,
    );
    expect(message).toContain("The SLEAP labels file 'not-sleap.h5'");
    expect(message).not.toContain("SECRET");
  });

  it("an embedded video goes by that name too", async () => {
    for (const url of [`${PKG}?f=a/b.slp&token=SECRET`, `${PKG}#a/SECRET`]) {
      const labels = await readSlpStreaming(url, { openVideos: false });
      expect(labels.videos[0].filename).toBe("pkg.slp");
    }
  });

  it("the file itself still opens", async () => {
    const labels = await readSlpStreaming(`${GOOD}?f=a/b.slp&token=SECRET`, {
      openVideos: false,
    });
    expect(labels.labeledFrames.length).toBe(1);
  });
});

describe("a URL as the hint for something else", () => {
  const masked = redactUrl(TOKEN_HINT);
  const maskedName = masked.split("/").pop() as string;

  it("a picked file is called by it, masked", async () => {
    quietConsole();
    const file = new PickedFile(JUNK_BYTES, "a.slp");
    for (const read of [
      () =>
        readSlpStreaming(file, { openVideos: false, filenameHint: TOKEN_HINT }),
      () =>
        loadSlp(file, { openVideos: false, h5: { filenameHint: TOKEN_HINT } }),
    ]) {
      expect(await messageOf(read())).toBe(cannotOpenAsHdf5Message(masked));
    }
  });

  it("so is a picked file named a URL, with no hint", async () => {
    quietConsole();
    const message = await messageOf(
      readSlpStreaming(new PickedFile(JUNK_BYTES, TOKEN_HINT), {
        openVideos: false,
      }),
    );
    expect(message).toBe(cannotOpenAsHdf5Message(masked));
  });

  it("HDF5's own account of the failure doesn't show it either", async () => {
    const said: string[] = [];
    h5wasmOutput = (...args) => said.push(args.join(" "));
    await messageOf(
      readSlpStreaming(new PickedFile(JUNK_BYTES, "a.slp"), {
        openVideos: false,
        filenameHint: TOKEN_HINT,
      }),
    );
    expect(said.length).toBeGreaterThan(0);
    expect(said.join("\n")).not.toContain("SECRET");
  });

  it("a byte buffer is called by what's left of it, masked", async () => {
    quietConsole();
    const message = await messageOf(
      readSlpStreaming(JUNK_BYTES.slice(), {
        openVideos: false,
        filenameHint: TOKEN_HINT,
      }),
    );
    expect(message).toBe(cannotOpenAsHdf5Message(maskedName));
    expect(message).not.toContain("SECRET");
  });

  it("a slash in its query can't uncover the token", async () => {
    quietConsole();
    for (const hint of [
      "https://example.test/get?f=a/y.slp&token=SECRET",
      "https://example.test/get?token=SECRET&f=a/y.slp",
    ]) {
      const message = await messageOf(
        readSlpStreaming(JUNK_BYTES.slice(), {
          openVideos: false,
          filenameHint: hint,
        }),
      );
      expect(message).toContain("Could not open");
      expect(message).not.toContain("SECRET");
    }
  });

  it("nor does a password show", async () => {
    quietConsole();
    const hint = "https://me:SECRET@example.test/y.slp";
    for (const source of [
      new PickedFile(JUNK_BYTES, "a.slp"),
      JUNK_BYTES.slice(),
    ]) {
      const message = await messageOf(
        readSlpStreaming(source, { openVideos: false, filenameHint: hint }),
      );
      expect(message).toContain("Could not open");
      expect(message).not.toContain("SECRET");
    }
  });

  it("loadSlp's retry on the main thread doesn't show it, or its warning", async () => {
    const warn = quietConsole();
    const message = await messageOf(
      loadSlp(JUNK_BYTES.slice(), {
        openVideos: false,
        h5: { filenameHint: TOKEN_HINT },
      }),
    );
    expect(message).toBe(cannotOpenAsHdf5Message(masked));
    expect(warn).toHaveBeenCalled();
    expect(JSON.stringify(warn.mock.calls)).not.toContain("SECRET");
  });

  it("readSlp, off the Worker, doesn't show it", async () => {
    const message = await messageOf(
      readSlp(JUNK_BYTES.slice(), {
        openVideos: false,
        h5: { filenameHint: TOKEN_HINT },
      }),
    );
    expect(message).toBe(cannotOpenAsHdf5Message(masked));
  });

  // A RangeSource's Worker blocks on Atomics.wait until this thread answers, so it
  // can't run in one thread: this looks at what's sent, which is what the Worker shows.
  it("what's sent to the Worker is masked, whichever the way in", async () => {
    vi.stubGlobal("Worker", ScriptedWorker);
    const sentName = (type: string) =>
      ScriptedWorker.last.sent.find((m) => m.type === type)?.payload?.filename;
    const options = { filenameHint: TOKEN_HINT };

    await new StreamingH5File().open(GOOD, options);
    expect(sentName("openUrl")).toBe(masked);

    await new StreamingH5File().openLocal(
      new PickedFile(JUNK_BYTES, "a.slp"),
      options,
    );
    expect(sentName("openLocal")).toBe(masked);

    await new StreamingH5File().openLocal(
      new PickedFile(JUNK_BYTES, TOKEN_HINT),
    );
    expect(sentName("openLocal")).toBe(masked);

    await new StreamingH5File().openBuffer(JUNK_BYTES.slice(), options);
    expect(sentName("openBuffer")).toBe(masked);

    await new StreamingH5File().openRange(
      { size: 4, readRange: async () => new Uint8Array(0) },
      options,
    );
    expect(sentName("openRange")).toBe(masked);
  });

  it("a hint that isn't a URL goes as it is", async () => {
    vi.stubGlobal("Worker", ScriptedWorker);
    const sentName = (type: string) =>
      ScriptedWorker.last.sent.find((m) => m.type === type)?.payload?.filename;
    for (const hint of ["/Users/me/b.slp", "a b#1?%.slp", "C:\\me\\b.slp"]) {
      const options = { filenameHint: hint };
      await new StreamingH5File().openBuffer(JUNK_BYTES.slice(), options);
      expect(sentName("openBuffer")).toBe(hint);
      await new StreamingH5File().openRange(
        { size: 4, readRange: async () => new Uint8Array(0) },
        options,
      );
      expect(sentName("openRange")).toBe(hint);
    }
  });

  it("what's left of a URL is masked too", async () => {
    vi.stubGlobal("Worker", ScriptedWorker);
    await new StreamingH5File().openBuffer(JUNK_BYTES.slice(), {
      filenameHint: "y.slp?token=SECRET",
    });
    const sent = ScriptedWorker.last.sent.find((m) => m.type === "openBuffer");
    expect(sent?.payload?.filename).toBe("y.slp?token=***");
  });

  it("the message itself masks a URL, and leaves anything else", () => {
    expect(cannotOpenAsHdf5Message(TOKEN_HINT)).toContain(`'${masked}'`);
    expect(cannotOpenAsHdf5Message(masked)).toContain(`'${masked}'`);
    expect(cannotOpenAsHdf5Message("y.slp?token=SECRET")).toContain(
      "'y.slp?token=***'",
    );
    for (const name of ["/Users/me/b.slp", "a b#1?%.slp", "C:\\me\\b.slp"]) {
      expect(cannotOpenAsHdf5Message(name)).toContain(`'${name}'`);
    }
  });
});

describe("what a Worker reports", () => {
  beforeEach(() => {
    vi.stubGlobal("Worker", ScriptedWorker);
  });

  const quoting = "failed for https://example.test/y.slp?token=SECRET.";
  const maskedQuote = `failed for ${redactUrl(TOKEN_HINT)}.`;

  it("is masked when it fails an open", async () => {
    ScriptedWorker.answer = () => ({ success: false, error: quoting });
    expect(await messageOf(new StreamingH5File().init())).toBe(maskedQuote);
    expect(await messageOf(new StreamingH5Writer().init())).toBe(maskedQuote);
  });

  it("is masked when it comes as an object", async () => {
    ScriptedWorker.answer = () => ({
      success: false,
      error: { why: quoting },
    });
    for (const opener of [new StreamingH5File(), new StreamingH5Writer()]) {
      const message = await messageOf(opener.init());
      expect(message).toContain(redactUrl(TOKEN_HINT));
      expect(message).not.toContain("SECRET");
    }
  });

  it("is masked when the Worker itself errors, on the console too", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    ScriptedWorker.answer = () => null;
    for (const Opener of [StreamingH5File, StreamingH5Writer]) {
      const reading = messageOf(new Opener().init());
      ScriptedWorker.last.onerror?.({ message: quoting });
      expect(await reading).toBe(`Worker error: ${maskedQuote}`);
    }
    expect(error).toHaveBeenCalledTimes(2);
    expect(JSON.stringify(error.mock.calls)).not.toContain("SECRET");
  });
});

describe("masking changes what's said, not what's asked for", () => {
  it("a signed URL opens, and every request carries it as it was given", async () => {
    for (const { query } of SIGNED_QUERIES) {
      const url = `${GOOD}?${query}`;
      for (const read of [
        async () =>
          (await readSlpStreaming(url, { openVideos: false })).labeledFrames
            .length,
        async () => (await loadSlp(url, { openVideos: false })).length,
      ]) {
        server.asked = [];
        expect(await read()).toBe(1);
        expect(server.asked.length).toBeGreaterThan(0);
        for (const asked of server.asked) expect(asked).toBe(url);
      }
    }
  });

  it("nor does a name that's only sometimes a secret change what's asked for", async () => {
    const url = `${GOOD}?key=a/b.slp&v=2`;
    const labels = await readSlpStreaming(url, { openVideos: false });
    expect(labels.labeledFrames.length).toBe(1);
    expect(new Set(server.asked)).toEqual(new Set([url]));
  });

  // What the Worker opens a picked file by has to be the file's own name.
  const OWN_NAMES = [
    "labels.slp?token=abc",
    "labels.slp?X-Amz-Signature=abc",
    "labels.slp?key=abc",
    "labels.slp&key=abc",
    "a#sig=b.slp",
    "a#key=b.slp",
    // What wget calls what it fetched from a signed URL.
    "labels.slp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=abc%2F20261031&X-Amz-Date=20261031T092209Z",
  ];
  const GOOD_BYTES = FILES["/good.slp"];

  it("a picked file opens whatever its own name holds", async () => {
    for (const name of OWN_NAMES) {
      for (const read of [
        async () =>
          (
            await readSlpStreaming(new PickedFile(GOOD_BYTES, name), {
              openVideos: false,
            })
          ).labeledFrames.length,
        async () =>
          (
            await readSlpStreaming(new PickedFile(GOOD_BYTES, name), {
              openVideos: false,
              filenameHint: name,
            })
          ).labeledFrames.length,
        async () =>
          (
            await loadSlp(new PickedFile(GOOD_BYTES, name), {
              openVideos: false,
            })
          ).length,
      ]) {
        expect(await read()).toBe(1);
      }
    }
  });

  it("and what's said about it is masked", async () => {
    quietConsole();
    for (const name of OWN_NAMES) {
      const message = await messageOf(
        readSlpStreaming(new PickedFile(JUNK_BYTES, name), {
          openVideos: false,
        }),
      );
      expect(message).toContain("Could not open");
      expect(message).toContain("***");
      expect(message).not.toContain("abc");
    }
  });

  it("the Worker is sent a file's own name, and only that as it is", async () => {
    vi.stubGlobal("Worker", ScriptedWorker);
    const sent = async (file: File, filenameHint?: string) => {
      await new StreamingH5File().openLocal(file, { filenameHint });
      return ScriptedWorker.last.sent.find((m) => m.type === "openLocal")
        ?.payload?.filename;
    };
    for (const name of OWN_NAMES) {
      const file = new PickedFile(JUNK_BYTES, name);
      expect(await sent(file)).toBe(name);
      expect(await sent(file, name)).toBe(name);
      expect(await sent(new PickedFile(JUNK_BYTES, "a.slp"), name)).not.toBe(
        name,
      );
    }
  });

  it("a buffer goes by its hint, masked, and opens", async () => {
    for (const filenameHint of OWN_NAMES) {
      const labels = await readSlpStreaming(GOOD_BYTES.slice(), {
        openVideos: false,
        filenameHint,
      });
      expect(labels.labeledFrames.length).toBe(1);
    }
  });
});

// These abort h5wasm in the middle of an open, as the browser does. It goes on working,
// but they come last.
describe("a URL that fails", () => {
  it("doesn't show its token, and still says what failed", async () => {
    quietConsole();
    const url = `${MISSING}?c=1&token=SECRET`;
    expect(await failureOf(url)).toContain(
      `Couldn't load ${redactUrl(url)}. Status: 404`,
    );
    // With the token first, what's after it can't be mistaken for it.
    const first = `${MISSING}?token=SECRET&c=1`;
    expect(await failureOf(first)).toContain(
      `Couldn't load ${redactUrl(first)}. Status: 404`,
    );
  });

  it("under any hint", async () => {
    quietConsole();
    for (const filenameHint of ["b.slp", "/Users/me/b.slp", TOKEN_HINT]) {
      const message = await failureOf(`${MISSING}?token=SECRET`, {
        filenameHint,
      });
      expect(message).toContain("Status: 404");
      expect(message).not.toContain("SECRET");
    }
  });

  it("whichever reader was asked", async () => {
    quietConsole();
    const url = `${MISSING}?token=SECRET`;
    for (const read of [
      () => loadSlp(url, { openVideos: false, h5: { stream: "range" } }),
      () => loadSlp(`${MISSING}#token=SECRET`, { openVideos: false }),
      () => openH5Worker(url),
    ]) {
      const message = await messageOf(read());
      expect(message).toContain("404");
      expect(message).not.toContain("SECRET");
    }
  });

  it("from a server that refuses HEAD", async () => {
    quietConsole();
    server.refusesHead = true;
    const message = await failureOf(`${GOOD}?token=SECRET`);
    expect(message).toContain("Status: 403");
    expect(message).not.toContain("SECRET");
  });

  it("nor does a password show", async () => {
    quietConsole();
    const message = await failureOf(
      "https://me:SECRET@files.example.test/missing.slp",
    );
    expect(message).toContain("Status: 404");
    expect(message).not.toContain("SECRET");
  });

  it("nor does a token in a gs:// URL, once it's resolved", async () => {
    quietConsole();
    const message = await failureOf("gs://bucket/missing.slp?token=SECRET");
    expect(message).toContain("storage.googleapis.com/bucket/missing.slp");
    expect(message).not.toContain("SECRET");
  });

  it("when the server goes away, in the browser's words", async () => {
    quietConsole();
    server.unreachable = true;
    const url = `${GOOD}?token=SECRET`;
    expect(await failureOf(url)).toBe(
      `Failed to execute 'send' on 'XMLHttpRequest': Failed to load '${redactUrl(url)}'.`,
    );
  });

  it("when a read fails after the file opened", async () => {
    quietConsole();
    const url = `${BIG}?token=SECRET`;
    const file = await openH5Worker(url);
    server.failsFrom = 1024 * 1024;
    const message = await messageOf(file.getDatasetValue("pred_points"));
    await file.close();
    expect(message).toContain(`Couldn't load ${redactUrl(url)}. Status: 500`);
  });

  // Text alone can't show where these URLs end.
  const AWKWARD_URLS = [
    `${MISSING}?name=a b&token=SECRET`,
    `${MISSING}?name=o'b&token=SECRET`,
    `${MISSING}?name=a"b&token=SECRET`,
    `${MISSING}?name=a<b&token=SECRET`,
    `${MISSING}?token=SECRET&name=a b`,
    `${ORIGIN}/o'brien/missing.slp?token=SECRET`,
    `${ORIGIN}/a b/missing.slp?token=SECRET`,
    `${ORIGIN}/o'brien/a b/missing.slp?token=SECRET`,
    `${ORIGIN}/o'brien/missing.slp?name=a b&token=SECRET`,
    "gs://bucket/o'brien/missing.slp?token=SECRET",
    `${MISSING}#token=SECRET`,
    `${ORIGIN}/${"a".repeat(600)}/y.slp?pad=${"b".repeat(600)}&token=SECRET`,
  ];

  it("a space or a quote in it doesn't leave the token behind", async () => {
    quietConsole();
    for (const url of AWKWARD_URLS) {
      const message = await failureOf(url);
      expect(message).toContain("Status: 404");
      expect(message).not.toContain("SECRET");
    }
  });

  it("nor when the browser quotes it its own way", async () => {
    quietConsole();
    server.unreachable = true;
    for (const url of AWKWARD_URLS) {
      const message = await failureOf(url);
      expect(message).toContain("Failed to load");
      expect(message).not.toContain("SECRET");
    }
    // The URL that's too long to quote whole really was shortened.
    const longest = AWKWARD_URLS[AWKWARD_URLS.length - 1];
    expect(asChromiumQuotes(longest)).not.toContain("?");
    expect(asChromiumQuotes(longest)).toContain("SECRET");
  });

  it("h5wasm doesn't print it either", async () => {
    const said: string[] = [];
    h5wasmOutput = (...args) => said.push(args.join(" "));
    vi.spyOn(console, "warn").mockImplementation(() => {});

    const message = await failureOf(`${MISSING}?token=SECRET`);
    expect(message).toContain("Status: 404");
    expect(message).not.toContain("-sASSERTIONS");

    server.refusesHead = true;
    await failureOf(`${GOOD}?token=SECRET`);
    server.refusesHead = false;

    const file = await openH5Worker(`${BIG}?token=SECRET`);
    server.failsFrom = 1024 * 1024;
    await messageOf(file.getDatasetValue("pred_points"));
    await file.close();

    expect(said.join("\n")).not.toContain("SECRET");
    expect(said.join("\n")).not.toContain("Couldn't load");
  });

  it("an abort that quotes no URL goes on as before", async () => {
    quietConsole();
    await (await openH5Worker(GOOD)).close();
    const { onAbort } = h5wasm.Module;
    expect(onAbort).toBeFunction();
    expect(() => onAbort?.("something else")).not.toThrow();
    expect(() => onAbort?.(undefined)).not.toThrow();
    expect(() => onAbort?.("Couldn't load https://x.test/y")).toThrow(
      "Aborted(Couldn't load https://x.test/y)",
    );
  });

  it("a build of h5wasm with no Module still starts", async () => {
    const scope: {
      postMessage: (data: unknown) => void;
      onmessage?: (e: { data: unknown }) => Promise<void>;
    } = { postMessage: vi.fn() };
    new Function("self", "importScripts", "h5wasm", H5_WORKER_CODE)(
      scope,
      () => {},
      { ready: Promise.resolve(), FS: {} },
    );
    await scope.onmessage?.({ data: { type: "init", id: 1, payload: {} } });
    expect(scope.postMessage).toHaveBeenCalledWith({ id: 1, success: true });
  });

  describe("a signed URL or an API key", () => {
    const LONG_PATH = `${ORIGIN}/${"a".repeat(600)}/missing.slp`;

    it("doesn't show when the file isn't there, and the rest of the URL does", async () => {
      quietConsole();
      for (const { query, secrets, kept } of SIGNED_QUERIES) {
        const message = await failureOf(`${MISSING}?${query}`);
        expect(message).toContain(`Couldn't load ${MISSING}?`);
        expect(message).toContain("Status: 404");
        expect(shownPart(message, secrets)).toBeNull();
        for (const param of kept) expect(message).toContain(param);
      }
    });

    it("from a server that refuses HEAD", async () => {
      quietConsole();
      server.refusesHead = true;
      for (const { query, secrets } of SIGNED_QUERIES) {
        const message = await failureOf(`${GOOD}?${query}`);
        expect(message).toContain("Status: 403");
        expect(shownPart(message, secrets)).toBeNull();
      }
    });

    it("whichever reader was asked", async () => {
      quietConsole();
      for (const { query, secrets } of SIGNED_QUERIES) {
        const url = `${MISSING}?${query}`;
        for (const read of [
          () => loadSlp(url, { openVideos: false }),
          () => loadSlp(url, { openVideos: false, h5: { stream: "range" } }),
          () => openH5Worker(url),
        ]) {
          const message = await messageOf(read());
          expect(message).toContain("404");
          expect(shownPart(message, secrets)).toBeNull();
        }
      }
    });

    it("when a read fails after the file opened", async () => {
      quietConsole();
      for (const name of ["S3", "Cloud Storage"]) {
        const { query, secrets } = signedQuery(name);
        server.failsFrom = Number.POSITIVE_INFINITY;
        const file = await openH5Worker(`${BIG}?${query}`);
        server.failsFrom = 1024 * 1024;
        const message = await messageOf(file.getDatasetValue("pred_points"));
        await file.close();
        expect(message).toContain("Status: 500");
        expect(shownPart(message, secrets)).toBeNull();
      }
    });

    it("when the server goes away, in the browser's words", async () => {
      quietConsole();
      server.unreachable = true;
      for (const { query, secrets } of SIGNED_QUERIES) {
        for (const path of [MISSING, LONG_PATH]) {
          const message = await failureOf(`${path}?${query}`);
          expect(message).toContain("Failed to load");
          expect(shownPart(message, secrets)).toBeNull();
        }
      }
    });

    it("when the browser's cut takes a long value's name with it", async () => {
      quietConsole();
      server.unreachable = true;
      for (const [name, lost] of [
        ["Cloud Storage", "X-Goog-Signature="],
        ["S3 with a session token", "X-Amz-Security-Token="],
        ["S3's older form with a session token", "x-amz-security-token="],
      ]) {
        const { query, secrets } = signedQuery(name);
        const url = `${LONG_PATH}?${query}`;
        // Going by the names alone would leave what follows the cut.
        expect(asChromiumQuotes(url)).not.toContain(lost);
        expect(shownPart(asChromiumQuotes(url), secrets)).not.toBeNull();

        const message = await failureOf(url);
        expect(message).toContain(`Failed to load '${ORIGIN}/aaa`);
        expect(message).toEndWith("...***'.");
        expect(shownPart(message, secrets)).toBeNull();
      }
    });

    it("what an ordinary key costs: its value in a message", async () => {
      quietConsole();
      const message = await failureOf(`${MISSING}?key=a/b.slp&v=2`);
      expect(message).toContain(`Couldn't load ${MISSING}?key=***&v=2.`);
    });
  });

  it("the Worker can still read the next URL", async () => {
    quietConsole();
    await failureOf(`${MISSING}?token=SECRET`);
    const labels = await readSlpStreaming(GOOD, { openVideos: false });
    expect(labels.labeledFrames.length).toBe(1);
  });
});
