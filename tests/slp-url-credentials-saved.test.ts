/**
 * A URL's credentials stay out of saved files.
 *
 * A video's `filename` can be a URL with its token: the URL its labels were opened from
 * (`readSlp(url)`), a URL given as the hint for what isn't one, or the video's own URL.
 * It stays whole in memory, because `Video.exists()` goes back to it, and a save used to
 * write it into a file that people pass around. Each test here searches the saved bytes.
 *
 * Reading a URL takes `createLazyFile`, which only works in h5wasm's browser build, so
 * like `slp-url-credentials.test.ts` these tests run that build and the real worker code
 * in this process, with a server in memory. That server refuses a request without the
 * token, so a read that works here has really used it. Replacing the Node open path
 * can't be undone, which is why this is its own file (see `bun-test.ts`). The real thing
 * was checked in Chromium when this was written.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadSlp, saveAnalysisH5ToBytes } from "../src/io/main.js";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { readSlpStreaming } from "../src/codecs/slp/read-streaming.js";
import { buildSourceVideoFromDict } from "../src/codecs/slp/source-video.js";
import { appendEncodedVideos } from "../src/codecs/slp/stream-embed.js";
import {
  buildExpectedSidecars,
  buildMetadataJson,
  buildSerializableEmbedPlan,
  checkInPlaceWritable,
  openSlpWriter,
  saveSlpMergedFromStores,
  saveSlpStructureToBytes,
  saveSlpToBytes,
} from "../src/codecs/slp/write.js";
import { H5_WORKER_CODE } from "../src/codecs/slp/h5-worker.js";
import { _registerNodeH5 } from "../src/codecs/slp/h5.js";
import { toAnalysisArrays } from "../src/io/analysis-h5.js";
import { Labels } from "../src/model/labels.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { setFsResolver } from "../src/model/matching.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";
import type { Video as VideoType } from "../src/model/video.js";
import { setImageBytesReader } from "../src/video/image-source.js";

const fixture = (name: string) =>
  new Uint8Array(
    readFileSync(fileURLToPath(new URL(`./data/${name}`, import.meta.url))),
  );

const PKG_BYTES = fixture("slp/minimal_instance.pkg.slp");

const ORIGIN = "https://files.example.test";
/** The server's real token: nothing opens without it. */
const PKG = `${ORIGIN}/pkg.slp?token=SECRET`;
/** A URL as the hint, for what isn't that URL. */
const HINT = "https://x.test/y.slp?token=SECRET";

const FILES: Record<string, Uint8Array> = {
  "/pkg.slp": PKG_BYTES,
  "/plain.slp": fixture("slp/minimal_instance.slp"),
};

type Request = { method: string; url: string; range: string | null };

/** The one server every request goes to. */
const server = {
  requests: [] as Request[],

  answer(request: Request) {
    server.requests.push(request);
    const { method, url, range } = request;
    const nothing = new Uint8Array(0);
    const parsed = new URL(url);
    if (parsed.searchParams.get("token") !== "SECRET") {
      return { status: 401, headers: {}, body: nothing };
    }
    const bytes = FILES[parsed.pathname];
    if (!bytes) return { status: 404, headers: {}, body: nothing };
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
    method: init?.method ?? "GET",
    url: String(url),
    range: null,
  });
  return new Response(body as BodyInit, { status, headers });
};

type BrowserH5wasm = {
  ready: Promise<unknown>;
  File: new (
    path: string,
    mode: string,
  ) => {
    get: (name: string) => {
      attrs: Record<string, { value: string }>;
      delete_attribute: (name: string) => void;
      create_attribute: (name: string, value: string) => void;
    };
    close: () => void;
  };
  FS: {
    writeFile: (path: string, bytes: Uint8Array) => void;
    readFile: (path: string) => Uint8Array;
    unlink: (path: string) => void;
  };
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
  const { error } = console;
  console.error = () => {};
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

_registerNodeH5(async () => h5wasm as never, null as never);

beforeEach(() => {
  server.requests = [];
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
});

const text = (bytes: Uint8Array | string) =>
  typeof bytes === "string" ? bytes : new TextDecoder("latin1").decode(bytes);

/** The strings a saved file's JSON is written as, seen the way the search sees them. */
const expectNoToken = (saved: Uint8Array | string, token = "SECRET") => {
  expect(text(saved)).not.toContain(token);
};

const openVideos = { openVideos: true };
const closedVideos = { openVideos: false };

/** Every way into `Labels` whose video can hold a URL with its token. */
const READERS: Record<string, () => Promise<Labels>> = {
  "readSlp, a URL": () => readSlp(PKG, closedVideos),
  "readSlp, a URL, videos open": () => readSlp(PKG, openVideos),
  "readSlpLazy, a URL": () => readSlpLazy(PKG, closedVideos),
  "readSlpLazy, a URL, videos open": () => readSlpLazy(PKG, openVideos),
  "readSlp, a URL as a file's name": () =>
    readSlp(PKG_BYTES.slice(), { ...closedVideos, h5: { filenameHint: HINT } }),
  "readSlpLazy, a URL as a file's name": () =>
    readSlpLazy(PKG_BYTES.slice(), {
      ...closedVideos,
      h5: { filenameHint: HINT },
    }),
  "readSlpStreaming, a URL as a URL's name": () =>
    readSlpStreaming(PKG, { ...closedVideos, filenameHint: HINT }),
  "readSlpStreaming, a URL as a URL's name, videos open": () =>
    readSlpStreaming(PKG, { ...openVideos, filenameHint: HINT }),
  "readSlpStreaming, a URL as a URL's name, lazy video metadata": () =>
    readSlpStreaming(PKG, {
      ...openVideos,
      lazyVideoMetadata: true,
      filenameHint: HINT,
    }),
  "readSlpStreaming, a URL as a file's name": () =>
    readSlpStreaming(PKG_BYTES.slice(), {
      ...closedVideos,
      filenameHint: HINT,
    }),
  "loadSlp, a URL, downloaded": () =>
    loadSlp(PKG, { ...closedVideos, h5: { stream: "download" } }),
  "loadSlp, a URL, downloaded, lazy": () =>
    loadSlp(PKG, { ...closedVideos, lazy: true, h5: { stream: "download" } }),
  "loadSlp, a URL, downloaded, videos open": () =>
    loadSlp(PKG, { ...openVideos, h5: { stream: "download" } }),
  "loadSlp, a URL as a URL's name": () =>
    loadSlp(PKG, { ...closedVideos, h5: { filenameHint: HINT } }),
  "loadSlp, a URL as a file's name": () =>
    loadSlp(PKG_BYTES.slice(), { ...closedVideos, h5: { filenameHint: HINT } }),
};

/** Every way out of `Labels` and into a file, or into what makes one. */
const SAVES: Record<string, (labels: Labels) => Promise<Uint8Array | string>> =
  {
    "embed unset": (labels) => saveSlpToBytes(labels),
    "embed false": (labels) => saveSlpToBytes(labels, { embed: false }),
    'embed "source"': (labels) => saveSlpToBytes(labels, { embed: "source" }),
    "embed true": (labels) => saveSlpToBytes(labels, { embed: true }),
    'embed "user"': (labels) => saveSlpToBytes(labels, { embed: "user" }),
    "structure only": (labels) => saveSlpStructureToBytes(labels),
    'structure only, embed "source"': (labels) =>
      saveSlpStructureToBytes(labels, { embed: "source" }),
    "the streaming writer's header": async (labels) => {
      const writer = await openSlpWriter({
        skeletons: labels.skeletons,
        videos: labels.videos,
        tracks: labels.tracks,
        provenance: labels.provenance,
      });
      return writer.close();
    },
    "the streaming embed plan": async (labels) =>
      JSON.stringify(
        await buildSerializableEmbedPlan(labels, "user", "source.pkg.slp", {
          allowEncode: true,
        }),
      ),
    "the metadata an in-place save writes": async (labels) =>
      buildMetadataJson(labels),
  };

const closeAll = (labels: Labels) => {
  for (const video of labels.videos) video.close();
};

describe("no reader and no way of saving writes a URL's token", () => {
  for (const [readerName, read] of Object.entries(READERS)) {
    for (const [saveName, save] of Object.entries(SAVES)) {
      it(`${readerName}; ${saveName}`, async () => {
        const labels = await read();
        // What makes this a test: the token was there to write.
        expect(String(labels.videos[0].filename)).toContain("SECRET");
        expectNoToken(await save(labels));
        closeAll(labels);
      });
    }
  }

  it("merging lazy stores into one file", async () => {
    const labels = await readSlpLazy(PKG, closedVideos);
    expectNoToken(
      await saveSlpMergedFromStores([labels._lazyDataStore!], {
        provenance: { filename: PKG },
      }),
    );
  });

  it("a package with no record of its source: embed source falls back to the video", async () => {
    const labels = await readSlp(PKG_BYTES.slice(), {
      ...closedVideos,
      h5: { filenameHint: HINT },
    });
    labels.videos[0].sourceVideo = null;
    expectNoToken(await saveSlpToBytes(labels, { embed: "source" }));

    const lazy = await readSlpLazy(PKG, closedVideos);
    lazy.videos[0].sourceVideo = null;
    expectNoToken(await saveSlpToBytes(lazy, { embed: "source" }));
  });

  it("request headers stay with the video in memory", async () => {
    const headers = { Authorization: "Bearer HEADER-SECRET" };
    for (const read of [readSlp, readSlpLazy]) {
      const labels = await read(PKG, { ...closedVideos, h5: { headers } });
      expect(labels.videos[0]._backendUrlHeaders()).toEqual(headers);
      expectNoToken(await saveSlpToBytes(labels), "HEADER-SECRET");
    }
  });

  it("a file that isn't a package names its video, not the URL it came from", async () => {
    for (const read of [readSlp, readSlpLazy]) {
      const labels = await read(
        `${ORIGIN}/plain.slp?token=SECRET`,
        closedVideos,
      );
      expect(labels.videos[0].filename).toBe(
        "tests/data/json_format_v1/centered_pair_low_quality.mp4",
      );
      const saved = await saveSlpToBytes(labels);
      expectNoToken(saved);
      expect(text(saved)).not.toContain("plain.slp");
    }
  });

  it("the search can see what's written: the masked URL is in the same bytes", async () => {
    const labels = await readSlp(PKG, closedVideos);
    const saved = text(await saveSlpToBytes(labels));
    expect(saved).toContain("/pkg.slp?token=***");
  });
});

/** Labels with one labeled frame on `video`, whatever it is. */
const labelsOn = (video: VideoType, provenance: Record<string, unknown> = {}) =>
  new Labels({
    skeletons: [new Skeleton({ name: "s", nodes: ["a", "b"] })],
    videos: [video],
    labeledFrames: [new LabeledFrame({ video, frameIdx: 0 })],
    provenance,
  });

describe("anything in a saved file that can quote a URL", () => {
  const URLS = [
    "https://x.test/v.mp4?token=SECRET",
    "https://x.test/v.mp4?a=1&access_token=SECRET&b=2",
    "https://x.test/v.mp4?TOKEN=SECRET",
    "https://x.test/v.mp4?%74oken=SECRET",
    "https://x.test/v.mp4?token=SEC/RET",
    "https://x.test/v.mp4?token=SEC%2FRET",
    "https://x.test/v.mp4?name=a b&token=SECRET",
    "https://x.test/v.mp4?name=o'b&token=SECRET",
    "https://x.test/v.mp4?token=SEC RET",
    "https://x.test/v.mp4?token=SEC'RET",
    "https://x.test/v.mp4#token=SECRET",
    "https://me:SECRET@x.test/v.mp4",
    "https://me:SEC RET@x.test/v.mp4",
    "https://me:SECRET@x.test",
    "https://me:SECRET@host:notaport/v.mp4",
    "https://me:SEC/RET@host:notaport/v.mp4",
    "gs://me:SECRET@bucket/v.mp4",
    "gs://bucket/v.mp4?sig=SECRET",
    "s3://bucket/v.mp4?x-amz-security-token=SECRET",
    "az://container/v.mp4?sas=SECRET",
    // A name cut from a URL no longer looks like one, and still says it.
    "v.mp4?token=SECRET",
    "/some/dir/v.mp4?a=1&token=SECRET",
    "v.mp4&token=SECRET",
    "v.mp4#token=SECRET",
  ];
  const tokenOf = (url: string) =>
    url.includes("SEC RET")
      ? "SEC RET"
      : url.includes("SEC'RET")
        ? "SEC'RET"
        : url.includes("SEC/RET")
          ? "SEC/RET"
          : url.includes("SEC%2FRET")
            ? "SEC%2FRET"
            : "SECRET";

  for (const url of URLS) {
    it(`the video's own filename: ${url}`, async () => {
      const video = new Video({ filename: url, openBackend: false });
      expectNoToken(await saveSlpToBytes(labelsOn(video)), tokenOf(url));
      expect(video.filename).toBe(url);
    });
  }

  it("what's around the token stays", async () => {
    const saved = text(
      await saveSlpToBytes(
        labelsOn(
          new Video({
            filename: "https://x.test/a/v.mp4?a=1&token=SECRET&b=2",
            openBackend: false,
          }),
        ),
      ),
    );
    expect(saved).toContain("https://x.test/a/v.mp4?a=1&token=***&b=2");
  });

  it("an image sequence's every filename", async () => {
    const urls = [0, 1, 2].map((i) => `https://x.test/${i}.png?token=SECRET`);
    const video = new Video({ filename: urls, openBackend: false });
    const saved = await saveSlpToBytes(labelsOn(video));
    expectNoToken(saved);
    expect(text(saved)).toContain("https://x.test/2.png?token=***");
    expect(video.filename).toEqual(urls);
  });

  it("anything in the backend's metadata, read from a file or not", async () => {
    const video = new Video({
      filename: "v.mp4",
      openBackend: false,
      backendMetadata: {
        filename: "https://x.test/1.mp4?token=SECRET",
        filenames: ["https://x.test/2.mp4?token=SECRET"],
        source_filename: "https://x.test/3.mp4?token=SECRET",
        nested: { deeper: ["https://x.test/4.mp4?token=SECRET"] },
      },
    });
    const saved = await saveSlpToBytes(labelsOn(video));
    expectNoToken(saved);
    for (const n of [1, 2, 3, 4]) {
      expect(text(saved)).toContain(`https://x.test/${n}.mp4?token=***`);
    }
  });

  it("a source video, at any depth, in videos_json", async () => {
    const original = new Video({
      filename: "https://x.test/original.mp4?token=SECRET",
      openBackend: false,
    });
    const middle = new Video({
      filename: "https://x.test/middle.mp4?token=SECRET",
      sourceVideo: original,
      openBackend: false,
    });
    const video = new Video({
      filename: "v.mp4",
      sourceVideo: middle,
      openBackend: false,
    });
    const saved = await saveSlpToBytes(labelsOn(video));
    expectNoToken(saved);
    expect(text(saved)).toContain("https://x.test/original.mp4?token=***");
    expect(text(saved)).toContain("https://x.test/middle.mp4?token=***");
  });

  it("a source video in an embedded video's own group", async () => {
    const labels = await readSlp(PKG, openVideos);
    const source = "https://x.test/original.mp4?token=SECRET";
    labels.videos[0].sourceVideo = new Video({
      filename: source,
      openBackend: false,
    });
    const saved = await saveSlpToBytes(labels);
    expectNoToken(saved);
    // Read back from `{group}/source_video`, which is what Python reads, not videos_json.
    const again = await readSlp(saved, closedVideos);
    expect(again.videos[0].sourceVideo?.filename).toBe(
      "https://x.test/original.mp4?token=***",
    );
    const plan = await buildSerializableEmbedPlan(labels, false, "x.pkg.slp");
    expect(plan.entries[0].sourceVideoJson?.filename).toBe(
      "https://x.test/original.mp4?token=***",
    );
    closeAll(labels);
  });

  it("a source that refers to the labels themselves", async () => {
    const video = new Video({
      filename: "v.mp4",
      openBackend: false,
      sourceVideo: buildSourceVideoFromDict(
        { filename: ".", backend: { filename: ".", dataset: "video0/video" } },
        HINT,
      ),
    });
    expect(video.sourceVideo?.filename).toBe(HINT);
    expectNoToken(await saveSlpToBytes(labelsOn(video)));
  });

  it("provenance, at any depth, keys included", async () => {
    const provenance = {
      filename: "https://x.test/1.slp?token=SECRET",
      merge_history: [
        { source_filename: "https://x.test/2.slp?token=SECRET", n: 1 },
      ],
      command: "sleap-track https://x.test/3.mp4?token=SECRET --batch-size 4",
      dlc_crops: { "https://x.test/4.mp4?token=SECRET": [0, 0, 10, 10] },
      when: new Date(0),
    };
    const labels = labelsOn(new Video({ filename: "v.mp4" }), provenance);
    const saved = await saveSlpToBytes(labels);
    expectNoToken(saved);
    expectNoToken(buildMetadataJson(labels));
    expect(text(saved)).toContain(
      "https://x.test/3.mp4?token=*** --batch-size 4",
    );
    expect(text(saved)).toContain('"when":"1970-01-01T00:00:00.000Z"');
    // Still there for whoever holds the labels.
    expect(labels.provenance).toBe(provenance);
    expect(provenance.filename).toContain("SECRET");
  });

  it("a SLEAP analysis file: its paths, provenance and backend metadata", async () => {
    const video = new Video({
      filename: "https://x.test/v.mp4?token=SECRET",
      openBackend: false,
      backendMetadata: { filename: "https://x.test/v.mp4?token=SECRET" },
    });
    const labels = labelsOn(video, {
      filename: "https://x.test/labels.slp?token=SECRET",
    });
    const saved = await saveAnalysisH5ToBytes(labels, {
      labelsPath: "https://x.test/labels.slp?token=SECRET",
    });
    expectNoToken(saved);
    expect(text(saved)).toContain("https://x.test/v.mp4?token=***");
  });
});

/** A file as this library used to write them: the mask put back, byte for byte. */
const putTokenBack = (saved: Uint8Array) => {
  const mask = new TextEncoder().encode("token=***");
  const put = new TextEncoder().encode("token=OLD");
  for (let i = 0; i < saved.length - mask.length; i++) {
    if (mask.every((b, j) => saved[i + j] === b)) saved.set(put, i);
  }
  return saved;
};

describe("a file that already holds a token", () => {
  const fileHoldingToken = async () => {
    const saved = await saveSlpToBytes(
      labelsOn(
        new Video({
          filename: "https://x.test/v.mp4?token=SECRET",
          openBackend: false,
          sourceVideo: new Video({
            filename: "https://x.test/source.mp4?token=SECRET",
            openBackend: false,
          }),
        }),
      ),
    );
    const path = "/holding-token.slp";
    h5wasm.FS.writeFile(path, putTokenBack(saved));
    const file = new h5wasm.File(path, "a");
    const metadata = file.get("metadata");
    const json = JSON.parse(metadata.attrs.json.value);
    json.provenance = { filename: "https://x.test/labels.slp?token=OLD" };
    metadata.delete_attribute("json");
    metadata.create_attribute("json", JSON.stringify(json));
    file.close();
    const bytes = h5wasm.FS.readFile(path).slice();
    h5wasm.FS.unlink(path);

    expect(text(bytes)).toContain("v.mp4?token=OLD");
    expect(text(bytes)).toContain("source.mp4?token=OLD");
    expect(text(bytes)).toContain("labels.slp?token=OLD");
    return bytes;
  };

  it("loses it the next time it's saved", async () => {
    const labels = await readSlp(await fileHoldingToken(), closedVideos);
    expect(labels.videos[0].filename).toContain("token=OLD");
    const saved = await saveSlpToBytes(labels);
    expectNoToken(saved, "OLD");
    expect(text(saved)).toContain("https://x.test/v.mp4?token=***");
  });

  it("isn't saved in place, which would leave it there", async () => {
    const labels = await readSlp(await fileHoldingToken(), closedVideos);
    const expected = buildExpectedSidecars(labels);
    const writable = checkInPlaceWritable({} as never, {} as never, {
      onDisk: expected,
      expected,
    });
    expect(writable.ok).toBe(false);
    expect(writable.ok ? "" : writable.reason).toContain("re-save");
    expect(writable.ok ? "" : writable.reason).not.toContain("OLD");
  });

  it("a file holding only a masked URL is saved in place as any other", async () => {
    const saved = await saveSlpToBytes(
      labelsOn(
        new Video({
          filename: "https://x.test/v.mp4?token=SECRET",
          openBackend: false,
        }),
      ),
    );
    const expected = buildExpectedSidecars(await readSlp(saved, closedVideos));
    expect(
      checkInPlaceWritable({} as never, {} as never, {
        onDisk: expected,
        expected,
      }).ok,
    ).toBe(true);
  });
});

describe("what a save leaves alone", () => {
  it("what's opened from a URL goes on working after a save", async () => {
    const labels = await readSlp(PKG, openVideos);
    const [video] = labels.videos;
    const frame = video.embeddedFrameIndices![0];
    expect(await video.getFrameBuffer(frame)).not.toBeNull();

    expectNoToken(await saveSlpToBytes(labels));

    expect(video.filename).toBe(PKG);
    expect(await video.getFrameBuffer(frame)).not.toBeNull();
    expect(await video.exists()).toBe(true);
    closeAll(labels);
  });

  it("every read of that URL had its token", async () => {
    const labels = await readSlp(PKG, openVideos);
    await labels.videos[0].getFrameBuffer(
      labels.videos[0].embeddedFrameIndices![0],
    );
    await saveSlpToBytes(labels);
    expect(server.requests.length).toBeGreaterThan(0);
    for (const { url } of server.requests) expect(url).toBe(PKG);
    closeAll(labels);
  });

  it("the server does turn away what has no token", async () => {
    const video = new Video({
      filename: `${ORIGIN}/pkg.slp?token=***`,
      openBackend: false,
    });
    expect(await video.exists()).toBe(false);
  });

  it("a saved package still has its images, and reads them", async () => {
    const labels = await readSlp(PKG, openVideos);
    const frame = labels.videos[0].embeddedFrameIndices![0];
    const stored = await labels.videos[0].getFrameBuffer(frame);
    const saved = await saveSlpToBytes(labels);
    closeAll(labels);

    const again = await readSlpStreaming(saved, openVideos);
    expect(again.videos[0].hasEmbeddedImages).toBe(true);
    expect(await again.videos[0].getFrameBuffer(frame)).toEqual(stored!);
    closeAll(again);
  });

  it("a package saved with its images closed still opens", async () => {
    const labels = await readSlp(PKG_BYTES.slice(), {
      ...closedVideos,
      h5: { filenameHint: HINT },
    });
    const count = labels.labeledFrames.length;
    const again = await readSlp(await saveSlpToBytes(labels), closedVideos);
    expect(again.labeledFrames.length).toBe(count);
    expect(again.videos[0].hasEmbeddedImages).toBe(true);
  });

  it("a video that's a URL with its token masked comes back as that", async () => {
    const video = new Video({
      filename: "https://x.test/v.mp4?token=SECRET",
      openBackend: false,
    });
    const again = await readSlp(
      await saveSlpToBytes(labelsOn(video)),
      closedVideos,
    );
    expect(again.videos[0].filename).toBe("https://x.test/v.mp4?token=***");
  });

  it("a filename with nothing to mask comes back as it was", async () => {
    for (const filename of [
      "https://x.test",
      "HTTP://X.TEST/a b.mp4?x=1 2",
      "https://x.test/a%20b.mp4?x=é#frag",
      "https://x.test/v.mp4?token=***",
      "gs://bucket/a b.mp4",
      "/Users/me/a b#1?%.mp4",
      "C:\\Users\\me\\b.mp4",
      "tests/data/videos/centered_pair_low_quality.mp4",
      "日本語.mp4",
    ]) {
      const saved = await saveSlpToBytes(
        labelsOn(new Video({ filename, openBackend: false }), {
          filename,
          nested: [{ filename }],
        }),
      );
      const again = await readSlp(saved, closedVideos);
      expect(again.videos[0].filename).toBe(filename);
      expect(again.provenance).toEqual({ filename, nested: [{ filename }] });
    }
  });

  it("a file with no URL in it saves the same strings as it holds", async () => {
    for (const name of [
      "minimal_instance.slp",
      "minimal_instance.pkg.slp",
      "labels.v002.rel_paths.slp",
      "predictions_1.2.7_provenance_and_tracking.slp",
      "imgvideo.slp",
    ]) {
      const labels = await readSlp(fixture(`slp/${name}`), closedVideos);
      const again = await readSlp(await saveSlpToBytes(labels), closedVideos);
      expect(again.provenance).toEqual(labels.provenance);
      expect(again.videos.map((v) => v.filename)).toEqual(
        labels.videos.map((v) => v.filename),
      );
    }
  });
});

describe("what says a video's name doesn't show its token", () => {
  const video = new Video({
    filename: "https://x.test/v.mp4?token=SECRET",
    openBackend: false,
  });

  const messageOf = async (doing: () => unknown): Promise<string> => {
    try {
      await doing();
    } catch (e) {
      return (e as Error).message;
    }
    throw new Error("It should have failed");
  };

  it("an analysis export of a video with no frames", async () => {
    const labels = new Labels({
      skeletons: [new Skeleton({ name: "s", nodes: ["a"] })],
      videos: [video],
    });
    const message = await messageOf(() =>
      toAnalysisArrays(labels, video, false, 0),
    );
    expect(message).toContain("No labeled frames in video:");
    expect(message).toContain("https://x.test/v.mp4?token=***");
    expectNoToken(message);
  });

  it("an analysis export, for an image sequence", async () => {
    const urls = ["https://x.test/0.png?token=SECRET", "1.png?token=SECRET"];
    const sequence = new Video({ filename: urls, openBackend: false });
    const labels = new Labels({
      skeletons: [new Skeleton({ name: "s", nodes: ["a"] })],
      videos: [sequence],
    });
    const message = await messageOf(() =>
      toAnalysisArrays(labels, sequence, false, 0),
    );
    expect(message).toContain("0.png?token=***,1.png?token=***");
    expectNoToken(message);
  });

  it("an embed list naming a video that isn't in the labels", async () => {
    const labels = labelsOn(new Video({ filename: "v.mp4" }));
    const message = await messageOf(() =>
      saveSlpToBytes(labels, { embed: [[video, 0]] }),
    );
    expect(message).toContain("isn't in these labels");
    expect(message).toContain("https://x.test/v.mp4?token=***");
    expectNoToken(message);
  });

  const twins = () => {
    const urls = ["https://x.test/a/v.mp4", "https://x.test/b/v.mp4"];
    return new Labels({
      skeletons: [new Skeleton({ name: "s", nodes: ["a"] })],
      videos: urls.map(
        (url) =>
          new Video({ filename: `${url}?token=SECRET`, openBackend: false }),
      ),
    });
  };

  it("two videos that a name could mean", async () => {
    const asked = "https://y.test/c/v.mp4?token=SECRET";
    for (const find of [
      (labels: Labels) => labels.find({ video: asked }),
      (labels: Labels) => labels.matchVideo(asked),
    ]) {
      const message = await messageOf(() => find(twins()));
      expect(message).toContain("Ambiguous video match");
      expect(message).toContain("https://y.test/c/v.mp4?token=***");
      expect(message).toContain("https://x.test/a/v.mp4?token=***");
      expectNoToken(message);
    }
  });

  it("a frame that can't be read to embed", async () => {
    const unreadable = new Video({
      filename: "https://x.test/v.mp4?token=SECRET",
      openBackend: false,
      backend: {
        filename: "https://x.test/v.mp4?token=SECRET",
        shape: [1, 4, 4, 1],
        getFrame: async () => null,
        close: () => {},
      } as never,
    });
    const message = await messageOf(() =>
      saveSlpToBytes(labelsOn(unreadable), { embed: true }),
    );
    expect(message).toContain("Couldn't read 1 frame image(s)");
    expect(message).toContain("https://x.test/v.mp4?token=***");
    expectNoToken(message);
  });

  it("a frame that can't be read to embed, on the streaming path", async () => {
    const unreadable = new Video({
      filename: "https://x.test/v.mp4?token=SECRET",
      openBackend: false,
      backend: {
        filename: "https://x.test/v.mp4?token=SECRET",
        shape: [1, 4, 4, 1],
        getFrame: async () => null,
        close: () => {},
      } as never,
    });
    const labels = labelsOn(unreadable);
    const { encodeEntries } = await buildSerializableEmbedPlan(
      labels,
      true,
      "",
      { allowEncode: true },
    );
    const message = await messageOf(() =>
      appendEncodedVideos(
        {
          beginEncodedVideo: async () => {},
          appendEncodedFrame: async () => {},
          endEncodedVideo: async () => ({ framesWritten: 0 }),
        },
        labels,
        encodeEntries,
      ),
    );
    expect(message).toContain("Couldn't read frame 0");
    expect(message).toContain("https://x.test/v.mp4?token=***");
    expectNoToken(message);
  });

  it("a video that didn't open", async () => {
    const labels = await readSlp(PKG_BYTES.slice(), {
      ...openVideos,
      h5: { filenameHint: `${ORIGIN}/missing.slp?token=SECRET` },
    });
    // The hint is a URL, so the embedded video tried to open it.
    expect(labels.videos[0].backendError?.message).toContain("status=404");
    expectNoToken(labels.videos[0].backendError?.message ?? "");
    closeAll(labels);
  });

  describe("an image sequence of URLs in a file that holds their token", () => {
    const fsResolver = {
      sameFile: async () => false,
      realpath: async (path: string) => path,
    };

    const sequenceHoldingToken = async () => {
      const saved = await saveSlpToBytes(
        labelsOn(
          new Video({
            filename: [0, 1].map((i) => `https://x.test/${i}.png?token=SECRET`),
            openBackend: false,
          }),
        ),
      );
      return putTokenBack(saved);
    };

    afterEach(() => {
      setFsResolver(null);
      setImageBytesReader(null);
    });

    it("that can't be found", async () => {
      setFsResolver({ ...fsResolver, exists: async () => false });
      const labels = await readSlp(await sequenceHoldingToken(), openVideos);
      const message = labels.videos[0].backendError?.message ?? "";
      expect(message).toContain("Image sequence not found");
      expect(message).toContain("https://x.test/0.png?token=***");
      expectNoToken(message, "OLD");
    });

    it("that fails to read, saying so in its own words", async () => {
      setFsResolver({ ...fsResolver, exists: async () => true });
      setImageBytesReader(async (path) => {
        throw new Error(`Couldn't read ${path}`);
      });
      const labels = await readSlp(await sequenceHoldingToken(), openVideos);
      const message = labels.videos[0].backendError?.message ?? "";
      expect(message).toContain("Couldn't read https://x.test/0.png?token=***");
      expectNoToken(message, "OLD");
    });
  });
});
