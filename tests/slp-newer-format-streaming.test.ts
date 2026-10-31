/**
 * What `readSlpStreaming` and `loadSlp` do after the streaming reader refuses a file
 * too new to read (`slp-newer-format.test.ts` has the refusal itself).
 *
 * Both need a browser Worker, so this file mocks the module that creates it, and
 * `loadSlp` gets a browser-looking environment. Module mocks are per process and
 * can't be undone, which is why this is its own test file (see `bun-test.ts`).
 */
import { describe, it, expect, afterEach, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as actualStreaming from "../src/codecs/slp/h5-streaming.js";

type FakeFile = {
  getAttrs: () => Promise<unknown>;
  close: () => Promise<void>;
};

let openedFile: FakeFile;
vi.mock("../src/codecs/slp/h5-streaming.js", () => ({
  ...actualStreaming,
  isStreamingSupported: () => true,
  openH5Worker: async () => openedFile,
}));

const { readSlpStreaming } = await import(
  "../src/codecs/slp/read-streaming.js"
);
const { loadSlp } = await import("../src/io/main.js");
const { UnsupportedSlpFormatError } = await import("../src/index.js");

/** A file whose metadata says `formatId`, recording whether it was closed. */
function fileWithFormat(formatId: number, close = async () => {}) {
  const fake = {
    getAttrs: async () => ({ format_id: { value: formatId } }),
    close: vi.fn(close) as unknown as () => Promise<void>,
  };
  openedFile = fake;
  return fake;
}

describe("readSlpStreaming closes the worker when a read fails", () => {
  for (const options of [
    {},
    { openVideos: true },
    { lazyVideoMetadata: true },
    { openVideos: true, lazyVideoMetadata: true },
  ]) {
    it(`options ${JSON.stringify(options)}`, async () => {
      const file = fileWithFormat(4);
      await expect(readSlpStreaming("mice.slp", options)).rejects.toThrow(
        UnsupportedSlpFormatError,
      );
      expect(file.close).toHaveBeenCalledTimes(1);
    });
  }

  it("a close that fails does not replace the clear error", async () => {
    fileWithFormat(4, async () => {
      throw new Error("worker close failed");
    });
    await expect(
      readSlpStreaming("mice.slp", { openVideos: true }),
    ).rejects.toThrow(UnsupportedSlpFormatError);
  });

  it("a close that never finishes does not hold the error back", async () => {
    // Nothing times out a message to the worker, so a stuck worker never answers.
    fileWithFormat(4, () => new Promise<void>(() => {}));
    await expect(
      readSlpStreaming("mice.slp", { openVideos: true }),
    ).rejects.toThrow(UnsupportedSlpFormatError);
  });
});

describe("loadSlp in a browser", () => {
  const versions = process.versions;
  const stubBrowser = () => {
    vi.stubGlobal("window", {});
    Object.defineProperty(process, "versions", {
      value: { ...versions, node: undefined },
      configurable: true,
    });
  };

  afterEach(() => {
    vi.unstubAllGlobals();
    Object.defineProperty(process, "versions", {
      value: versions,
      configurable: true,
    });
    vi.restoreAllMocks();
  });

  it("does not retry a file too new to read on the main thread", async () => {
    stubBrowser();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const file = fileWithFormat(4);
    // A retry would read these bytes, which are not an HDF5 file, and fail with
    // an error about that instead.
    await expect(
      loadSlp(new Uint8Array(16).buffer, {
        h5: { filenameHint: "mice.slp" },
      }),
    ).rejects.toThrow(UnsupportedSlpFormatError);
    expect(warn).not.toHaveBeenCalled();
    expect(file.close).toHaveBeenCalledTimes(1);
  });

  it("still retries on the main thread after any other streaming failure", async () => {
    stubBrowser();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    openedFile = {
      getAttrs: async () => {
        throw new Error("Path not found: metadata");
      },
      close: vi.fn(async () => {}) as unknown as () => Promise<void>,
    };
    const bytes = readFileSync(
      fileURLToPath(
        new URL("./data/slp/minimal_instance.slp", import.meta.url),
      ),
    );
    const labels = await loadSlp(new Uint8Array(bytes).buffer, {
      openVideos: false,
    });
    expect(labels.labeledFrames.length).toBe(1);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("falling back");
  });
});
