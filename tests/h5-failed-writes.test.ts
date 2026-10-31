/**
 * A write that fails is an error, however it fails and whenever it comes to light.
 *
 * The last of a file is written when it closes, and h5wasm drops what HDF5 says about
 * that, as it does for a flush or a slice. The streaming writer's worker also swallowed
 * anything thrown while closing, and `StreamingH5Writer.close` only logged what got past
 * that. So a save could report success with the end of its file missing.
 */
import {
  afterEach,
  describe,
  expect,
  it,
  setDefaultTimeout,
  vi,
} from "./bun-test";
import { existsSync, readFileSync } from "node:fs";
import { ready, File as H5File } from "h5wasm/node";
import { StreamingH5Writer } from "../src/codecs/slp/h5-streaming.js";
import { H5_WORKER_CODE } from "../src/codecs/slp/h5-worker.js";
import { getH5Module } from "../src/codecs/slp/h5.js";
import { readSlp } from "../src/codecs/slp/read.js";
import { openSlpWriter, saveSlpToBytes } from "../src/codecs/slp/write.js";
import {
  datasetIn,
  emptyFile,
  H5WASM_ON_THREAD,
  memorySink,
  onRealWorkerThreads,
  rejection,
  tmpPath,
} from "./worker-file";

// A worker thread compiles its own h5wasm, which takes a while when the suite is busy.
setDefaultTimeout(120_000);

const EMBEDDED = "py_written_30_embedded_legacy.pkg.slp";
const fixture = (name: string): Uint8Array =>
  new Uint8Array(readFileSync(new URL(`./data/slp/${name}`, import.meta.url)));

const COULD_NOT_WRITE = /Could not write \d+ bytes at offset \d+ of the file/;
const FAILED_WRITE = /Could not finish writing the file/;
const OUT_OF_STORAGE =
  /Could not write \d+ bytes at offset \d+ of the file\. The browser may be out of storage/;

afterEach(() => {
  vi.restoreAllMocks();
});

/** The expected "writeAt failed" log, which would otherwise fill the test output. */
const quietErrors = () =>
  vi.spyOn(console, "error").mockImplementation(() => {});

/** What copying the fixture's one embedded video takes. */
const copyOfVideo = async () => {
  const numbers = await datasetIn(fixture(EMBEDDED), "video0/frame_numbers");
  return {
    videoIndex: 0,
    sourceGroup: "video0",
    frameNumbers: Array.from(numbers?.value as ArrayLike<number>, Number),
    format: "png",
    channelOrder: "RGB",
  } as any;
};

/** Each way a save can open a file, with what it does before closing it. */
const SAVES: Array<
  [
    string,
    (
      writer: StreamingH5Writer,
    ) => Promise<{ dest: ReturnType<typeof memorySink>; step: () => unknown }>,
  ]
> = [
  [
    "a copy of a file's videos",
    async (writer) => {
      const source = fixture(EMBEDDED);
      const dest = memorySink(await emptyFile());
      await writer.openAppend(
        dest.sink,
        "dest.slp",
        dest.bytes().length,
        {
          size: source.length,
          readRange: async (offset, length) =>
            source.slice(offset, offset + length),
        },
        "source.slp",
        H5WASM_ON_THREAD,
      );
      const entry = await copyOfVideo();
      return { dest, step: () => writer.appendEmbeddedVideos([entry]) };
    },
  ],
  [
    "a save in place",
    async (writer) => {
      const start = fixture(EMBEDDED);
      const dest = memorySink(start);
      await writer.openWrite(
        dest.sink,
        "dest.slp",
        start.length,
        H5WASM_ON_THREAD,
      );
      return {
        dest,
        step: () =>
          writer.updateLabelsInPlace({ metadataJson: '{"edited":1}' } as any),
      };
    },
  ],
  [
    "newly embedded frames",
    async (writer) => {
      const dest = memorySink(await emptyFile());
      await writer.openWrite(
        dest.sink,
        "dest.slp",
        dest.bytes().length,
        H5WASM_ON_THREAD,
      );
      return {
        dest,
        step: async () => {
          await writer.beginEncodedVideo(
            {
              videoIndex: 0,
              format: "png",
              channelOrder: "RGB",
              frameNumbers: [0],
              rows: 1,
            } as any,
            false,
          );
          await writer.appendEncodedFrame(0, {
            kind: "png",
            bytes: new Uint8Array(5000).fill(7),
          } as any);
          await writer.endEncodedVideo();
        },
      };
    },
  ],
];

describe("a save writing through the bridge", () => {
  onRealWorkerThreads();

  for (const [what, open] of SAVES) {
    it(`${what}: closes without complaint when nothing fails`, async () => {
      const writer = new StreamingH5Writer();
      const { step } = await open(writer);
      await step();
      expect(await rejection(writer.close())).toBe("it resolved");
    });

    it(`${what}: a write that fails while the file closes rejects`, async () => {
      quietErrors();
      const writer = new StreamingH5Writer();
      const { dest, step } = await open(writer);
      await step();
      dest.failWrites(true);
      expect(await rejection(writer.close())).toMatch(COULD_NOT_WRITE);
    });

    it(`${what}: the step rejects, without waiting for the file to close`, async () => {
      quietErrors();
      const writer = new StreamingH5Writer();
      const { dest, step } = await open(writer);
      dest.failWrites(true);
      expect(await rejection(Promise.resolve().then(step))).toMatch(
        COULD_NOT_WRITE,
      );
      dest.failWrites(false);
      expect(await rejection(writer.close())).toMatch(COULD_NOT_WRITE);
    });
  }

  it("a truncate that fails while the file closes rejects", async () => {
    quietErrors();
    // Bytes past what the file says is its end, which HDF5 cuts off as it closes.
    const start = await emptyFile();
    const padded = new Uint8Array(start.length + 4096);
    padded.set(start);
    const dest = memorySink(padded);
    const writer = new StreamingH5Writer();
    await writer.openWrite(
      dest.sink,
      "dest.slp",
      padded.length,
      H5WASM_ON_THREAD,
    );
    dest.failTruncates(true);
    expect(await rejection(writer.close())).toMatch(
      /Could not change the length of the file/,
    );
    expect(dest.truncates()).toBe(1);
  });

  it("nothing more goes to the file after a write fails, and later steps are refused", async () => {
    quietErrors();
    const start = fixture(EMBEDDED);
    const dest = memorySink(start);
    const writer = new StreamingH5Writer();
    await writer.openWrite(
      dest.sink,
      "dest.slp",
      start.length,
      H5WASM_ON_THREAD,
    );
    const metadata = { metadataJson: '{"edited":1}' } as any;
    dest.failWrites(true);
    const failure = await rejection(writer.updateLabelsInPlace(metadata));
    expect(failure).toMatch(COULD_NOT_WRITE);
    dest.failWrites(false);
    const writes = dest.writes();

    expect(await rejection(writer.updateLabelsInPlace(metadata))).toBe(failure);
    expect(await rejection(writer.close())).toBe(failure);
    expect(dest.writes()).toBe(writes);
  });

  it("ends the worker whether or not closing failed", async () => {
    quietErrors();
    const writer = new StreamingH5Writer();
    const { dest, step } = await SAVES[1][1](writer);
    await step();
    const terminate = vi.spyOn((writer as any).worker, "terminate");
    dest.failWrites(true);
    await rejection(writer.close());
    expect(terminate).toHaveBeenCalledTimes(1);
  });

  it("what a healthy save wrote reads back", async () => {
    const writer = new StreamingH5Writer();
    const { dest, step } = await SAVES[0][1](writer);
    await step();
    await writer.close();
    const copied = await datasetIn(dest.bytes(), "video0/video");
    const original = await datasetIn(fixture(EMBEDDED), "video0/video");
    expect(copied?.shape).toEqual(original?.shape);
  });
});

/** An OPFS file in memory, whose sync access handle fails on request. */
function memoryOpfs() {
  let data = new Uint8Array(0);
  const failing = { write: false, flush: false, close: false, quota: false };
  const quota = () =>
    Object.assign(new Error("the storage quota is used up"), {
      name: "QuotaExceededError",
    });
  const handle = {
    read(view: Uint8Array, { at }: { at: number }) {
      const part = data.subarray(at, at + view.length);
      view.set(part);
      return part.length;
    },
    write(view: Uint8Array, { at }: { at: number }) {
      if (failing.write) throw quota();
      // Chromium 153, out of quota, writes nothing and returns -8 as an unsigned number.
      if (failing.quota) return 4294967288;
      if (at + view.length > data.length) {
        const grown = new Uint8Array(at + view.length);
        grown.set(data);
        data = grown;
      }
      data.set(view, at);
      return view.length;
    },
    truncate(length: number) {
      const resized = new Uint8Array(length);
      resized.set(data.subarray(0, length));
      data = resized;
    },
    flush() {
      if (failing.flush) throw quota();
    },
    close() {
      if (failing.close) throw quota();
    },
  };
  const directory = {
    getFileHandle: async () => ({
      createSyncAccessHandle: async () => handle,
    }),
  };
  return {
    storage: { getDirectory: async () => directory },
    failing,
    bytes: () => data,
  };
}

/**
 * A writer over the worker code on this thread. An OPFS handle needs no bridge, so nothing
 * has to block, and a test can stand in for the browser's OPFS.
 */
async function writerOverOpfs(storage: unknown): Promise<StreamingH5Writer> {
  Object.defineProperty(globalThis.navigator, "storage", {
    value: storage,
    configurable: true,
  });
  const scope: any = {
    postMessage: (message: unknown) =>
      (writer as any).handleMessage({ data: message }),
  };
  const importScripts = (path: string) => {
    new Function(
      `${readFileSync(path, "utf8")}\nglobalThis.h5wasm = h5wasm;`,
    )();
  };
  new Function("self", "importScripts", H5_WORKER_CODE)(scope, importScripts);
  const writer: StreamingH5Writer = Object.create(StreamingH5Writer.prototype);
  Object.assign(writer, {
    worker: {
      postMessage: (message: unknown) => scope.onmessage({ data: message }),
      terminate: () => {},
    },
    messageId: 0,
    pendingMessages: new Map(),
  });
  return writer;
}

describe("a save writing to OPFS", () => {
  for (const failure of ["write", "flush", "close"] as const) {
    it(`the handle's ${failure} failing as the file closes rejects`, async () => {
      const opfs = memoryOpfs();
      const writer = await writerOverOpfs(opfs.storage);
      await writer.openWriteOpfs("dest.slp", 0, H5WASM_ON_THREAD);
      opfs.failing[failure] = true;
      expect(await rejection(writer.close())).toMatch(
        /storage quota is used up/,
      );
    });
  }

  it("running out of storage is an error, though the handle doesn't throw", async () => {
    const opfs = memoryOpfs();
    const writer = await writerOverOpfs(opfs.storage);
    await writer.openWriteOpfs("dest.slp", 0, H5WASM_ON_THREAD);
    opfs.failing.quota = true;
    expect(await rejection(writer.close())).toMatch(OUT_OF_STORAGE);
  });

  it("a step that runs out of storage rejects, and so does closing the file", async () => {
    const opfs = memoryOpfs();
    const writer = await writerOverOpfs(opfs.storage);
    await writer.openWriteOpfs("dest.slp", 0, H5WASM_ON_THREAD);
    await writer.beginEncodedVideo(
      {
        videoIndex: 0,
        format: "png",
        channelOrder: "RGB",
        frameNumbers: [0],
        rows: 1,
      } as any,
      false,
    );
    opfs.failing.quota = true;
    const step = writer.appendEncodedFrame(0, {
      kind: "png",
      bytes: new Uint8Array(1 << 20).fill(7),
    } as any);
    expect(await rejection(step.then(() => writer.endEncodedVideo()))).toMatch(
      OUT_OF_STORAGE,
    );
    expect(await rejection(writer.close())).toMatch(OUT_OF_STORAGE);
  });

  it("a file that closes without complaint can be read", async () => {
    const opfs = memoryOpfs();
    const writer = await writerOverOpfs(opfs.storage);
    await writer.openWriteOpfs("dest.slp", 0, H5WASM_ON_THREAD);
    expect(await rejection(writer.close())).toBe("it resolved");

    const module = await ready;
    const path = await tmpPath();
    module.FS.writeFile(path, opfs.bytes());
    const file = new H5File(path, "r");
    expect(Number(file.file_id)).toBeGreaterThan(0);
    file.close();
    module.FS.unlink(path);
  });
});

describe("a write that fails on this thread", () => {
  /** Fail the device's writes from now on, as a full disk would. */
  async function failDeviceWrites() {
    const module = (await getH5Module()) as any;
    const noSpace = () => new module.FS.ErrnoError(28);
    const { FS } = module;
    return vi.spyOn(FS, "write").mockImplementation(() => {
      throw noSpace();
    });
  }

  it("a save rejects, instead of returning part of a file", async () => {
    const labels = await readSlp(
      new URL("./data/slp/minimal_instance.slp", import.meta.url).pathname,
      { openVideos: false },
    );
    await failDeviceWrites();
    expect(await rejection(saveSlpToBytes(labels))).toMatch(FAILED_WRITE);
  });

  it("a stream writer throws, and leaves no temporary file behind", async () => {
    const labels = await readSlp(
      new URL("./data/slp/minimal_instance.slp", import.meta.url).pathname,
      { openVideos: false },
    );
    const writer = await openSlpWriter({
      videos: labels.videos,
      skeletons: labels.skeletons,
    } as any);
    const path = (writer as any).memPath;
    await failDeviceWrites();
    expect(() => writer.close()).toThrow(FAILED_WRITE);
    vi.restoreAllMocks();
    expect(existsSync(path)).toBe(false);
  });

  it("a flush and a slice of data throw too", async () => {
    await getH5Module();
    const path = await tmpPath();
    const file = new H5File(path, "w");
    const dataset: any = file.create_dataset({
      name: "values",
      data: new Float64Array(1 << 16),
      shape: [1 << 16],
      dtype: "<f8",
    });
    await failDeviceWrites();
    expect(() => file.flush()).toThrow(FAILED_WRITE);
    expect(() =>
      dataset.write_slice([[0, 1 << 16]], new Float64Array(1 << 16)),
    ).toThrow(FAILED_WRITE);
    expect(() => file.close()).toThrow(FAILED_WRITE);
    vi.restoreAllMocks();
    (await ready).FS.unlink(path);
  });

  it("closing a file that was only being read never throws", async () => {
    await getH5Module();
    const never = new H5File("/no/such/folder/mice.slp", "r");
    expect(Number(never.file_id)).toBeLessThan(0);
    expect(() => never.close()).not.toThrow();
  });
});
