/**
 * A range source whose read fails is an error, not zeros.
 *
 * `readRange` is how an app hands over a file it reads itself, as sleap-app does for
 * large files on desktop. A failed read used to be answered as "no bytes", which the
 * worker took for the end of the file. HDF5 fills that in with zeros, so the file opened
 * with whatever the zeros decoded to. The same bridge carries the reads of a save, where
 * the zeros would be copied into the new file as video frames.
 *
 * These run on real worker threads, because the bridge's worker blocks until the main
 * thread answers.
 */
import {
  afterEach,
  describe,
  expect,
  it,
  setDefaultTimeout,
  vi,
} from "./bun-test";
import { readFileSync } from "node:fs";
import { ready, File as H5File } from "h5wasm/node";
import { readSlp } from "../src/codecs/slp/read.js";
import { readSlpStreaming } from "../src/codecs/slp/read-streaming.js";
import {
  type RangeSink,
  type RangeSource,
  StreamingH5File,
  StreamingH5Writer,
} from "../src/codecs/slp/h5-streaming.js";
import { StreamingHdf5VideoBackend } from "../src/video/streaming-hdf5-video.js";
import {
  H5WASM_ON_THREAD,
  onRealWorkerThreads,
  poses,
  tmpPath,
} from "./worker-file";

// A worker thread compiles its own h5wasm, which takes a while when the suite is busy.
setDefaultTimeout(120_000);

const fixture = (name: string): Uint8Array =>
  new Uint8Array(readFileSync(new URL(`./data/slp/${name}`, import.meta.url)));

const PYTHON_2X = "py_written_30_twin_legacy.slp";
const PYTHON_30 = "py_written_30_twin.slp";
const EMBEDDED = "py_written_30_embedded_legacy.pkg.slp";

const options = { h5wasmUrl: H5WASM_ON_THREAD, filenameHint: "mice.slp" };
const FAILED_READ = /Could not read \d+ bytes at offset \d+ of the file/;

afterEach(() => {
  vi.restoreAllMocks();
});

/**
 * What a promise that needs a worker thread rejects with. Not `expect().rejects`: that
 * waits in a way that can keep the worker's answer from ever being delivered.
 */
const rejection = (promise: Promise<unknown>): Promise<string> =>
  promise.then(
    () => "it resolved",
    (error: Error) => error.message,
  );

/** The expected "readRange failed" log, which would otherwise fill the test output. */
const quietErrors = () =>
  vi.spyOn(console, "error").mockImplementation(() => {});

/** A range source over `bytes` whose reads fail while `failing` says so. */
function rangeSource(
  bytes: Uint8Array,
  failing: (offset: number, length: number) => boolean = () => false,
): RangeSource & { failedReads: () => number } {
  let failedReads = 0;
  return {
    size: bytes.length,
    readRange: async (offset, length) => {
      if (failing(offset, length)) {
        failedReads += 1;
        throw new Error("the drive went away");
      }
      return bytes.slice(offset, offset + length);
    },
    failedReads: () => failedReads,
  };
}

/** Open and read `bytes`, failing every read while the worker answers one request. */
async function readFailingDuring(
  bytes: Uint8Array,
  type: string,
  path: string,
) {
  let armed = false;
  const prototype = StreamingH5File.prototype as any;
  const send = prototype.send;
  vi.spyOn(prototype, "send").mockImplementation(function (
    this: unknown,
    t: unknown,
    payload: any,
  ) {
    if (t !== type || payload?.path !== path) {
      return send.call(this, t, payload);
    }
    armed = true;
    return send.call(this, t, payload).finally(() => {
      armed = false;
    });
  } as any);
  const source = rangeSource(bytes, () => armed);
  const outcome = await readSlpStreaming(source, options).then(
    (labels) => ({ labels, error: undefined }),
    (error: Error) => ({ labels: undefined, error }),
  );
  return { ...outcome, failedReads: source.failedReads() };
}

describe("a range source whose read fails", () => {
  onRealWorkerThreads();

  for (const [file, reads] of [
    [
      PYTHON_2X,
      [
        ["getDatasetValue", "frames"],
        ["getDatasetValue", "instances"],
        ["getDatasetValue", "points"],
        ["getDatasetValue", "pred_points"],
        ["getDatasetValue", "negative_frames"],
        ["getDatasetValue", "tracks_json"],
        ["getDatasetValue", "videos_json"],
        ["getDatasetValue", "suggestions_json"],
        ["getDatasetValue", "sessions_json"],
        ["getDatasetValue", "provenance_json"],
        ["getDatasetValue", "identity/name"],
        ["getKeys", "session_data"],
      ],
    ],
    [
      PYTHON_30,
      [
        ["getDatasetValue", "frames"],
        ["getDatasetValue", "instances"],
        ["getDatasetValue", "points"],
        ["getDatasetValue", "pred_points"],
        ["getDatasetValue", "suggestions_json"],
      ],
    ],
  ] as const) {
    it(`${file}: reads as the eager reader does when nothing fails`, async () => {
      const bytes = fixture(file);
      const labels = await readSlpStreaming(rangeSource(bytes), options);
      const eager = await readSlp(bytes.slice().buffer, {
        openVideos: false,
        h5: { filenameHint: "mice.slp" },
      });
      expect(poses(labels)).toEqual(poses(eager));
      expect(labels.tracks.length).toBe(eager.tracks.length);
      expect(labels.videos.length).toBe(eager.videos.length);
    });

    for (const [type, path] of reads) {
      it(`${file}: failing during ${type} ${path} is an error`, async () => {
        quietErrors();
        const { error, failedReads } = await readFailingDuring(
          fixture(file),
          type,
          path,
        );
        expect(failedReads).toBeGreaterThan(0);
        expect(error?.message).toMatch(FAILED_READ);
      });
    }
  }

  it("no block of a file can fail without an error", async () => {
    quietErrors();
    const bytes = fixture("py_written_25_identity.slp");
    const BLOCK = 4096;
    for (let start = 0; start < bytes.length; start += BLOCK) {
      const source = rangeSource(
        bytes,
        (offset, length) => offset < start + BLOCK && offset + length > start,
      );
      const outcome = await rejection(readSlpStreaming(source, options));
      expect(source.failedReads()).toBeGreaterThan(0);
      expect(outcome).toMatch(FAILED_READ);
    }
  });

  it("a file shorter than the source says is an error", async () => {
    const bytes = fixture(PYTHON_2X);
    const cut = bytes.slice(0, Math.floor(bytes.length / 2));
    expect(
      await rejection(
        readSlpStreaming(
          {
            size: bytes.length,
            readRange: async (offset, length) =>
              cut.slice(offset, offset + length),
          },
          options,
        ),
      ),
    ).toMatch(FAILED_READ);
  });

  it("a source that returns less than it was asked for is fine", async () => {
    const bytes = fixture(PYTHON_2X);
    const labels = await readSlpStreaming(
      {
        size: bytes.length,
        readRange: async (offset, length) =>
          bytes.slice(offset, offset + Math.min(length, 100)),
      },
      options,
    );
    expect(poses(labels)).toEqual(
      poses(
        await readSlp(bytes.slice().buffer, {
          openVideos: false,
          h5: { filenameHint: "mice.slp" },
        }),
      ),
    );
  });

  // HDF5 stops mid-call on a failed read, with its buffers set up for bytes that never
  // came. Asked again, it answers from them: this read used to come back as zeros.
  for (const [what, read] of [
    ["a table", (f: StreamingH5File) => f.getDatasetValue("frames")],
    ["strings", (f: StreamingH5File) => f.getDatasetValue("videos_json")],
    ["a listing", (f: StreamingH5File) => f.getKeys("session_data")],
  ] as const) {
    it(`after a failed read of ${what}, nothing more comes out of the file`, async () => {
      quietErrors();
      let armed = false;
      const file = new StreamingH5File();
      await file.openRange(
        rangeSource(fixture(PYTHON_2X), () => armed),
        options,
      );
      await file.getDatasetMeta("frames").catch(() => {});
      await file.getDatasetMeta("videos_json").catch(() => {});

      armed = true;
      const failure = await rejection(read(file));
      expect(failure).toMatch(FAILED_READ);

      armed = false;
      expect(await rejection(read(file))).toBe(failure);
      expect(await rejection(file.getDatasetValue("points"))).toBe(failure);
      expect(await rejection(file.getAttrs("metadata"))).toBe(failure);
      await file.close();
    });
  }

  it("a video's frame reads as nothing after a failed read, never as zeros", async () => {
    quietErrors();
    let armed = false;
    const bytes = fixture(EMBEDDED);
    const file = new StreamingH5File();
    await file.openRange(
      rangeSource(bytes, () => armed),
      options,
    );
    const backend = new StreamingHdf5VideoBackend({
      filename: "mice.slp",
      h5file: file,
      datasetPath: "video0/video",
      deferred: true,
    });
    await backend.ensureLoaded();
    const frame = backend.frameNumbers[0];
    const healthy = await backend.getFrameBuffer(frame);
    expect(healthy?.some((byte) => byte !== 0)).toBe(true);

    const other = new StreamingH5File();
    await other.openRange(
      rangeSource(bytes, () => armed),
      options,
    );
    const failing = new StreamingHdf5VideoBackend({
      filename: "mice.slp",
      h5file: other,
      datasetPath: "video0/video",
      deferred: true,
    });
    await failing.ensureLoaded();
    armed = true;
    expect(await failing.getFrameBuffer(frame)).toBeNull();
    armed = false;
    expect(await failing.getFrameBuffer(frame)).toBeNull();

    await file.close();
    await other.close();
  });
});

/** A file in memory, as a native write handle would be on disk. */
function memorySink(initial: Uint8Array) {
  let data = initial.slice();
  let failing = false;
  let writes = 0;
  const sink: RangeSink = {
    writeAt: async (offset, bytes) => {
      writes += 1;
      if (offset + bytes.length > data.length) {
        const grown = new Uint8Array(offset + bytes.length);
        grown.set(data);
        data = grown;
      }
      data.set(bytes, offset);
    },
    readAt: async (offset, length) => {
      if (failing) throw new Error("the drive went away");
      return data.slice(offset, offset + length);
    },
    truncate: async (length) => {
      const resized = new Uint8Array(length);
      resized.set(data.subarray(0, length));
      data = resized;
    },
    close: async () => {},
  };
  return {
    sink,
    bytes: () => data,
    writes: () => writes,
    failReads: (on: boolean) => {
      failing = on;
    },
  };
}

/** Read one dataset from a file's bytes, on this thread. */
async function datasetIn(bytes: Uint8Array, name: string) {
  const module = await ready;
  const p = await tmpPath();
  module.FS.writeFile(p, bytes);
  const f = new H5File(p, "r");
  try {
    const dataset = f.get(name) as any;
    return dataset ? { value: dataset.value, shape: dataset.shape } : null;
  } finally {
    f.close();
    module.FS.unlink(p);
  }
}

async function emptyFile(): Promise<Uint8Array> {
  const module = await ready;
  const p = await tmpPath();
  const f = new H5File(p, "w");
  f.create_group("metadata");
  f.close();
  const bytes = new Uint8Array(module.FS.readFile(p));
  module.FS.unlink(p);
  return bytes;
}

describe("a save reading through the bridge", () => {
  onRealWorkerThreads();

  const entry = async () => {
    const numbers = await datasetIn(fixture(EMBEDDED), "video0/frame_numbers");
    return {
      videoIndex: 0,
      sourceGroup: "video0",
      frameNumbers: Array.from(numbers?.value as ArrayLike<number>, Number),
      format: "png",
      channelOrder: "RGB",
    } as any;
  };

  it("copies a video's frames when nothing fails", async () => {
    const source = fixture(EMBEDDED);
    const dest = memorySink(await emptyFile());
    const writer = new StreamingH5Writer();
    await writer.openAppend(
      dest.sink,
      "dest.slp",
      dest.bytes().length,
      rangeSource(source),
      "source.slp",
      H5WASM_ON_THREAD,
    );
    await writer.appendEmbeddedVideos([await entry()]);
    await writer.close();
    const copied = await datasetIn(dest.bytes(), "video0/video");
    const original = await datasetIn(source, "video0/video");
    expect(copied?.shape).toEqual(original?.shape);
    expect(Array.from(copied?.value as ArrayLike<number>)).toEqual(
      Array.from(original?.value as ArrayLike<number>),
    );
  });

  it("a failed read of the source rejects, instead of copying zeros as frames", async () => {
    quietErrors();
    let armed = false;
    const dest = memorySink(await emptyFile());
    const writer = new StreamingH5Writer();
    await writer.openAppend(
      dest.sink,
      "dest.slp",
      dest.bytes().length,
      rangeSource(fixture(EMBEDDED), () => armed),
      "source.slp",
      H5WASM_ON_THREAD,
    );
    const copy = await entry();
    armed = true;
    expect(await rejection(writer.appendEmbeddedVideos([copy]))).toMatch(
      FAILED_READ,
    );
    await writer.close();
  });

  it("a failed read of the file being saved in place rejects, and stops writing to it", async () => {
    quietErrors();
    const start = fixture(EMBEDDED);
    const frames = await datasetIn(start, "frames");
    const dest = memorySink(start);
    const writer = new StreamingH5Writer();
    await writer.openWrite(
      dest.sink,
      "dest.slp",
      start.length,
      H5WASM_ON_THREAD,
    );
    const writesAtOpen = dest.writes();

    dest.failReads(true);
    expect(
      await rejection(
        writer.updateLabelsInPlace({ metadataJson: "{}" } as any),
      ),
    ).toMatch(FAILED_READ);
    dest.failReads(false);
    await writer.close();

    expect(dest.writes()).toBe(writesAtOpen);
    expect(await datasetIn(dest.bytes(), "frames")).toEqual(frames);
  });

  it("a healthy in-place save closes the file as it should", async () => {
    const start = fixture(EMBEDDED);
    const dest = memorySink(start);
    const writer = new StreamingH5Writer();
    await writer.openWrite(
      dest.sink,
      "dest.slp",
      start.length,
      H5WASM_ON_THREAD,
    );
    await writer.updateLabelsInPlace({} as any);
    await writer.close();
    expect(Buffer.compare(dest.bytes(), start)).toBe(0);
  });
});
