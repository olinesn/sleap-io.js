/**
 * A damaged compressed chunk is an error, not whatever was in memory.
 *
 * HDF5 fails the read, but h5wasm drops what it says and hands back the buffer it
 * allocated, so the table read as leftovers from earlier allocations. That went unnoticed
 * unless they happened to break a check further on, and on strings h5wasm went on to free
 * pointers from that buffer, which could abort the whole module.
 */
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  setDefaultTimeout,
  vi,
} from "./bun-test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inflateSync } from "node:zlib";
import { readSlp } from "../src/codecs/slp/read.js";
import { readSlpStreaming } from "../src/codecs/slp/read-streaming.js";
import {
  StreamingH5File,
  StreamingH5Writer,
} from "../src/codecs/slp/h5-streaming.js";
import { StreamingHdf5VideoBackend } from "../src/video/streaming-hdf5-video.js";
import {
  datasetIn,
  emptyFile,
  H5WASM_ON_THREAD,
  memorySink,
  onRealWorkerThreads,
  poses,
  rejection,
} from "./worker-file";

// A worker thread compiles its own h5wasm, which takes a while when the suite is busy.
setDefaultTimeout(120_000);

const fixture = (name: string): Uint8Array =>
  new Uint8Array(readFileSync(new URL(`./data/slp/${name}`, import.meta.url)));

const options = { h5wasmUrl: H5WASM_ON_THREAD, filenameHint: "mice.slp" };
const COULD_NOT_READ = /Could not read \S+ from the file/;
const COULD_NOT_READ_OR_UNPACK =
  /Could not (read \S+|unpack a score map) from the file/;

beforeEach(() => {
  // What HDF5 prints about each failure goes to stderr; the warnings are the test's own.
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

/**
 * Where a file's zlib streams are. They are its compressed chunks, and any blob it stores
 * as already-compressed bytes: the score maps, each of which HDF5 sees as plain data.
 */
function compressedChunks(
  bytes: Uint8Array,
): Array<{ offset: number; length: number }> {
  const found: Array<{ offset: number; length: number }> = [];
  for (let i = 0; i + 2 < bytes.length; i++) {
    if (bytes[i] !== 0x78 || ((bytes[i] << 8) | bytes[i + 1]) % 31 !== 0) {
      continue;
    }
    try {
      const { engine } = inflateSync(bytes.subarray(i), {
        info: true,
      } as any) as any;
      const length = engine.bytesWritten as number;
      if (length > 8) {
        found.push({ offset: i, length });
        i += length - 1;
      }
    } catch {
      // not a stream after all
    }
  }
  return found;
}

function damaged(
  bytes: Uint8Array,
  chunk: { offset: number; length: number },
): Uint8Array {
  const copy = bytes.slice();
  for (let i = chunk.offset + 2; i < chunk.offset + chunk.length; i++) {
    copy[i] ^= 0xff;
  }
  return copy;
}

const arrayBuffer = (bytes: Uint8Array): ArrayBuffer =>
  bytes.slice().buffer as ArrayBuffer;

const eagerRead = (bytes: Uint8Array) =>
  readSlp(arrayBuffer(bytes) as any, { openVideos: false });

const rangeSource = (bytes: Uint8Array) => ({
  size: bytes.length,
  readRange: async (offset: number, length: number) =>
    bytes.slice(offset, offset + length),
});

describe("a damaged compressed chunk, read on this thread", () => {
  for (const [name, scoreMaps] of [
    ["handbuilt_30_gzip.slp", 2],
    ["handbuilt_30_2x.slp", 2],
    ["py_written_28_multiview3d.slp", 0],
  ] as const) {
    it(`${name}: every chunk is an error, and the reader survives them all`, async () => {
      const bytes = fixture(name);
      const clean = poses(await eagerRead(bytes));
      const chunks = compressedChunks(bytes);
      expect(chunks.length).toBeGreaterThan(scoreMaps);

      let failedReads = 0;
      for (const chunk of chunks) {
        const message = await rejection(eagerRead(damaged(bytes, chunk)));
        expect(message).toMatch(COULD_NOT_READ_OR_UNPACK);
        if (COULD_NOT_READ.test(message)) failedReads += 1;
      }
      // HDF5 sees nothing wrong with a damaged score map: what unpacks it does.
      expect(failedReads).toBe(chunks.length - scoreMaps);

      // Freeing garbage would have aborted the module, or corrupted what's read next.
      expect(poses(await eagerRead(bytes))).toEqual(clean);
    });
  }

  it("says which table it couldn't read", async () => {
    const bytes = fixture("handbuilt_30_gzip.slp");
    const messages = new Set<string>();
    for (const chunk of compressedChunks(bytes)) {
      messages.add(await rejection(eagerRead(damaged(bytes, chunk))));
    }
    expect(messages).toContain(
      "Could not read /points/x from the file. Part of it may be damaged.",
    );
    expect(messages).toContain(
      "Could not read /frames/frame_idx from the file. Part of it may be damaged.",
    );
  });

  it("a frame of a compressed video never reads as what was in memory", async () => {
    const bytes = fixture("cropped_format_2_3.pkg.slp");
    const readFrames = async (source: Uint8Array) => {
      const dir = mkdtempSync(join(tmpdir(), "damaged-"));
      const path = join(dir, "mice.pkg.slp");
      writeFileSync(path, source);
      const labels = await readSlp(path, { openVideos: true });
      const video = labels.videos[0];
      const frames = video.embeddedFrameIndices as number[];
      return Promise.all(
        frames.map((frame) =>
          Promise.resolve(video.getFrameBuffer(frame)).catch(() => null),
        ),
      );
    };
    const clean = await readFrames(bytes);
    expect(clean.length).toBeGreaterThan(0);
    expect(clean).not.toContain(null);

    const read = await readFrames(damaged(bytes, compressedChunks(bytes)[0]));
    expect(read).toContain(null);
    read.forEach((frame, i) => {
      if (frame) expect(Buffer.compare(frame, clean[i]!)).toBe(0);
    });
  });
});

describe("a damaged compressed chunk, read in the worker", () => {
  onRealWorkerThreads();

  it("handbuilt_30_2x.slp: every chunk is an error", async () => {
    const bytes = fixture("handbuilt_30_2x.slp");
    const chunks = compressedChunks(bytes);
    expect(chunks.length).toBeGreaterThan(0);
    let failedReads = 0;
    for (const chunk of chunks) {
      const bad = damaged(bytes, chunk);
      const message = await rejection(
        readSlpStreaming(rangeSource(bad), options),
      );
      expect(message).toMatch(COULD_NOT_READ_OR_UNPACK);
      if (COULD_NOT_READ.test(message)) failedReads += 1;
    }
    // HDF5 sees nothing wrong with a damaged score map: what unpacks it does.
    expect(failedReads).toBe(chunks.length - 2);
  });

  it("handbuilt_30_gzip.slp: a chunk of each kind of table is an error", async () => {
    const bytes = fixture("handbuilt_30_gzip.slp");
    const chunks = compressedChunks(bytes);
    for (let i = 0; i < chunks.length; i += 12) {
      const bad = damaged(bytes, chunks[i]);
      expect(
        await rejection(readSlpStreaming(rangeSource(bad), options)),
      ).toMatch(COULD_NOT_READ);
    }
  });

  it("the rest of the file can still be read: HDF5 didn't lose its place", async () => {
    const bytes = fixture("handbuilt_30_gzip.slp");
    const bad = damaged(bytes, compressedChunks(bytes)[0]);
    const file = new StreamingH5File();
    await file.openRange(rangeSource(bad), options);

    const failures: string[] = [];
    let reads = 0;
    for (const table of ["identity/meta_owner", "frames/frame_idx"]) {
      await file.getDatasetValue(table).then(
        () => {
          reads += 1;
        },
        (error: Error) => {
          failures.push(error.message);
        },
      );
    }
    expect(failures.length).toBe(1);
    expect(failures[0]).toMatch(COULD_NOT_READ);
    expect(reads).toBe(1);
    await file.close();
  });

  it("a frame reads as nothing, never as what was in memory", async () => {
    const bytes = fixture("cropped_format_2_3.pkg.slp");
    const open = async (source: Uint8Array) => {
      const file = new StreamingH5File();
      await file.openRange(rangeSource(source), options);
      const backend = new StreamingHdf5VideoBackend({
        filename: "mice.slp",
        h5file: file,
        datasetPath: "video0/video",
        deferred: true,
      });
      await backend.ensureLoaded();
      return { file, backend };
    };

    const good = await open(bytes);
    const frame = good.backend.frameNumbers[0];
    expect((await good.backend.getFrameBuffer(frame))?.length).toBeGreaterThan(
      0,
    );
    await good.file.close();

    const bad = await open(damaged(bytes, compressedChunks(bytes)[0]));
    expect(await bad.backend.getFrameBuffer(frame)).toBeNull();
    await bad.file.close();
  });

  it("a save copying a damaged frame rejects, instead of copying what was in memory", async () => {
    const bytes = fixture("cropped_format_2_3.pkg.slp");
    const numbers = await datasetIn(bytes, "video0/frame_numbers");
    const dest = memorySink(await emptyFile());
    const writer = new StreamingH5Writer();
    await writer.openAppend(
      dest.sink,
      "dest.slp",
      dest.bytes().length,
      rangeSource(damaged(bytes, compressedChunks(bytes)[0])),
      "source.slp",
      H5WASM_ON_THREAD,
    );
    const failure = await rejection(
      writer.appendEmbeddedVideos([
        {
          videoIndex: 0,
          sourceGroup: "video0",
          frameNumbers: Array.from(numbers?.value as ArrayLike<number>, Number),
          format: "png",
          channelOrder: "RGB",
        } as any,
      ]),
    );
    expect(failure).toMatch(COULD_NOT_READ);
    await writer.close();
  });
});
