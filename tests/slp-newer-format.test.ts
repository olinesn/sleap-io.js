/**
 * An `.slp` file newer than this library can read must be refused, not read as empty.
 *
 * Python sleap-io 0.10 writes SLP format 3.0 by default: every table is a group of
 * columns instead of a compound dataset. Read with the 2.x code those tables come
 * back empty, so such a file opened as 0 frames with no error, and saving it then
 * wrote the empty project. Every reader now checks `format_id` first, with the same
 * rule as Python's `_check_format_id`.
 *
 * `py_written_30_columnar.slp` is a real file from Python's release branch (see
 * `tests/data/slp/gen_format_30_fixture.py`). The files with other format ids are
 * real 2.x fixtures with only that attribute changed, so they prove the check goes
 * by `format_id` and not by what the tables happen to look like.
 *
 * The `loadSlp` fallback and the worker cleanup need module mocks, so they are in
 * `slp-newer-format-streaming.test.ts`.
 */
import { describe, it, expect } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { ready, File as H5File } from "h5wasm/node";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { readFromStreamingFile } from "../src/codecs/slp/read-streaming.js";
import type { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import { UnsupportedSlpFormatError } from "../src/index.js";
import { UnsupportedSlpFormatError as BrowserError } from "../src/index.browser.js";
import { loadSlp } from "../src/io/main.js";
import {
  loadSlpMetadata,
  validateSlpBuffer,
  UnsupportedSlpFormatError as LiteError,
} from "../src/lite.js";

const fixtureRoot = fileURLToPath(new URL("./data/slp", import.meta.url));
const fixturePath = (name: string) => path.join(fixtureRoot, name);
const fixtureBytes = (name: string): ArrayBuffer =>
  new Uint8Array(readFileSync(fixturePath(name))).buffer as ArrayBuffer;

const COLUMNAR = "py_written_30_columnar.slp";
/** A real 2.x file, with this many frames, to change `format_id` on. */
const REAL_2X = "py_written_25_identity.slp";
const REAL_2X_FRAMES = 2;

/** A copy of a real fixture with `format_id` changed, or removed when `null`. */
async function withFormatId(
  name: string,
  formatId: number | null,
): Promise<ArrayBuffer> {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    /* already exists */
  }
  const p = `/tmp/newer_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(p, new Uint8Array(fixtureBytes(name)));
  const f = new H5File(p, "a");
  const meta = f.get("metadata") as unknown as {
    delete_attribute: (n: string) => void;
    create_attribute: (n: string, v: number) => void;
  };
  meta.delete_attribute("format_id");
  if (formatId !== null) meta.create_attribute("format_id", formatId);
  f.close();
  const out = module.FS.readFile(p);
  module.FS.unlink(p);
  return new Uint8Array(out).buffer as ArrayBuffer;
}

/**
 * A fake `StreamingH5File` over the real file, wrapping attributes in `{ value }`
 * like the worker. `readSlpStreaming` needs a Worker, which the Node suite doesn't
 * have, so its orchestrator `readFromStreamingFile` is driven with this instead.
 *
 * The pose tables come from the lazy reader when given, because h5wasm/node returns
 * compound rows in a different shape from the worker's columns. Without them a table
 * reads as whatever the file really holds, which for 3.0 is a group with no value:
 * that is how the file used to open as 0 frames.
 */
async function streamingFake(
  bytes: ArrayBuffer,
  columns: Record<string, unknown> = {},
): Promise<{ fake: StreamingH5File; close: () => void }> {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    /* already exists */
  }
  const p = `/tmp/newer_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(p, new Uint8Array(bytes));
  const h5 = new H5File(p, "r");
  const get = (dp: string) => h5.get(dp) as unknown as Record<string, unknown>;
  const fake = {
    keys: () => h5.keys(),
    getKeys: async () => h5.keys(),
    getAttrs: async (dp: string) => {
      const item = get(dp);
      if (!item) throw new Error(`Path not found: ${dp}`);
      const attrs = (item.attrs ?? {}) as Record<string, { value: unknown }>;
      return Object.fromEntries(
        Object.entries(attrs).map(([k, a]) => [k, { value: a.value }]),
      );
    },
    getDatasetMeta: async (dp: string) => {
      const d = get(dp);
      return { shape: (d?.shape as number[]) ?? [], dtype: d?.dtype ?? "" };
    },
    getDatasetValue: async (dp: string) => {
      if (dp in columns) return { value: columns[dp], shape: [], dtype: "" };
      const d = get(dp);
      return {
        value: d?.value,
        shape: (d?.shape as number[]) ?? [],
        dtype: d?.dtype ?? "",
      };
    },
  } as unknown as StreamingH5File;
  return {
    fake,
    close: () => {
      h5.close();
      module.FS.unlink(p);
    },
  };
}

async function readStreaming(
  bytes: ArrayBuffer,
  name: string,
  columns?: Record<string, unknown>,
) {
  const { fake, close } = await streamingFake(bytes, columns);
  try {
    return await readFromStreamingFile(fake, `https://x.test/${name}?a=1`);
  } finally {
    close();
  }
}

/** The pose tables of a 2.x file, in the shape the worker returns them. */
async function workerColumns(bytes: ArrayBuffer) {
  const labels = await readSlpLazy(bytes, { openVideos: false });
  const store = labels._lazyDataStore;
  if (!store) throw new Error("readSlpLazy did not produce a lazy store");
  return {
    frames: store.framesData,
    instances: store.instancesData,
    points: store.pointsData,
    pred_points: store.predPointsData,
  };
}

/** Every reader, as (buffer, file name) -> promise, to run the same checks on. */
const READERS: Array<
  [string, (bytes: ArrayBuffer, name: string) => Promise<unknown>]
> = [
  [
    "readSlp",
    (bytes, name) =>
      readSlp(bytes, { openVideos: false, h5: { filenameHint: name } }),
  ],
  [
    "readSlpLazy",
    (bytes, name) =>
      readSlpLazy(bytes, { openVideos: false, h5: { filenameHint: name } }),
  ],
  ["readFromStreamingFile", (bytes, name) => readStreaming(bytes, name)],
  [
    "loadSlp",
    (bytes, name) =>
      loadSlp(bytes, { openVideos: false, h5: { filenameHint: name } }),
  ],
  [
    "loadSlp lazy",
    (bytes, name) =>
      loadSlp(bytes, {
        openVideos: false,
        lazy: true,
        h5: { filenameHint: name },
      }),
  ],
  [
    "loadSlpMetadata",
    (bytes, name) => loadSlpMetadata(bytes, { filename: name }),
  ],
];

/** What a reader threw, or a failure showing what it returned instead. */
async function refusal(read: () => Promise<unknown>): Promise<Error> {
  let result: unknown;
  try {
    result = await read();
  } catch (e) {
    return e as Error;
  }
  const frames = (result as { labeledFrames?: { length: number } })
    ?.labeledFrames?.length;
  throw new Error(`expected a refusal, but it read (frames: ${frames})`);
}

describe("a file in SLP format 3.0, written by Python, is refused", () => {
  it("the fixture really is the new layout, with poses in it", async () => {
    const module = await ready;
    const { fake, close } = await streamingFake(fixtureBytes(COLUMNAR));
    try {
      expect(
        ((await fake.getAttrs("metadata")).format_id as { value: number })
          .value,
      ).toBe(3);
    } finally {
      close();
    }
    const p = `/tmp/newer_layout_${Date.now()}.slp`;
    module.FS.writeFile(p, new Uint8Array(fixtureBytes(COLUMNAR)));
    const h5 = new H5File(p, "r");
    try {
      const x = h5.get("points/x") as unknown as { value: ArrayLike<number> };
      expect(Array.from(x.value)).toEqual([1, 3]);
      expect(h5.get("frames")?.constructor.name).toBe("Group");
    } finally {
      h5.close();
      module.FS.unlink(p);
    }
  });

  for (const [name, read] of READERS) {
    it(`${name} throws a clear error instead of returning 0 frames`, async () => {
      const e = await refusal(() => read(fixtureBytes(COLUMNAR), "mice.slp"));
      expect(e).toBeInstanceOf(UnsupportedSlpFormatError);
      expect(e.name).toBe("UnsupportedSlpFormatError");
      expect((e as UnsupportedSlpFormatError).formatId).toBe(3);
      expect(e.message).toContain("mice.slp uses SLP format 3.0");
      expect(e.message).toContain("newer than this version of sleap-io.js");
      expect(e.message).toContain("max 2.x");
      expect(e.message).toContain("legacy_format=True");
    });
  }

  it("readSlp given a path names that path", async () => {
    const e = await refusal(() =>
      readSlp(fixturePath(COLUMNAR), { openVideos: false }),
    );
    expect(e).toBeInstanceOf(UnsupportedSlpFormatError);
    expect(e.message).toContain(fixturePath(COLUMNAR));
  });

  it("the streaming reader names the file without the URL's query", async () => {
    const e = await refusal(() =>
      readStreaming(fixtureBytes(COLUMNAR), "mice.slp"),
    );
    expect(e).toBeInstanceOf(UnsupportedSlpFormatError);
    expect(e.message).toContain("mice.slp uses SLP format 3.0");
    expect(e.message).not.toContain("a=1");
  });

  it("the error stops before anything but the metadata is read", async () => {
    const stages: Array<string | undefined> = [];
    const e = await refusal(() =>
      readSlp(fixtureBytes(COLUMNAR), {
        openVideos: false,
        onProgress: (_i, _n, message) => stages.push(message),
      }),
    );
    expect(e).toBeInstanceOf(UnsupportedSlpFormatError);
    expect(stages).toEqual(["Reading metadata"]);
  });

  it("validateSlpBuffer does not call the file valid", () => {
    let caught: unknown;
    try {
      validateSlpBuffer(fixtureBytes(COLUMNAR));
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(UnsupportedSlpFormatError);
    expect((caught as Error).message).toContain(
      "This file uses SLP format 3.0",
    );
  });

  it("the browser and lite entry points export the same error class", () => {
    expect(BrowserError).toBe(UnsupportedSlpFormatError);
    expect(LiteError).toBe(UnsupportedSlpFormatError);
  });
});

describe("the check goes by format_id", () => {
  for (const [name, read] of READERS) {
    it(`${name} refuses a 2.x layout marked 3.0 and later formats`, async () => {
      for (const [id, shown] of [
        [3, "3.0"],
        [3.1, "3.1"],
        [4, "4.0"],
      ] as const) {
        const bytes = await withFormatId(REAL_2X, id);
        const e = await refusal(() => read(bytes, "newer.slp"));
        expect(e).toBeInstanceOf(UnsupportedSlpFormatError);
        expect((e as UnsupportedSlpFormatError).formatId).toBe(id);
        expect(e.message).toContain(`newer.slp uses SLP format ${shown},`);
      }
    });
  }

  it("a format below 3.0 reads in every reader", async () => {
    const bytes = await withFormatId(REAL_2X, 2.99);
    const eager = await readSlp(bytes, { openVideos: false });
    expect(eager.labeledFrames.length).toBe(REAL_2X_FRAMES);
    const lazy = await readSlpLazy(bytes, { openVideos: false });
    expect(lazy.length).toBe(REAL_2X_FRAMES);
    const streamed = await readStreaming(
      bytes,
      "ok.slp",
      await workerColumns(bytes),
    );
    expect(streamed.labeledFrames.length).toBe(REAL_2X_FRAMES);
    expect((await loadSlpMetadata(bytes)).counts.labeledFrames).toBe(
      REAL_2X_FRAMES,
    );
    expect(validateSlpBuffer(bytes)).toBe(true);
  });

  it("a file with no format_id still reads, as it did before", async () => {
    const bytes = await withFormatId(REAL_2X, null);
    const eager = await readSlp(bytes, { openVideos: false });
    expect(eager.labeledFrames.length).toBe(REAL_2X_FRAMES);
    const lazy = await readSlpLazy(bytes, { openVideos: false });
    expect(lazy.length).toBe(REAL_2X_FRAMES);
    const streamed = await readStreaming(
      bytes,
      "ok.slp",
      await workerColumns(bytes),
    );
    expect(streamed.labeledFrames.length).toBe(REAL_2X_FRAMES);
    expect((await loadSlpMetadata(bytes)).counts.labeledFrames).toBe(
      REAL_2X_FRAMES,
    );
  });

  it("the streaming reader takes format_id with or without the { value } wrapper", async () => {
    // The worker wraps attributes in { value }; other callers of this exported
    // function may pass them bare.
    for (const format_id of [{ value: 3 }, 3]) {
      const stub = {
        getAttrs: async () => ({ format_id }),
      } as unknown as StreamingH5File;
      const e = await refusal(() => readFromStreamingFile(stub, "stub.slp"));
      expect(e).toBeInstanceOf(UnsupportedSlpFormatError);
    }
  });

  it("a newer file with no metadata JSON gets this error, not the corruption one", async () => {
    // A later format may not keep the JSON attribute where 2.x has it, so the
    // format check has to come first.
    const stub = {
      getAttrs: async () => ({ format_id: { value: 4 } }),
    } as unknown as StreamingH5File;
    const e = await refusal(() => readFromStreamingFile(stub, "stub.slp"));
    expect(e).toBeInstanceOf(UnsupportedSlpFormatError);
  });

  it("a newer file with other tables gets this error, not 'missing /frames'", async () => {
    const module = await ready;
    try {
      module.FS.mkdir("/tmp");
    } catch {
      /* already exists */
    }
    const p = `/tmp/newer_tables_${Date.now()}.slp`;
    module.FS.writeFile(p, new Uint8Array(0));
    const f = new H5File(p, "w");
    const meta = f.create_group("metadata") as unknown as {
      create_attribute: (n: string, v: unknown) => void;
    };
    meta.create_attribute("format_id", 4.0);
    meta.create_attribute("json", "{}");
    f.close();
    const bytes = new Uint8Array(module.FS.readFile(p)).buffer as ArrayBuffer;
    module.FS.unlink(p);

    const e = await refusal(() => loadSlpMetadata(bytes));
    expect(e).toBeInstanceOf(UnsupportedSlpFormatError);
    let caught: unknown;
    try {
      validateSlpBuffer(bytes);
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(UnsupportedSlpFormatError);
  });
});

describe("the error message", () => {
  it("never shows credentials from a URL", () => {
    const { message } = new UnsupportedSlpFormatError(
      3,
      "https://me:secret@data.test/mice.slp?token=abc123&part=1",
    );
    expect(message).not.toContain("secret");
    expect(message).not.toContain("abc123");
    expect(message).toContain("data.test/mice.slp");
    expect(message).toContain("part=1");
  });

  it("leaves local paths alone, Windows ones included", () => {
    for (const p of ["/data/mice.slp", "C:\\data\\mice.slp", "mice.slp"]) {
      expect(new UnsupportedSlpFormatError(3, p).message).toContain(
        `${p} uses SLP format 3.0`,
      );
    }
  });

  it("says 'This file' when there is no name", () => {
    for (const name of [undefined, ""]) {
      expect(new UnsupportedSlpFormatError(3, name).message).toMatch(
        /^This file uses SLP format 3\.0/,
      );
    }
  });
});

describe("2.x fixtures still read", () => {
  // [fixture, format_id it holds, labeled frames it holds]
  const FIXTURES: Array<[string, number, number]> = [
    ["test_grid_labels.legacy.slp", 1.0, 1],
    ["minimal_instance.slp", 1.1, 1],
    ["minimal_instance.pkg.slp", 1.2, 1],
    ["typical.slp", 1.2, 1],
    ["labels.v002.slp", 1.2, 10],
    ["centered_pair_predictions.slp", 1.2, 1100],
    ["predictions_1.2.7_provenance_and_tracking.slp", 1.2, 101],
    ["cropped_format_2_3.pkg.slp", 2.3, 1],
    ["py_written_25_identity.slp", 2.5, 2],
    ["py_written_27_category.slp", 2.7, 3],
    ["py_written_28_multiview3d.slp", 2.8, 2],
  ];

  for (const [name, formatId, frames] of FIXTURES) {
    it(`${name} (format ${formatId})`, async () => {
      const bytes = fixtureBytes(name);
      const metadata = await loadSlpMetadata(bytes, { filename: name });
      expect(metadata.formatId).toBeCloseTo(formatId, 5);
      expect(metadata.counts.labeledFrames).toBe(frames);
      expect(validateSlpBuffer(bytes)).toBe(true);

      const eager = await readSlp(bytes, { openVideos: false });
      expect(eager.labeledFrames.length).toBe(frames);
      const lazy = await readSlpLazy(bytes, { openVideos: false });
      expect(lazy.length).toBe(frames);
      const streamed = await readStreaming(
        bytes,
        name,
        await workerColumns(bytes),
      );
      expect(streamed.labeledFrames.length).toBe(frames);
    });
  }
});
