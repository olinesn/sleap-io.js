/**
 * Regression tests for the helpful error raised when an `.slp` file's
 * `/metadata` group exists but is missing its required `json` attribute (the
 * JSON-encoded metadata blob the reader needs to recover skeletons, provenance,
 * etc.). Such files are typically truncated, foreign, or otherwise corrupt.
 *
 * Ports Python sleap-io PR #446 (closes JS #149). Before the fix, the reader
 * silently parsed the missing attribute as `null` and failed later with an
 * opaque error; now it throws a clear, actionable message naming the file and
 * the missing attribute. The eager (`readSlp`), lazy (`readSlpLazy`), and
 * streaming readers all funnel through the shared `parseMetadataJson` helper, so
 * they produce the same error.
 *
 * Fixtures are constructed in-test with h5wasm (mirroring
 * tests/h5-attrs-and-cleanup.test.ts) so no checked-in corrupt file is needed.
 */
import { describe, it, expect } from "./bun-test";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { readFromStreamingFile } from "../src/codecs/slp/read-streaming.js";
import type { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import {
  parseMetadataJson,
  missingMetadataJsonError,
} from "../src/codecs/slp/parsers.js";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const fixtureRoot = fileURLToPath(new URL("./data", import.meta.url));

/** A unique MEMFS path so concurrent fixtures never collide. */
function memPath(): string {
  return (
    "/slp-metadata-error-" +
    Date.now() +
    "-" +
    Math.random().toString(36).slice(2) +
    ".slp"
  );
}

/**
 * Build a minimal in-memory `.slp` buffer. When `includeJson` is false the
 * `metadata` group is created (with an unrelated `format_id` attr) but its
 * `json` attribute is omitted, reproducing the corrupt-file scenario. When a
 * `json` string is provided verbatim (e.g. malformed), it is written as-is.
 */
async function buildFixture(opts: {
  includeJson: boolean;
  rawJson?: string;
}): Promise<ArrayBuffer> {
  const h5 = await import("h5wasm");
  await h5.ready;
  const FS = h5.FS;
  const p = memPath();
  FS.writeFile(p, new Uint8Array(0));
  const f = new h5.File(p, "w");

  const meta = f.create_group("metadata") as {
    create_attribute: (n: string, v: unknown) => void;
  };
  // Group exists but (when includeJson is false) carries only an unrelated attr,
  // mirroring Python's test that sets `format_id` without `json`.
  meta.create_attribute("format_id", 1.1);
  if (opts.rawJson !== undefined) {
    meta.create_attribute("json", opts.rawJson);
  } else if (opts.includeJson) {
    meta.create_attribute("json", JSON.stringify({ skeletons: [], nodes: [] }));
  }

  // The reader touches these datasets after metadata; include empties so that,
  // absent the metadata error, reads would proceed (proving the throw is what
  // stops them) rather than tripping over a missing dataset first.
  f.create_dataset({ name: "tracks_json", data: [], shape: [0], dtype: "S" });
  f.create_dataset({
    name: "suggestions_json",
    data: [],
    shape: [0],
    dtype: "S",
  });
  f.create_dataset({ name: "videos_json", data: [], shape: [0], dtype: "S" });
  f.close();

  const out = FS.readFile(p);
  FS.unlink(p);
  return new Uint8Array(out).buffer;
}

/**
 * Build a minimal in-memory `.slp` buffer with NO `metadata` group at all,
 * reproducing the case Python's
 * `test_read_metadata_missing_metadata_group_raises_valueerror` exercises
 * (`f.require_group("other")` with no `metadata`). The non-metadata datasets
 * are still present so that, absent the metadata error, the reader could
 * otherwise proceed — proving the throw is specifically about the absent group.
 */
async function buildNoMetadataGroupFixture(): Promise<ArrayBuffer> {
  const h5 = await import("h5wasm");
  await h5.ready;
  const FS = h5.FS;
  const p = memPath();
  FS.writeFile(p, new Uint8Array(0));
  const f = new h5.File(p, "w");

  // An unrelated group, but deliberately NO `metadata` group.
  f.create_group("other");
  f.create_dataset({ name: "tracks_json", data: [], shape: [0], dtype: "S" });
  f.create_dataset({
    name: "suggestions_json",
    data: [],
    shape: [0],
    dtype: "S",
  });
  f.create_dataset({ name: "videos_json", data: [], shape: [0], dtype: "S" });
  f.close();

  const out = FS.readFile(p);
  FS.unlink(p);
  return new Uint8Array(out).buffer;
}

/**
 * Copy a real `.slp` fixture into MEMFS, delete only the `metadata/json`
 * attribute, and re-serialize. Mirrors Python's
 * `test_read_labels_missing_json_attr_raises_valueerror`, which corrupts a copy
 * of a working file so every other dataset is genuinely present.
 */
async function corruptRealFixture(filename: string): Promise<ArrayBuffer> {
  const h5 = await import("h5wasm");
  await h5.ready;
  const FS = h5.FS;
  const p = memPath();
  const data = new Uint8Array(
    fs.readFileSync(path.join(fixtureRoot, "slp", filename)),
  );
  FS.writeFile(p, data);
  const f = new h5.File(p, "a");
  const meta = f.get("metadata") as { delete_attribute: (n: string) => void };
  meta.delete_attribute("json");
  f.close();

  const out = FS.readFile(p);
  FS.unlink(p);
  return new Uint8Array(out).buffer;
}

const CORRUPT_MESSAGE = "missing its required metadata JSON blob";

describe("missing /metadata json attribute raises a helpful error", () => {
  it("readSlp (eager) throws naming the file and missing attribute", async () => {
    const buf = await buildFixture({ includeJson: false });
    await expect(
      readSlp(buf, { openVideos: false, h5: { filenameHint: "broken.slp" } }),
    ).rejects.toThrow(CORRUPT_MESSAGE);
    await expect(
      readSlp(buf, { openVideos: false, h5: { filenameHint: "broken.slp" } }),
    ).rejects.toThrow("broken.slp");
  });

  it("readSlpLazy throws the same helpful error", async () => {
    const buf = await buildFixture({ includeJson: false });
    await expect(
      readSlpLazy(buf, {
        openVideos: false,
        h5: { filenameHint: "broken.slp" },
      }),
    ).rejects.toThrow(CORRUPT_MESSAGE);
  });

  it("the message is actionable (mentions corruption and the h5py recovery)", async () => {
    const buf = await buildFixture({ includeJson: false });
    let caught: unknown;
    try {
      await readSlp(buf, { openVideos: false });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    const msg = (caught as Error).message;
    expect(msg).toContain("likely corrupt");
    expect(msg).toContain("metadata");
    expect(msg).toContain("h5py");
  });

  it("a corrupted copy of a real fixture (json attr deleted) throws", async () => {
    // Every other dataset is present, so the throw is specifically about the
    // missing json attribute, not an unrelated missing group.
    const buf = await corruptRealFixture("minimal_instance.slp");
    await expect(
      readSlp(buf, {
        openVideos: false,
        h5: { filenameHint: "minimal_instance.slp" },
      }),
    ).rejects.toThrow(CORRUPT_MESSAGE);
  });

  it("a healthy fixture still loads (no false positive)", async () => {
    const buf = await buildFixture({ includeJson: true });
    const labels = await readSlp(buf, { openVideos: false });
    expect(labels).toBeDefined();
    expect(labels.skeletons.length).toBe(0);
  });
});

describe("a missing /metadata GROUP raises the same helpful error", () => {
  // Mirrors Python's test_read_metadata_missing_metadata_group_raises_valueerror.
  // Python catches the KeyError from BOTH a missing `metadata` group AND a
  // missing `json` attr and maps both to the same ValueError; the JS readers
  // unify these through `missingMetadataJsonError` too.
  it("readSlp (eager) throws the corruption error when the group is absent", async () => {
    const buf = await buildNoMetadataGroupFixture();
    await expect(
      readSlp(buf, { openVideos: false, h5: { filenameHint: "nogroup.slp" } }),
    ).rejects.toThrow(CORRUPT_MESSAGE);
    await expect(
      readSlp(buf, { openVideos: false, h5: { filenameHint: "nogroup.slp" } }),
    ).rejects.toThrow("nogroup.slp");
  });

  it("readSlpLazy throws the same error when the group is absent", async () => {
    const buf = await buildNoMetadataGroupFixture();
    await expect(
      readSlpLazy(buf, {
        openVideos: false,
        h5: { filenameHint: "nogroup.slp" },
      }),
    ).rejects.toThrow(CORRUPT_MESSAGE);
  });
});

describe("the streaming reader funnels through the same helpful error", () => {
  // The full streaming reader (`readSlpStreaming`) is browser/Worker-gated and
  // unreachable from the all-Node suite (see streaming-field-names.test.ts), so
  // we drive `readFromStreamingFile` directly with a minimal `getAttrs`-only
  // stub. The metadata read is the first thing the function does, so no other
  // `file` method is reached before the error is thrown.

  /** A StreamingH5File stub exposing only `getAttrs` (all the reader needs to
   * reach the metadata error). `getAttrs` either returns the serialized attrs
   * or throws, mirroring the real worker's behavior. */
  function stubFile(
    getAttrs: (path: string) => Promise<Record<string, unknown>>,
  ): StreamingH5File {
    return { getAttrs } as unknown as StreamingH5File;
  }

  it("throws when the worker reports a missing /metadata group", async () => {
    // The worker's getAttrs throws "Path not found: metadata" for an absent
    // group; the reader maps that to the helpful corruption error.
    const file = stubFile(async (path) => {
      throw new Error(`Path not found: ${path}`);
    });
    await expect(
      readFromStreamingFile(file, "https://example.com/stream.slp"),
    ).rejects.toThrow(CORRUPT_MESSAGE);
    await expect(
      readFromStreamingFile(file, "https://example.com/stream.slp"),
    ).rejects.toThrow("stream.slp");
  });

  it("doesn't call a file corrupt when reading /metadata failed", async () => {
    const file = stubFile(async () => {
      throw new Error("Could not read 512 bytes at offset 800 of the file.");
    });
    await expect(
      readFromStreamingFile(file, "https://example.com/stream.slp"),
    ).rejects.toThrow("Could not read 512 bytes at offset 800 of the file.");
  });

  it("throws when the /metadata group is present but `json` is missing", async () => {
    // Group present (only an unrelated attr), json attr deleted — the corrupt
    // fixture scenario, but driven through the streaming code path.
    const file = stubFile(async () => ({
      format_id: { value: 1.1 },
    }));
    await expect(
      readFromStreamingFile(
        file,
        "https://example.com/x.slp",
        "corrupt-stream.slp",
      ),
    ).rejects.toThrow(CORRUPT_MESSAGE);
    await expect(
      readFromStreamingFile(
        file,
        "https://example.com/x.slp",
        "corrupt-stream.slp",
      ),
    ).rejects.toThrow("corrupt-stream.slp");
  });

  it("does NOT mask malformed-but-present json as corruption (streaming)", async () => {
    // A present (non-empty) but invalid json attr must fail at the parse step,
    // not be re-reported as missing/corrupt — Python's
    // test_read_metadata_malformed_json_not_remasked, via the streaming path.
    const file = stubFile(async () => ({
      json: { value: "{not valid json" },
    }));
    let caught: unknown;
    try {
      await readFromStreamingFile(file, "https://example.com/x.slp");
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).not.toContain(CORRUPT_MESSAGE);
  });
});

describe("malformed-but-present json is not masked as corruption", () => {
  it("readSlp surfaces a JSON parse error, not the corruption message", async () => {
    // Mirrors Python's test_read_metadata_malformed_json_not_remasked: a present
    // (non-empty) json attribute that is invalid JSON must fail at the parse
    // step, NOT be re-reported as a missing/corrupt attribute.
    const buf = await buildFixture({
      includeJson: false,
      rawJson: "{not valid json",
    });
    let caught: unknown;
    try {
      await readSlp(buf, { openVideos: false });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).not.toContain(CORRUPT_MESSAGE);
  });
});

describe("parseMetadataJson helper (shared by all readers)", () => {
  it("throws the helpful error for an undefined attribute", () => {
    expect(() => parseMetadataJson(undefined, "ghost.slp")).toThrow(
      CORRUPT_MESSAGE,
    );
    expect(() => parseMetadataJson(undefined, "ghost.slp")).toThrow(
      "ghost.slp",
    );
  });

  it("treats an empty string / empty buffer as missing", () => {
    expect(() => parseMetadataJson("", "empty.slp")).toThrow(CORRUPT_MESSAGE);
    expect(() => parseMetadataJson({ value: "" }, "empty.slp")).toThrow(
      CORRUPT_MESSAGE,
    );
    expect(() => parseMetadataJson(new Uint8Array(0), "empty.slp")).toThrow(
      CORRUPT_MESSAGE,
    );
  });

  it("parses present JSON (string and { value } wrapper)", () => {
    expect(parseMetadataJson('{"a":1}', "ok.slp")).toEqual({ a: 1 });
    expect(parseMetadataJson({ value: '{"b":2}' }, "ok.slp")).toEqual({ b: 2 });
  });

  it("lets malformed-but-present JSON throw the underlying parse error", () => {
    expect(() => parseMetadataJson("{bad", "ok.slp")).toThrow();
    expect(() => parseMetadataJson("{bad", "ok.slp")).not.toThrow(
      CORRUPT_MESSAGE,
    );
  });

  it("missingMetadataJsonError embeds the labels path", () => {
    const err = missingMetadataJsonError("/data/foo.slp");
    expect(err).toBeInstanceOf(Error);
    expect(err.message).toContain("/data/foo.slp");
    expect(err.message).toContain("likely corrupt");
  });
});
