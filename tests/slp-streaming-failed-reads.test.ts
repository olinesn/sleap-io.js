/**
 * A table the streaming reader can't read is an error, in any layout.
 *
 * The worker can fail on a table that exists: a dropped connection, a file that went
 * away, a compression filter it refuses. Reading that as a table with no rows opens the
 * project with poses missing, skips `loadSlp`'s retry, and hands the loss to the next
 * save. A table the file doesn't have is a different thing, and still reads as empty.
 */
import { afterEach, describe, expect, it, vi } from "./bun-test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { ready, File as H5File } from "h5wasm/node";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import {
  saveSlpMergedFromStores,
  saveSlpToBytes,
} from "../src/codecs/slp/write.js";
import { Labels } from "../src/index.js";
import { readRecording, tmpPath } from "./worker-file";

const fixture = (name: string): ArrayBuffer =>
  new Uint8Array(
    readFileSync(
      path.join(fileURLToPath(new URL("./data/slp", import.meta.url)), name),
    ),
  ).buffer as ArrayBuffer;

const PYTHON_2X = "py_written_30_twin_legacy.slp";
const PYTHON_30 = "py_written_30_twin.slp";

afterEach(() => {
  vi.restoreAllMocks();
});

const quietWarnings = () =>
  vi.spyOn(console, "warn").mockImplementation(() => {});

const bufferOf = (bytes: Uint8Array): ArrayBuffer =>
  new Uint8Array(bytes).buffer as ArrayBuffer;

/** What the frame tables hold, in a form `toEqual` can compare. */
const poses = (labels: Labels) =>
  Array.from(labels.labeledFrames).map((frame) => ({
    frameIdx: frame.frameIdx,
    isNegative: frame.isNegative,
    instances: frame.instances.map((instance) => ({
      type: instance.constructor.name,
      points: instance.points.map((point) => point.xy),
    })),
  }));

/** Saved by `saveSlpToBytes`: some tables flat, with an attribute naming the columns. */
const saved = async () =>
  bufferOf(
    await saveSlpToBytes(
      await readSlp(fixture(PYTHON_2X), { openVideos: false }),
    ),
  );

/** Saved from a lazy project: every table flat. */
const savedLazy = async () => {
  const labels = await readSlpLazy(fixture(PYTHON_2X), { openVideos: false });
  return bufferOf(await saveSlpMergedFromStores([labels._lazyDataStore!]));
};

const POSE_TABLES = ["frames", "instances", "points", "pred_points"];
const SESSION_TABLES = [
  "session_data/frame_groups",
  "session_data/instance_groups",
  "session_data/instance_group_members",
];

describe("a failed read of a table that exists is an error", () => {
  const READS = ["getDatasetMeta", "getDatasetValue", "getAttrs"];
  const failing =
    (read: string, table: string) => (type: string, requested: string) =>
      type === read && requested === table
        ? new Error(`could not read ${table}`)
        : undefined;

  const LAYOUTS: Array<[string, () => Promise<ArrayBuffer>, string[]]> = [
    [
      "Python's 2.x layout",
      async () => fixture(PYTHON_2X),
      [...POSE_TABLES, "negative_frames", ...SESSION_TABLES],
    ],
    [
      "what saveSlpToBytes writes",
      saved,
      [...POSE_TABLES, "negative_frames", ...SESSION_TABLES],
    ],
    [
      "what a lazy project saves",
      savedLazy,
      [...POSE_TABLES, "negative_frames"],
    ],
    [
      "the 3.0 layout",
      async () => fixture(PYTHON_30),
      [...POSE_TABLES, ...SESSION_TABLES],
    ],
  ];

  for (const [layout, make, tables] of LAYOUTS) {
    for (const table of tables) {
      it(`${layout}: /${table}, whichever request fails`, async () => {
        quietWarnings();
        const bytes = await make();
        const { requests } = await readRecording(bytes);
        const reads = new Set(
          requests
            .filter(([type, p]) => p === table && READS.includes(type))
            .map(([type]) => type),
        );
        expect(reads.has("getDatasetValue")).toBe(true);
        for (const read of reads) {
          for (const lazy of [false, true]) {
            await expect(
              readRecording(bytes, failing(read, table), lazy),
            ).rejects.toThrow(`could not read ${table}`);
          }
        }
      });
    }
  }

  it("a flat table can't be read without the attribute naming its columns", async () => {
    quietWarnings();
    const bytes = await savedLazy();
    for (const table of [...POSE_TABLES, "negative_frames"]) {
      const { requests } = await readRecording(bytes);
      expect(requests).toContainEqual(["getAttrs", table]);
      await expect(
        readRecording(bytes, failing("getAttrs", table)),
      ).rejects.toThrow(`could not read ${table}`);
    }
  });

  it("a compound table names its columns itself, so asks for no attribute", async () => {
    const { requests } = await readRecording(fixture(PYTHON_2X));
    for (const table of [...POSE_TABLES, "negative_frames"]) {
      expect(requests).not.toContainEqual(["getAttrs", table]);
    }
  });

  describe("a real file with a filter the worker refuses", () => {
    const LZF = "handbuilt_30_2x_lzf.slp";
    for (const lazy of [false, true]) {
      it(`gives the reason, not an empty project${lazy ? " (lazy)" : ""}`, async () => {
        await expect(
          readRecording(fixture(LZF), undefined, lazy),
        ).rejects.toThrow(/\/frames is compressed with "lzf".*Re-save/);
      });
    }
    it("the same as on the main thread, where loadSlp would retry it", async () => {
      await expect(
        readSlp(fixture(LZF), { openVideos: false }),
      ).rejects.toThrow(/\/frames is compressed with "lzf"/);
    });
  });
});

describe("a table the file doesn't have still reads as empty", () => {
  /** Three frames: one of a user instance, one of a prediction, one negative. */
  const TABLES: Record<string, [string[], number[][]]> = {
    frames: [
      [
        "frame_id",
        "video",
        "frame_idx",
        "instance_id_start",
        "instance_id_end",
      ],
      [
        [0, 0, 3, 0, 1],
        [1, 0, 4, 1, 2],
        [2, 0, 5, 2, 2],
      ],
    ],
    instances: [
      [
        "instance_id",
        "instance_type",
        "frame_id",
        "skeleton",
        "track",
        "from_predicted",
        "score",
        "point_id_start",
        "point_id_end",
        "tracking_score",
      ],
      [
        [0, 0, 0, 0, -1, -1, 0, 0, 1, 0],
        [1, 1, 1, 0, -1, -1, 0.5, 0, 1, 0],
      ],
    ],
    points: [["x", "y", "visible", "complete"], [[1.5, 2.5, 1, 0]]],
    pred_points: [
      ["x", "y", "visible", "complete", "score"],
      [[3.5, 4.5, 1, 0, 0.75]],
    ],
    negative_frames: [["video_id", "frame_idx"], [[0, 5]]],
  };

  /** A small 2.x file holding these tables, flat. */
  async function small2xFile(
    tables: Record<string, [string[], number[][]]>,
  ): Promise<ArrayBuffer> {
    const module = await ready;
    const p = await tmpPath();
    module.FS.writeFile(p, new Uint8Array(0));
    const f: any = new H5File(p, "w");
    const meta = f.create_group("metadata");
    meta.create_attribute("format_id", 2.8);
    meta.create_attribute(
      "json",
      JSON.stringify({
        version: "2.0.0",
        skeletons: [{ graph: { name: "s" }, links: [], nodes: [{ id: 0 }] }],
        nodes: [{ name: "a" }],
      }),
    );
    f.create_dataset({
      name: "videos_json",
      data: [
        JSON.stringify({ filename: "a.mp4", backend: { filename: "a.mp4" } }),
      ],
    });
    for (const [name, [fields, rows]] of Object.entries(tables)) {
      f.create_dataset({
        name,
        data: new Float64Array(rows.flat()),
        shape: [rows.length, fields.length],
        dtype: "<d",
      });
      f.get(name).create_attribute("field_names", JSON.stringify(fields));
    }
    f.close();
    const out = module.FS.readFile(p);
    module.FS.unlink(p);
    return new Uint8Array(out).buffer as ArrayBuffer;
  }

  it("with all of them, the small file reads as written", async () => {
    const { labels } = await readRecording(await small2xFile(TABLES));
    expect(poses(labels)).toEqual([
      {
        frameIdx: 3,
        isNegative: false,
        instances: [{ type: "Instance", points: [[1.5, 2.5]] }],
      },
      {
        frameIdx: 4,
        isNegative: false,
        instances: [{ type: "PredictedInstance", points: [[3.5, 4.5]] }],
      },
      { frameIdx: 5, isNegative: true, instances: [] },
    ]);
  });

  for (const table of [...POSE_TABLES, "negative_frames"]) {
    it(`/${table}: opens as the eager reader does, asking nothing about it`, async () => {
      const { [table]: _left, ...rest } = TABLES;
      const bytes = await small2xFile(rest);
      const eager = poses(await readSlp(bytes, { openVideos: false }));
      expect(eager.length).toBe(table === "frames" ? 0 : 3);

      const { labels, requests } = await readRecording(bytes);
      expect(requests.filter(([, p]) => p === table)).toEqual([]);
      expect(poses(labels)).toEqual(eager);

      const lazy = await readRecording(bytes, undefined, true);
      expect(lazy.requests.filter(([, p]) => p === table)).toEqual([]);
      lazy.labels.materialize();
      expect(poses(lazy.labels)).toEqual(eager);
    });
  }

  it("a table with no rows isn't a failure either", async () => {
    // `minimal_instance.slp` has no predictions: its compound /pred_points has 0 rows,
    // which a lazy save writes as a flat table with 0 rows.
    const bytes = fixture("minimal_instance.slp");
    const lazily = await readSlpLazy(bytes, { openVideos: false });
    for (const file of [
      bytes,
      bufferOf(await saveSlpMergedFromStores([lazily._lazyDataStore!])),
    ]) {
      const { labels } = await readRecording(file);
      expect(labels.labeledFrames[0].instances.length).toBe(2);
    }
    const empty = await readRecording(
      bufferOf(await saveSlpToBytes(new Labels())),
    );
    expect(empty.labels.labeledFrames.length).toBe(0);
  });
});
