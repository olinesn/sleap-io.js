/**
 * Anything the streaming reader can't read is an error, in any layout.
 *
 * The worker can fail on something that exists: a dropped connection, a file that went
 * away, a compression filter it refuses. Reading that as a table with no rows, or a file
 * with no videos, opens the project with data missing, skips `loadSlp`'s retry, and hands
 * the loss to the next save. What the file doesn't have is a different thing, and still
 * reads as empty.
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
import { fileURLToPath } from "node:url";
import path from "node:path";
import { ready, File as H5File } from "h5wasm/node";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import {
  saveSlpMergedFromStores,
  saveSlpToBytes,
} from "../src/codecs/slp/write.js";
import { Labels } from "../src/index.js";
import { poses, readRecording, tmpPath } from "./worker-file";

// Failing every request in turn reads a file a hundred times, more slowly under coverage.
setDefaultTimeout(120_000);

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
              readRecording(bytes, failing(read, table), { lazy }),
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
          readRecording(fixture(LZF), undefined, { lazy }),
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

      const lazy = await readRecording(bytes, undefined, { lazy: true });
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

describe("any request the reader makes", () => {
  const WITHOUT_VIDEOS = [
    PYTHON_2X,
    PYTHON_30,
    "py_written_25_identity.slp",
    "py_written_27_category.slp",
    "py_written_28_multiview3d.slp",
    "py_written_annotations.slp",
    "predictions_1.2.7_provenance_and_tracking.slp",
    "js_flat_annotations.slp",
    "multiview.slp",
    "imgvideo.slp",
    "typical.slp",
  ];
  const WITH_VIDEOS = [
    "minimal_instance.pkg.slp",
    "spilled_source_video.pkg.slp",
    "cropped_format_2_3.pkg.slp",
    "py_written_30_embedded.pkg.slp",
    "py_written_30_embedded_legacy.pkg.slp",
  ];
  const cases: Array<
    [string, () => ArrayBuffer | Promise<ArrayBuffer>, object]
  > = [
    ...WITHOUT_VIDEOS.map(
      (file) =>
        [file, () => fixture(file), {}] as [string, () => ArrayBuffer, object],
    ),
    ...WITH_VIDEOS.flatMap((file) =>
      [{}, { openVideos: true }, { lazyVideoMetadata: true }].map(
        (options) =>
          [
            `${file} ${JSON.stringify(options)}`,
            () => fixture(file),
            options,
          ] as [string, () => ArrayBuffer, object],
      ),
    ),
    [
      "saved here, so with identities_json",
      async () =>
        bufferOf(
          await saveSlpToBytes(
            await readSlp(fixture("py_written_25_identity.slp"), {
              openVideos: false,
            }),
          ),
        ),
      {},
    ],
  ];

  for (const [name, bytes, options] of cases) {
    it(`${name}: failing any of them is an error`, async () => {
      quietWarnings();
      const file = await bytes();
      const { requests, refused } = await readRecording(
        file,
        undefined,
        options,
      );
      // Everything asked for is there, so a request that fails can't be taken for one
      // that found nothing.
      expect(refused).toEqual([]);

      const distinct = new Map(requests.map((r) => [r.join(" "), r]));
      expect(distinct.size).toBeGreaterThan(0);
      for (const [request, [type, path]] of distinct) {
        const message = `could not read: ${request}`;
        await expect(
          readRecording(
            file,
            (t, p) =>
              t === type && p === path ? new Error(message) : undefined,
            options,
          ),
        ).rejects.toThrow(message);
      }
    });
  }
});

describe("what a file doesn't have isn't a failed read", () => {
  /** A small file: `build` adds to its root. */
  async function smallFile(
    videos: object[],
    build: (f: any) => void = () => {},
  ): Promise<ArrayBuffer> {
    const module = await ready;
    const p = await tmpPath();
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
      data: videos.map((video) => JSON.stringify(video)),
    });
    build(f);
    f.close();
    const out = module.FS.readFile(p);
    module.FS.unlink(p);
    return new Uint8Array(out).buffer as ArrayBuffer;
  }

  for (const options of [
    {},
    { openVideos: true },
    { lazyVideoMetadata: true },
  ]) {
    const label = JSON.stringify(options);

    it(`a video in another file, whose dataset isn't here ${label}`, async () => {
      const { labels, refused } = await readRecording(
        await smallFile([
          {
            filename: "other.h5",
            backend: { filename: "other.h5", dataset: "box" },
          },
        ]),
        undefined,
        options,
      );
      expect(refused).toEqual([]);
      expect(labels.videos.length).toBe(1);
    });

    it(`an embedded video whose group isn't here ${label}`, async () => {
      const { labels, refused } = await readRecording(
        await smallFile([
          {
            filename: ".",
            backend: { filename: ".", dataset: "video7/video" },
          },
          { filename: ".", backend: { filename: "." } },
        ]),
        undefined,
        options,
      );
      // Opening videos tries to read a frame's shape from each one with a dataset.
      expect(refused.map(([, path]) => path)).toEqual(
        "openVideos" in options ? ["video7/video"] : [],
      );
      expect(labels.videos.length).toBe(2);
    });
  }

  it("a dataset path with a leading slash still finds the dataset", async () => {
    const { labels, refused } = await readRecording(
      await smallFile(
        [
          {
            filename: ".",
            backend: { filename: ".", dataset: "/video0/video" },
          },
        ],
        (f) => {
          const group = f.create_group("video0");
          group.create_dataset({
            name: "video",
            data: new Uint8Array(8),
            shape: [1, 8],
          });
          group.get("video").create_attribute("channel_order", "BGR");
        },
      ),
    );
    expect(refused).toEqual([]);
    expect(labels.videos[0].backendMetadata.channel_order).toBe("BGR");
  });

  it("identities with some of their metadata columns", async () => {
    const bytes = await smallFile([{ filename: "a.mp4", backend: {} }], (f) => {
      const identity = f.create_group("identity");
      identity.create_dataset({ name: "name", data: ["mouse"] });
      identity.create_dataset({
        name: "meta_owner",
        data: new Int32Array([0]),
      });
    });
    const { labels, refused } = await readRecording(bytes);
    expect(refused).toEqual([]);
    const eager = await readSlp(bytes, { openVideos: false });
    expect(labels.identities.map((i) => i.name)).toEqual(["mouse"]);
    expect(eager.identities.map((i) => i.name)).toEqual(["mouse"]);
  });
});
