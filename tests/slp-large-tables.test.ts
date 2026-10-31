/**
 * Pose tables too big to hold as JS rows: the full save writes them a chunk at a time,
 * and the browser reader reads a flat table in slices into typed columns. A real
 * 8-camera project has 63 million point rows; one array per row ran the tab out of its
 * ~4 GB JS heap on both paths.
 */
import { describe, it, expect } from "./bun-test";
import h5wasm from "h5wasm/node";
import { rmSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  Instance,
  LabeledFrame,
  Labels,
  PredictedInstance,
  Skeleton,
  Track,
  Video,
  loadSlp,
  saveSlpToBytes,
  buildLabelTableRows,
} from "../src/index.js";
import { readStructDatasetStreaming } from "../src/codecs/slp/read-streaming.js";

const NODES = ["a", "b", "c"];

function project(predictedFrames: number, opts?: { users?: boolean }): Labels {
  const skeleton = new Skeleton({ nodes: NODES });
  const video = new Video({ filename: "cam.mp4" });
  const tracks = [new Track("t0"), new Track("t1")];
  const frames: LabeledFrame[] = [];
  for (let f = 0; f < predictedFrames; f += 1) {
    const instances: Array<Instance | PredictedInstance> = [];
    for (let k = 0; k < 2; k += 1) {
      const pred = PredictedInstance.fromArray(
        NODES.map((_, n) => [f + k + n / 10, f - k + n / 100]),
        skeleton,
        0.5,
      );
      pred.track = tracks[k];
      pred.trackingScore = 0.25;
      instances.push(pred);
    }
    frames.push(new LabeledFrame({ video, frameIdx: f, instances }));
  }
  if (opts?.users) {
    const source = frames[1].instances[1] as PredictedInstance;
    const user = Instance.fromArray(
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      skeleton,
    );
    user.fromPredicted = source;
    user.track = tracks[1];
    frames[1].instances.push(user);
  }
  return new Labels({
    labeledFrames: frames,
    videos: [video],
    skeletons: [skeleton],
    tracks,
  });
}

/** The file's datasets, through h5wasm. */
async function withFile<T>(
  bytes: Uint8Array,
  fn: (file: any) => T,
): Promise<T> {
  await h5wasm.ready;
  const dir = mkdtempSync(join(tmpdir(), "slp-large-"));
  const path = join(dir, "out.slp");
  writeFileSync(path, bytes);
  const file = new h5wasm.File(path, "r");
  try {
    return fn(file);
  } finally {
    file.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

/** A compound table as rows of the given members. */
function rowsOf(file: any, name: string, members: string[]): number[][] {
  const ds = file.get(name);
  const out: number[][] = [];
  for (const row of ds.value as unknown[][]) {
    out.push(row.slice(0, members.length).map(Number));
  }
  return out;
}

describe("the full save writes pose tables a chunk at a time", () => {
  it("writes the same rows as the row builder, with links and tracks", async () => {
    const labels = project(4, { users: true });
    const want = buildLabelTableRows(labels);
    const bytes = await saveSlpToBytes(labels);
    await withFile(bytes, (file) => {
      expect(rowsOf(file, "instances", Array(10).fill(""))).toEqual(
        want.instances,
      );
      expect(rowsOf(file, "pred_points", Array(5).fill(""))).toEqual(
        want.predPoints,
      );
      expect(rowsOf(file, "points", Array(4).fill(""))).toEqual(want.points);
      expect(
        Array.from(file.get("frames").value as ArrayLike<number>).map(Number),
      ).toEqual(want.frames.flat());
    });
    expect(want.instances[4][5]).toBe(3); // the user instance links to its source
    const back = await loadSlp(bytes, { openVideos: false });
    const user = back.labeledFrames[1].instances[2] as Instance;
    expect(user.fromPredicted).toBe(back.labeledFrames[1].instances[1]);
  });

  it("carries a table past its buffer, chunked and uncompressed", async () => {
    const labels = project(2000); // 4,000 instances, 12,000 point rows
    const bytes = await saveSlpToBytes(labels);
    await withFile(bytes, (file) => {
      const ds = file.get("pred_points");
      expect(Number(ds.shape[0])).toBe(12000);
      expect(ds.metadata.chunks[0]).toBe(8192);
      expect(ds.metadata.maxshape[0]).toBeGreaterThan(1e18); // unlimited
      expect(ds.filters.length).toBe(0);
    });
    const back = await loadSlp(bytes, { openVideos: false });
    expect(back.labeledFrames.length).toBe(2000);
    const inst = back.labeledFrames[1999].instances[1] as PredictedInstance;
    expect(inst.track?.name).toBe("t1");
    expect(inst.points[2].xy[0]).toBeCloseTo(1999 + 1 + 0.2, 6);
    expect(inst.score).toBeCloseTo(0.5, 6);
    expect(inst.trackingScore).toBeCloseTo(0.25, 6);
  });

  it("gzips a table of a million rows, and reads it back the same", async () => {
    const labels = project(170_000); // 340,000 instances, 1,020,000 point rows
    const bytes = await saveSlpToBytes(labels);
    await withFile(bytes, (file) => {
      const points = file.get("pred_points");
      expect(Number(points.shape[0])).toBe(1_020_000);
      expect(points.filters.map((f: { name: string }) => f.name)).toContain(
        "deflate",
      );
      expect(file.get("instances").filters.length).toBe(0); // under the threshold
    });
    // Against 1,020,000 × 26 B of points + 340,000 × 45 B of instances, uncompressed.
    expect(bytes.byteLength).toBeLessThan(
      (1_020_000 * 26 + 340_000 * 45) * 0.8,
    );
    const back = await loadSlp(bytes, { openVideos: false });
    const inst = back.labeledFrames[169_999].instances[0] as PredictedInstance;
    expect(inst.points[1].xy[1]).toBeCloseTo(169_999 + 0.01, 6);
  });
});

/** A fake open file holding one flat 2-D table, counting its reads. */
function fakeMatrixFile(
  values: Float32Array | Float64Array | Int32Array | BigInt64Array,
  rowCount: number,
  fields: string[],
) {
  const cols = fields.length;
  const reads: Array<[number, number] | "whole"> = [];
  return {
    reads,
    keys: () => ["table"],
    getDatasetMeta: async () => ({
      shape: [rowCount, cols],
      dtype: "<f4",
      metadata: {},
    }),
    getAttrs: async () => ({ field_names: { value: JSON.stringify(fields) } }),
    getDatasetValue: async (
      _path: string,
      slice?: Array<[number, number] | []>,
    ) => {
      if (!slice) {
        reads.push("whole");
        return { value: values, shape: [rowCount, cols], dtype: "<f4" };
      }
      const [a, b] = slice[0] as [number, number];
      reads.push([a, b]);
      return {
        value: (values as Float64Array).slice(a * cols, b * cols),
        shape: [b - a, cols],
        dtype: "<f4",
      };
    },
  };
}

describe("the browser reader reads a flat table in slices", () => {
  const fields = ["x", "y", "visible", "complete", "score"];
  const rowCount = 300_000; // more than one slice

  it("gives typed columns with every value in place, without a whole read", async () => {
    const values = new Float32Array(rowCount * 5);
    for (let i = 0; i < values.length; i += 1) values[i] = i % 7919;
    const file = fakeMatrixFile(values, rowCount, fields);
    const cols = await readStructDatasetStreaming(file as never, "table");
    expect(file.reads.includes("whole")).toBe(false);
    expect(file.reads.length).toBeGreaterThan(1);
    expect(cols.x).toBeInstanceOf(Float32Array);
    for (const [c, name] of fields.entries()) {
      const col = cols[name] as unknown as Float32Array;
      expect(col.length).toBe(rowCount);
      for (const r of [0, 1, 104856, 104857, rowCount - 1]) {
        expect(col[r]).toBe(values[r * 5 + c]);
      }
    }
  });

  it("keeps integer tables integers, and turns 64-bit ones into numbers", async () => {
    const ints = new Int32Array(40).map((_, i) => i);
    const f1 = fakeMatrixFile(ints, 8, fields);
    const c1 = await readStructDatasetStreaming(f1 as never, "table");
    expect(c1.y).toBeInstanceOf(Int32Array);
    expect((c1.y as unknown as Int32Array)[3]).toBe(16);

    const big = new BigInt64Array(40).map((_, i) => BigInt(i) + 2n ** 40n);
    const f2 = fakeMatrixFile(big, 8, fields);
    const c2 = await readStructDatasetStreaming(f2 as never, "table");
    expect(c2.x).toBeInstanceOf(Float64Array);
    expect((c2.x as unknown as Float64Array)[1]).toBe(5 + 2 ** 40);
  });

  it("reads a table whose field names do not match its columns whole, as before", async () => {
    const values = new Float32Array(12);
    const file = fakeMatrixFile(values, 3, ["only", "three", "names"]);
    const meta = file.getDatasetMeta;
    file.getDatasetMeta = async () => ({ ...(await meta()), shape: [3, 4] });
    await readStructDatasetStreaming(file as never, "table");
    expect(file.reads).toEqual(["whole"]);
  });
});

describe("the browser reader reads a big compound table in slices", () => {
  const members = [
    { name: "x", type: 1, size: 8 },
    { name: "flag", type: 0, size: 1, signed: false },
    { name: "enum", type: 8, size: 1 },
    { name: "small", type: 0, size: 1, signed: true },
    { name: "id", type: 0, size: 8 },
    { name: "score", type: 1, size: 4 },
  ];
  const rowCount = 300_000;

  function fakeCompoundFile(rows = rowCount) {
    const reads: Array<[number, number] | "whole"> = [];
    return {
      reads,
      keys: () => ["table"],
      getDatasetMeta: async () => ({
        shape: [rows],
        dtype: members.map((m) => [m.name, "?"]),
        metadata: { compound_type: { members } },
      }),
      getAttrs: async () => ({}),
      getDatasetValue: async (
        _p: string,
        slice?: Array<[number, number] | []>,
      ) => {
        const [a, b] = (slice?.[0] as [number, number]) ?? [0, rows];
        reads.push(slice ? [a, b] : "whole");
        const col = (f: (r: number) => number) =>
          Float64Array.from({ length: b - a }, (_, i) => f(a + i));
        return {
          value: {
            x: col((r) => r / 2),
            flag: col((r) => r % 2),
            enum: col((r) => r % 2),
            small: col(() => -1),
            id: col((r) => r + 2 ** 40),
            score: col((r) => r % 4),
          },
          shape: [rows],
          dtype: "",
        };
      },
    };
  }

  it("sizes each column to its member, with every value in place", async () => {
    const file = fakeCompoundFile();
    const cols = await readStructDatasetStreaming(file as never, "table");
    expect(file.reads.includes("whole")).toBe(false);
    expect(file.reads.length).toBeGreaterThan(1);
    expect(cols.x).toBeInstanceOf(Float64Array);
    expect(cols.flag).toBeInstanceOf(Uint8Array);
    expect(cols.enum).toBeInstanceOf(Uint8Array);
    expect(cols.small).toBeInstanceOf(Float64Array);
    expect((cols.small as unknown as Float64Array)[7]).toBe(-1);
    expect(cols.id).toBeInstanceOf(Float64Array);
    expect(cols.score).toBeInstanceOf(Float32Array);
    for (const r of [0, 131071, 131072, rowCount - 1]) {
      expect((cols.x as unknown as Float64Array)[r]).toBe(r / 2);
      expect((cols.flag as unknown as Uint8Array)[r]).toBe(r % 2);
      expect((cols.id as unknown as Float64Array)[r]).toBe(r + 2 ** 40);
      expect((cols.score as unknown as Float32Array)[r]).toBe(r % 4);
    }
  });

  it("reads a small compound table whole, as before", async () => {
    const file = fakeCompoundFile(1000);
    const cols = await readStructDatasetStreaming(file as never, "table");
    expect(file.reads).toEqual(["whole"]);
    expect((cols.id as unknown as Float64Array)[999]).toBe(999 + 2 ** 40);
  });
});
