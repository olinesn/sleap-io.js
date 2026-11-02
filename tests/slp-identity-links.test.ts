/**
 * Per-detection identity links + embeddings (SLP 2.5, PR-B I3/I4).
 *
 * `Instance.identity` / `identityScore` / `identityEmbedding` persist to
 * `/identity/links` (flat-2D `<d`+field_names, the h5wasm compound stand-in) +
 * `/embeddings` (plain `(N,D)` float + owner columns), joined by owner_id = the
 * global per-modality index. Format bumps to 2.5 when a detection carries either.
 */
import { describe, it, expect } from "./bun-test";
import { readSlp } from "../src/codecs/slp/read.js";
import { saveSlpToBytes } from "../src/codecs/slp/write.js";
import { openH5File } from "../src/codecs/slp/h5.js";
import { Instance } from "../src/model/instance.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";
import { Labels } from "../src/model/labels.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { Identity } from "../src/model/identity.js";
import { Embedding } from "../src/model/embedding.js";
import { readFromStreamingFile } from "../src/codecs/slp/read-streaming.js";
import type { StreamingH5File } from "../src/codecs/slp/h5-streaming.js";
import { ready, File as H5File } from "h5wasm/node";

const SK = new Skeleton({ nodes: ["A", "B"], edges: [] });

/** Back a StreamingH5File with a real h5wasm/node file opened from `bytes`. */
async function makeStreamingFake(
  bytes: Uint8Array,
): Promise<{ fake: StreamingH5File; close: () => void }> {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    /* exists */
  }
  const p = `/tmp/idstream_${Date.now()}_${Math.random().toString(16).slice(2)}.slp`;
  module.FS.writeFile(p, bytes);
  const h5 = new H5File(p, "r");
  const get = (dp: string) => h5.get(dp) as any;
  const fake = {
    keys: () => h5.keys() as string[],
    getKeys: async (dp: string) => {
      const g = get(dp);
      return typeof g?.keys === "function" ? (g.keys() as string[]) : [];
    },
    getAttrs: async (dp: string) => {
      const a = (get(dp)?.attrs as Record<string, any>) ?? {};
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(a)) out[k] = v?.value ?? v;
      return out;
    },
    getDatasetMeta: async (dp: string) => {
      const d = get(dp);
      return {
        shape: (d?.shape as number[]) ?? [],
        dtype: (d?.dtype as string) ?? "",
      };
    },
    getDatasetValue: async (dp: string) => {
      const d = get(dp);
      return {
        value: d?.value,
        shape: (d?.shape as number[]) ?? [],
        dtype: (d?.dtype as string) ?? "",
      };
    },
  } as unknown as StreamingH5File;
  return {
    fake,
    close: () => {
      h5.close();
      try {
        module.FS.unlink(p);
      } catch {
        /* ignore */
      }
    },
  };
}

function labelsWithInstanceIdentities(): {
  labels: Labels;
  identities: Identity[];
} {
  const video = new Video({ filename: "v.mp4" });
  const idA = new Identity({ name: "A" });
  const idB = new Identity({ name: "B" });
  const identities = [idA, idB];
  const frames: LabeledFrame[] = [];
  // 3 frames, alternating identities, one instance each + one 2-instance frame.
  const specs: Array<Array<{ id: Identity; score: number; emb: number[] }>> = [
    [{ id: idA, score: 0.9, emb: [0.1, 0.2, 0.3] }],
    [
      { id: idB, score: 0.8, emb: [0.4, 0.5, 0.6] },
      { id: idA, score: 0.7, emb: [0.7, 0.8, 0.9] },
    ],
    [{ id: idB, score: 0.6, emb: [1.0, 1.1, 1.2] }],
  ];
  specs.forEach((frameSpec, f) => {
    const insts = frameSpec.map((s, k) => {
      const inst = Instance.fromArray(
        [
          [f * 10 + k, 1],
          [f * 10 + k, 2],
        ],
        SK,
      );
      inst.identity = s.id;
      inst.identityScore = s.score;
      inst.identityEmbedding = new Embedding(s.emb);
      return inst;
    });
    frames.push(new LabeledFrame({ video, frameIdx: f, instances: insts }));
  });
  return {
    labels: new Labels({
      labeledFrames: frames,
      videos: [video],
      skeletons: [SK],
      identities,
    }),
    identities,
  };
}

describe("Per-detection identity links + embeddings (SLP 2.5)", () => {
  it("round-trips Instance.identity / identityScore / identityEmbedding", async () => {
    const { labels } = labelsWithInstanceIdentities();
    const bytes = new Uint8Array(await saveSlpToBytes(labels));
    const loaded = await readSlp(bytes.buffer as ArrayBuffer, {
      openVideos: false,
    });

    const all = loaded.labeledFrames.flatMap((f) => f.instances);
    expect(all).toHaveLength(4);
    // Frame 0 inst 0 -> A / 0.9 / [0.1,0.2,0.3]
    expect(all[0].identity?.name).toBe("A");
    expect(all[0].identityScore).toBe(0.9);
    expect(all[0].identityEmbedding?.vector).toEqual([0.1, 0.2, 0.3]);
    // Frame 1 has two instances (global ids 1,2): B/0.8 then A/0.7
    expect(all[1].identity?.name).toBe("B");
    expect(all[1].identityScore).toBe(0.8);
    expect(all[2].identity?.name).toBe("A");
    expect(all[2].identityEmbedding?.vector).toEqual([0.7, 0.8, 0.9]);
    // Frame 2 inst -> B / 0.6
    expect(all[3].identity?.name).toBe("B");
    expect(all[3].identityScore).toBe(0.6);
    // Resolved identities are the catalog objects.
    expect(all[0].identity).toBe(loaded.identities[0]);
    expect(all[1].identity).toBe(loaded.identities[1]);
  });

  it("bumps format_id to 2.5 and writes /identity/links + /embeddings", async () => {
    const { labels } = labelsWithInstanceIdentities();
    const bytes = new Uint8Array(await saveSlpToBytes(labels));
    const { file, close } = await openH5File(bytes.buffer as ArrayBuffer);
    try {
      const attrs = (file.get("metadata") as any).attrs ?? {};
      const formatId = Number(attrs.format_id?.value ?? attrs.format_id);
      expect(formatId).toBeCloseTo(2.5);
      const links = file.get("identity/links") as any;
      expect(links).toBeTruthy();
      expect((links.shape as number[])[0]).toBe(4); // one row per identified inst
      const vectors = file.get("embeddings/vectors") as any;
      expect(vectors).toBeTruthy();
      expect(vectors.shape as number[]).toEqual([4, 3]); // 4 embeddings, dim 3
    } finally {
      close();
    }
  });

  it("reads a Python-written 2.5 file (compound /identity/links + /embeddings)", async () => {
    const { fileURLToPath } = await import("node:url");
    const { readFileSync } = await import("node:fs");
    const nodePath = await import("node:path");
    const fixtureRoot = fileURLToPath(new URL("./data", import.meta.url));
    const p = nodePath.join(fixtureRoot, "slp", "py_written_25_identity.slp");
    const loaded = await readSlp(
      new Uint8Array(readFileSync(p)).buffer as ArrayBuffer,
      { openVideos: false },
    );
    // Catalog came from the Python /identity group (no identities_json).
    expect(loaded.identities.map((i) => i.name)).toEqual(["A", "B"]);
    const all = loaded.labeledFrames.flatMap((f) => f.instances);
    expect(all).toHaveLength(3);
    expect(all[0].identity?.name).toBe("A");
    expect(all[0].identityScore).toBeCloseTo(0.9, 5); // f4 on disk
    expect(all[0].identityEmbedding?.vector[0]).toBeCloseTo(0.1, 5);
    expect(all[1].identity?.name).toBe("B");
    expect(all[1].identityScore).toBeCloseTo(0.8, 5);
    expect(all[2].identity?.name).toBe("A");
    expect(all[2].identityEmbedding?.vector).toHaveLength(3);
    expect(all[2].identityEmbedding?.vector[2]).toBeCloseTo(0.9, 5);
  });

  it("streaming reader attaches per-instance identity + embedding (JS + Python files)", async () => {
    // JS-written file, streamed.
    const { labels } = labelsWithInstanceIdentities();
    const jsBytes = new Uint8Array(await saveSlpToBytes(labels));
    let f = await makeStreamingFake(jsBytes);
    try {
      const loaded = await readFromStreamingFile(
        f.fake,
        "t.slp",
        "t.slp",
        false,
      );
      const all = loaded.labeledFrames.flatMap((fr) => fr.instances);
      expect(all).toHaveLength(4);
      expect(all[0].identity?.name).toBe("A");
      expect(all[0].identityScore).toBe(0.9);
      expect(all[0].identityEmbedding?.vector).toEqual([0.1, 0.2, 0.3]);
      expect(all[1].identity?.name).toBe("B");
    } finally {
      f.close();
    }

    // Python-written file (compound links + float32 embeddings), streamed.
    const { readFileSync } = await import("node:fs");
    const { fileURLToPath } = await import("node:url");
    const nodePath = await import("node:path");
    const fixtureRoot = fileURLToPath(new URL("./data", import.meta.url));
    const pyBytes = new Uint8Array(
      readFileSync(
        nodePath.join(fixtureRoot, "slp", "py_written_25_identity.slp"),
      ),
    );
    f = await makeStreamingFake(pyBytes);
    try {
      const loaded = await readFromStreamingFile(
        f.fake,
        "t.slp",
        "t.slp",
        false,
      );
      expect(loaded.identities.map((i) => i.name)).toEqual(["A", "B"]);
      const all = loaded.labeledFrames.flatMap((fr) => fr.instances);
      expect(all).toHaveLength(3);
      expect(all[0].identity?.name).toBe("A");
      expect(all[0].identityScore).toBeCloseTo(0.9, 5);
      expect(all[2].identity?.name).toBe("A");
    } finally {
      f.close();
    }
  });

  it("a lazy store from the streaming reader saves its links", async () => {
    const { labels } = labelsWithInstanceIdentities();
    const f = await makeStreamingFake(
      new Uint8Array(await saveSlpToBytes(labels)),
    );
    try {
      const lazy = await readFromStreamingFile(
        f.fake,
        "t.slp",
        "t.slp",
        false,
        undefined,
        false,
        true,
      );
      expect(lazy.isLazy).toBe(true);
      const bytes = new Uint8Array(await saveSlpToBytes(lazy));
      const back = await readSlp(bytes.buffer as ArrayBuffer, {
        openVideos: false,
      });
      const all = back.labeledFrames.flatMap((fr) => fr.instances);
      expect(all.map((i) => i.identity?.name)).toEqual(["A", "B", "A", "B"]);
      expect(all[0].identityScore).toBe(0.9);
      expect(all[3].identityEmbedding?.vector).toEqual([1.0, 1.1, 1.2]);
    } finally {
      f.close();
    }
  });

  it("lazy read attaches per-instance identity on materialization", async () => {
    const { readSlpLazy } = await import("../src/codecs/slp/read.js");
    const { labels } = labelsWithInstanceIdentities();
    const bytes = new Uint8Array(await saveSlpToBytes(labels));
    const lazy = await readSlpLazy(bytes.buffer as ArrayBuffer, {
      openVideos: false,
    });
    // Materialize each frame on demand (attaches identity per global instance_id).
    const n = lazy._lazyFrameList?.length ?? 0;
    const all: Instance[] = [];
    for (let i = 0; i < n; i += 1) {
      const fr = lazy.frameAt(i);
      if (fr) for (const inst of fr.instances) all.push(inst as Instance);
    }
    expect(all).toHaveLength(4);
    expect(all[0].identity?.name).toBe("A");
    expect(all[0].identityScore).toBe(0.9);
    expect(all[0].identityEmbedding?.vector).toEqual([0.1, 0.2, 0.3]);
    expect(all[1].identity?.name).toBe("B");
    expect(all[2].identity?.name).toBe("A");
    expect(all[3].identity?.name).toBe("B");
  });

  it("does not bump format_id when identities exist but no detection is linked", async () => {
    const video = new Video({ filename: "v.mp4" });
    const inst = Instance.fromArray(
      [
        [1, 2],
        [3, 4],
      ],
      SK,
    );
    const lf = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const labels = new Labels({
      labeledFrames: [lf],
      videos: [video],
      skeletons: [SK],
      identities: [new Identity({ name: "unused" })],
    });
    const bytes = new Uint8Array(await saveSlpToBytes(labels));
    const { file, close } = await openH5File(bytes.buffer as ArrayBuffer);
    try {
      const attrs = (file.get("metadata") as any).attrs ?? {};
      const formatId = Number(attrs.format_id?.value ?? attrs.format_id);
      expect(formatId).toBeLessThan(2.5);
      expect(file.get("identity/links")).toBeNull();
      expect(file.get("embeddings/vectors")).toBeNull();
    } finally {
      close();
    }
  });
});
