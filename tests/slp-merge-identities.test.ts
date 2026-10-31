/**
 * Identity and category ride on an instance, not its constructor, so anything that
 * rebuilds instances loses them unless it copies them: a merge bringing back a model's
 * predictions, Labels.copy() and extract().
 */
import { describe, it, expect } from "./bun-test";
import { Labels } from "../src/model/labels.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { Instance, PredictedInstance, Track } from "../src/model/instance.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";
import { Identity } from "../src/model/identity.js";
import { Embedding } from "../src/model/embedding.js";
import {
  Camera,
  CameraGroup,
  FrameGroup,
  InstanceGroup,
  RecordingSession,
} from "../src/model/camera.js";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { openSlpWriter, saveSlpToBytes } from "../src/codecs/slp/write.js";

const skeleton = new Skeleton({ nodes: ["a", "b"], edges: [["a", "b"]] });

function detection(x: number, identity?: Identity): PredictedInstance {
  const inst = PredictedInstance.fromArray(
    [
      [x, x],
      [x + 1, x + 1],
    ],
    skeleton,
  );
  if (identity) {
    inst.identity = identity;
    inst.identityScore = x / 1000;
    inst.identityEmbedding = new Embedding([x, x + 2]);
    inst.category = "mouse";
    inst.categoryScore = x / 2000;
    inst.categoryEmbedding = new Embedding([x + 3]);
  }
  return inst;
}

/** Two predictions per frame, labelled by an identity model. */
function identityModelOutput(names = ["female", "male"], n = 3): Labels {
  const video = new Video({ filename: "v.mp4" });
  const identities = names.map(
    (name) => new Identity({ name, metadata: { strain: `${name}_strain` } }),
  );
  return new Labels({
    labeledFrames: [...Array(n).keys()].map(
      (f) =>
        new LabeledFrame({
          video,
          frameIdx: f,
          instances: [
            detection(10, identities[0]),
            detection(100, identities[1]),
          ],
        }),
    ),
    videos: [video],
    skeletons: [skeleton],
    identities,
  });
}

function project(n = 3): Labels {
  const video = new Video({ filename: "v.mp4" });
  return new Labels({
    labeledFrames: [...Array(n).keys()].map(
      (f) =>
        new LabeledFrame({
          video,
          frameIdx: f,
          instances: [detection(10), detection(100)],
        }),
    ),
    videos: [video],
    skeletons: [skeleton],
  });
}

async function reopen(labels: Labels): Promise<Labels> {
  const bytes = await saveSlpToBytes(labels);
  return readSlp(bytes.buffer as ArrayBuffer, { openVideos: false });
}

function expectLabelled(labels: Labels, f = 0): void {
  const [a, b] = labels.labeledFrames[f].instances;
  expect(a.identity?.name).toBe("female");
  expect(a.identityScore).toBeCloseTo(0.01);
  expect(a.identityEmbedding?.vector).toEqual([10, 12]);
  expect(a.category).toBe("mouse");
  expect(a.categoryScore).toBeCloseTo(0.005);
  expect(a.categoryEmbedding?.vector).toEqual([13]);
  expect(b.identity?.name).toBe("male");
  expect(a.identity).toBe(labels.identities[0]);
  expect(b.identity).toBe(labels.identities[1]);
  expect(labels.identities.map((i) => i.name)).toEqual(["female", "male"]);
  expect(labels.identities[0].metadata.strain).toBe("female_strain");
}

describe("Merging an identity model's predictions", () => {
  for (const frame of ["replace_predictions", "keep_new", "keep_both"]) {
    it(`keeps their identities and categories (${frame})`, async () => {
      const labels = project();
      await labels.merge(identityModelOutput(), {
        video: "basename",
        frame,
      });
      const back = await reopen(labels);
      const offset = frame === "keep_both" ? 2 : 0;
      const [a, b] = back.labeledFrames[1].instances.slice(offset);
      expect(a.identity?.name).toBe("female");
      expect(a.identityScore).toBeCloseTo(0.01);
      expect(a.identityEmbedding?.vector).toEqual([10, 12]);
      expect(a.category).toBe("mouse");
      expect(b.identity?.name).toBe("male");
      expect(back.identities[0].metadata.strain).toBe("female_strain");
    });
  }

  it("keeps them on frames the project did not have", async () => {
    const labels = project(1);
    await labels.merge(identityModelOutput(), {
      video: "basename",
      frame: "replace_predictions",
    });
    expectLabelled(await reopen(labels), 2);
  });

  it("uses one catalog entry per name, keeping the project's", async () => {
    const labels = project();
    const female = new Identity({ name: "female", color: "#123456" });
    labels.identities.push(female);
    await labels.merge(identityModelOutput(), {
      video: "basename",
      frame: "replace_predictions",
    });
    expect(labels.identities.map((i) => i.name)).toEqual(["female", "male"]);
    expect(labels.labeledFrames[0].instances[0].identity).toBe(female);
    const back = await reopen(labels);
    expect(back.identities.map((i) => i.name)).toEqual(["female", "male"]);
    expect(back.identities[0].color).toBe("#123456");
  });

  it("merging the same file twice adds nothing to the catalog", async () => {
    const labels = project();
    const model = identityModelOutput();
    await labels.merge(model, {
      video: "basename",
      frame: "replace_predictions",
    });
    await labels.merge(model, {
      video: "basename",
      frame: "replace_predictions",
    });
    expect(labels.identities).toHaveLength(2);
  });
});

describe("Labels.copy() and extract()", () => {
  it("a copy saves its detections' identities and categories", async () => {
    const copy = identityModelOutput().copy();
    expectLabelled(await reopen(copy));
  });

  it("a copy has its own real Identity objects", () => {
    const labels = identityModelOutput();
    const copy = labels.copy();
    expect(copy.identities[0]).toBeInstanceOf(Identity);
    expect(copy.identities[0]).not.toBe(labels.identities[0]);
    expect(copy.labeledFrames[0].instances[0].identity).toBe(
      copy.identities[0],
    );
    copy.identities[0].metadata.strain = "changed";
    expect(labels.identities[0].metadata.strain).toBe("female_strain");
  });

  it("a copy of a project with groups saves their identities", async () => {
    const labels = identityModelOutput();
    const cam = new Camera({ name: "c", rvec: [0, 0, 0], tvec: [0, 0, 0] });
    const session = new RecordingSession({
      cameraGroup: new CameraGroup({ cameras: [cam] }),
    });
    session.addVideo(labels.videos[0], cam);
    const lf = labels.labeledFrames[0];
    session.frameGroupByFrameIdx.set(
      0,
      new FrameGroup({
        frameIdx: 0,
        instanceGroups: [
          new InstanceGroup({
            instanceByCamera: new Map([[cam, lf.instances[1] as Instance]]),
            identity: labels.identities[1],
          }),
        ],
        labeledFrameByCamera: new Map([[cam, lf]]),
      }),
    );
    labels.sessions.push(session);
    const reread = await reopen(labels);
    const back = await reopen(reread.copy());
    const group =
      back.sessions[0].frameGroupByFrameIdx.get(0)!.instanceGroups[0];
    expect(group.identity?.name).toBe("male");
    expect(group.identity).toBe(back.identities[1]);
  });

  it("a copy keeps an identity on a tracked user instance too", async () => {
    const labels = identityModelOutput();
    const user = Instance.fromArray(
      [
        [1, 1],
        [2, 2],
      ],
      skeleton,
    );
    user.track = new Track("t");
    user.identity = labels.identities[1];
    labels.labeledFrames[0].instances.push(user);
    const back = await reopen(labels.copy());
    expect(back.labeledFrames[0].instances[2].identity?.name).toBe("male");
  });

  it("extract() saves them, copying or not", async () => {
    const labels = identityModelOutput();
    const inds = [[labels.videos[0], 1]] as Array<[Video, number]>;
    for (const copy of [true, false]) {
      const back = await reopen(labels.extract(inds, copy));
      expect(back.labeledFrames).toHaveLength(1);
      expect(back.labeledFrames[0].instances[1].identity?.name).toBe("male");
      expect(back.identities.map((i) => i.name)).toEqual(["female", "male"]);
    }
  });
});

describe("The stream writer", () => {
  const storeOf = async (labels: Labels) => {
    const bytes = await saveSlpToBytes(labels);
    const lazy = await readSlpLazy(bytes.buffer as ArrayBuffer, {
      openVideos: false,
    });
    return { lazy, store: lazy._lazyDataStore! };
  };

  it("keeps a store's identities, categories, scores and embeddings", async () => {
    const { lazy, store } = await storeOf(identityModelOutput());
    const writer = await openSlpWriter({
      skeletons: lazy.skeletons,
      videos: lazy.videos,
      tracks: lazy.tracks,
      identities: lazy.identities,
    });
    writer.appendStore(store);
    const bytes = writer.close();
    expectLabelled(
      await readSlp(new Uint8Array(bytes).buffer, { openVideos: false }),
      2,
    );
  });

  it("puts them on the right rows after an earlier store and a skipped frame", async () => {
    const first = await storeOf(identityModelOutput(["female", "male"], 2));
    const second = await storeOf(identityModelOutput(["female", "male"], 3));
    const writer = await openSlpWriter({
      skeletons: first.lazy.skeletons,
      videos: first.lazy.videos,
      tracks: [],
      identities: first.lazy.identities,
    });
    writer.appendStore(first.store);
    writer.appendStore(second.store);
    const back = await readSlp(new Uint8Array(writer.close()).buffer, {
      openVideos: false,
    });
    expect(back.labeledFrames).toHaveLength(3);
    for (let f = 0; f < 3; f++) expectLabelled(back, f);
  });

  it("follows identityMap, and leaves out what its catalog lacks", async () => {
    const { lazy, store } = await storeOf(identityModelOutput());
    const renamed = new Identity({ name: "renamed" });
    const writer = await openSlpWriter({
      skeletons: lazy.skeletons,
      videos: lazy.videos,
      tracks: lazy.tracks,
      identities: [renamed],
    });
    writer.appendStore(store, {
      identityMap: new Map([[lazy.identities[0], renamed]]),
    });
    const back = await readSlp(new Uint8Array(writer.close()).buffer, {
      openVideos: false,
    });
    const [a, b] = back.labeledFrames[0].instances;
    expect(a.identity?.name).toBe("renamed");
    expect(b.identity ?? null).toBeNull();
  });

  it("raises the format for what it wrote, and writes nothing for a store with none", async () => {
    const { lazy, store } = await storeOf(identityModelOutput());
    const writer = await openSlpWriter({
      skeletons: lazy.skeletons,
      videos: lazy.videos,
      tracks: lazy.tracks,
      identities: lazy.identities,
    });
    writer.appendStore(store);
    const { ready, File: H5File } = await import("h5wasm/node");
    const { writeFileSync, unlinkSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    await ready;
    const path = `${tmpdir()}/stream-labels-${process.pid}.slp`;
    const formatOf = (b: Uint8Array): number => {
      writeFileSync(path, b);
      const f = new H5File(path, "r");
      const id = Number(f.get("metadata").attrs.format_id.value);
      const links = !!f.get("identity/links");
      f.close();
      return links ? id : -id;
    };
    expect(formatOf(writer.close())).toBeGreaterThanOrEqual(2.7);

    const plain = await storeOf(project());
    const plainWriter = await openSlpWriter({
      skeletons: plain.lazy.skeletons,
      videos: plain.lazy.videos,
      tracks: plain.lazy.tracks,
    });
    plainWriter.appendStore(plain.store);
    expect(formatOf(plainWriter.close())).toBeLessThan(0);
    unlinkSync(path);
  });
});

describe("Saving an identity only its detection or group knows about", () => {
  it("keeps a detection's", async () => {
    const labels = project();
    labels.labeledFrames[0].instances[0].identity = new Identity({ name: "x" });
    const back = await reopen(labels);
    expect(back.labeledFrames[0].instances[0].identity?.name).toBe("x");
    expect(back.identities.map((i) => i.name)).toEqual(["x"]);
  });

  it("keeps a group's", async () => {
    const labels = project();
    const cam = new Camera({ name: "c", rvec: [0, 0, 0], tvec: [0, 0, 0] });
    const session = new RecordingSession({
      cameraGroup: new CameraGroup({ cameras: [cam] }),
    });
    session.addVideo(labels.videos[0], cam);
    const lf = labels.labeledFrames[0];
    session.frameGroupByFrameIdx.set(
      0,
      new FrameGroup({
        frameIdx: 0,
        instanceGroups: [
          new InstanceGroup({
            instanceByCamera: new Map([[cam, lf.instances[0] as Instance]]),
            identity: new Identity({ name: "y" }),
          }),
        ],
        labeledFrameByCamera: new Map([[cam, lf]]),
      }),
    );
    labels.sessions.push(session);
    const back = await reopen(labels);
    const group =
      back.sessions[0].frameGroupByFrameIdx.get(0)!.instanceGroups[0];
    expect(group.identity?.name).toBe("y");
  });
});
