/**
 * A multi-camera group points at its detections by position: frame number in the
 * file, detection number in that frame. An edit moves things, so a save must write
 * each member where it is now, or the group ends up on another animal.
 */
import { describe, it, expect } from "./bun-test";
import { Labels } from "../src/model/labels.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { Instance, PredictedInstance, Track } from "../src/model/instance.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";
import {
  Camera,
  CameraGroup,
  FrameGroup,
  InstanceGroup,
  RecordingSession,
} from "../src/model/camera.js";
import { Identity } from "../src/model/identity.js";
import { Instance3D } from "../src/model/instance3d.js";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import {
  buildExpectedSidecars,
  checkInPlaceWritable,
  saveSlpToBytes,
} from "../src/codecs/slp/write.js";

const N_CAMERAS = 2;
const N_FRAMES = 4;
const N_ANIMALS = 3;
const NODES = ["a", "b", "c", "d"];

/** Each detection's first x names it: camera, frame, animal. */
function firstX(cam: number, frame: number, animal: number): number {
  return 1000 * cam + 100 * frame + 10 * animal;
}

function pointsOf(cam: number, frame: number, animal: number): number[][] {
  const x = firstX(cam, frame, animal);
  return NODES.map((_, n) => [x + n, x + n + 0.5]);
}

/** Save and read back, so every group holds as-read positions like a real file. */
async function makeProject(): Promise<Labels> {
  const skeleton = new Skeleton({ nodes: NODES, edges: [["a", "b"]] });
  const videos = [...Array(N_CAMERAS).keys()].map(
    (c) => new Video({ filename: `cam${c}.mp4` }),
  );
  const cameras = videos.map(
    (_, c) => new Camera({ name: `cam${c}`, rvec: [0, 0, 0], tvec: [c, 0, 0] }),
  );
  const tracks = [...Array(N_ANIMALS).keys()].map(
    (a) => new Track(`track_${a}`),
  );
  const identities = [...Array(N_ANIMALS).keys()].map(
    (a) => new Identity({ name: `id_${a}` }),
  );
  const frames: LabeledFrame[] = [];
  const session = new RecordingSession({
    cameraGroup: new CameraGroup({ cameras }),
  });
  cameras.forEach((cam, c) => {
    session.addVideo(videos[c], cam);
  });
  const byCamFrame = new Map<string, LabeledFrame>();
  for (let c = 0; c < N_CAMERAS; c++) {
    for (let f = 0; f < N_FRAMES; f++) {
      const instances = [...Array(N_ANIMALS).keys()].map((a) => {
        const inst = PredictedInstance.fromArray(pointsOf(c, f, a), skeleton);
        inst.track = tracks[a];
        return inst;
      });
      const lf = new LabeledFrame({
        video: videos[c],
        frameIdx: f,
        instances,
      });
      frames.push(lf);
      byCamFrame.set(`${c}:${f}`, lf);
    }
  }
  for (let f = 0; f < N_FRAMES; f++) {
    const groups = [...Array(N_ANIMALS).keys()].map((a) => {
      const instanceByCamera = new Map<Camera, Instance>();
      cameras.forEach((cam, c) => {
        instanceByCamera.set(
          cam,
          byCamFrame.get(`${c}:${f}`)!.instances[a] as Instance,
        );
      });
      return new InstanceGroup({
        instanceByCamera,
        identity: identities[a],
        instance3d: new Instance3D({
          points: NODES.map((_, n) => [f, a, n]),
          skeleton,
        }),
      });
    });
    const labeledFrameByCamera = new Map<Camera, LabeledFrame>();
    cameras.forEach((cam, c) => {
      labeledFrameByCamera.set(cam, byCamFrame.get(`${c}:${f}`)!);
    });
    session.frameGroupByFrameIdx.set(
      f,
      new FrameGroup({
        frameIdx: f,
        instanceGroups: groups,
        labeledFrameByCamera,
      }),
    );
  }
  const labels = new Labels({
    labeledFrames: frames,
    videos,
    skeletons: [skeleton],
    tracks,
    sessions: [session],
    identities,
  });
  return reopen(labels);
}

async function reopen(labels: Labels): Promise<Labels> {
  const bytes = await saveSlpToBytes(labels);
  return readSlp(bytes.buffer as ArrayBuffer, { openVideos: false });
}

/** Every group member in a saved file: `frame/group/camera` → the detection's first x. */
function membersOf(labels: Labels): Record<string, number> {
  const out: Record<string, number> = {};
  const session = labels.sessions[0];
  for (const [f, fg] of session.frameGroupByFrameIdx) {
    fg.instanceGroups.forEach((group, g) => {
      for (const [cam, inst] of group.instanceByCamera) {
        out[`${f}/${g}/${cam.name}`] = inst._xy[0];
      }
    });
  }
  return out;
}

/** What every member should be, before any edit. */
function untouched(): Record<string, number> {
  const out: Record<string, number> = {};
  for (let f = 0; f < N_FRAMES; f++) {
    for (let a = 0; a < N_ANIMALS; a++) {
      for (let c = 0; c < N_CAMERAS; c++) {
        out[`${f}/${a}/cam${c}`] = firstX(c, f, a);
      }
    }
  }
  return out;
}

function frameOf(labels: Labels, cam: number, f: number): LabeledFrame {
  const video = labels.videos[cam];
  return labels.labeledFrames.find(
    (lf) => lf.video === video && lf.frameIdx === f,
  )!;
}

/** The way an app's undo snapshot copies an instance: same values, new object. */
function copyOf(inst: Instance): Instance {
  const points = inst.points.map((p) => ({
    xy: [p.xy[0], p.xy[1]] as [number, number],
    visible: p.visible,
    complete: p.complete,
    name: p.name,
    score: 0.5,
  }));
  return new PredictedInstance({
    skeleton: inst.skeleton,
    points: points as never,
    track: inst.track,
    score: 0.5,
  });
}

describe("Multi-camera groups after an edit", () => {
  it("an untouched project saves every member where it was", async () => {
    const labels = await makeProject();
    expect(membersOf(await reopen(labels))).toEqual(untouched());
  });

  it("deleting a detection leaves the rest of its frame on the right animals", async () => {
    const labels = await makeProject();
    frameOf(labels, 1, 2).instances.splice(0, 1);
    const expected = untouched();
    delete expected["2/0/cam1"];
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("emptying a frame leaves every later frame on the right animals", async () => {
    const labels = await makeProject();
    const gone = frameOf(labels, 0, 1);
    labels.labeledFrames = labels.labeledFrames.filter((lf) => lf !== gone);
    const expected = untouched();
    for (let a = 0; a < N_ANIMALS; a++) delete expected[`1/${a}/cam0`];
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("a filter that replaces the frame's list keeps the right animals", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 0, 3);
    lf.instances = lf.instances.filter((i) => i._xy[0] !== firstX(0, 3, 1));
    const expected = untouched();
    delete expected["3/1/cam0"];
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("reordering a frame's detections keeps the right animals", async () => {
    const labels = await makeProject();
    frameOf(labels, 1, 0).instances.reverse();
    expect(membersOf(await reopen(labels))).toEqual(untouched());
  });

  it("an added detection does not join any group", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 0, 0);
    lf.instances.unshift(
      Instance.fromArray(
        NODES.map((_, n) => [5 + n, 5]),
        lf.instances[0].skeleton,
      ),
    );
    const back = await reopen(labels);
    expect(membersOf(back)).toEqual(untouched());
    expect(frameOf(back, 0, 0).instances.length).toBe(N_ANIMALS + 1);
  });

  it("an accepted prediction stays in the group, even with its nodes moved", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 0, 2);
    const predicted = lf.instances[1] as PredictedInstance;
    const accepted = Instance.fromArray(
      NODES.map((_, n) => [7000 + n, 7000]),
      predicted.skeleton,
    );
    accepted.fromPredicted = predicted;
    lf.instances.splice(1, 1, accepted);
    const expected = untouched();
    expected["2/1/cam0"] = 7000;
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("undo putting back copies keeps every member", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 1, 1);
    lf.instances = lf.instances.map((i) => copyOf(i as Instance));
    expect(membersOf(await reopen(labels))).toEqual(untouched());
  });

  it("undo of a dragged node, whose copy differs from the object in the group", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 1, 1);
    const copies = lf.instances.map((i) => copyOf(i as Instance));
    // The drag moves one node of the original; undo restores the copies from before it.
    (lf.instances[2] as Instance).points[0].xy = [9999, 9999];
    lf.instances = copies;
    expect(membersOf(await reopen(labels))).toEqual(untouched());
  });

  it("a linked copy keeps the member, however far either has moved", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 1, 1);
    const copies = lf.instances.map((i) => {
      const copy = copyOf(i as Instance);
      copy.copyOf = i as Instance;
      return copy;
    });
    // Every node of the original moves in place, then undo puts the copies back.
    (lf.instances[2] as Instance)._xy.fill(5000);
    lf.instances = copies;
    expect(membersOf(await reopen(labels))).toEqual(untouched());

    // Then every node of the restored copy moves too, then another undo.
    (lf.instances[2] as Instance)._xy.fill(6000);
    const second = copyOf(lf.instances[2] as Instance);
    second.copyOf = (lf.instances[2] as Instance).copyOf;
    lf.instances[2] = second;
    const expected = untouched();
    expected["1/2/cam1"] = 6000;
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("an accepted copy of a copy stays in the group", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 0, 2);
    const original = lf.instances[1] as PredictedInstance;
    const copy = copyOf(original) as PredictedInstance;
    copy.copyOf = original;
    const accepted = Instance.fromArray(
      NODES.map((_, n) => [7000 + n, 7000]),
      original.skeleton,
    );
    accepted.fromPredicted = copy;
    lf.instances.splice(1, 1, accepted);
    const expected = untouched();
    expected["2/1/cam0"] = 7000;
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("an unlinked copy of a member since moved is dropped, not guessed", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 1, 1);
    const copies = lf.instances.map((i) => copyOf(i as Instance));
    (lf.instances[2] as Instance)._xy.fill(5000);
    lf.instances = copies;
    const expected = untouched();
    delete expected["1/2/cam1"];
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("a missing node saved as (0, 0) does not make two detections look alike", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 0, 0);
    const member = lf.instances[0] as Instance;
    const stranger = Instance.fromArray(
      NODES.map(() => [0, 0]),
      member.skeleton,
    );
    stranger._xy[0] = 4321;
    stranger._xy[1] = 4321;
    for (let n = 1; n < NODES.length; n++) {
      member._xy[2 * n] = 0;
      member._xy[2 * n + 1] = 0;
      member._visible[n] = 0;
      stranger._visible[n] = 0;
    }
    lf.instances.push(stranger);
    const saved = await reopen(labels);
    frameOf(saved, 0, 0).instances.splice(0, 1);
    const expected = untouched();
    delete expected["0/0/cam0"];
    expect(membersOf(await reopen(saved))).toEqual(expected);
  });

  it("a save that puts its frames on other videos keeps the members", async () => {
    const labels = await makeProject();
    for (const video of labels.videos) {
      video.sourceVideo = new Video({ filename: `source_${video.filename}` });
    }
    const bytes = await saveSlpToBytes(labels, { embed: "source" });
    const back = await readSlp(bytes.buffer as ArrayBuffer, {
      openVideos: false,
    });
    const members = Object.values(membersOf(back));
    expect(members.length).toBe(N_FRAMES * N_ANIMALS * N_CAMERAS);
    expect(back.videos[0].filename).toBe("source_cam0.mp4");
  });

  it("a member removed from the group's map stays removed", async () => {
    const labels = await makeProject();
    const group =
      labels.sessions[0].frameGroupByFrameIdx.get(1)!.instanceGroups[0];
    const cam = labels.sessions[0].cameras[0];
    group.instanceByCamera.delete(cam);
    const expected = untouched();
    delete expected["1/0/cam0"];
    expect(membersOf(await reopen(labels))).toEqual(expected);
    expect(membersOf(await reopen(labels.copy()))).toEqual(expected);
  });

  it("members swapped in the groups' maps stay swapped, in a copy too", async () => {
    const labels = await makeProject();
    const groups =
      labels.sessions[0].frameGroupByFrameIdx.get(1)!.instanceGroups;
    const cam = labels.sessions[0].cameras[1];
    const a = groups[0].instanceByCamera.get(cam)!;
    const b = groups[1].instanceByCamera.get(cam)!;
    groups[0].instanceByCamera.set(cam, b);
    groups[1].instanceByCamera.set(cam, a);
    const expected = untouched();
    expected["1/0/cam1"] = firstX(1, 1, 1);
    expected["1/1/cam1"] = firstX(1, 1, 0);
    expect(membersOf(await reopen(labels))).toEqual(expected);
    expect(membersOf(await reopen(labels.copy()))).toEqual(expected);
  });

  it("a whole-project undo that rebuilds every frame keeps every member", async () => {
    const labels = await makeProject();
    labels.labeledFrames = labels.labeledFrames.map(
      (lf) =>
        new LabeledFrame({
          video: lf.video,
          frameIdx: lf.frameIdx,
          instances: lf.instances.map((i) => copyOf(i as Instance)),
        }),
    );
    expect(membersOf(await reopen(labels))).toEqual(untouched());
  });

  it("deleting every frame leaves the groups with no members", async () => {
    const labels = await makeProject();
    labels.labeledFrames = [];
    expect(membersOf(await reopen(labels))).toEqual({});
  });

  it("two members never land on one detection", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 0, 0);
    // Member 0 is deleted, and what is left in its frame looks like member 0 but is
    // already member 1's detection.
    const member1 = lf.instances[1] as Instance;
    member1._xy.set((lf.instances[0] as Instance)._xy);
    lf.instances.splice(0, 1);
    const expected = untouched();
    delete expected["0/0/cam0"];
    expected["0/1/cam0"] = firstX(0, 0, 0);
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("a group that loses every member keeps its 3D and identity", async () => {
    const labels = await makeProject();
    frameOf(labels, 0, 0).instances.splice(0, 1);
    frameOf(labels, 1, 0).instances.splice(0, 1);
    const back = await reopen(labels);
    const emptied =
      back.sessions[0].frameGroupByFrameIdx.get(0)!.instanceGroups[0];
    expect(emptied.identity?.name).toBe("id_0");
    expect(emptied.instanceByCamera.size).toBe(0);
    expect(emptied.instance3d?.points?.[1]).toEqual([0, 0, 1]);
  });

  it("the group's getter still names the member after an edit", async () => {
    const labels = await makeProject();
    const lf = frameOf(labels, 1, 2);
    const kept = lf.instances[2];
    lf.instances.splice(0, 1);
    const fg = labels.sessions[0].frameGroupByFrameIdx.get(2)!;
    const group = fg.instanceGroups[2];
    expect(group.instanceByCamera.get(labels.sessions[0].cameras[1])).toBe(
      kept as Instance,
    );
    expect(fg.getFrame(labels.sessions[0].cameras[1])).toBe(lf);
  });

  it("a member read through the getter is written where it is now", async () => {
    const labels = await makeProject();
    const fg = labels.sessions[0].frameGroupByFrameIdx.get(2)!;
    for (const g of fg.instanceGroups) expect(g.instanceByCamera.size).toBe(2);
    expect(fg.labeledFrameByCamera.size).toBe(2);
    frameOf(labels, 1, 2).instances.splice(0, 1);
    const expected = untouched();
    delete expected["2/0/cam1"];
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("a copy made after an edit saves the right animals, and so does the original", async () => {
    const labels = await makeProject();
    const gone = frameOf(labels, 0, 1);
    labels.labeledFrames = labels.labeledFrames.filter((lf) => lf !== gone);
    frameOf(labels, 1, 2).instances.splice(0, 1);
    const expected = untouched();
    for (let a = 0; a < N_ANIMALS; a++) delete expected[`1/${a}/cam0`];
    delete expected["2/0/cam1"];
    const copy = labels.copy();
    expect(membersOf(await reopen(copy))).toEqual(expected);
    expect(membersOf(await reopen(labels))).toEqual(expected);
    // An edit on the copy moves only the copy's members.
    frameOf(copy, 1, 3).instances.splice(0, 1);
    const copyExpected = { ...expected };
    delete copyExpected["3/0/cam1"];
    expect(membersOf(await reopen(copy))).toEqual(copyExpected);
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("a second save after an edit is the same as the first", async () => {
    const labels = await makeProject();
    frameOf(labels, 1, 2).instances.splice(0, 1);
    const expected = untouched();
    delete expected["2/0/cam1"];
    expect(membersOf(await reopen(labels))).toEqual(expected);
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("a lazy project edited after it is materialized saves the right animals", async () => {
    const saved = await saveSlpToBytes(await makeProject());
    const labels = await readSlpLazy(saved.buffer as ArrayBuffer, {
      openVideos: false,
    });
    expect(labels.isLazy).toBe(true);
    labels.materialize();
    const gone = frameOf(labels, 0, 1);
    labels.labeledFrames = labels.labeledFrames.filter((lf) => lf !== gone);
    frameOf(labels, 1, 2).instances.splice(0, 1);
    const expected = untouched();
    for (let a = 0; a < N_ANIMALS; a++) delete expected[`1/${a}/cam0`];
    delete expected["2/0/cam1"];
    expect(membersOf(await reopen(labels))).toEqual(expected);
  });

  it("a lazy project saved unopened keeps every member, without building a frame", async () => {
    const saved = await saveSlpToBytes(await makeProject());
    const labels = await readSlpLazy(saved.buffer as ArrayBuffer, {
      openVideos: false,
    });
    const bytes = await saveSlpToBytes(labels);
    expect(labels.isLazy).toBe(true);
    const back = await readSlp(bytes.buffer as ArrayBuffer, {
      openVideos: false,
    });
    expect(membersOf(back)).toEqual(untouched());
  });

  it("the in-place save refuses a project with groups, whose rows it cannot rewrite", async () => {
    const labels = await makeProject();
    const expected = buildExpectedSidecars(labels);
    const gate = checkInPlaceWritable({}, {}, { onDisk: expected, expected });
    expect(gate.ok).toBe(false);
    expect((gate as { reason: string }).reason).toContain("multi-camera");

    const plain = new Labels({ labeledFrames: labels.labeledFrames.slice() });
    const plainExpected = buildExpectedSidecars(plain);
    expect(
      checkInPlaceWritable(
        {},
        {},
        { onDisk: plainExpected, expected: plainExpected },
      ).ok,
    ).toBe(true);
  });

  it("the in-place save refuses anything else that points at a detection by position", async () => {
    const gate = (labels: Labels) => {
      const expected = buildExpectedSidecars(labels);
      return checkInPlaceWritable({}, {}, { onDisk: expected, expected });
    };
    const fresh = async () =>
      new Labels({ labeledFrames: (await makeProject()).labeledFrames });
    const withCatalog = await fresh();
    withCatalog.identities.push(new Identity({ name: "x" }));
    expect(gate(withCatalog).ok).toBe(true);

    const withIdentity = await fresh();
    withIdentity.labeledFrames[0].instances[0].identity = new Identity({
      name: "x",
    });
    expect((gate(withIdentity) as { reason: string }).reason).toContain(
      "identities",
    );

    const withCategory = await fresh();
    withCategory.labeledFrames[0].instances[0].category = "mouse";
    expect((gate(withCategory) as { reason: string }).reason).toContain(
      "categories",
    );

    const withAnnotation = await fresh();
    const lf = withAnnotation.labeledFrames[0];
    lf.centroids.push({ instance: lf.instances[0] } as never);
    expect((gate(withAnnotation) as { reason: string }).reason).toContain(
      "annotations",
    );
  });
});
