/**
 * Lossless, lazy-native recording sessions (issue #197).
 *
 * Verifies the model faithfully round-trips cross-camera grouping via as-read
 * index refs (no rawJson needed), that the lazy reader serializes grouping with
 * ZERO frame materialization, that the written shape matches Python `sleap-io`
 * (calibration keyed `cam_N`, camcorder maps keyed by integer index, no
 * format_id bump), that legacy files re-save to that canonical shape, that a
 * mutation reached via `instanceByCamera` alone still derives correct indices,
 * that `Labels.copy()` preserves the typed session model, and that eager / lazy
 * / (Worker-free) streaming reads all produce the same sessions_json.
 */
import { describe, it, expect } from "./bun-test";
import { readSlp, readSlpLazy } from "../src/codecs/slp/read.js";
import { readSessionsStreaming } from "../src/codecs/slp/read-streaming.js";
import { saveSlpToBytes } from "../src/codecs/slp/write.js";
import { Labels } from "../src/model/labels.js";
import { injectSessionFrameResolver } from "../src/model/camera.js";
import { ready, File as H5File } from "h5wasm/node";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import path from "node:path";

const fixtureRoot = fileURLToPath(new URL("./data", import.meta.url));
const MULTIVIEW = path.join(fixtureRoot, "slp", "multiview.slp");

/** Decode an HDF5 attribute (string / bytes / {value} / array) to a string. */
function attrStr(attrs: any, name: string): string | undefined {
  const a = attrs?.[name];
  const v = a?.value ?? a;
  if (typeof v === "string") return v;
  if (v instanceof Uint8Array) return new TextDecoder().decode(v);
  if (Array.isArray(v)) return String(v[0]);
  return undefined;
}

/** Read a flat-2D + `field_names` struct dataset into column arrays of numbers. */
function readStructCols(
  file: any,
  name: string,
): Record<string, number[]> | null {
  const ds = file.get(name);
  if (!ds) return null;
  const flat = ds.value as ArrayLike<number>;
  const shape = ds.shape as number[];
  const fn = attrStr(ds.attrs, "field_names");
  if (!fn) return null;
  const fields = JSON.parse(fn) as string[];
  const [nrows, ncols] = shape;
  const cols: Record<string, number[]> = {};
  fields.forEach((f, j) => {
    const col = new Array<number>(nrows);
    for (let i = 0; i < nrows; i++) col[i] = Number(flat[i * ncols + j]);
    cols[f] = col;
  });
  return cols;
}

interface ColumnarSessionData {
  frameGroups: Record<string, number[]>;
  instanceGroups: Record<string, number[]>;
  members: Record<string, number[]>;
}

/** Read the raw sessions_json entries, format_id, and (SLP 2.8) the columnar
 * /session_data struct tables via a fresh h5wasm open. */
async function inspect(bytes: Uint8Array): Promise<{
  sessionsJson: Array<Record<string, unknown>>;
  formatId: number;
  sessionData: ColumnarSessionData | null;
}> {
  const module = await ready;
  // Ensure /tmp exists (see inspectSessions in slp-raw-sessions.test.ts): the
  // shared h5wasm MEMFS can transiently lack it under `bun test --parallel`.
  try {
    module.FS.mkdir("/tmp");
  } catch {
    // already exists
  }
  const p = `/tmp/lossless_${Date.now()}_${Math.random()
    .toString(16)
    .slice(2)}.slp`;
  module.FS.writeFile(p, bytes);
  const file = new H5File(p, "r");
  try {
    const keys = file.keys() as string[];
    const sessionsJson = keys.includes("sessions_json")
      ? (file.get("sessions_json") as { value: string[] }).value.map((s) =>
          JSON.parse(s),
        )
      : [];
    const meta = file.get("metadata") as {
      attrs: Record<string, { value?: unknown }>;
    };
    const formatId = Number(
      (meta.attrs.format_id as { value?: unknown })?.value ??
        meta.attrs.format_id,
    );
    const fg = readStructCols(file, "session_data/frame_groups");
    const ig = readStructCols(file, "session_data/instance_groups");
    const mem = readStructCols(file, "session_data/instance_group_members");
    const sessionData =
      fg && ig && mem
        ? { frameGroups: fg, instanceGroups: ig, members: mem }
        : null;
    return { sessionsJson, formatId, sessionData };
  } finally {
    file.close();
    module.FS.unlink(p);
  }
}

/** The verbatim on-disk sessions_json string of the multiview fixture. */
async function fixtureSessionsJson(): Promise<string> {
  const module = await ready;
  try {
    module.FS.mkdir("/tmp");
  } catch {
    // already exists
  }
  const p = `/tmp/mvsrc_${Date.now()}.slp`;
  module.FS.writeFile(p, readFileSync(MULTIVIEW));
  const file = new H5File(p, "r");
  try {
    return (file.get("sessions_json") as { value: string[] }).value[0];
  } finally {
    file.close();
    module.FS.unlink(p);
  }
}

describe("Lossless lazy-native sessions (#197)", () => {
  it("byte-stable canonical grouping round-trip (eager)", async () => {
    const labels = await readSlp(MULTIVIEW, { openVideos: false });
    const firstBytes = new Uint8Array(await saveSlpToBytes(labels));
    const first = await inspect(firstBytes);

    // Re-read the canonical bytes and save again — must be byte-stable.
    const reloaded = await readSlp(firstBytes.buffer as ArrayBuffer, {
      openVideos: false,
    });
    const second = await inspect(
      new Uint8Array(await saveSlpToBytes(reloaded)),
    );

    expect(second.sessionsJson).toEqual(first.sessionsJson);
    // The columnar /session_data tables are byte-stable across the round-trip too.
    expect(second.sessionData).toEqual(first.sessionData);
    // Grouping survived: the first instance group has one member per camera (8).
    const ig = first.sessionData!.instanceGroups;
    expect(ig.member_end[0] - ig.member_start[0]).toBe(8);
  });

  it("eager, lazy, and streaming reads produce identical canonical sessions_json", async () => {
    const eager = await readSlp(MULTIVIEW, { openVideos: false });
    const eagerJson = (
      await inspect(new Uint8Array(await saveSlpToBytes(eager)))
    ).sessionsJson;

    const lazy = await readSlpLazy(MULTIVIEW, { openVideos: false });
    const lazyJson = (await inspect(new Uint8Array(await saveSlpToBytes(lazy))))
      .sessionsJson;
    expect(lazyJson).toEqual(eagerJson);

    // Worker-free streaming: drive readSessionsStreaming with the fixture's raw
    // sessions_json, wrap the parsed sessions in a Labels with the eager frame
    // table, inject the resolver, and serialize.
    const rawSessions = await fixtureSessionsJson();
    const fakeFile = {
      keys: () => ["sessions_json"],
      getDatasetValue: async (name: string) =>
        name === "sessions_json" ? { value: [rawSessions] } : { value: [] },
    } as unknown as Parameters<typeof readSessionsStreaming>[0];
    const streamedSessions = await readSessionsStreaming(
      fakeFile,
      eager.videos,
      eager.skeletons,
      undefined,
      false,
      eager.labeledFrames.map((lf) => lf.frameIdx),
    );
    const streamedLabels = new Labels({
      labeledFrames: eager.labeledFrames,
      videos: eager.videos,
      skeletons: eager.skeletons,
      sessions: streamedSessions,
    });
    injectSessionFrameResolver(streamedLabels);
    const streamedJson = (
      await inspect(new Uint8Array(await saveSlpToBytes(streamedLabels)))
    ).sessionsJson;
    expect(streamedJson).toEqual(eagerJson);
  });

  it("NO frame materialization when serializing lazy sessions", async () => {
    const labels = await readSlpLazy(MULTIVIEW, { openVideos: false });
    expect(labels._lazyFrameList).not.toBeNull();
    expect(labels._lazyFrameList!.materializedCount).toBe(0);

    const bytes = new Uint8Array(await saveSlpToBytes(labels));

    // Grouping never forced a single frame to materialize.
    expect(labels._lazyFrameList!.materializedCount).toBe(0);

    // ...yet the full member grouping (one row per camera) survived to the
    // columnar /session_data tables, built purely from the as-read index refs.
    const { sessionData } = await inspect(bytes);
    expect(sessionData).not.toBeNull();
    const ig = sessionData!.instanceGroups;
    expect(ig.member_end[0] - ig.member_start[0]).toBe(8);
  });

  it("writes Python-canonical cam_N calibration keys + integer camcorder maps", async () => {
    const labels = await readSlp(MULTIVIEW, { openVideos: false });
    const { sessionsJson, formatId, sessionData } = await inspect(
      new Uint8Array(await saveSlpToBytes(labels)),
    );
    // Frame groups present -> the columnar /session_data group is written and the
    // format bumps to 2.8 (SLP 2.8), matching Python's group-presence gate.
    expect(formatId).toBeCloseTo(2.8);

    const s0 = sessionsJson[0];
    const calib = s0.calibration as Record<string, Record<string, unknown>>;
    const calibKeys = Object.keys(calib).filter((k) => k !== "metadata");
    // Calibration keyed `cam_0`..`cam_7` (Python `camera_group_to_dict` parity),
    // each camera dict keeping its `name`.
    expect(
      calibKeys.sort((a, b) => Number(a.slice(4)) - Number(b.slice(4))),
    ).toEqual([
      "cam_0",
      "cam_1",
      "cam_2",
      "cam_3",
      "cam_4",
      "cam_5",
      "cam_6",
      "cam_7",
    ]);
    for (const k of calibKeys) {
      expect(k).toMatch(/^cam_\d+$/);
      expect(typeof calib[k].name).toBe("string");
    }
    // camcorder_to_video_idx_map keys are bare integer indices (Python parity),
    // a DIFFERENT key space than calibration — both resolve by camera order.
    for (const k of Object.keys(
      s0.camcorder_to_video_idx_map as Record<string, unknown>,
    )) {
      expect(k).toMatch(/^\d+$/);
    }
    // The columnarized member map (camera, lf, inst) holds NUMBERS.
    expect(sessionData).not.toBeNull();
    expect(typeof sessionData!.members.camera[0]).toBe("number");
    expect(typeof sessionData!.members.lf[0]).toBe("number");
    expect(typeof sessionData!.members.inst[0]).toBe("number");
  });

  it("legacy file re-saves to the canonical Python shape (converter)", async () => {
    // The fixture is format 1.2 with cam_N-keyed calibration and STRING ref
    // pairs and no labeled_frame_by_camera.
    const src = await inspect(readFileSync(MULTIVIEW));
    expect(src.formatId).toBeCloseTo(1.2);
    const srcCalibKeys = Object.keys(
      src.sessionsJson[0].calibration as Record<string, unknown>,
    ).filter((k) => k !== "metadata");
    expect(srcCalibKeys[0]).toBe("cam_0");
    // Source stores ref pairs as STRINGS (legacy shape).
    const srcPair = Object.values(
      (src.sessionsJson[0].frame_group_dicts as any[])[0].instance_groups[0]
        .camcorder_to_lf_and_inst_idx_map,
    )[0] as unknown[];
    expect(typeof srcPair[0]).toBe("string");

    const labels = await readSlp(MULTIVIEW, { openVideos: false });
    const bytes = new Uint8Array(await saveSlpToBytes(labels));
    const out = await inspect(bytes);

    // Re-save emits the columnar SLP 2.8 layout -> format bumps to 2.8...
    expect(out.formatId).toBeCloseTo(2.8);
    // ...calibration stays cam_N in the slim sessions_json (Python parity)...
    const calibKeys = Object.keys(
      out.sessionsJson[0].calibration as Record<string, unknown>,
    ).filter((k) => k !== "metadata");
    expect(calibKeys.every((k) => /^cam_\d+$/.test(k))).toBe(true);
    // ...and the legacy inline STRING ref pairs are normalized to NUMBERS in the
    // columnar member table.
    expect(out.sessionData).not.toBeNull();
    expect(typeof out.sessionData!.members.lf[0]).toBe("number");
    expect(typeof out.sessionData!.members.inst[0]).toBe("number");

    // Reloading the upgraded file yields the same grouping sizes as the source.
    const reloaded = await readSlp(bytes.buffer as ArrayBuffer, {
      openVideos: false,
    });
    const srcFg = labels.sessions[0].frameGroups.get(0)!;
    const outFg = reloaded.sessions[0].frameGroups.get(0)!;
    expect(outFg.instanceGroups.length).toBe(srcFg.instanceGroups.length);
    expect(outFg.instanceGroups[0].instanceByCamera.size).toBe(
      srcFg.instanceGroups[0].instanceByCamera.size,
    );
    expect(outFg.labeledFrameByCamera.size).toBe(
      srcFg.labeledFrameByCamera.size,
    );
  });

  it("mutation is reflected: editing a group's instance survives the round-trip", async () => {
    const labels = await readSlp(MULTIVIEW, { openVideos: false });
    const fg = labels.sessions[0].frameGroups.get(0)!;
    const ig = fg.instanceGroups[0];

    // Reach the instance through the group's resolved map and mutate a coordinate.
    const [firstCam, firstInst] = [...ig.instanceByCamera][0];
    firstInst.points[0].xy = [12345, 67890];
    const camIdx = labels.sessions[0].cameraGroup.cameras.indexOf(firstCam);

    // Round-trip: the member ref still points at the (now-mutated) 2D instance in
    // /points, so the reloaded session resolves to the mutated coordinate.
    const bytes = new Uint8Array(await saveSlpToBytes(labels));
    const reloaded = await readSlp(bytes.buffer as ArrayBuffer, {
      openVideos: false,
    });
    const rCam = reloaded.sessions[0].cameraGroup.cameras[camIdx];
    const rInst = reloaded.sessions[0].frameGroups
      .get(0)!
      .instanceGroups[0].instanceByCamera.get(rCam)!;
    expect(rInst.points[0].xy[0]).toBe(12345);
    expect(rInst.points[0].xy[1]).toBe(67890);
  });

  it("mutation via instanceByCamera alone (no frame-map access) derives the correct member ref", async () => {
    // Reviewer repro: reach the mutation through ONLY the instance group's map —
    // never touching fg.labeledFrameByCamera — then shift the instance's index.
    // The written member row must reflect the NEW (lf, inst) index (derive-first via
    // the frame-group GETTER), not the stale as-read ref.
    const { Instance } = await import("../src/model/instance.js");
    const labels = await readSlp(MULTIVIEW, { openVideos: false });
    const session = labels.sessions[0];
    const fg = session.frameGroups.get(0)!;
    const ig = fg.instanceGroups[0];

    const [camera, instance] = [...ig.instanceByCamera][0];
    const lf = labels.labeledFrames.find((f) =>
      f.instances.includes(instance),
    )!;
    const nodeCount = instance.skeleton.nodeNames.length;
    const sentinel = Instance.fromArray(
      Array.from({ length: nodeCount }, () => [0, 0]),
      instance.skeleton,
    );
    lf.instances.unshift(sentinel); // shifts `instance` one slot later
    const expectedLfIdx = labels.labeledFrames.indexOf(lf);
    const expectedInstIdx = lf.instances.indexOf(instance);

    const { sessionData } = await inspect(
      new Uint8Array(await saveSlpToBytes(labels)),
    );
    expect(sessionData).not.toBeNull();
    const camIdx = session.cameraGroup.cameras.indexOf(camera);
    // Find this camera's member row in the first instance group (fg 0, ig 0).
    const igCols = sessionData!.instanceGroups;
    const mem = sessionData!.members;
    let found: [number, number] | undefined;
    for (let m = igCols.member_start[0]; m < igCols.member_end[0]; m++) {
      if (mem.camera[m] === camIdx) found = [mem.lf[m], mem.inst[m]];
    }
    expect(found).toEqual([expectedLfIdx, expectedInstIdx]);
  });

  it("Labels.copy() preserves the typed session model and re-saves byte-identically", async () => {
    const labels = await readSlp(MULTIVIEW, { openVideos: false });
    const copy = labels.copy();

    // Prototypes survived: the grouping getters work on the copy.
    const cfg = copy.sessions[0].frameGroups.get(0)!;
    expect(cfg.instanceGroups.length).toBeGreaterThan(0);
    const origSize =
      labels.sessions[0].frameGroups.get(0)!.instanceGroups[0].instanceByCamera
        .size;
    expect(cfg.instanceGroups[0].instanceByCamera.size).toBe(origSize);
    expect(origSize).toBeGreaterThan(0);

    // The copy re-saves to the same sessions_json as the original (ref-backed
    // grouping re-resolved against the copy's own frames).
    const origBytes = new Uint8Array(await saveSlpToBytes(labels));
    const copyBytes = new Uint8Array(await saveSlpToBytes(copy));
    const orig = await inspect(origBytes);
    const out = await inspect(copyBytes);
    expect(out.sessionsJson).toEqual(orig.sessionsJson);
  });

  it("in-memory construction still exposes concrete maps (no refs needed)", async () => {
    const { Camera, CameraGroup, InstanceGroup, FrameGroup } = await import(
      "../src/model/camera.js"
    );
    const { Instance } = await import("../src/model/instance.js");
    const { Skeleton } = await import("../src/model/skeleton.js");
    const { LabeledFrame } = await import("../src/model/labeled-frame.js");
    const { Video } = await import("../src/model/video.js");

    const skeleton = new Skeleton({ nodes: ["A"], edges: [] });
    const video = new Video({ filename: "v.mp4" });
    const cam = new Camera({ name: "cam", rvec: [0, 0, 0], tvec: [0, 0, 0] });
    const inst = Instance.fromArray([[1, 2]], skeleton);
    const lf = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    void CameraGroup;

    const ig = new InstanceGroup({
      instanceByCamera: new Map([[cam, inst]]),
    });
    expect(ig.instanceByCamera.size).toBe(1);
    expect(ig.instances).toHaveLength(1);

    const fg = new FrameGroup({
      frameIdx: 0,
      instanceGroups: [ig],
      labeledFrameByCamera: new Map([[cam, lf]]),
    });
    expect(fg.labeledFrameByCamera.size).toBe(1);
    expect(fg.cameras).toHaveLength(1);
    expect(fg.getFrame(cam)).toBe(lf);
  });
});

describe("Frame numbers of SLEAP's own multi-view groups", () => {
  // multiview.slp is a real SLEAP file: its frame groups have no frame_idx. These
  // numbers are what Python sleap-io reads from it.
  const PYTHON_GROUPS_BY_FRAME: Record<number, number> = { 0: 2, 1: 2, 2: 2 };

  const groupsByFrame = (labels: Labels): Record<number, number> => {
    const out: Record<number, number> = {};
    for (const [f, fg] of labels.sessions[0].frameGroups) {
      expect(fg.frameIdx).toBe(f);
      out[f] = fg.instanceGroups.length;
    }
    return out;
  };

  it("come from the members, as in Python, on the eager and lazy readers", async () => {
    const eager = await readSlp(MULTIVIEW, { openVideos: false });
    expect(groupsByFrame(eager)).toEqual(PYTHON_GROUPS_BY_FRAME);
    const lazy = await readSlpLazy(MULTIVIEW, { openVideos: false });
    expect(groupsByFrame(lazy)).toEqual(PYTHON_GROUPS_BY_FRAME);
  });

  it("come from the members on the streaming reader", async () => {
    const eager = await readSlp(MULTIVIEW, { openVideos: false });
    const rawSessions = await fixtureSessionsJson();
    const fakeFile = {
      keys: () => ["sessions_json"],
      getDatasetValue: async (name: string) =>
        name === "sessions_json" ? { value: [rawSessions] } : { value: [] },
    } as unknown as Parameters<typeof readSessionsStreaming>[0];
    const sessions = await readSessionsStreaming(
      fakeFile,
      eager.videos,
      eager.skeletons,
      undefined,
      false,
      eager.labeledFrames.map((lf) => lf.frameIdx),
    );
    const labels = new Labels({ videos: eager.videos, sessions });
    expect(groupsByFrame(labels)).toEqual(PYTHON_GROUPS_BY_FRAME);
  });

  it("every member of a group is on that group's frame, and a save keeps them", async () => {
    const labels = await readSlp(MULTIVIEW, { openVideos: false });
    const saved = await readSlp(
      (await saveSlpToBytes(labels)).buffer as ArrayBuffer,
      { openVideos: false },
    );
    expect(groupsByFrame(saved)).toEqual(PYTHON_GROUPS_BY_FRAME);
    expect(saved.labeledFrames.length).toBe(labels.labeledFrames.length);
    for (const check of [labels, saved]) {
      for (const [f, fg] of check.sessions[0].frameGroups) {
        for (const [camera, frame] of fg.labeledFrameByCamera) {
          expect(frame.frameIdx).toBe(f);
          expect(frame.video).toBe(check.sessions[0].getVideo(camera));
        }
      }
    }
  });
});
