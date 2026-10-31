/**
 * Generate SLP test files and verify Python sleap-io can read them.
 * This test creates temporary .slp files written by JS, then invokes
 * Python sleap-io to read them back, verifying cross-language compatibility.
 */
import { describe, it, expect, setDefaultTimeout } from "./bun-test";
import { saveSlpToBytes } from "../src/codecs/slp/write.js";
import { Labels } from "../src/model/labels.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";
import { Instance } from "../src/model/instance.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { ROI } from "../src/model/roi.js";
import { SegmentationMask } from "../src/model/mask.js";
import { UserBoundingBox } from "../src/model/bbox.js";
import { UserCentroid, PredictedCentroid } from "../src/model/centroid.js";
import { Track } from "../src/model/instance.js";
import { ready, File as H5File } from "h5wasm/node";
import { writeFileSync, unlinkSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

// These tests shell out to Python via `uv run --with sleap-io`. On a cold uv
// cache (e.g. fresh CI) the first invocation downloads sleap-io and its deps
// (numpy/pandas/h5py), which easily exceeds bun's 5s default per-test timeout.
// Give them generous headroom; warm runs still finish in well under a second.
setDefaultTimeout(120_000);

function tmpSlp(): string {
  return join(
    tmpdir(),
    `sleap-io-js-test-${Date.now()}-${Math.random().toString(16).slice(2)}.slp`,
  );
}

// Centroid annotations landed in sleap-io after the current PyPI release
// (0.6.x has no `sleap_io.model.centroid`); they ship in sleap-io main / the
// version sleap-nn bundles. Probe once so the Python-direction centroid tests
// SKIP (rather than fail) until `uv run --with sleap-io` resolves a build that
// has them, at which point they activate automatically.
let _pyHasCentroids: boolean | undefined;
function pythonHasCentroids(): boolean {
  if (_pyHasCentroids === undefined) {
    try {
      execFileSync(
        "uv",
        [
          "run",
          "--with",
          "sleap-io",
          "python",
          "-c",
          "import sleap_io.model.centroid",
        ],
        { timeout: 120000, stdio: "pipe" },
      );
      _pyHasCentroids = true;
    } catch {
      _pyHasCentroids = false;
    }
  }
  return _pyHasCentroids;
}

describe("Python compatibility (#76)", () => {
  it("all string attributes are fixed-length strings (not vlen)", async () => {
    const skeleton = new Skeleton({ name: "fly", nodes: ["head", "thorax"] });
    const video = new Video({ filename: "test.mp4" });
    const track = new Track("track0");
    const inst = new Instance({
      skeleton,
      points: [
        { xy: [100, 200], visible: true, complete: true },
        { xy: [150, 250], visible: true, complete: true },
      ],
    });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const roi = ROI.fromBbox(10, 20, 100, 200, {
      name: "roi1",
      category: "arena",
      source: "manual",
      video,
      track,
    });

    const maskData = new Uint8Array(10 * 10);
    maskData[0] = 1;
    maskData[1] = 1;
    const mask = SegmentationMask.fromArray(maskData, 10, 10, {
      name: "mask1",
      category: "cell",
      source: "model",
    });

    const bbox = new UserBoundingBox({
      x1: 0,
      y1: 20,
      x2: 100,
      y2: 100,
      track,
      category: "animal",
      name: "bb1",
      source: "manual",
    });

    frame.masks.push(mask);
    frame.bboxes.push(bbox);
    const labels = new Labels({
      skeletons: [skeleton],
      videos: [video],
      labeledFrames: [frame],
      tracks: [track],
      rois: [roi],
    });

    const bytes = await saveSlpToBytes(labels);

    // Verify ALL string attributes are fixed-length H5T_STRING (not vlen).
    // h5py reads fixed-length strings as `bytes` (has .decode()), but
    // vlen strings as `str` (no .decode()) — this is what caused #76.
    const module = await ready;
    const memPath = `/tmp/attr_check_${Date.now()}.slp`;
    module.FS.writeFile(memPath, bytes);
    const file = new H5File(memPath, "r");

    function expectFixedString(ds: any, attrName: string) {
      const attr = ds.attrs[attrName];
      expect(
        attr.metadata.type,
        `${ds.path}@${attrName} should be H5T_STRING`,
      ).toBe(3);
      expect(
        attr.metadata.vlen,
        `${ds.path}@${attrName} should not be vlen`,
      ).toBe(false);
    }

    try {
      expectFixedString(file.get("metadata"), "json");
      expectFixedString(file.get("frames"), "field_names");
      expectFixedString(file.get("instances"), "field_names");
      expectFixedString(file.get("points"), "field_names");

      expectFixedString(file.get("rois"), "field_names");
      expectFixedString(file.get("masks"), "field_names");
    } finally {
      file.close();
      module.FS.unlink(memPath);
    }
  });

  it("metadata JSON attribute decodes to valid metadata", async () => {
    const skeleton = new Skeleton({ name: "test", nodes: ["A", "B", "C"] });
    skeleton.addEdge("A", "B");
    const video = new Video({ filename: "video.mp4" });
    const track = new Track("animal1");
    const inst = new Instance({
      skeleton,
      points: [
        { xy: [10, 20], visible: true, complete: true },
        { xy: [30, 40], visible: true, complete: true },
        { xy: [50, 60], visible: false, complete: false },
      ],
      track,
    });
    const frame = new LabeledFrame({ video, frameIdx: 5, instances: [inst] });
    const labels = new Labels({
      skeletons: [skeleton],
      videos: [video],
      labeledFrames: [frame],
      tracks: [track],
    });

    const bytes = await saveSlpToBytes(labels);
    const module = await ready;
    const memPath = `/tmp/metadata_check_${Date.now()}.slp`;
    module.FS.writeFile(memPath, bytes);
    const file = new H5File(memPath, "r");
    try {
      const md = file.get("metadata") as any;
      const jsonAttr = md.attrs["json"];
      // Fixed-length string: h5py reads as bytes, h5wasm reads as string
      expect(jsonAttr.metadata.type).toBe(3); // H5T_STRING
      expect(jsonAttr.metadata.vlen).toBe(false);

      const metadata = JSON.parse(jsonAttr.value);
      expect(metadata.version).toBeDefined();
      expect(metadata.skeletons).toHaveLength(1);
      expect(metadata.skeletons[0].graph.name).toBe("test");
      expect(metadata.skeletons[0].nodes).toHaveLength(3);
      expect(metadata.nodes).toHaveLength(3);
    } finally {
      file.close();
      module.FS.unlink(memPath);
    }
  });

  it("field_names attribute decodes to correct column names", async () => {
    const skeleton = new Skeleton({ name: "s", nodes: ["A"] });
    const video = new Video({ filename: "v.mp4" });
    const inst = new Instance({
      skeleton,
      points: [{ xy: [1, 2], visible: true, complete: true }],
    });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const labels = new Labels({
      skeletons: [skeleton],
      videos: [video],
      labeledFrames: [frame],
    });

    const bytes = await saveSlpToBytes(labels);
    const module = await ready;
    const memPath = `/tmp/fieldnames_check_${Date.now()}.slp`;
    module.FS.writeFile(memPath, bytes);
    const file = new H5File(memPath, "r");
    try {
      const frames = file.get("frames") as any;
      const frameFields = JSON.parse(frames.attrs["field_names"].value);
      expect(frameFields).toEqual([
        "frame_id",
        "video",
        "frame_idx",
        "instance_id_start",
        "instance_id_end",
      ]);

      const instances = file.get("instances") as any;
      const instFields = JSON.parse(instances.attrs["field_names"].value);
      expect(instFields).toContain("instance_id");
      expect(instFields).toContain("skeleton");
      expect(instFields).toContain("track");

      const points = file.get("points") as any;
      const pointFields = JSON.parse(points.attrs["field_names"].value);
      expect(pointFields).toEqual(["x", "y", "visible", "complete"]);
    } finally {
      file.close();
      module.FS.unlink(memPath);
    }
  });

  it("Python can decode metadata and skeletons from JS-written SLP", async () => {
    const skeleton = new Skeleton({
      name: "fly",
      nodes: ["head", "thorax", "abdomen"],
    });
    skeleton.addEdge("head", "thorax");
    skeleton.addEdge("thorax", "abdomen");
    const video = new Video({ filename: "test.mp4" });
    const inst = new Instance({
      skeleton,
      points: [
        { xy: [100, 200], visible: true, complete: true },
        { xy: [150, 250], visible: true, complete: true },
        { xy: [200, 300], visible: true, complete: true },
      ],
    });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    const labels = new Labels({
      skeletons: [skeleton],
      videos: [video],
      labeledFrames: [frame],
    });

    const bytes = await saveSlpToBytes(labels);
    const slpPath = tmpSlp();
    const pyPath = slpPath.replace(/\.slp$/, ".py");

    try {
      writeFileSync(slpPath, bytes);

      // Test metadata reading (#76) and skeleton decoding (format fix). Full
      // sio.load_slp() now works too (JS writes the pose tables as compound
      // datasets with integer id columns — #218; end-to-end coverage lives in
      // skeleton-edgeless-nodes.test.ts); this test stays scoped to the
      // metadata/skeleton readers it was written for.
      //
      // Write the script to a file and invoke via execFileSync (no shell), so
      // the SLP path interpolation and multiline script body don't have to
      // survive shell quoting — JSON.stringify produces a valid Python string
      // literal (same backslash-escape rules) on both POSIX and Windows.
      const pyScript = `
import h5py, json
from sleap_io.io.slp import read_metadata, read_skeletons

slp_path = ${JSON.stringify(slpPath)}

# Test 1: metadata reads without .decode() error (issue #76)
md = read_metadata(slp_path)
assert "version" in md, f"Missing version in metadata"
assert len(md["skeletons"]) == 1, f"Expected 1 skeleton, got {len(md['skeletons'])}"
assert len(md["nodes"]) == 3, f"Expected 3 nodes, got {len(md['nodes'])}"

# Test 2: skeletons decode correctly (format fix)
skeletons = read_skeletons(slp_path)
assert len(skeletons) == 1, f"Expected 1 skeleton, got {len(skeletons)}"
skel = skeletons[0]
assert skel.name == "fly", f"Expected name 'fly', got '{skel.name}'"
assert len(skel.nodes) == 3, f"Expected 3 nodes, got {len(skel.nodes)}"
node_names = [n.name for n in skel.nodes]
assert node_names == ["head", "thorax", "abdomen"], f"Wrong nodes: {node_names}"
assert len(skel.edges) == 2, f"Expected 2 edges, got {len(skel.edges)}"

print("OK: Python reads metadata and skeletons from JS-written SLP")
`;
      writeFileSync(pyPath, pyScript);
      const result = execFileSync(
        "uv",
        ["run", "--with", "sleap-io", "python", pyPath],
        {
          encoding: "utf-8",
          timeout: 30000,
        },
      );
      expect(result).toContain(
        "OK: Python reads metadata and skeletons from JS-written SLP",
      );
    } finally {
      try {
        unlinkSync(slpPath);
      } catch {}
      try {
        unlinkSync(pyPath);
      } catch {}
    }
  });
});

describe("ImageVideo serialization (#221)", () => {
  const IMG_LIST = [
    "frames/img_000.png",
    "frames/img_001.png",
    "frames/img_002.png",
  ];

  function imageSeqLabels(): Labels {
    const skeleton = new Skeleton({ name: "fly", nodes: ["head", "thorax"] });
    const video = new Video({ filename: IMG_LIST });
    const inst = new Instance({
      skeleton,
      points: [
        { xy: [10, 20], visible: true, complete: true },
        { xy: [30, 40], visible: true, complete: true },
      ],
    });
    const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
    return new Labels({
      skeletons: [skeleton],
      videos: [video],
      labeledFrames: [frame],
    });
  }

  it("serializes the frame list under `filenames` (plural) + a scalar `filename`", async () => {
    const bytes = await saveSlpToBytes(imageSeqLabels());
    const module = await ready;
    const memPath = `/tmp/imgseq_${Date.now()}.slp`;
    module.FS.writeFile(memPath, bytes);
    const file = new H5File(memPath, "r");
    try {
      const raw = (file.get("videos_json") as { value: unknown }).value;
      const entry = Array.isArray(raw) ? raw[0] : raw;
      const s =
        typeof entry === "string"
          ? entry
          : new TextDecoder().decode(entry as Uint8Array);
      const meta = JSON.parse(s.replace(/[\s\0]+$/, "")) as {
        backend: { filename: unknown; filenames: unknown };
      };
      // Canonical Python shape: scalar first frame + full list under `filenames`.
      expect(meta.backend.filename).toBe(IMG_LIST[0]);
      expect(meta.backend.filenames).toEqual(IMG_LIST);
    } finally {
      file.close();
      module.FS.unlink(memPath);
    }
  });

  it("round-trips: sleap-io.js reads its own image sequence as the full list", async () => {
    const { readSlp } = await import("../src/codecs/slp/read.js");
    const loaded = await readSlp(await saveSlpToBytes(imageSeqLabels()), {
      openVideos: false,
    });
    expect(loaded.videos[0].filename).toEqual(IMG_LIST);
  });

  it("Python sleap-io reads the JS-written image sequence (would crash pre-#221)", async () => {
    const slpPath = tmpSlp();
    const pyPath = slpPath.replace(/\.slp$/, ".py");
    try {
      writeFileSync(slpPath, await saveSlpToBytes(imageSeqLabels()));
      // Exercise the video-reading path (`read_videos` → `make_video`), which is
      // exactly where #221 crashed: pre-fix the frame list landed in the scalar
      // `filename`, so `make_video` did `Path([...])` → TypeError. Scoped to
      // `read_videos` on purpose — this test is about ImageVideo serialization,
      // not the full instance read path. open_backend=False since the image
      // files don't exist on disk (we only assert the filename metadata).
      const pyScript = `
from pathlib import Path
from sleap_io.io.slp import read_videos
videos = read_videos(${JSON.stringify(slpPath)}, open_backend=False)
fn = videos[0].filename
assert isinstance(fn, list), f"expected a list, got {type(fn).__name__}"
assert len(fn) == 3, f"expected 3 frames, got {len(fn)}"
assert [Path(p).name for p in fn] == ["img_000.png", "img_001.png", "img_002.png"], (
    f"wrong frames/order: {fn}"
)
print("OK: Python reads JS-written image sequence")
`;
      writeFileSync(pyPath, pyScript);
      const result = execFileSync(
        "uv",
        ["run", "--with", "sleap-io", "python", pyPath],
        { encoding: "utf-8", timeout: 120000 },
      );
      expect(result).toContain("OK: Python reads JS-written image sequence");
    } finally {
      try {
        unlinkSync(slpPath);
      } catch {}
      try {
        unlinkSync(pyPath);
      } catch {}
    }
  });
});

describe("Centroid annotation cross-compat (#centroids)", () => {
  // Build a project with a pose instance + a UserCentroid linked to it, and a
  // standalone PredictedCentroid on another frame. Exercises the `/centroids`
  // group layout that Python sleap-io reads (`grp["x"]`, `grp["instance"]`...).
  function centroidLabels(): Labels {
    const skeleton = new Skeleton({
      name: "rodent",
      nodes: ["snout", "neck", "tailbase"],
    });
    const video = new Video({ filename: "fake.mp4" });
    const inst = new Instance({
      skeleton,
      points: [
        { xy: [10, 10], visible: true, complete: true },
        { xy: [12, 12], visible: true, complete: true },
        { xy: [14, 14], visible: true, complete: true },
      ],
    });
    const uc = new UserCentroid({ x: 12, y: 12, instance: inst, name: "cm" });
    const pc = new PredictedCentroid({ x: 99, y: 88, score: 0.77 });
    return new Labels({
      skeletons: [skeleton],
      videos: [video],
      labeledFrames: [
        new LabeledFrame({
          video,
          frameIdx: 0,
          instances: [inst],
          centroids: [uc],
        }),
        new LabeledFrame({
          video,
          frameIdx: 1,
          instances: [],
          centroids: [pc],
        }),
      ],
    });
  }

  it("round-trips: sleap-io.js reads its own centroids", async () => {
    const { readSlp } = await import("../src/codecs/slp/read.js");
    const loaded = await readSlp(await saveSlpToBytes(centroidLabels()), {
      openVideos: false,
    });
    const cents = loaded.labeledFrames.flatMap((lf) => lf.centroids);
    expect(cents.length).toBe(2);
    const user = cents.find((c) => !c.isPredicted)!;
    const pred = cents.find((c) => c.isPredicted)!;
    expect(user.xy).toEqual([12, 12]);
    expect(user.name).toBe("cm");
    expect(user.instance).not.toBeNull(); // linked pose instance survives
    expect((pred as PredictedCentroid).score).toBeCloseTo(0.77, 5);
  });

  it("Python sleap-io reads JS-written centroids (would crash pre-fix)", async () => {
    if (!pythonHasCentroids()) {
      console.warn(
        "[skip] Python sleap-io lacks centroid support (PyPI 0.6.x); activates once released.",
      );
      return;
    }
    const slpPath = tmpSlp();
    const pyPath = slpPath.replace(/\.slp$/, ".py");
    try {
      writeFileSync(slpPath, await saveSlpToBytes(centroidLabels()));
      // Pre-fix, JS wrote `/centroids` as a 2-D matrix dataset, so Python's
      // group-expecting reader raised "Field names only allowed for compound
      // types" and load_slp failed. This asserts the group layout is readable.
      const pyScript = `
import sleap_io as sio
labels = sio.load_slp(${JSON.stringify(slpPath)})
cents = [c for lf in labels.labeled_frames for c in lf.centroids]
assert len(cents) == 2, f"Expected 2 centroids, got {len(cents)}"
user = [c for c in cents if not c.is_predicted]
pred = [c for c in cents if c.is_predicted]
assert len(user) == 1 and len(pred) == 1, f"Wrong user/pred split: {len(user)}/{len(pred)}"
assert tuple(user[0].xy) == (12.0, 12.0), f"Bad user xy: {user[0].xy}"
assert user[0].name == "cm", f"Bad name: {user[0].name!r}"
assert user[0].instance is not None, "User centroid lost its instance link"
assert abs(pred[0].score - 0.77) < 1e-4, f"Bad predicted score: {pred[0].score}"
print("OK: Python reads JS-written centroids")
`;
      writeFileSync(pyPath, pyScript);
      const result = execFileSync(
        "uv",
        ["run", "--with", "sleap-io", "python", pyPath],
        { encoding: "utf-8", timeout: 120000 },
      );
      expect(result).toContain("OK: Python reads JS-written centroids");
    } finally {
      try {
        unlinkSync(slpPath);
      } catch {}
      try {
        unlinkSync(pyPath);
      } catch {}
    }
  });

  it("sleap-io.js reads Python-written centroids", async () => {
    if (!pythonHasCentroids()) {
      console.warn(
        "[skip] Python sleap-io lacks centroid support (PyPI 0.6.x); activates once released.",
      );
      return;
    }
    const { readSlp } = await import("../src/codecs/slp/read.js");
    const slpPath = tmpSlp();
    const pyPath = slpPath.replace(/\.slp$/, ".py");
    try {
      // Python writes the canonical `/centroids` group; JS must read it (pre-fix
      // JS silently returned 0 centroids from a Python file).
      const pyScript = `
import numpy as np
import sleap_io as sio
from sleap_io.model.centroid import UserCentroid, PredictedCentroid

skel = sio.Skeleton(["snout", "neck", "tailbase"], name="rodent")
video = sio.Video.from_filename("fake.mp4")
inst = sio.Instance.from_numpy(np.array([[10, 10], [12, 12], [14, 14]], "float32"), skeleton=skel)
uc = UserCentroid(x=12.0, y=12.0, instance=inst, name="cm")
pc = PredictedCentroid(x=99.0, y=88.0, score=0.77)
lf0 = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst], centroids=[uc])
lf1 = sio.LabeledFrame(video=video, frame_idx=1, centroids=[pc])
labels = sio.Labels(videos=[video], skeletons=[skel], labeled_frames=[lf0, lf1])
sio.save_slp(labels, ${JSON.stringify(slpPath)}, embed=False)
print("OK: Python wrote centroids")
`;
      writeFileSync(pyPath, pyScript);
      const out = execFileSync(
        "uv",
        ["run", "--with", "sleap-io", "python", pyPath],
        { encoding: "utf-8", timeout: 120000 },
      );
      expect(out).toContain("OK: Python wrote centroids");

      const loaded = await readSlp(new Uint8Array(readFileSync(slpPath)), {
        openVideos: false,
      });
      const cents = loaded.labeledFrames.flatMap((lf) => lf.centroids);
      expect(cents.length).toBe(2);
      const user = cents.find((c) => !c.isPredicted)!;
      const pred = cents.find((c) => c.isPredicted)!;
      expect(user.xy).toEqual([12, 12]);
      expect(user.name).toBe("cm");
      expect((pred as PredictedCentroid).score).toBeCloseTo(0.77, 4);
    } finally {
      try {
        unlinkSync(slpPath);
      } catch {}
      try {
        unlinkSync(pyPath);
      } catch {}
    }
  });
});
