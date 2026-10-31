#!/usr/bin/env python
"""Regenerate the ``handbuilt_30*.slp`` fixtures: one project in two SLP layouts.

``handbuilt_30.slp`` is in format 3.0 (every table a group of columns) and
``handbuilt_30_2x.slp`` holds the same project in the 2.x layout, so a test can check
that the two read the same. They are NOT written by Python sleap-io: this script
follows ``_TABLE_SPECS`` and the pose dtypes in ``sleap_io/io/slp.py`` on release/0.10
by hand, using only h5py. That lets the project hold what Python's writer never
produces together, and what a reader must cope with anyway:

- embedded videos whose ``videoN`` numbers are out of list order, with frames in only
  some of them (in 3.0 ``/frames/video`` is the list index; in 2.x it is ``N``);
- unknown columns and per-record ``metadata/`` groups in ``handbuilt_30.slp``.

Python's own reader on release/0.10 opens all of these with the expected contents, and
their tables match ``py_written_30_twin.slp`` (the same project from Python's own writer,
see ``gen_format_30_twins_fixture.py``) in dtype, chunking and compression.

Also written: ``handbuilt_30_gzip.slp`` (columns compressed as with Python's
``compression="gzip"``, and videos that aren't embedded), ``handbuilt_30_lzf.slp``
(the same with ``"lzf"``, which sleap-io.js refuses because its HDF5 build can't
decompress it), ``handbuilt_30_2x_lzf.slp`` (the 2.x file with its pose tables in
``"lzf"``) and
``handbuilt_30_mixed.slp`` (columnar ``/points``, the other tables in the 2.x layout).

Run from this folder:

    uv run --no-project --with h5py --with numpy python gen_format_30_handbuilt_fixture.py
"""

import json
import zlib
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent
STR = h5py.string_dtype()

SKELETON = {
    "directed": True,
    "graph": {"name": "mouse", "num_edges_inserted": 2},
    "links": [
        {
            "edge_insert_idx": 0,
            "key": 0,
            "source": 0,
            "target": 1,
            "type": {"py/reduce": [{"py/type": "sleap.skeleton.EdgeType"}, {"py/tuple": [1]}]},
        },
        {"edge_insert_idx": 1, "key": 0, "source": 1, "target": 2, "type": {"py/id": 1}},
    ],
    "multigraph": True,
    "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
}
METADATA = {
    "version": "2.0.0",
    "skeletons": [SKELETON],
    "nodes": [{"name": "head"}, {"name": "body"}, {"name": "tail"}],
    "videos": [],
    "tracks": [],
    "suggestions": [],
    "negative_anchors": {},
    "provenance": {"sleap_version": "stand-in"},
}

# Embedded videos, deliberately NOT in `videoN` order: list index 0 is video2.
VIDEO_GROUPS = ["video2", "video0", "video1"]
RAW_VIDEO_ID = [2, 0, 1]

NAN = float("nan")

# x, y, visible, verified, labeled
POINTS = [
    (1.5, 2.5, True, True, True),  # U0: visible
    (3.5, 4.5, False, True, True),  # U0: hidden, keeps coordinates (occluded)
    (NAN, NAN, False, False, True),  # U0: absent
    (10.25, 11.25, True, False, True),  # U1
    (12.25, 13.25, True, False, True),  # U1
    (NAN, NAN, False, False, False),  # U1: unlabeled
]
# x, y, visible, verified, labeled, score
PRED_POINTS = [
    (1.0, 2.0, True, False, True, 0.25),  # P0
    (3.0, 4.0, True, False, True, 0.5),  # P0
    (5.0, 6.0, True, False, True, 0.75),  # P0
    (7.0, 8.0, True, False, True, 0.125),  # P1
    (NAN, NAN, False, False, True, NAN),  # P1
    (9.0, 1.0, True, False, True, 1.0),  # P1
]
# instance_type, frame_id, skeleton, track, from_predicted, score, start, end, tracking_score
INSTANCES = [
    (1, 0, 0, 0, -1, 0.1, 0, 3, 0.75),  # P0
    (0, 0, 0, 0, 0, 0.0, 0, 3, NAN),  # U0 from P0
    (1, 1, 0, 1, -1, 0.9, 3, 6, 0.5),  # P1
    (0, 4, 0, -1, -1, 0.0, 3, 6, NAN),  # U1
]
# video (list index), frame_idx, start, end, is_negative
FRAMES = [
    (0, 5, 0, 2, False),
    (1, 7, 2, 3, False),
    (1, 9, 3, 3, True),  # negative
    (0, 11, 3, 3, False),  # empty, not negative
    (0, 13, 3, 4, False),
]


def wkb_point(x, y):
    return b"\x01" + (1).to_bytes(4, "little") + np.array([x, y], dtype="<f8").tobytes()


WKB = [wkb_point(1, 2), wkb_point(3, 4), wkb_point(5, 6)]
WKB_FLAT = np.frombuffer(b"".join(WKB), dtype=np.uint8)
ROI_SPANS = [(0, 21), (21, 42), (42, 63)]
# annotation_type, video, frame_idx, track, instance, is_predicted, score, tracking_score, name, category, source
ROIS = [
    (0, 0, 5, 0, 1, False, NAN, NAN, "roi-a", "cat-a", "src-a"),
    (0, 1, 7, -1, 2, True, 0.5, 0.25, "roi-b", "", "src-b"),
    (0, -1, -1, -1, -1, False, NAN, NAN, "static", "cat-s", ""),  # static ROI
]

RLE = [np.array([1, 2, 3], dtype="<u4"), np.array([4, 2], dtype="<u4")]
RLE_FLAT = np.frombuffer(b"".join(r.tobytes() for r in RLE), dtype=np.uint8)
# height, width, video, frame_idx, track, instance, from_predicted, is_predicted, score,
# tracking_score, rle_start, rle_end, scale_x, scale_y, offset_x, offset_y, name, category, source
MASKS = [
    (2, 3, 0, 5, 0, 0, -1, True, 0.5, 0.75, 0, 12, 1.0, 1.0, 0.0, 0.0, "m-pred", "c", "s"),
    (2, 3, 0, 5, -1, -1, 0, False, NAN, NAN, 12, 20, 2.0, 0.5, 1.5, 2.5, "m-user", "", ""),
]
MASK_SCORE = zlib.compress(np.arange(6, dtype="<f4").tobytes())

LABEL_PIXELS = [
    np.array([[0, 1], [1, 2]], dtype="<i4"),
    np.array([[2, 2], [0, 1]], dtype="<i4"),
]
LI_BLOB = [zlib.compress(p.tobytes()) for p in LABEL_PIXELS]
# video, frame_idx, height, width, n_objects, objects_start, is_predicted, score, source
LABEL_IMAGES = [
    (0, 5, 2, 2, 2, 0, False, NAN, "li-user"),
    (1, 7, 2, 2, 1, 2, True, 0.5, "li-pred"),
]
# label_id, track, instance, score, tracking_score, name, category
LI_OBJECTS = [
    (1, 0, 0, 0.5, 0.25, "o1", "c1"),
    (2, -1, -1, NAN, NAN, "o2", ""),
    (2, 1, 2, 0.75, NAN, "", "c2"),
]
LI_SCORE = zlib.compress(np.arange(4, dtype="<f4").tobytes())

# owner_type, owner_id, catalog index, score
ID_LINKS = [(0, 0, 0, 0.5), (0, 2, 1, NAN), (2, 1, 0, 0.25), (3, 0, 1, 0.75), (4, 0, 0, NAN), (5, 0, 1, 0.5)]
CAT_LINKS = [(0, 1, 0, 0.125), (2, 0, 0, NAN), (3, 1, 0, 0.5)]

# frame_idx, ig_start, ig_end
FRAME_GROUPS = [(5, 0, 2), (7, 2, 3)]
# identity_idx, score, instance_3d_score, pts3d_start, pts3d_end, pts3d_predicted, member_start, member_end
INSTANCE_GROUPS = [
    (0, 0.5, 0.25, 0, 3, False, 0, 2),
    (-1, NAN, NAN, -1, -1, False, 2, 3),
    (1, 0.75, 0.5, 0, 3, True, 3, 4),
]
# camera, lf, inst
MEMBERS = [(0, 0, 1), (1, 1, 0), (0, 0, 0), (1, 1, 0)]
POINTS_3D = np.array([[1.5, 2.5, 3.5], [NAN, NAN, NAN], [7.5, 8.5, 9.5]], dtype="f8")
PRED_POINTS_3D = np.array(
    [[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.25], [NAN, NAN, NAN, NAN]], dtype="f8"
)
FG_META = [json.dumps({"fg": 1}), ""]
IG_META = ["", json.dumps({"ig": "x"}), ""]

CALIBRATION = {
    "cam_0": {
        "name": "camA",
        "size": [640, 480],
        "matrix": [[100.0, 0, 320.0], [0, 100.0, 240.0], [0, 0, 1.0]],
        "distortions": [0, 0, 0, 0, 0],
        "rotation": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
    },
    "cam_1": {
        "name": "camB",
        "size": [640, 480],
        "matrix": [[100.0, 0, 320.0], [0, 100.0, 240.0], [0, 0, 1.0]],
        "distortions": [0, 0, 0, 0, 0],
        "rotation": [0.0, 0.5, 0.0],
        "translation": [10.0, 0.0, 0.0],
    },
    "metadata": {"board": "x"},
}
SESSIONS = [
    json.dumps(
        {
            "calibration": CALIBRATION,
            "camcorder_to_video_idx_map": {"0": 0, "1": 1},
            "name": "s0",
            "fg_start": 0,
            "fg_end": 2,
        }
    )
]


def common(f, *, format_id, embedded=True):
    grp = f.create_group("metadata")
    grp.attrs["format_id"] = format_id
    grp.attrs["json"] = np.bytes_(json.dumps(METADATA, separators=(",", ":")))
    f.create_dataset(
        "provenance_json", data=np.bytes_(json.dumps(METADATA["provenance"]))
    )
    videos = []
    for i, name in enumerate(VIDEO_GROUPS):
        if not embedded:
            clip = f"clip{i}.mp4"
            videos.append(json.dumps({"filename": clip, "backend": {"filename": clip}}))
            continue
        g = f.create_group(name)
        d = g.create_dataset("video", data=np.zeros((2, 4), dtype="i1"))
        d.attrs["format"] = "png"
        d.attrs["channel_order"] = "RGB"
        d.attrs["frames"] = 30
        d.attrs["height"] = 480
        d.attrs["width"] = 640
        d.attrs["channels"] = 1
        g.create_dataset("frame_numbers", data=np.array([5, 7], dtype="i8"))
        videos.append(
            json.dumps(
                {
                    "filename": ".",
                    "backend": {
                        "type": "HDF5Video",
                        "filename": ".",
                        "dataset": f"{name}/video",
                        "input_format": "channels_last",
                        "convert_range": False,
                        "format": "png",
                        "shape": [30, 480, 640, 1],
                    },
                }
            )
        )
    f.create_dataset("videos_json", data=videos, dtype=STR, maxshape=(None,))
    f.create_dataset(
        "tracks_json",
        data=[json.dumps([0, "mouse1"]), json.dumps([0, "mouse2"])],
        dtype=STR,
        maxshape=(None,),
    )
    f.create_dataset(
        "suggestions_json",
        data=[json.dumps({"video": "1", "frame_idx": 3, "group": 0})],
        dtype=STR,
        maxshape=(None,),
    )
    f.create_dataset("sessions_json", data=SESSIONS, dtype=STR, maxshape=(None,))

    ident = f.create_group("identity")
    ident.create_dataset("name", data=["animal_A", "animal_B"], dtype=STR, maxshape=(None,))
    ident.create_dataset(
        "meta_owner", data=np.array([0], "i4"), maxshape=(None,), compression="gzip"
    )
    ident.create_dataset("meta_key", data=["color"], dtype=STR, maxshape=(None,), compression="gzip")
    ident.create_dataset("meta_val", data=["#112233"], dtype=STR, maxshape=(None,), compression="gzip")
    cat = f.create_group("categories")
    cat.create_dataset("name", data=["rodent"], dtype=STR, maxshape=(None,))

    emb = f.create_group("embeddings")
    emb.create_dataset(
        "vectors",
        data=np.array([[0.5, 0.25], [1.0, 2.0]], "f4"),
        maxshape=(None, 2),
        chunks=(2, 2),
        compression="gzip",
    )
    emb.create_dataset("owner_type", data=np.array([0, 2], "u1"), maxshape=(None,), compression="gzip")
    emb.create_dataset("owner_id", data=np.array([1, 0], "i8"), maxshape=(None,), compression="gzip")
    emb.create_dataset(
        "category_vectors",
        data=np.array([[1.5, 2.5, 3.5]], "f4"),
        maxshape=(None, 3),
        chunks=(1, 3),
        compression="gzip",
    )
    emb.create_dataset("category_owner_type", data=np.array([0], "u1"), maxshape=(None,), compression="gzip")
    emb.create_dataset("category_owner_id", data=np.array([3], "i8"), maxshape=(None,), compression="gzip")

    c = f.create_group("centroids")
    c.create_dataset("x", data=np.array([1.5, 2.5], "f8"))
    c.create_dataset("y", data=np.array([3.5, 4.5], "f8"))
    c.create_dataset("z", data=np.array([NAN, 5.5], "f8"))
    c.create_dataset("video", data=np.array([0, 1], "i4"))
    c.create_dataset("frame_idx", data=np.array([5, 7], "i8"))
    c.create_dataset("track", data=np.array([0, -1], "i4"))
    c.create_dataset("instance", data=np.array([1, -1], "i4"))
    c.create_dataset("is_predicted", data=np.array([0, 1], "u1"))
    c.create_dataset("score", data=np.array([NAN, 0.5], "f4"))
    c.create_dataset("tracking_score", data=np.array([NAN, 0.25], "f4"))
    c.create_dataset("category", data=["c", ""], dtype=STR)
    c.create_dataset("name", data=["n", ""], dtype=STR)
    c.create_dataset("source", data=["manual", "center_of_mass"], dtype=STR)

    b = f.create_group("bboxes")
    b.create_dataset("x1", data=np.array([1.0, 2.0], "f8"))
    b.create_dataset("y1", data=np.array([3.0, 4.0], "f8"))
    b.create_dataset("x2", data=np.array([5.0, 6.0], "f8"))
    b.create_dataset("y2", data=np.array([7.0, 8.0], "f8"))
    b.create_dataset("angle", data=np.array([0.0, 0.5], "f8"))
    b.create_dataset("video", data=np.array([0, 1], "i4"))
    b.create_dataset("frame_idx", data=np.array([5, 7], "i8"))
    b.create_dataset("track", data=np.array([0, -1], "i4"))
    b.create_dataset("instance", data=np.array([0, -1], "i4"))
    b.create_dataset("is_predicted", data=np.array([0, 1], "u1"))
    b.create_dataset("score", data=np.array([NAN, 0.5], "f4"))
    b.create_dataset("tracking_score", data=np.array([0.25, NAN], "f4"))
    b.create_dataset("category", data=["bc", ""], dtype=STR)
    b.create_dataset("name", data=["bn", ""], dtype=STR)
    b.create_dataset("source", data=["bs", ""], dtype=STR)
    return b


def write_columns(parent, name, cols, *, resizable=False, compression=None):
    g = parent.require_group(name)
    for key, (values, dtype) in cols.items():
        kwargs = {}
        if compression and len(values):
            kwargs.update(compression=compression, shuffle=True, chunks=(min(len(values), 65536),))
            if compression == "gzip":
                kwargs["compression_opts"] = 1
        if resizable:
            kwargs["maxshape"] = (None,)
        if dtype == "str":
            g.create_dataset(key, data=list(values), dtype=STR, **kwargs)
        else:
            g.create_dataset(key, data=np.array(list(values), dtype=dtype), **kwargs)
    return g


def col(rows, i, dtype):
    return [r[i] for r in rows], dtype


def write_30(path, *, compression=None, unknown=True, embedded=True):
    def columns(parent, name, cols, **kwargs):
        # Python's `compression` applies to the columns of every table.
        return write_columns(parent, name, cols, compression=compression, **kwargs)

    with h5py.File(path, "w") as f:
        bboxes = common(f, format_id=3.0, embedded=embedded)
        columns(
            f,
            "points",
            {
                "x": col(POINTS, 0, "f8"),
                "y": col(POINTS, 1, "f8"),
                "visible": col(POINTS, 2, "?"),
                "verified": col(POINTS, 3, "?"),
                "labeled": col(POINTS, 4, "?"),
            },
        )
        columns(
            f,
            "pred_points",
            {
                "x": col(PRED_POINTS, 0, "f8"),
                "y": col(PRED_POINTS, 1, "f8"),
                "visible": col(PRED_POINTS, 2, "?"),
                "verified": col(PRED_POINTS, 3, "?"),
                "labeled": col(PRED_POINTS, 4, "?"),
                "score": col(PRED_POINTS, 5, "f8"),
            },
        )
        inst = columns(
            f,
            "instances",
            {
                "instance_type": col(INSTANCES, 0, "u1"),
                "frame_id": col(INSTANCES, 1, "i8"),
                "skeleton": col(INSTANCES, 2, "i8"),
                "track": col(INSTANCES, 3, "i8"),
                "from_predicted": col(INSTANCES, 4, "i8"),
                "score": col(INSTANCES, 5, "f8"),
                "point_id_start": col(INSTANCES, 6, "i8"),
                "point_id_end": col(INSTANCES, 7, "i8"),
                "tracking_score": col(INSTANCES, 8, "f8"),
            },
        )
        fr = columns(
            f,
            "frames",
            {
                "video": col(FRAMES, 0, "i8"),
                "frame_idx": col(FRAMES, 1, "i8"),
                "instance_id_start": col(FRAMES, 2, "i8"),
                "instance_id_end": col(FRAMES, 3, "i8"),
                "is_negative": col(FRAMES, 4, "?"),
            },
        )
        if unknown:
            f["points"].create_dataset("brightness", data=np.arange(len(POINTS), dtype="f4"))
            inst.create_dataset("future", data=np.arange(len(INSTANCES), dtype="i2"))
            md = inst.create_group("metadata").create_group("note")
            md.create_dataset("values", data=["x", "", "z", ""], dtype=STR)
            md.create_dataset("present", data=np.array([True, False, True, False]))
            frmd = fr.create_group("metadata").create_group("quality")
            frmd.create_dataset("values", data=np.arange(len(FRAMES), dtype="f8"))
            bmd = bboxes.create_group("metadata").create_group("flagged")
            bmd.create_dataset("values", data=np.array([True, False]))

        rois = columns(
            f,
            "rois",
            {
                "annotation_type": col(ROIS, 0, "u1"),
                "video": col(ROIS, 1, "i8"),
                "frame_idx": col(ROIS, 2, "i8"),
                "track": col(ROIS, 3, "i8"),
                "instance": col(ROIS, 4, "i8"),
                "is_predicted": col(ROIS, 5, "?"),
                "score": col(ROIS, 6, "f8"),
                "tracking_score": col(ROIS, 7, "f8"),
                "wkb_start": ([s[0] for s in ROI_SPANS], "i8"),
                "wkb_end": ([s[1] for s in ROI_SPANS], "i8"),
                "name": col(ROIS, 8, "str"),
                "category": col(ROIS, 9, "str"),
                "source": col(ROIS, 10, "str"),
            },
        )
        rois.create_dataset("wkb", data=WKB_FLAT, chunks=True, compression="gzip", compression_opts=1)

        masks = columns(
            f,
            "masks",
            {
                "height": col(MASKS, 0, "i8"),
                "width": col(MASKS, 1, "i8"),
                "annotation_type": ([2, 2], "u1"),
                "video": col(MASKS, 2, "i8"),
                "frame_idx": col(MASKS, 3, "i8"),
                "track": col(MASKS, 4, "i8"),
                "instance": col(MASKS, 5, "i8"),
                "from_predicted": col(MASKS, 6, "i8"),
                "is_predicted": col(MASKS, 7, "?"),
                "score": col(MASKS, 8, "f8"),
                "tracking_score": col(MASKS, 9, "f8"),
                "rle_start": col(MASKS, 10, "i8"),
                "rle_end": col(MASKS, 11, "i8"),
                "scale_x": col(MASKS, 12, "f8"),
                "scale_y": col(MASKS, 13, "f8"),
                "offset_x": col(MASKS, 14, "f8"),
                "offset_y": col(MASKS, 15, "f8"),
                "name": col(MASKS, 16, "str"),
                "category": col(MASKS, 17, "str"),
                "source": col(MASKS, 18, "str"),
            },
        )
        masks.create_dataset("rle", data=RLE_FLAT, chunks=True, compression="gzip", compression_opts=1)
        sm = columns(
            masks,
            "score_maps",
            {
                "mask_idx": ([0], "i8"),
                "data_start": ([0], "i8"),
                "data_end": ([len(MASK_SCORE)], "i8"),
                "height": ([2], "i8"),
                "width": ([3], "i8"),
                "scale_x": ([1.0], "f8"),
                "scale_y": ([1.0], "f8"),
                "offset_x": ([0.0], "f8"),
                "offset_y": ([0.0], "f8"),
            },
        )
        sm.create_dataset("data", data=np.frombuffer(MASK_SCORE, np.uint8), chunks=True)

        li = columns(
            f,
            "label_images",
            {
                "video": col(LABEL_IMAGES, 0, "i8"),
                "frame_idx": col(LABEL_IMAGES, 1, "i8"),
                "height": col(LABEL_IMAGES, 2, "i8"),
                "width": col(LABEL_IMAGES, 3, "i8"),
                "n_objects": col(LABEL_IMAGES, 4, "i8"),
                "objects_start": col(LABEL_IMAGES, 5, "i8"),
                "data_start": ([0, 0], "i8"),
                "data_end": ([0, 0], "i8"),
                "is_predicted": col(LABEL_IMAGES, 6, "?"),
                "score": col(LABEL_IMAGES, 7, "f8"),
                "scale_x": ([1.0, 1.0], "f8"),
                "scale_y": ([1.0, 1.0], "f8"),
                "offset_x": ([0.0, 0.0], "f8"),
                "offset_y": ([0.0, 0.0], "f8"),
                "source": col(LABEL_IMAGES, 8, "str"),
            },
        )
        li.create_dataset(
            "data",
            data=np.stack(LABEL_PIXELS),
            chunks=(1, 2, 2),
            compression="gzip",
            compression_opts=1,
        )
        columns(
            li,
            "objects",
            {
                "label_id": col(LI_OBJECTS, 0, "i8"),
                "track": col(LI_OBJECTS, 1, "i8"),
                "instance": col(LI_OBJECTS, 2, "i8"),
                "score": col(LI_OBJECTS, 3, "f8"),
                "tracking_score": col(LI_OBJECTS, 4, "f8"),
                "name": col(LI_OBJECTS, 5, "str"),
                "category": col(LI_OBJECTS, 6, "str"),
            },
        )
        lism = columns(
            li,
            "score_maps",
            {
                "li_idx": ([1], "i8"),
                "data_start": ([0], "i8"),
                "data_end": ([len(LI_SCORE)], "i8"),
                "height": ([2], "i8"),
                "width": ([2], "i8"),
                "scale_x": ([1.0], "f8"),
                "scale_y": ([1.0], "f8"),
                "offset_x": ([0.0], "f8"),
                "offset_y": ([0.0], "f8"),
            },
        )
        lism.create_dataset("data", data=np.frombuffer(LI_SCORE, np.uint8), chunks=True)

        for parent, rows, idx, score in (
            ("identity", ID_LINKS, "identity_idx", "identity_score"),
            ("categories", CAT_LINKS, "category_idx", "category_score"),
        ):
            columns(
                f[parent],
                "links",
                {
                    "owner_type": col(rows, 0, "u1"),
                    "owner_id": col(rows, 1, "i8"),
                    idx: col(rows, 2, "i8"),
                    score: col(rows, 3, "f8"),
                },
                resizable=True,
            )

        sd = f.create_group("session_data")
        columns(
            sd,
            "frame_groups",
            {
                "frame_idx": col(FRAME_GROUPS, 0, "i8"),
                "ig_start": col(FRAME_GROUPS, 1, "i8"),
                "ig_end": col(FRAME_GROUPS, 2, "i8"),
            },
            resizable=True,
        )
        columns(
            sd,
            "instance_groups",
            {
                "identity_idx": col(INSTANCE_GROUPS, 0, "i8"),
                "score": col(INSTANCE_GROUPS, 1, "f8"),
                "instance_3d_score": col(INSTANCE_GROUPS, 2, "f8"),
                "pts3d_start": col(INSTANCE_GROUPS, 3, "i8"),
                "pts3d_end": col(INSTANCE_GROUPS, 4, "i8"),
                "pts3d_predicted": col(INSTANCE_GROUPS, 5, "?"),
                "member_start": col(INSTANCE_GROUPS, 6, "i8"),
                "member_end": col(INSTANCE_GROUPS, 7, "i8"),
            },
            resizable=True,
        )
        columns(
            sd,
            "instance_group_members",
            {
                "camera": col(MEMBERS, 0, "i8"),
                "lf": col(MEMBERS, 1, "i8"),
                "inst": col(MEMBERS, 2, "i8"),
            },
            resizable=True,
        )
        write_session_matrices(sd)


def write_session_matrices(sd):
    sd.create_dataset("points_3d", data=POINTS_3D, maxshape=(None, 3), chunks=True, compression="gzip")
    sd.create_dataset("pred_points_3d", data=PRED_POINTS_3D, maxshape=(None, 4), chunks=True, compression="gzip")
    sd.create_dataset("frame_group_meta", data=FG_META, dtype=STR, maxshape=(None,))
    sd.create_dataset("instance_group_meta", data=IG_META, dtype=STR, maxshape=(None,))


def table(f, name, rows, dtype, compression=None):
    f.create_dataset(name, data=np.array(rows, dtype=dtype), compression=compression)


LEGACY_POINT = np.dtype([("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")])
LEGACY_PRED = np.dtype(LEGACY_POINT.descr + [("score", "f8")])
LEGACY_INSTANCE = np.dtype(
    [
        ("instance_id", "i8"),
        ("instance_type", "u1"),
        ("frame_id", "u8"),
        ("skeleton", "u4"),
        ("track", "i4"),
        ("from_predicted", "i8"),
        ("score", "f4"),
        ("point_id_start", "u8"),
        ("point_id_end", "u8"),
        ("tracking_score", "f4"),
    ]
)
LEGACY_FRAME = np.dtype(
    [
        ("frame_id", "u8"),
        ("video", "u4"),
        ("frame_idx", "u8"),
        ("instance_id_start", "u8"),
        ("instance_id_end", "u8"),
    ]
)


def strings(f, name, values):
    f.create_dataset(name, data=list(values), dtype=h5py.special_dtype(vlen=str))


def write_legacy(path, *, compression=None):
    with h5py.File(path, "w") as f:
        common(f, format_id=2.8)
        table(f, "points", [r[:4] for r in POINTS], LEGACY_POINT, compression)
        table(
            f,
            "pred_points",
            [r[:4] + (r[5],) for r in PRED_POINTS],
            LEGACY_PRED,
            compression,
        )
        table(
            f,
            "instances",
            [(i,) + r for i, r in enumerate(INSTANCES)],
            LEGACY_INSTANCE,
            compression,
        )
        table(
            f,
            "frames",
            [(i, RAW_VIDEO_ID[r[0]], r[1], r[2], r[3]) for i, r in enumerate(FRAMES)],
            LEGACY_FRAME,
            compression,
        )
        negative = [(RAW_VIDEO_ID[r[0]], r[1]) for r in FRAMES if r[4]]
        table(
            f,
            "negative_frames",
            negative,
            [("video_id", "u4"), ("frame_idx", "u8")],
            compression,
        )

        table(
            f,
            "rois",
            [
                (r[0], r[1], r[2], r[3], r[5], r[6], r[7], s[0], s[1], r[4])
                for r, s in zip(ROIS, ROI_SPANS)
            ],
            [
                ("annotation_type", "u1"),
                ("video", "i4"),
                ("frame_idx", "i8"),
                ("track", "i4"),
                ("is_predicted", "u1"),
                ("score", "f4"),
                ("tracking_score", "f4"),
                ("wkb_start", "u8"),
                ("wkb_end", "u8"),
                ("instance", "i4"),
            ],
        )
        f.create_dataset("roi_wkb", data=WKB_FLAT, chunks=True, compression="gzip", compression_opts=1)
        strings(f, "roi_names", [r[8] for r in ROIS])
        strings(f, "roi_categories", [r[9] for r in ROIS])
        strings(f, "roi_sources", [r[10] for r in ROIS])

        table(
            f,
            "masks",
            [(r[0], r[1], 2) + r[2:16] for r in MASKS],
            [
                ("height", "u4"),
                ("width", "u4"),
                ("annotation_type", "u1"),
                ("video", "i4"),
                ("frame_idx", "i8"),
                ("track", "i4"),
                ("instance", "i4"),
                ("from_predicted", "i4"),
                ("is_predicted", "u1"),
                ("score", "f4"),
                ("tracking_score", "f4"),
                ("rle_start", "u8"),
                ("rle_end", "u8"),
                ("scale_x", "f4"),
                ("scale_y", "f4"),
                ("offset_x", "f4"),
                ("offset_y", "f4"),
            ],
        )
        f.create_dataset("mask_rle", data=RLE_FLAT, chunks=True, compression="gzip", compression_opts=1)
        strings(f, "mask_names", [r[16] for r in MASKS])
        strings(f, "mask_categories", [r[17] for r in MASKS])
        strings(f, "mask_sources", [r[18] for r in MASKS])
        table(
            f,
            "mask_score_map_index",
            [(0, 0, len(MASK_SCORE), 2, 3, 1.0, 1.0, 0.0, 0.0)],
            [
                ("mask_idx", "u4"),
                ("data_start", "u8"),
                ("data_end", "u8"),
                ("height", "u4"),
                ("width", "u4"),
                ("scale_x", "f4"),
                ("scale_y", "f4"),
                ("offset_x", "f4"),
                ("offset_y", "f4"),
            ],
        )
        f.create_dataset("mask_score_maps", data=np.frombuffer(MASK_SCORE, np.uint8), chunks=True)

        f.create_dataset(
            "label_image_data",
            data=np.stack(LABEL_PIXELS),
            chunks=(1, 2, 2),
            compression="gzip",
            compression_opts=1,
        )
        table(
            f,
            "label_images",
            [
                (r[0], r[1], r[2], r[3], r[4], r[5], 0, 0, r[6], r[7], 1.0, 1.0, 0.0, 0.0)
                for r in LABEL_IMAGES
            ],
            [
                ("video", "i4"),
                ("frame_idx", "i8"),
                ("height", "u4"),
                ("width", "u4"),
                ("n_objects", "u4"),
                ("objects_start", "u4"),
                ("data_start", "u8"),
                ("data_end", "u8"),
                ("is_predicted", "u1"),
                ("score", "f4"),
                ("scale_x", "f4"),
                ("scale_y", "f4"),
                ("offset_x", "f4"),
                ("offset_y", "f4"),
            ],
        )
        strings(f, "label_image_sources", [r[8] for r in LABEL_IMAGES])
        table(
            f,
            "label_image_objects",
            [r[:5] for r in LI_OBJECTS],
            [
                ("label_id", "i4"),
                ("track", "i4"),
                ("instance", "i4"),
                ("score", "f4"),
                ("tracking_score", "f4"),
            ],
        )
        strings(f, "label_image_obj_names", [r[5] for r in LI_OBJECTS])
        strings(f, "label_image_obj_categories", [r[6] for r in LI_OBJECTS])
        table(
            f,
            "label_image_score_map_index",
            [(1, 0, len(LI_SCORE), 2, 2, 1.0, 1.0, 0.0, 0.0)],
            [
                ("li_idx", "u4"),
                ("data_start", "u8"),
                ("data_end", "u8"),
                ("height", "u4"),
                ("width", "u4"),
                ("scale_x", "f4"),
                ("scale_y", "f4"),
                ("offset_x", "f4"),
                ("offset_y", "f4"),
            ],
        )
        f.create_dataset("label_image_score_maps", data=np.frombuffer(LI_SCORE, np.uint8), chunks=True)

        for parent, rows, idx, score in (
            ("identity", ID_LINKS, "identity_idx", "identity_score"),
            ("categories", CAT_LINKS, "category_idx", "category_score"),
        ):
            f[parent].create_dataset(
                "links",
                data=np.array(
                    rows,
                    dtype=[("owner_type", "u1"), ("owner_id", "i8"), (idx, "i4"), (score, "f4")],
                ),
                maxshape=(None,),
            )

        sd = f.create_group("session_data")
        sd.create_dataset(
            "frame_groups",
            data=np.array(FRAME_GROUPS, dtype=[("frame_idx", "i8"), ("ig_start", "u8"), ("ig_end", "u8")]),
            maxshape=(None,),
        )
        sd.create_dataset(
            "instance_groups",
            data=np.array(
                INSTANCE_GROUPS,
                dtype=[
                    ("identity_idx", "i4"),
                    ("score", "f8"),
                    ("instance_3d_score", "f8"),
                    ("pts3d_start", "i8"),
                    ("pts3d_end", "i8"),
                    ("pts3d_predicted", "u1"),
                    ("member_start", "u8"),
                    ("member_end", "u8"),
                ],
            ),
            maxshape=(None,),
        )
        sd.create_dataset(
            "instance_group_members",
            data=np.array(MEMBERS, dtype=[("camera", "u4"), ("lf", "i8"), ("inst", "u4")]),
            maxshape=(None,),
        )
        write_session_matrices(sd)


def write_mixed(path):
    write_legacy(path)
    with h5py.File(path, "a") as f:
        f["metadata"].attrs["format_id"] = 3.0
        del f["points"]
        write_columns(
            f,
            "points",
            {
                "x": col(POINTS, 0, "f8"),
                "y": col(POINTS, 1, "f8"),
                "visible": col(POINTS, 2, "?"),
                "verified": col(POINTS, 3, "?"),
                "labeled": col(POINTS, 4, "?"),
            },
        )


write_30(OUT / "handbuilt_30.slp")
write_30(OUT / "handbuilt_30_gzip.slp", compression="gzip", unknown=False, embedded=False)
write_30(OUT / "handbuilt_30_lzf.slp", compression="lzf", unknown=False, embedded=False)
write_legacy(OUT / "handbuilt_30_2x.slp")
write_legacy(OUT / "handbuilt_30_2x_lzf.slp", compression="lzf")
write_mixed(OUT / "handbuilt_30_mixed.slp")
print("wrote", sorted(p.name for p in OUT.glob("handbuilt_30*.slp")))
