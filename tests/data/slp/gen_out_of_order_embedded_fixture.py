#!/usr/bin/env python
"""Regenerate ``out_of_order_embedded.pkg.slp`` for the JS save tests.

A package whose embedded videos are numbered out of list order, with frames in only
some of them, and with real images. ``/frames`` holds each video's ``videoN`` number.
Readers go by those numbers here, because the stored ids aren't ``0..n-1``:

    list index  group    embedded frames  labeled frames
    0           video2   5, 11            5, 11
    1           video0   7, 9             7, 9 (negative)
    2           video1   3                none

A save that writes list indices beside these numbers moves every frame to another
video. ``handbuilt_30_2x.slp`` has the same numbering but its images are all zeros, so
a save with open videos refuses to copy them. Each video here also records a source
video, for ``embed: "source"``.

Written with h5py alone, then checked against Python sleap-io (the contract the JS
readers must match). Run from this folder:

    uv run --no-project --python 3.12 --with "sleap-io==0.9.5" --with h5py --with numpy \
        python gen_out_of_order_embedded_fixture.py
"""

import json
import struct
import zlib
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / "out_of_order_embedded.pkg.slp"
STR = h5py.string_dtype()

HEIGHT, WIDTH = 2, 2


def png(value: int) -> bytes:
    """A small grayscale PNG with every pixel `value` (so never an all-zero blob)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    scanlines = b"".join(b"\x00" + bytes([value]) * WIDTH for _ in range(HEIGHT))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


# (group, {frame: pixel value}, source file). List order is deliberately not N order.
VIDEOS = [
    ("video2", {5: 20, 11: 21}, "clip_a.mp4"),
    ("video0", {7: 40, 9: 41}, "clip_b.mp4"),
    ("video1", {3: 60}, "clip_c.mp4"),
]

SKELETON = {
    "directed": True,
    "graph": {"name": "mouse", "num_edges_inserted": 1},
    "links": [
        {
            "edge_insert_idx": 0,
            "key": 0,
            "source": 0,
            "target": 1,
            "type": {"py/reduce": [{"py/type": "sleap.skeleton.EdgeType"}, {"py/tuple": [1]}]},
        }
    ],
    "multigraph": True,
    "nodes": [{"id": 0}, {"id": 1}],
}
METADATA = {
    "version": "2.0.0",
    "skeletons": [SKELETON],
    "nodes": [{"name": "head"}, {"name": "tail"}],
    "videos": [],
    "tracks": [],
    "suggestions": [],
    "negative_anchors": {},
    "provenance": {},
}

POINT = np.dtype([("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")])
PRED_POINT = np.dtype(POINT.descr + [("score", "f8")])
INSTANCE = np.dtype(
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
FRAME = np.dtype(
    [
        ("frame_id", "u8"),
        ("video", "u4"),
        ("frame_idx", "u8"),
        ("instance_id_start", "u8"),
        ("instance_id_end", "u8"),
    ]
)
NEGATIVE = np.dtype([("video_id", "u4"), ("frame_idx", "u8")])

POINTS = [(1.0, 2.0, True, True), (3.0, 4.0, True, True)]
PRED_POINTS = [(5.0, 6.0, True, True, 0.5), (7.0, 8.0, True, True, 0.75)]
INSTANCES = [
    (0, 0, 0, 0, -1, -1, 0.0, 0, 2, 0.0),  # user, on frame row 0
    (1, 1, 1, 0, -1, -1, 0.9, 0, 2, 0.0),  # predicted, on frame row 1
]
# frame_id, video (the videoN number), frame_idx, instance_id_start, instance_id_end
FRAMES = [
    (0, 2, 5, 0, 1),
    (1, 0, 7, 1, 2),
    (2, 0, 9, 2, 2),
    (3, 2, 11, 2, 2),
]
NEGATIVE_FRAMES = [(0, 9)]


def write() -> None:
    with h5py.File(OUT, "w") as f:
        grp = f.create_group("metadata")
        grp.attrs["format_id"] = 1.4
        grp.attrs["json"] = np.bytes_(json.dumps(METADATA, separators=(",", ":")))

        videos_json = []
        for group, images, source in VIDEOS:
            blobs = [png(value) for value in images.values()]
            widest = max(len(blob) for blob in blobs)
            rows = np.zeros((len(blobs), widest), dtype="i1")
            for row, blob in enumerate(blobs):
                rows[row, : len(blob)] = np.frombuffer(blob, dtype="i1")
            g = f.create_group(group)
            d = g.create_dataset("video", data=rows)
            d.attrs["format"] = "png"
            d.attrs["channel_order"] = "RGB"
            d.attrs["frames"] = 20
            d.attrs["height"] = HEIGHT
            d.attrs["width"] = WIDTH
            d.attrs["channels"] = 1
            g.create_dataset("frame_numbers", data=np.array(list(images), dtype="i8"))
            source_video = {
                "filename": source,
                "backend": {"filename": source, "shape": [20, HEIGHT, WIDTH, 1]},
            }
            g.create_group("source_video").attrs["json"] = json.dumps(source_video)
            videos_json.append(
                json.dumps(
                    {
                        "filename": ".",
                        "backend": {
                            "type": "HDF5Video",
                            "shape": [len(images), HEIGHT, WIDTH, 1],
                            "filename": ".",
                            "dataset": f"{group}/video",
                            "input_format": "channels_last",
                            "convert_range": False,
                            "has_embedded_images": True,
                            "grayscale": True,
                            "fps": None,
                        },
                        "source_video": source_video,
                    }
                )
            )
        f.create_dataset("videos_json", data=videos_json, dtype=STR, maxshape=(None,))
        for name in ("tracks_json", "suggestions_json", "sessions_json"):
            f.create_dataset(name, data=[], dtype=STR, maxshape=(None,))

        f.create_dataset("points", data=np.array(POINTS, dtype=POINT))
        f.create_dataset("pred_points", data=np.array(PRED_POINTS, dtype=PRED_POINT))
        f.create_dataset("instances", data=np.array(INSTANCES, dtype=INSTANCE))
        f.create_dataset("frames", data=np.array(FRAMES, dtype=FRAME))
        f.create_dataset("negative_frames", data=np.array(NEGATIVE_FRAMES, dtype=NEGATIVE))


def check() -> None:
    import sleap_io as sio

    labels = sio.load_slp(str(OUT))
    assert [v.backend.dataset for v in labels.videos] == [
        "video2/video",
        "video0/video",
        "video1/video",
    ]
    where = [
        (labels.videos.index(lf.video), lf.frame_idx, lf.is_negative)
        for lf in labels.labeled_frames
    ]
    assert where == [(0, 5, False), (1, 7, False), (1, 9, True), (0, 11, False)], where
    print(f"wrote {OUT.name}: {where} (sleap-io {sio.__version__})")


if __name__ == "__main__":
    write()
    check()
