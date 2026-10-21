#!/usr/bin/env python
"""Regenerate ``sparse_video_negative.pkg.slp`` for the JS negative-frames readers.

Python sleap-io writes ``/negative_frames`` rows as ``(video_id, frame_idx)``
where ``video_id`` is the *HDF5 group id* of an embedded video (``video2/video``
→ 2), the same id ``/frames`` uses — not the video's position in ``videos``.
Group ids go sparse once a video is removed from a package (``video0`` and
``video2`` left → positions 0 and 1). A reader that compares the raw id with the
position misses or misplaces every negative flag on such a file.

Base file: ``minimal_instance.pkg.slp`` (one embedded video under ``/video0``,
one labeled frame at frame 0 with two instances).

What this does:
1. Copies the base package and moves ``/video0`` → ``/video2`` (and points
   ``videos_json``'s ``dataset`` and the ``/frames`` video column at 2).
2. Appends an empty frame row for frame 1 and a ``/negative_frames`` row
   ``(2, 1)`` in Python's dtype ``[("video_id", u4), ("frame_idx", u8)]``.
3. Loads the result with Python sleap-io and checks frame 1 reads as negative
   and frame 0 doesn't — the contract the JS readers must match.

Run with:
    uv run --python 3.12 --with "sleap-io>=0.9.3" --with h5py --with numpy python gen_sparse_video_negative_fixture.py
"""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "minimal_instance.pkg.slp"
OUT = HERE / "sparse_video_negative.pkg.slp"
SPARSE_ID = 2


def main() -> None:
    shutil.copyfile(BASE, OUT)
    with h5py.File(OUT, "a") as f:
        f.move("video0", f"video{SPARSE_ID}")

        videos = [json.loads(v) for v in f["videos_json"][:]]
        videos[0]["backend"]["dataset"] = f"video{SPARSE_ID}/video"
        del f["videos_json"]
        f.create_dataset("videos_json", data=[json.dumps(v) for v in videos], maxshape=(None,))

        frames = f["frames"][:]
        frames["video"] = SPARSE_ID
        empty = np.zeros(1, dtype=frames.dtype)
        empty["frame_id"] = 1
        empty["video"] = SPARSE_ID
        empty["frame_idx"] = 1
        end = frames["instance_id_end"][-1]
        empty["instance_id_start"] = end
        empty["instance_id_end"] = end
        del f["frames"]
        f.create_dataset("frames", data=np.concatenate([frames, empty]), maxshape=(None,))

        neg_dtype = np.dtype([("video_id", "u4"), ("frame_idx", "u8")])
        f.create_dataset("negative_frames", data=np.array([(SPARSE_ID, 1)], dtype=neg_dtype))

    import sleap_io as sio

    labels = sio.load_slp(str(OUT))
    by_idx = {lf.frame_idx: lf for lf in labels.labeled_frames}
    assert labels.videos[0].backend.dataset == f"video{SPARSE_ID}/video"
    assert by_idx[1].is_negative and len(by_idx[1].instances) == 0
    assert not by_idx[0].is_negative and len(by_idx[0].instances) == 2
    print(f"wrote {OUT.name}: frame 1 negative under video id {SPARSE_ID} (sleap-io {sio.__version__})")


if __name__ == "__main__":
    main()
