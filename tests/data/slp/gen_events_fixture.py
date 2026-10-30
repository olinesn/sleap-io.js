#!/usr/bin/env python
"""Regenerate ``py_written_26_events.slp``, a format-2.6 file with events.

The JS reader must read this exactly as Python wrote it (``tests/slp-events.test.ts``),
so the file pins the ``/event_types`` + ``/events`` layout: bool ``is_predicted``,
int64 ids, vlen strings, gzip metadata tables, and the optional ``score`` and
``scores`` + ``score_offsets`` columns.

Run from this folder:

    uv run --with "sleap-io==0.9.5" python gen_events_fixture.py
"""

import numpy as np
import sleap_io as sio

skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])
video0 = sio.Video(filename="clip0.mp4", open_backend=False)
video1 = sio.Video(filename="clip1.mp4", open_backend=False)
mouse1, mouse2 = sio.Track(name="mouse1"), sio.Track(name="mouse2")
animal = sio.Identity(name="animal_A", metadata={"color": "#112233"})

frame = sio.LabeledFrame(
    video=video0,
    frame_idx=0,
    instances=[
        sio.Instance.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton, track=mouse1
        )
    ],
)

attack = sio.EventType(
    name="attack",
    description="One mouse attacks another.",
    metadata={"color": "#e6194b", "social": "true"},
)
rear = sio.EventType(name="rear")
unused = sio.EventType(name="never used", description="in the catalog, no events")

labels = sio.Labels(
    labeled_frames=[frame],
    videos=[video0, video1],
    skeletons=[skeleton],
    tracks=[mouse1, mouse2],
    identities=[animal],
    event_types=[attack, rear, unused],
    events=[
        sio.UserEvent(
            type=attack,
            video=video0,
            start_frame=100,
            end_frame=140,
            subject=mouse1,
            target=mouse2,
            name="first attack",
            source="manual",
            metadata={"note": "ünïcode ok"},
        ),
        sio.UserEvent(type=rear, video=video1, start_frame=50, subject=animal),
        sio.PredictedEvent(
            type=attack,
            video=video0,
            start_frame=200,
            end_frame=204,
            subject=mouse2,
            scores=[0.6, 0.8, 0.9, 0.7, 0.5],
            score=0.74,
            source="classifier",
        ),
        sio.PredictedEvent(type=rear, video=video0, start_frame=7, end_frame=9),
        sio.UserEvent(type=rear, video=video0, start_frame=300, end_frame=301),
    ],
)
labels.save("py_written_26_events.slp", embed=False)
print("wrote py_written_26_events.slp")
