#!/usr/bin/env python
"""Regenerate the ``py_written_30_twin*.slp`` fixtures: one project saved both ways.

Python sleap-io 0.10 writes format 3.0 by default, and format 2.x with
``legacy_format=True``. ``tests/slp-format-30.test.ts`` checks that the two files read
to the same values through every reader, which is also what makes
``legacy_format=True`` good advice for anything that can't read 3.0.

Also saved both ways, from ``sparse_video_negative.pkg.slp`` (whose embedded video sits
at ``/video2``, with a negative frame):

- ``py_written_30_sparse*.slp`` refer to that package without embedding, so the video
  keeps its number. There ``/frames/video`` differs between the layouts: the list index
  (0) in 3.0, and the ``videoN`` number (2) in 2.x.
- ``py_written_30_embedded*.pkg.slp`` embed its images again, which renumbers the
  video to ``/video0``. These hold real embedded frames.

The project holds a little of everything: user and predicted instances, a
``from_predicted`` link, hidden, absent and unlabeled points, tracks, a negative frame,
an empty frame, suggestions, identity and category links, embeddings, ROIs, masks with
a score map, label images, boxes, centroids, a session with 3D points, and per-record
metadata (which the 2.x file can't store, so Python drops it with a warning).

Run from this folder:

    uv run --no-project --python 3.12 \
        --with "sleap-io @ git+https://github.com/talmolab/sleap-io@release/0.10" \
        --with shapely python gen_format_30_twins_fixture.py

The committed files came from release/0.10 at e5bf2b170 (2026-10-30). Point the command
at a tag once sleap-io 0.10 is released.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import shapely
import sleap_io as sio
from sleap_io.model.label_image import LabelImage

HERE = Path(__file__).resolve().parent
NAN = float("nan")


def build_labels() -> sio.Labels:
    skeleton = sio.Skeleton(
        nodes=["head", "body", "tail"],
        edges=[("head", "body"), ("body", "tail")],
        name="mouse",
    )
    videos = [
        sio.Video(filename=f"clip{i}.mp4", open_backend=False) for i in range(3)
    ]
    mouse1, mouse2 = sio.Track(name="mouse1"), sio.Track(name="mouse2")
    animal_a = sio.Identity(name="animal_A", metadata={"color": "#112233"})
    animal_b = sio.Identity(name="animal_B")
    rodent = sio.Category(name="rodent")

    p0 = sio.PredictedInstance.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.25, 0.5, 0.75]),
        score=0.1,
        track=mouse1,
        tracking_score=0.75,
        identity=animal_a,
        identity_score=0.5,
    )
    p0.metadata["note"] = "first"
    u0 = sio.Instance.from_numpy(
        np.array([[1.5, 2.5], [3.5, 4.5], [NAN, NAN]]),
        skeleton=skeleton,
        track=mouse1,
        from_predicted=p0,
        category=rodent,
        category_score=0.125,
        identity_embedding=sio.Embedding(np.array([0.5, 0.25], dtype="f4")),
        # A hidden point that keeps its coordinates, then an absent one.
        visible=np.array([True, False, False]),
        verified=np.array([True, True, False]),
    )
    p1 = sio.PredictedInstance.from_numpy(
        np.array([[7.0, 8.0], [NAN, NAN], [9.0, 1.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.125, NAN, 1.0]),
        score=0.9,
        track=mouse2,
        tracking_score=0.5,
        identity=animal_b,
    )
    u1 = sio.Instance.from_numpy(
        np.array([[10.25, 11.25], [12.25, 13.25], [NAN, NAN]]),
        skeleton=skeleton,
        # The last point is unlabeled, which only the 3.0 layout can store.
        labeled=np.array([True, True, False]),
        visible=np.array([True, True, False]),
        category=rodent,
        category_embedding=sio.Embedding(np.array([1.5, 2.5, 3.5], dtype="f4")),
    )

    f0 = sio.LabeledFrame(video=videos[0], frame_idx=5, instances=[p0, u0])
    f0.metadata["quality"] = 0.5
    f1 = sio.LabeledFrame(video=videos[1], frame_idx=7, instances=[p1])
    negative = sio.LabeledFrame(video=videos[1], frame_idx=9, is_negative=True)
    empty = sio.LabeledFrame(video=videos[0], frame_idx=11)
    f4 = sio.LabeledFrame(video=videos[0], frame_idx=13, instances=[u1])

    f0.rois.append(
        sio.UserROI(
            geometry=shapely.Point(1, 2),
            name="roi-a",
            category="cat-a",
            source="src-a",
            video=videos[0],
            track=mouse1,
            instance=u0,
            identity=animal_b,
            identity_score=0.5,
        )
    )
    f1.rois.append(
        sio.PredictedROI(
            geometry=shapely.Point(3, 4),
            name="roi-b",
            source="src-b",
            video=videos[1],
            instance=p1,
            score=0.5,
            tracking_score=0.25,
        )
    )
    static_roi = sio.UserROI(geometry=shapely.Point(5, 6), name="static", category="s")

    predicted_mask = sio.PredictedSegmentationMask(
        rle_counts=np.array([1, 2, 3], dtype=np.uint32),
        height=2,
        width=3,
        name="m-pred",
        category="c",
        source="s",
        track=mouse1,
        tracking_score=0.75,
        instance=p0,
        identity=animal_b,
        identity_score=0.75,
        score=0.5,
        score_map=np.arange(6, dtype=np.float32).reshape(2, 3),
    )
    user_mask = sio.UserSegmentationMask(
        rle_counts=np.array([4, 2], dtype=np.uint32),
        height=2,
        width=3,
        name="m-user",
        scale=(2.0, 0.5),
        offset=(1.5, 2.5),
        from_predicted=predicted_mask,
        category=rodent,
        category_score=0.5,
    )
    f0.masks.extend([predicted_mask, user_mask])

    f0.label_images.append(
        sio.UserLabelImage(
            data=np.array([[0, 1], [1, 2]], dtype=np.int32),
            objects={
                1: LabelImage.Info(
                    track=mouse1,
                    name="o1",
                    category="c1",
                    instance=p0,
                    score=0.5,
                    tracking_score=0.25,
                ),
                2: LabelImage.Info(name="o2"),
            },
            source="li-user",
        )
    )
    f1.label_images.append(
        sio.PredictedLabelImage(
            data=np.array([[2, 2], [0, 1]], dtype=np.int32),
            objects={
                2: LabelImage.Info(
                    track=mouse2, category="c2", instance=p1, score=0.75
                )
            },
            source="li-pred",
            score=0.5,
            score_map=np.arange(4, dtype=np.float32).reshape(2, 2),
        )
    )

    box = sio.UserBoundingBox(
        x1=1.0,
        y1=3.0,
        x2=5.0,
        y2=7.0,
        track=mouse1,
        tracking_score=0.25,
        instance=p0,
        category="bc",
        name="bn",
        source="bs",
        identity=animal_a,
    )
    box.metadata["flagged"] = True
    f0.bboxes.append(box)
    f1.bboxes.append(
        sio.PredictedBoundingBox(x1=2.0, y1=4.0, x2=6.0, y2=8.0, angle=0.5, score=0.5)
    )
    f0.centroids.append(
        sio.UserCentroid(
            x=1.5, y=3.5, track=mouse1, instance=u0, category="c", name="n", source="manual"
        )
    )
    f1.centroids.append(
        sio.PredictedCentroid(
            x=2.5,
            y=4.5,
            z=5.5,
            score=0.5,
            tracking_score=0.25,
            source="center_of_mass",
            identity=animal_a,
            identity_score=0.25,
        )
    )

    cam_a = sio.Camera(
        name="camA",
        matrix=np.array([[100.0, 0, 320.0], [0, 100.0, 240.0], [0, 0, 1.0]]),
        dist=np.zeros(5),
        size=(640, 480),
        rvec=np.zeros(3),
        tvec=np.zeros(3),
    )
    cam_b = sio.Camera(
        name="camB",
        matrix=np.array([[100.0, 0, 320.0], [0, 100.0, 240.0], [0, 0, 1.0]]),
        dist=np.zeros(5),
        size=(640, 480),
        rvec=np.array([0.0, 0.5, 0.0]),
        tvec=np.array([10.0, 0.0, 0.0]),
    )
    group_a = sio.InstanceGroup(
        instance_by_camera={cam_a: u0, cam_b: p1},
        score=0.5,
        identity=animal_a,
        instance_3d=sio.Instance3D(
            points=np.array([[1.5, 2.5, 3.5], [NAN, NAN, NAN], [7.5, 8.5, 9.5]]),
            skeleton=skeleton,
            score=0.25,
        ),
    )
    group_b = sio.InstanceGroup(instance_by_camera={cam_a: p0}, metadata={"ig": "x"})
    group_c = sio.InstanceGroup(
        instance_by_camera={cam_b: p1},
        score=0.75,
        identity=animal_b,
        instance_3d=sio.PredictedInstance3D(
            points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [NAN, NAN, NAN]]),
            skeleton=skeleton,
            score=0.5,
            point_scores=np.array([0.5, 0.25, NAN]),
        ),
    )
    session = sio.RecordingSession(
        camera_group=sio.CameraGroup(cameras=[cam_a, cam_b], metadata={"board": "x"}),
        video_by_camera={cam_a: videos[0], cam_b: videos[1]},
        camera_by_video={videos[0]: cam_a, videos[1]: cam_b},
        frame_group_by_frame_idx={
            5: sio.FrameGroup(
                frame_idx=5,
                instance_groups=[group_a, group_b],
                labeled_frame_by_camera={cam_a: f0, cam_b: f1},
                metadata={"fg": 1},
            ),
            7: sio.FrameGroup(
                frame_idx=7,
                instance_groups=[group_c],
                labeled_frame_by_camera={cam_b: f1},
            ),
        },
        metadata={"name": "s0"},
    )

    return sio.Labels(
        labeled_frames=[f0, f1, negative, empty, f4],
        videos=videos,
        skeletons=[skeleton],
        tracks=[mouse1, mouse2],
        identities=[animal_a, animal_b],
        categories=[rodent],
        suggestions=[sio.SuggestionFrame(video=videos[1], frame_idx=3)],
        sessions=[session],
        provenance={"made_by": "gen_format_30_twins_fixture.py"},
        rois=[static_roi],
    )


def poses(path: str) -> list:
    """The point values Python itself reads back, to compare the two layouts."""
    labels = sio.load_slp(path, open_videos=False)
    return [
        (lf.frame_idx, lf.is_negative, [inst.numpy().tolist() for inst in lf])
        for lf in labels.labeled_frames
    ]


def save_both(labels: sio.Labels, columnar: str, legacy: str, **options) -> None:
    """Save one project in both layouts and check Python reads back the same poses."""
    sio.save_slp(labels, columnar, **options)
    with warnings.catch_warnings():
        # Metadata is dropped, and an unlabeled point saved as absent, on purpose.
        warnings.simplefilter("ignore")
        sio.save_slp(labels, legacy, legacy_format=True, **options)
    assert str(poses(columnar)) == str(poses(legacy))


def main() -> None:
    # File names are relative so that no machine's paths end up in the fixtures.
    os.chdir(HERE)
    save_both(
        build_labels(),
        "py_written_30_twin.slp",
        "py_written_30_twin_legacy.slp",
        embed=False,
        save_embedding_vectors=True,
        unlabeled_as="absent",
    )
    source = "sparse_video_negative.pkg.slp"
    save_both(
        sio.load_slp(source),
        "py_written_30_sparse.slp",
        "py_written_30_sparse_legacy.slp",
        embed=False,
        restore_original_videos=False,
    )
    save_both(
        sio.load_slp(source),
        "py_written_30_embedded.pkg.slp",
        "py_written_30_embedded_legacy.pkg.slp",
        embed=None,
    )
    print(f"wrote the twin fixtures (sleap-io {sio.__version__})")


if __name__ == "__main__":
    main()
