#!/usr/bin/env python
"""Regenerate ``py_written_annotations.slp``: every annotation kind, written by Python.

The reference for what a load and save in the JS port must keep. It holds user and
predicted ROIs, masks, bounding boxes, centroids and label images, with what makes
them awkward:

- track, instance and ``from_predicted`` links
- identities and categories with scores and embeddings on detections (formats 2.5
  and 2.7)
- score maps of another size than their mask or label image (4x6 on 2x3), with a
  scale and an offset (a flat array in JS doesn't say its own shape)
- a static ROI tied to a video but no frame

Run with a pinned sleap-io, so that the layout on disk stays the 2.x one:

    uv run --no-project --with "sleap-io==0.9.5" --with shapely \\
        python -I tests/data/slp/gen_annotations_fixture.py
"""

from pathlib import Path

import numpy as np
import shapely
import sleap_io as sio
from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
from sleap_io.model.centroid import PredictedCentroid, UserCentroid
from sleap_io.model.embedding import Embedding
from sleap_io.model.identity import Identity
from sleap_io.model.label_image import LabelImage, PredictedLabelImage, UserLabelImage
from sleap_io.model.mask import PredictedSegmentationMask, UserSegmentationMask
from sleap_io.model.roi import PredictedROI, UserROI

OUT = Path(__file__).parent / "py_written_annotations.slp"

skeleton = sio.Skeleton(["a", "b"], name="s")
video_a = sio.Video.from_filename("a.mp4")
video_b = sio.Video.from_filename("b.mp4")
track = sio.Track("t0")
instance = sio.Instance.from_numpy(
    np.array([[1, 2], [3, 4]], "float32"), skeleton=skeleton, track=track
)
predicted_instance = sio.PredictedInstance.from_numpy(
    np.array([[5, 6], [7, 8]], "float32"),
    skeleton=skeleton,
    score=0.5,
    point_scores=np.array([0.25, 0.75]),
)
identity = Identity(name="mouse-1")
raster = np.array([[0, 1, 1], [1, 1, 0]], dtype=bool)
score_map = np.arange(24, dtype=np.float32).reshape(4, 6) / 32

predicted_mask = PredictedSegmentationMask.from_numpy(
    raster,
    score=0.75,
    score_map=score_map,
    score_map_scale=(2.0, 2.0),
    score_map_offset=(1.0, 1.0),
    name="predicted mask",
    category="cell",
    source="model",
    track=track,
    tracking_score=0.25,
    instance=predicted_instance,
    identity=identity,
    identity_score=0.5,
    identity_embedding=Embedding(vector=np.array([0.5, 0.25], dtype=np.float32)),
    category_score=0.25,
    category_embedding=Embedding(vector=np.array([1.5, 2.5, 3.5], dtype=np.float32)),
)
user_mask = UserSegmentationMask.from_numpy(
    raster, name="user mask", instance=instance, from_predicted=predicted_mask
)
frame_a = sio.LabeledFrame(
    video=video_a,
    frame_idx=3,
    instances=[instance, predicted_instance],
    rois=[
        UserROI(
            geometry=shapely.Polygon([(0, 0), (4, 0), (4, 3)]),
            name="user roi",
            category="arena",
            source="manual",
            track=track,
            tracking_score=0.5,
            instance=instance,
        )
    ],
    masks=[predicted_mask, user_mask],
    bboxes=[
        UserBoundingBox(
            x1=1.5,
            y1=2.5,
            x2=3.25,
            y2=4.75,
            angle=0.5,
            name="user box",
            category="mouse",
            source="manual",
            track=track,
            tracking_score=0.5,
            instance=instance,
        )
    ],
    centroids=[UserCentroid(x=1.0, y=2.0, name="user centroid", instance=instance)],
    label_images=[
        PredictedLabelImage(
            data=np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32),
            objects={
                1: LabelImage.Info(
                    track=track,
                    category="neuron",
                    name="first",
                    instance=instance,
                    score=0.5,
                    tracking_score=0.25,
                ),
                2: LabelImage.Info(
                    category="glia", name="second", instance=predicted_instance
                ),
            },
            source="cellpose",
            score=0.75,
            score_map=score_map,
            score_map_scale=(2.0, 2.0),
            score_map_offset=(1.0, 1.0),
        )
    ],
)
frame_b = sio.LabeledFrame(
    video=video_b,
    frame_idx=7,
    rois=[
        PredictedROI(
            geometry=shapely.Point(1.5, 2.5),
            score=0.75,
            identity=identity,
            identity_score=0.25,
        )
    ],
    bboxes=[
        PredictedBoundingBox(
            x1=10,
            y1=20,
            x2=30,
            y2=40,
            score=0.75,
            category="rat",
            category_score=0.25,
            identity=identity,
            identity_score=0.5,
            identity_embedding=Embedding(vector=np.array([0.75, 0.5], dtype=np.float32)),
        )
    ],
    centroids=[PredictedCentroid(x=3.0, y=4.0, score=0.5, identity=identity)],
    label_images=[UserLabelImage(data=np.array([[0, 0]], dtype=np.int32))],
)
labels = sio.Labels(
    labeled_frames=[frame_a, frame_b],
    videos=[video_a, video_b],
    skeletons=[skeleton],
    tracks=[track],
    identities=[identity],
    rois=[
        UserROI(
            geometry=shapely.LineString([(0, 1), (2, 3)]),
            name="static roi",
            video=video_b,
        )
    ],
)
sio.save_slp(labels, OUT, embed=False, save_embedding_vectors=True)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
