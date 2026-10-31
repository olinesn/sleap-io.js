/**
 * A project holding every annotation kind, with the awkward values each one can
 * carry: empty and non-ASCII strings, user and predicted, links to tracks and
 * instances, scores, spatial transforms, score maps, and a frame number past 2^24.
 *
 * Shared by the tests that read it back through Python sleap-io and through this
 * library, and used to write `tests/data/slp/js_flat_annotations.slp`.
 */
import { UserBoundingBox, PredictedBoundingBox } from "../src/model/bbox.js";
import { UserCentroid } from "../src/model/centroid.js";
import { Instance, PredictedInstance, Track } from "../src/model/instance.js";
import {
  type LabelImageObjectInfo,
  PredictedLabelImage,
  UserLabelImage,
} from "../src/model/label-image.js";
import { LabeledFrame } from "../src/model/labeled-frame.js";
import { Labels } from "../src/model/labels.js";
import {
  PredictedSegmentationMask,
  UserSegmentationMask,
} from "../src/model/mask.js";
import { PredictedROI, UserROI } from "../src/model/roi.js";
import { Skeleton } from "../src/model/skeleton.js";
import { Video } from "../src/model/video.js";

/** The annotation kinds, by their names on `LabeledFrame`. */
export const ANNOTATION_KINDS = [
  "rois",
  "masks",
  "labelImages",
  "bboxes",
  "centroids",
] as const;
export type AnnotationKind = (typeof ANNOTATION_KINDS)[number];

/** A frame number a float32 column would round (to 2^24). */
export const FRAME_PAST_2_24 = 16_777_217;

/** Non-ASCII, so a string that goes through the wrong encoding shows. */
export const UNICODE_NAME = "souris é ✓";

/**
 * Build the project. With `only`, every other annotation kind is left out, so that
 * a kind that can't be read in another program shows up on its own. `farFrameIdx` is
 * the frame number of the last frame.
 */
export function annotatedLabels(
  only?: AnnotationKind,
  farFrameIdx: number = FRAME_PAST_2_24,
): Labels {
  const keep = (kind: AnnotationKind) => !only || only === kind;
  const skeleton = new Skeleton({ name: "s", nodes: ["a", "b"] });
  const videoA = new Video({ filename: "a.mp4" });
  const videoB = new Video({ filename: "b.mp4" });
  const trackA = new Track("track a");
  const trackB = new Track("track b");
  const userInstance = new Instance({
    skeleton,
    track: trackA,
    points: [
      { xy: [1, 2], visible: true, complete: true },
      { xy: [3, 4], visible: true, complete: true },
    ],
  });
  const predictedInstance = new PredictedInstance({
    skeleton,
    score: 0.5,
    track: trackB,
    points: [
      { xy: [5, 6], visible: true, complete: false, score: 0.25 },
      { xy: [7, 8], visible: true, complete: false, score: 0.75 },
    ],
  });
  const laterInstance = new Instance({
    skeleton,
    points: [
      { xy: [9, 10], visible: true, complete: true },
      { xy: [11, 12], visible: true, complete: true },
    ],
  });

  const frame0 = new LabeledFrame({
    video: videoA,
    frameIdx: 3,
    instances: [userInstance, predictedInstance],
  });
  const frame1 = new LabeledFrame({
    video: videoB,
    frameIdx: 7,
    instances: [laterInstance],
  });
  const farFrame = new LabeledFrame({
    video: videoB,
    frameIdx: farFrameIdx,
  });
  const staticRois: UserROI[] = [];

  if (keep("rois")) {
    frame0.rois.push(
      new UserROI({
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [0, 0],
              [4, 0],
              [4, 3],
              [0, 0],
            ],
          ],
        },
        name: UNICODE_NAME,
        category: "arena",
        source: "manual",
        track: trackA,
        trackingScore: 0.5,
        instance: userInstance,
      }),
    );
    frame1.rois.push(
      new PredictedROI({
        geometry: { type: "Point", coordinates: [1.5, 2.5] },
        score: 0.75,
        instance: laterInstance,
      }),
    );
    farFrame.rois.push(
      new UserROI({
        geometry: { type: "Point", coordinates: [8, 9] },
        name: "far",
      }),
    );
    staticRois.push(
      new UserROI({
        geometry: {
          type: "LineString",
          coordinates: [
            [0, 1],
            [2, 3],
          ],
        },
        name: "static",
        video: videoB,
      }),
    );
  }

  if (keep("masks")) {
    const predicted = new PredictedSegmentationMask({
      rleCounts: new Uint32Array([1, 2, 3]),
      height: 2,
      width: 3,
      name: UNICODE_NAME,
      category: "cell",
      source: "model",
      track: trackB,
      trackingScore: 0.25,
      instance: predictedInstance,
      scale: [2, 4],
      offset: [10, 20],
      score: 0.75,
      scoreMap: new Float32Array([0, 0.25, 0.5, 0.75, 1, 0.5]),
      scoreMapScale: [0.5, 0.25],
      scoreMapOffset: [1, 2],
    });
    frame0.masks.push(
      predicted,
      new UserSegmentationMask({
        rleCounts: new Uint32Array([6]),
        height: 2,
        width: 3,
        fromPredicted: predicted,
        instance: userInstance,
      }),
    );
    farFrame.masks.push(
      new PredictedSegmentationMask({
        rleCounts: new Uint32Array([0, 4]),
        height: 2,
        width: 2,
        score: 0.5,
      }),
    );
  }

  if (keep("labelImages")) {
    frame0.labelImages.push(
      new PredictedLabelImage({
        data: new Int32Array([0, 1, 1, 2, 2, 0]),
        height: 2,
        width: 3,
        objects: new Map<number, LabelImageObjectInfo>([
          [
            1,
            {
              track: trackA,
              category: "neuron",
              name: UNICODE_NAME,
              instance: userInstance,
              score: 0.5,
              trackingScore: 0.25,
            },
          ],
          [
            2,
            {
              track: null,
              category: "",
              name: "",
              instance: predictedInstance,
            },
          ],
        ]),
        source: "cellpose",
        score: 0.75,
        scale: [2, 4],
        offset: [10, 20],
        scoreMap: new Float32Array([0, 0.25, 0.5, 0.75, 1, 0.5]),
        scoreMapScale: [0.5, 0.25],
        scoreMapOffset: [1, 2],
      }),
    );
    // A different size and no objects: the object table holds only the first's.
    farFrame.labelImages.push(
      new UserLabelImage({
        data: new Int32Array([0, 0, 0, 0]),
        height: 2,
        width: 2,
      }),
    );
  }

  if (keep("bboxes")) {
    frame0.bboxes.push(
      new UserBoundingBox({
        x1: 1.5,
        y1: 2.5,
        x2: 3.25,
        y2: 4.75,
        angle: 0.5,
        name: UNICODE_NAME,
        category: "mouse",
        source: "manual",
        track: trackA,
        trackingScore: 0.5,
        instance: userInstance,
      }),
    );
    farFrame.bboxes.push(
      new PredictedBoundingBox({
        x1: 10,
        y1: 20,
        x2: 30,
        y2: 40,
        score: 0.75,
        track: trackB,
      }),
    );
  }

  if (keep("centroids")) {
    frame1.centroids.push(
      new UserCentroid({ x: 1, y: 2, name: "c", instance: laterInstance }),
    );
  }

  return new Labels({
    skeletons: [skeleton],
    videos: [videoA, videoB],
    tracks: [trackA, trackB],
    labeledFrames: [frame0, frame1, farFrame],
    rois: staticRois,
  });
}
