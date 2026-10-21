import type { LabeledFrame } from "./labeled-frame.js";
import type { Video } from "./video.js";
import { buildVideoIdMap } from "./video-id-map.js";

/**
 * Raw `"<video_id>:<frame_idx>"` keys for the rows of a `/negative_frames`
 * dataset (columns `video_id` — or legacy `video` — and `frame_idx`).
 *
 * `video_id` is the HDF5 group id of the video (`video2/video` → 2), the same
 * id the `/frames` table uses, which sleap-io writes sparse once a video has
 * been removed. Keys stay raw here: the lazy store compares them with raw
 * `/frames` ids and copies them through on save unchanged.
 */
export function negativeFrameKeys(
  columns: Record<string, unknown[]>,
): Set<string> {
  const videoIds = columns.video_id ?? columns.video ?? [];
  const frameIdxs = columns.frame_idx ?? [];
  const keys = new Set<string>();
  for (let i = 0; i < frameIdxs.length; i++) {
    keys.add(`${Number(videoIds[i])}:${Number(frameIdxs[i])}`);
  }
  return keys;
}

/**
 * Set `isNegative` on the eagerly built frames named by raw `/negative_frames`
 * keys, resolving each raw video id to its `videos` index exactly as
 * `/frames` is resolved ({@link buildVideoIdMap}).
 */
export function markNegativeFrames(
  labeledFrames: LabeledFrame[],
  rawKeys: Set<string>,
  framesData: Record<string, unknown[]>,
  videos: Video[],
): void {
  if (rawKeys.size === 0) return;
  const videoIdToIndex = buildVideoIdMap(framesData, videos);
  const byIndex = new Set<string>();
  for (const key of rawKeys) {
    const [rawVideoId, frameIdx] = key.split(":").map(Number);
    byIndex.add(`${videoIdToIndex.get(rawVideoId) ?? rawVideoId}:${frameIdx}`);
  }
  for (const frame of labeledFrames) {
    if (byIndex.has(`${videos.indexOf(frame.video)}:${frame.frameIdx}`)) {
      frame.isNegative = true;
    }
  }
}
