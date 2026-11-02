import type { Video } from "./video.js";

/**
 * Map each raw `frames.video` id to the index of its `Video` in the `videos`
 * array. SLP stores a frame's video as the integer id of its `videoN` HDF5
 * group, which is NOT necessarily the video's position in `videos` — after a
 * remove-video / merge / export the group ids can be sparse or non-contiguous
 * (e.g. `video0`, `video2` → array indices 0, 1). Callers resolve a frame's
 * video with `videos[map.get(rawId) ?? rawId]`.
 *
 * Fast path: when the ids are exactly `0..videos.length-1` (the common case) an
 * identity map is returned. Otherwise each video's group id is parsed from its
 * `dataset` ("videoN/video" → N), preferring the live backend's dataset and
 * falling back to `backendMetadata.dataset`.
 *
 * In SLP 3.0, `video` is always the list index, so the guess above must not run: it
 * would send a frame to the wrong video when only some videos have frames and
 * their group ids aren't in list order. A 3.0 `/frames` table always comes with an
 * `is_negative` column (see `./columnar.ts`), and a 2.x one never does.
 *
 * Single source of truth for the eager readers (`buildLabeledFrames`) and the
 * lazy store ({@link LazyDataStore}), which previously carried drifting copies.
 */
export function buildVideoIdMap(
  framesData: Record<string, unknown[]>,
  videos: Video[],
): Map<number, number> {
  if (framesData.is_negative) {
    return new Map(videos.map((_, index) => [index, index]));
  }
  const videoIds = new Set<number>();
  for (const value of framesData.video ?? []) {
    videoIds.add(Number(value));
  }
  if (!videoIds.size) return new Map();

  const maxId = Math.max(...Array.from(videoIds));
  if (videoIds.size === videos.length && maxId === videos.length - 1) {
    const identity = new Map<number, number>();
    for (let i = 0; i < videos.length; i += 1) {
      identity.set(i, i);
    }
    return identity;
  }

  const map = new Map<number, number>();
  for (let index = 0; index < videos.length; index += 1) {
    const video = videos[index];
    const dataset =
      video.backend?.dataset ??
      (video.backendMetadata?.dataset as string | undefined) ??
      "";
    const parsedId = parseVideoIdFromDataset(dataset);
    if (parsedId != null) {
      map.set(parsedId, index);
    }
  }
  return map;
}

/** Parse the integer id from a `"videoN/..."` HDF5 dataset path, or `null`. */
export function parseVideoIdFromDataset(dataset: string): number | null {
  if (!dataset) return null;
  const group = dataset.split("/")[0];
  if (!group.startsWith("video")) return null;
  const id = Number(group.slice(5));
  return Number.isNaN(id) ? null : id;
}
