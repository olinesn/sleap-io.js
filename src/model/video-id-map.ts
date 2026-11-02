import type { Video } from "./video.js";

/**
 * Each video's `videoN` number (see {@link parseVideoIdFromDataset}), in list
 * order; `null` for a video that has none.
 */
export type VideoNumbers = Array<number | null>;

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
 * {@link frameVideoIds} is its inverse, for the writers.
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
  return resolveVideoIds(
    videoIds,
    videos.map((video) =>
      parseVideoIdFromDataset(
        video.backend?.dataset ??
          (video.backendMetadata?.dataset as string | undefined) ??
          "",
      ),
    ),
  );
}

/** The rule of {@link buildVideoIdMap}, over `videoN` numbers instead of videos. */
function resolveVideoIds(
  videoIds: Set<number>,
  videoNumbers: VideoNumbers,
): Map<number, number> {
  if (!videoIds.size) return new Map();

  const maxId = Math.max(...Array.from(videoIds));
  if (
    videoIds.size === videoNumbers.length &&
    maxId === videoNumbers.length - 1
  ) {
    return new Map(videoNumbers.map((_, index) => [index, index]));
  }
  return indicesByNumber(videoNumbers);
}

/** Each number's video, the last one if several hold it. */
function indicesByNumber(videoNumbers: VideoNumbers): Map<number, number> {
  const map = new Map<number, number>();
  videoNumbers.forEach((number, index) => {
    if (number != null) map.set(number, index);
  });
  return map;
}

/**
 * Whether a 2.x `/frames` table holding these `[stored id, list index]` rows reads
 * back as those list indices, under a `videos_json` with these numbers.
 */
export function videoIdsReadBack(
  rows: Array<[id: number, index: number]>,
  videoNumbers: VideoNumbers,
): boolean {
  const map = resolveVideoIds(new Set(rows.map(([id]) => id)), videoNumbers);
  return rows.every(([id, index]) => (map.get(id) ?? id) === index);
}

/**
 * The id to store in a 2.x `/frames` table for each video, the inverse of
 * {@link buildVideoIdMap}: what a reader turns back into that video's list index.
 *
 * All list indices if those read back: every video has frames (so the ids are exactly
 * `0..n-1`, which readers take as list indices), or no video's number is in the way.
 * Otherwise each video with frames keeps its list index if that still reads back, and
 * takes its number if not. Python's `_written_video_id_map` always stores the number,
 * which goes wrong when every video has frames and the numbers are a shuffle of
 * `0..n-1`: they read as list indices.
 *
 * @param usedIndices The list index of every video with a row in the table.
 * @param videoNumbers The numbers in the `videos_json` written with the table.
 * @returns One id per video. An index past the list (none in a valid table) has none.
 * @throws Error if a video's frames can't read back under any id: not every video has
 *   frames, so readers go by the numbers, and another video holds the only id it has.
 */
export function frameVideoIds(
  usedIndices: Set<number>,
  videoNumbers: VideoNumbers,
): number[] {
  const listIndices = videoNumbers.map((_, index) => index);
  const asListIndices = Array.from(usedIndices, (index): [number, number] => [
    index,
    index,
  ]);
  if (videoIdsReadBack(asListIndices, videoNumbers)) return listIndices;

  // Not every video has frames (those ids would have read back), so this is what
  // readers go by, whichever ids are chosen here.
  const byNumber = indicesByNumber(videoNumbers);
  return videoNumbers.map((number, index) => {
    if (!usedIndices.has(index)) return number ?? index;
    if ((byNumber.get(index) ?? index) === index) return index;
    if (number != null && byNumber.get(number) === index) return number;
    const id = number ?? index;
    throw new Error(
      `Cannot save: the frames of video ${index} would load on video ${byNumber.get(id)}, ` +
        `because both are stored under id ${id} and not every video has frames. ` +
        "Nothing was written. Remove the videos that have no frames, or open the " +
        "videos so embedded ones can be renumbered, then save again.",
    );
  });
}

/** Parse the integer id from a `"videoN/..."` HDF5 dataset path, or `null`. */
export function parseVideoIdFromDataset(dataset: string): number | null {
  const match = /^video(\d+)(?:\/|$)/.exec(dataset);
  return match ? Number(match[1]) : null;
}
