import {
  frameVideoIds,
  parseVideoIdFromDataset,
  type VideoNumbers,
} from "../../model/video-id-map.js";
import { parseJsonEntry } from "./parsers.js";
import type { LabelTable, LabelTableUpdate } from "./write.js";

/** The `videoN` number of each `videos_json` entry, as held in a file. */
export function videoNumbersOfEntries(entries: unknown): VideoNumbers {
  const list = Array.isArray(entries)
    ? entries
    : ArrayBuffer.isView(entries)
      ? Array.from(entries as unknown as ArrayLike<unknown>)
      : [entries];
  return list.map((entry) => {
    const parsed = parseJsonEntry(entry) as {
      backend?: { dataset?: unknown };
    } | null;
    const dataset = parsed?.backend?.dataset;
    return typeof dataset === "string"
      ? parseVideoIdFromDataset(dataset)
      : null;
  });
}

/**
 * `update` (from `buildLabelTableUpdate`, whose tables hold list indices) with
 * `frames` and `negative_frames` holding what reads back as those indices under
 * the `videos_json` of the file being patched, which an in-place save leaves as it
 * is. That file's numbers, not the ones held in memory: a full re-save renumbers
 * embedded videos on disk and leaves the `Video` objects as they were.
 */
export function withFrameVideoIds(
  update: LabelTableUpdate,
  videoNumbers: VideoNumbers,
): LabelTableUpdate {
  const videoColumn = update.frames.fields.indexOf("video");
  const ids = frameVideoIds(
    new Set(update.frames.rows.map((row) => row[videoColumn])),
    videoNumbers,
  );
  if (ids.every((id, index) => id === index)) return update;

  const rewrite = (table: LabelTable, field: string): LabelTable => {
    const column = table.fields.indexOf(field);
    return {
      ...table,
      rows: table.rows.map((row) => {
        const out = [...row];
        out[column] = ids[row[column]] ?? row[column];
        return out;
      }),
    };
  };
  return {
    ...update,
    frames: rewrite(update.frames, "video"),
    negativeFrames:
      update.negativeFrames && rewrite(update.negativeFrames, "video_id"),
  };
}
