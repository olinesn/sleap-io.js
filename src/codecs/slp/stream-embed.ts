/**
 * Main-thread half of the streaming NEW-embed: read each frame a
 * {@link SerializableEncodeEntry} embeds and hand it to the streaming writer's
 * worker, which PNG-encodes it and appends it straight to the destination file
 * (`embed-encode-worker.ts`). The UI thread only reads frames (an mp4 already
 * decodes in its own worker) and passes them on, so a long export never holds
 * it, and only a few frames are ever in memory at once.
 *
 * Order of use: write the structure with `saveSlpStructureToBytes`, open the
 * written file with `StreamingH5Writer.openWrite` (or `openAppend` when raw
 * entries need copying too), then call {@link appendEncodedVideos} with the
 * `encodeEntries` of `buildSerializableEmbedPlan(..., { allowEncode: true })`.
 *
 * @module
 */
import type { Labels } from "../../model/labels.js";
import type { Video } from "../../model/video.js";
import type { VideoBackend, VideoFrame } from "../../video/backend.js";
import { GrayscaleVideoBackend } from "../../video/grayscale-backend.js";
import {
  decodeEncoded,
  isEncodedBytes,
  isImageBitmapLike,
  isImageDataLike,
} from "../../video/image-decode.js";
import {
  ExportCancelled,
  type SerializableEncodeEntry,
  type SlpWriteOptions,
} from "./write.js";

/** One frame handed to the worker. Its buffer or bitmap is transferred. */
export type EncodeFramePayload =
  | { kind: "bitmap"; bitmap: ImageBitmap }
  | {
      kind: "pixels";
      data: Uint8Array;
      width: number;
      height: number;
      channels: number;
      /** RGBA by construction (an `ImageData`): the worker drops the alpha. */
      rgba: boolean;
    }
  | { kind: "png"; bytes: Uint8Array };

/** Where encoded videos are written; `StreamingH5Writer` implements it. */
export interface EncodedVideoSink {
  beginEncodedVideo(
    entry: SerializableEncodeEntry,
    collapseToGray: boolean,
  ): Promise<void>;
  appendEncodedFrame(
    frameIdx: number,
    frame: EncodeFramePayload,
  ): Promise<void>;
  endEncodedVideo(): Promise<{ framesWritten: number }>;
}

export type AppendEncodedVideosOptions = {
  /**
   * As {@link SlpWriteOptions.onProgress}: called with frames done of all
   * frames (`message` "Embedding frames"); return `false` to cancel, which
   * throws {@link ExportCancelled}.
   */
  onProgress?: SlpWriteOptions["onProgress"];
  /** Frames handed to the worker but not yet written (default 4). */
  maxInFlight?: number;
  /**
   * Supply a video's frames yourself — e.g. PNGs a native tool (ffmpeg) is
   * writing — instead of decoding them from its backend. Called once per video
   * with the frame numbers it will embed, in order; return a reader for them,
   * or null to decode from the backend. Encoded PNG bytes are stored as-is.
   */
  frameReader?: (
    video: Video,
    frameNumbers: readonly number[],
  ) => Promise<ExternalFrameReader | null>;
};

/** Frames of one video from outside its backend (see `frameReader`). */
export interface ExternalFrameReader {
  /** The frame (e.g. PNG bytes), asked for in the order announced. */
  read(frameIdx: number): Promise<VideoFrame | null>;
  /** Called when the video is done, or the export fails or is cancelled. */
  close(): Promise<void> | void;
}

// As write.ts's EMBED_READ_ATTEMPTS: a pinned read can still come back empty
// once while another read of the same decoder is cancelled.
const READ_ATTEMPTS = 3;
const READ_OPTS = { pinned: true, prefetch: false } as const;

function videoName(video: Video): string {
  return Array.isArray(video.filename)
    ? (video.filename[0] ?? "")
    : video.filename;
}

/**
 * The backend to read frames from, and whether the worker must collapse them
 * to grayscale. A grayscale video's wrapper would collapse each frame on this
 * thread (a canvas readback per frame), so read its inner backend instead and
 * let the worker do it.
 */
async function frameSource(
  backend: VideoBackend,
  firstFrame: number | undefined,
): Promise<{ source: VideoBackend; collapseToGray: boolean }> {
  if (!(backend instanceof GrayscaleVideoBackend)) {
    return { source: backend, collapseToGray: false };
  }
  // Autodetect resolves on the wrapper's first read.
  if (backend.grayscale === null && firstFrame !== undefined) {
    await backend.getFrame(firstFrame, READ_OPTS);
  }
  return { source: backend.inner, collapseToGray: backend.grayscale === true };
}

function isPng(bytes: Uint8Array): boolean {
  return (
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  );
}

/** Turn a decoded frame into a payload the worker owns (copied/cloned). */
export async function toEncodeFramePayload(
  frame: VideoFrame,
  rawShape?: [number, number, number, number],
): Promise<EncodeFramePayload> {
  if (isImageBitmapLike(frame)) {
    // The backend may cache this bitmap and draw it again, so hand over a copy.
    return {
      kind: "bitmap",
      bitmap: await createImageBitmap(frame as unknown as ImageBitmap),
    };
  }
  if (isImageDataLike(frame)) {
    const f = frame as {
      data: Uint8Array | Uint8ClampedArray;
      width: number;
      height: number;
      channels?: number;
    };
    return {
      kind: "pixels",
      data: new Uint8Array(f.data),
      width: f.width,
      height: f.height,
      channels: f.channels ?? f.data.length / (f.width * f.height),
      rgba: f.channels === undefined,
    };
  }
  const bytes =
    frame instanceof ArrayBuffer
      ? new Uint8Array(frame)
      : (frame as unknown as Uint8Array);
  if (isEncodedBytes(bytes)) {
    if (isPng(bytes)) return { kind: "png", bytes: bytes.slice() };
    // Stored as PNG like every other frame, so re-encode (e.g. a JPEG).
    const img = await decodeEncoded(bytes);
    return {
      kind: "pixels",
      data: new Uint8Array(img.data),
      width: img.width,
      height: img.height,
      channels: 4,
      rgba: true,
    };
  }
  if (!rawShape) {
    throw new Error(
      "Can't embed raw pixel bytes from a backend with no resolved shape.",
    );
  }
  const [, height, width, channels] = rawShape;
  return {
    kind: "pixels",
    data: bytes.slice(),
    width,
    height,
    channels,
    rgba: false,
  };
}

async function readFrame(
  video: Video,
  source: VideoBackend,
  frameIdx: number,
): Promise<VideoFrame> {
  let frame: VideoFrame | null = null;
  for (let attempt = 0; attempt < READ_ATTEMPTS && !frame; attempt++) {
    frame = await source.getFrame(frameIdx, READ_OPTS);
  }
  if (!frame) {
    throw new Error(
      `Couldn't read frame ${frameIdx} from "${videoName(video)}" to embed. ` +
        "The package was not written.",
    );
  }
  return frame;
}

/**
 * Read, hand over and append every frame of each new-embed entry, one video at
 * a time, through `sink` (an open `StreamingH5Writer`). Throws if any frame
 * can't be read or written (the caller discards the partial file), and
 * {@link ExportCancelled} when `onProgress` returns `false`.
 */
export async function appendEncodedVideos(
  sink: EncodedVideoSink,
  labels: Labels,
  entries: SerializableEncodeEntry[],
  options: AppendEncodedVideosOptions = {},
): Promise<void> {
  const maxInFlight = Math.max(1, options.maxInFlight ?? 4);
  const total = entries.reduce((n, e) => n + e.frameNumbers.length, 0);
  let done = 0;

  for (const entry of entries) {
    const video = labels.videos[entry.videoIndex];
    const external =
      video && entry.frameNumbers.length > 0
        ? ((await options.frameReader?.(video, entry.frameNumbers)) ?? null)
        : null;
    try {
      await appendOneVideo(entry, video, external);
    } finally {
      await external?.close();
    }
  }

  async function appendOneVideo(
    entry: SerializableEncodeEntry,
    video: Video | undefined,
    external: ExternalFrameReader | null,
  ): Promise<void> {
    let source: VideoBackend | null = null;
    let collapseToGray = false;
    if (entry.frameNumbers.length > 0 && !external) {
      if (!video?.backend) {
        throw new Error(
          `Can't embed frames from "${video ? videoName(video) : `video ${entry.videoIndex}`}": its video isn't open.`,
        );
      }
      ({ source, collapseToGray } = await frameSource(
        video.backend,
        entry.frameNumbers[0],
      ));
    }
    await sink.beginEncodedVideo(entry, collapseToGray);

    const inFlight: Promise<void>[] = [];
    const settleOne = async () => {
      await inFlight.shift();
      done += 1;
      if (options.onProgress?.(done, total, "Embedding frames") === false) {
        throw new ExportCancelled();
      }
    };
    for (const frameIdx of entry.frameNumbers) {
      const frame = external
        ? await external.read(frameIdx)
        : await readFrame(video!, source!, frameIdx);
      if (!frame) {
        throw new Error(
          `Couldn't read frame ${frameIdx} from "${video ? videoName(video) : `video ${entry.videoIndex}`}" to embed. ` +
            "The package was not written.",
        );
      }
      const pending = sink.appendEncodedFrame(
        frameIdx,
        await toEncodeFramePayload(frame, source?.shape),
      );
      // Awaited in order below; this only keeps a failure that lands while an
      // earlier frame is awaited from surfacing as an unhandled rejection.
      pending.catch(() => {});
      inFlight.push(pending);
      if (inFlight.length >= maxInFlight) await settleOne();
    }
    while (inFlight.length > 0) await settleOne();

    const { framesWritten } = await sink.endEncodedVideo();
    if (framesWritten !== entry.frameNumbers.length) {
      throw new Error(
        `embedding video${entry.videoIndex}: wrote ${framesWritten} of ` +
          `${entry.frameNumbers.length} planned frame(s) - refusing to write a ` +
          "file with dropped images.",
      );
    }
  }
}
