/**
 * Frame-spanning event annotations (SLP format 2.6+). Port of Python sleap-io's
 * `sleap_io/model/event.py`.
 *
 * Unlike every other annotation (`Instance`, `Centroid`, `BoundingBox`, ...),
 * which is strictly per-frame and lives on one `LabeledFrame`, an `Event` has a
 * temporal extent: a `(video, startFrame, endFrame, type)` interval. Events model
 * anything with a duration (behavior bouts, stimulus epochs, review flags) and are
 * stored on `Labels.events`, not on any one frame.
 *
 * - `EventType`: a catalog / controlled-vocabulary entry (the "ethogram"),
 *   lightweight and matched by name across files, like `Track` / `Identity`.
 * - `Event`: abstract base with the interval, participants and metadata.
 * - `UserEvent`: a human-annotated event (ground truth).
 * - `PredictedEvent`: a model-predicted event with optional confidence score(s).
 *
 * `startFrame` and `endFrame` are both INCLUSIVE: an event covers every frame in
 * `[startFrame, endFrame]` and spans `endFrame - startFrame + 1` frames. An event
 * with `endFrame === startFrame` (or no `endFrame`) is instantaneous.
 */
import type { Identity } from "./identity.js";
import type { Track } from "./instance.js";
import type { Video } from "./video.js";

/** Who an event is about, or directed at: a within-video `Track` or a cross-video `Identity`. */
export type EventParticipant = Track | Identity;

/**
 * A type of event in the catalog (controlled vocabulary). Referenced by `Event`s
 * and matched across separately-loaded files and merges by `name`.
 */
export class EventType {
  name: string;
  description: string;
  /** Arbitrary string-keyed, string-valued metadata (e.g. `{ color: "#e6194b" }`). */
  metadata: Record<string, string>;

  constructor(options?: {
    name?: string;
    description?: string;
    metadata?: Record<string, string>;
  }) {
    this.name = options?.name ?? "";
    this.description = options?.description ?? "";
    this.metadata = options?.metadata ?? {};
  }

  /** Does this type match `other`, by name (default) or by object identity? */
  matches(other: EventType, method: "name" | "identity" = "name"): boolean {
    if (method === "name") return this.name === other.name;
    if (method === "identity") return this === other;
    throw new Error(`Unknown matching method: ${method}`);
  }
}

/** Constructor options shared by `UserEvent` and `PredictedEvent`. */
export interface EventOptions {
  /** The catalog entry; a bare string is promoted to `new EventType({ name })`. */
  type: EventType | string;
  /** The video the event occurs in. `null` only when read from a file that has none. */
  video: Video | null;
  /** First frame of the event (inclusive). */
  startFrame: number;
  /** Last frame of the event (inclusive). Defaults to `startFrame`. */
  endFrame?: number | null;
  /** Who the event is about. `null` means a frame-level event with no individual. */
  subject?: EventParticipant | null;
  /** Who the event is directed at. `null` means non-directed / individual. */
  target?: EventParticipant | null;
  /** Optional human-readable name for this specific event. */
  name?: string;
  /** Annotation source identifier. */
  source?: string;
  /** Arbitrary string-keyed, string-valued metadata. */
  metadata?: Record<string, string>;
}

/**
 * A labeled event spanning a range of frames in a video. Abstract: use
 * {@link UserEvent} or {@link PredictedEvent}.
 */
export abstract class Event {
  type: EventType;
  video: Video | null;
  startFrame: number;
  endFrame: number;
  subject: EventParticipant | null;
  target: EventParticipant | null;
  name: string;
  source: string;
  metadata: Record<string, string>;

  constructor(options: EventOptions) {
    this.type =
      typeof options.type === "string"
        ? new EventType({ name: options.type })
        : options.type;
    this.video = options.video;
    this.startFrame = Math.trunc(options.startFrame);
    this.endFrame =
      options.endFrame === undefined || options.endFrame === null
        ? this.startFrame
        : Math.trunc(options.endFrame);
    if (this.endFrame < this.startFrame) {
      throw new Error(
        `Expected endFrame >= startFrame, got startFrame=${this.startFrame}, endFrame=${this.endFrame}.`,
      );
    }
    this.subject = options.subject ?? null;
    this.target = options.target ?? null;
    this.name = options.name ?? "";
    this.source = options.source ?? "";
    this.metadata = options.metadata ?? {};
  }

  /** Is this event a prediction? */
  abstract get isPredicted(): boolean;

  /** Is this event directed at a `target` (vs. non-directed / self)? */
  get isDirected(): boolean {
    return this.target !== null;
  }

  /** Does this event span a single frame? */
  get isInstantaneous(): boolean {
    return this.endFrame === this.startFrame;
  }

  /** Number of frames spanned (inclusive). */
  get nFrames(): number {
    return this.endFrame - this.startFrame + 1;
  }

  /** Does `frameIdx` fall within this event's inclusive span? */
  contains(frameIdx: number): boolean {
    return this.startFrame <= frameIdx && frameIdx <= this.endFrame;
  }

  /**
   * Does this event overlap `other` in the same video? Events in different videos
   * never overlap, regardless of their frame indices.
   */
  overlaps(other: Event): boolean {
    if (this.video !== other.video) return false;
    return (
      this.startFrame <= other.endFrame && other.startFrame <= this.endFrame
    );
  }
}

/** A human-annotated event (ground truth). */
export class UserEvent extends Event {
  get isPredicted(): boolean {
    return false;
  }
}

/**
 * A model-predicted event. Its two confidence fields are independent and optional:
 * a predictor sets whichever it produces, and neither is derived from the other.
 */
export class PredictedEvent extends Event {
  /** Framewise confidence, one value per frame of `[startFrame, endFrame]`. */
  scores: Float32Array | null;
  /** Event-level confidence. */
  score: number | null;

  constructor(
    options: EventOptions & {
      scores?: ArrayLike<number> | null;
      score?: number | null;
    },
  ) {
    super(options);
    this.scores =
      options.scores === undefined || options.scores === null
        ? null
        : Float32Array.from(options.scores);
    this.score = options.score ?? null;
    if (this.scores !== null && this.scores.length !== this.nFrames) {
      throw new Error(
        `Framewise event scores must have length nFrames (${this.nFrames}), got ${this.scores.length}.`,
      );
    }
  }

  get isPredicted(): boolean {
    return true;
  }
}

/**
 * A copy of `event` with some of its references replaced (a copied catalog, a
 * restored video, a merged track...). Metadata and scores are copied too.
 */
export function cloneEvent(
  event: Event,
  refs?: Partial<Pick<EventOptions, "type" | "video" | "subject" | "target">>,
): Event {
  const options: EventOptions = {
    type: refs?.type ?? event.type,
    video: refs?.video !== undefined ? refs.video : event.video,
    startFrame: event.startFrame,
    endFrame: event.endFrame,
    subject: refs?.subject !== undefined ? refs.subject : event.subject,
    target: refs?.target !== undefined ? refs.target : event.target,
    name: event.name,
    source: event.source,
    metadata: { ...event.metadata },
  };
  return event instanceof PredictedEvent
    ? new PredictedEvent({
        ...options,
        scores: event.scores,
        score: event.score,
      })
    : new UserEvent(options);
}
