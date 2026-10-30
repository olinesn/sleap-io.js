/**
 * Shared reconstruction of frame-spanning events from the `/event_types` and
 * `/events` groups (SLP format 2.6+), the same layout Python sleap-io reads and
 * writes (`read_event_types` / `read_events` in `sleap_io/io/slp.py`).
 *
 * Both the synchronous eager/lazy reader (`read.ts`) and the async streaming reader
 * (`read-streaming.ts`) gather the same per-field columns; this turns those columns
 * into `EventType` / `Event` objects so the construction logic lives in one place.
 */
import {
  type Event,
  type EventParticipant,
  EventType,
  PredictedEvent,
  UserEvent,
} from "../../model/event.js";
import type { Identity } from "../../model/identity.js";
import type { Track } from "../../model/instance.js";
import type { Video } from "../../model/video.js";

/** Participant kinds in the `subject_kind` / `target_kind` columns. */
export const EVENT_PARTICIPANT_NONE = 0;
export const EVENT_PARTICIPANT_TRACK = 1;
export const EVENT_PARTICIPANT_IDENTITY = 2;

/** The decoded columns of the `/event_types` group. */
export interface EventTypeColumns {
  name: string[];
  description?: string[];
  metaOwner?: ArrayLike<number | bigint>;
  metaKey?: string[];
  metaVal?: string[];
}

/** The decoded columns of the `/events` group. Numeric columns may be int64. */
export interface EventColumns {
  video: ArrayLike<number | bigint>;
  startFrame: ArrayLike<number | bigint>;
  endFrame: ArrayLike<number | bigint>;
  type: ArrayLike<number | bigint>;
  subjectKind: ArrayLike<number | bigint>;
  targetKind: ArrayLike<number | bigint>;
  subjectIdx: ArrayLike<number | bigint>;
  targetIdx: ArrayLike<number | bigint>;
  /** Plain ints, or h5wasm's reading of an HDF5 bool. */
  isPredicted: ArrayLike<number | bigint | boolean>;
  name?: string[];
  source?: string[];
  score?: ArrayLike<number>;
  scores?: ArrayLike<number>;
  scoreOffsets?: ArrayLike<number | bigint>;
  metaOwner?: ArrayLike<number | bigint>;
  metaKey?: string[];
  metaVal?: string[];
}

/** Rebuild per-owner metadata dicts from an entity-attribute-value table. */
function metadataFromEav(
  n: number,
  owners: ArrayLike<number | bigint> | undefined,
  keys: string[] | undefined,
  vals: string[] | undefined,
): Record<string, string>[] {
  const metadata: Record<string, string>[] = Array.from(
    { length: n },
    () => ({}),
  );
  if (!owners || !keys || !vals) return metadata;
  for (let i = 0; i < owners.length; i++) {
    const owner = metadata[Number(owners[i])];
    if (owner) owner[keys[i]] = vals[i];
  }
  return metadata;
}

/** Build the event-type catalog from the `/event_types` columns. */
export function buildEventTypes(cols: EventTypeColumns): EventType[] {
  const metadata = metadataFromEav(
    cols.name.length,
    cols.metaOwner,
    cols.metaKey,
    cols.metaVal,
  );
  return cols.name.map(
    (name, i) =>
      new EventType({
        name,
        description: cols.description?.[i] ?? "",
        metadata: metadata[i],
      }),
  );
}

/**
 * Build events from the `/events` columns. A row whose span is impossible
 * (`end_frame < start_frame`) is skipped with a warning rather than failing the
 * whole file, so one bad row never stops a project from opening.
 */
export function buildEvents(
  cols: EventColumns,
  videos: Video[],
  eventTypes: EventType[],
  tracks: Track[],
  identities: Identity[],
): Event[] {
  const n = cols.video.length;
  const metadata = metadataFromEav(
    n,
    cols.metaOwner,
    cols.metaKey,
    cols.metaVal,
  );

  const participant = (kind: number, idx: number): EventParticipant | null => {
    if (kind === EVENT_PARTICIPANT_TRACK) return tracks[idx] ?? null;
    if (kind === EVENT_PARTICIPANT_IDENTITY) return identities[idx] ?? null;
    return null;
  };

  const events: Event[] = [];
  for (let i = 0; i < n; i++) {
    const start = Number(cols.startFrame[i]);
    const end = Number(cols.endFrame[i]);
    if (end < start) {
      console.warn(
        `[sleap-io] skipping event row ${i}: end_frame ${end} < start_frame ${start}`,
      );
      continue;
    }
    const options = {
      // An event with no catalog entry should not happen (the catalog is collected
      // on save); fall back to an empty type, as the Python reader does.
      type: eventTypes[Number(cols.type[i])] ?? new EventType({ name: "" }),
      video: videos[Number(cols.video[i])] ?? null,
      startFrame: start,
      endFrame: end,
      subject: participant(
        Number(cols.subjectKind[i]),
        Number(cols.subjectIdx[i]),
      ),
      target: participant(
        Number(cols.targetKind[i]),
        Number(cols.targetIdx[i]),
      ),
      name: cols.name?.[i] ?? "",
      source: cols.source?.[i] ?? "",
      metadata: metadata[i],
    };
    if (!cols.isPredicted[i]) {
      events.push(new UserEvent(options));
      continue;
    }
    let scores: ArrayLike<number> | null = null;
    if (cols.scores && cols.scoreOffsets) {
      const lo = Number(cols.scoreOffsets[i]);
      const hi = Number(cols.scoreOffsets[i + 1]);
      if (hi > lo) {
        scores = Array.prototype.slice.call(cols.scores, lo, hi) as number[];
      }
    }
    const rawScore = cols.score?.[i];
    const score =
      rawScore === undefined || Number.isNaN(Number(rawScore))
        ? null
        : Number(rawScore);
    events.push(new PredictedEvent({ ...options, scores, score }));
  }
  return events;
}

/** The columns `/event_types` holds for this catalog. */
export function eventTypeColumnsOf(eventTypes: EventType[]): EventTypeColumns {
  const metaOwner: number[] = [];
  const metaKey: string[] = [];
  const metaVal: string[] = [];
  eventTypes.forEach((t, i) => {
    for (const [k, v] of Object.entries(t.metadata)) {
      metaOwner.push(i);
      metaKey.push(String(k));
      metaVal.push(String(v));
    }
  });
  return {
    name: eventTypes.map((t) => t.name),
    description: eventTypes.some((t) => t.description)
      ? eventTypes.map((t) => t.description)
      : undefined,
    metaOwner: metaOwner.length ? metaOwner : undefined,
    metaKey: metaOwner.length ? metaKey : undefined,
    metaVal: metaOwner.length ? metaVal : undefined,
  };
}

/**
 * The columns `/events` holds for these events. A reference that is not in its
 * catalog is written as -1 (kind 0 for a participant), like the Python writer.
 * Optional columns are present only when some event uses them.
 */
export function eventColumnsOf(
  events: Event[],
  videos: Video[],
  eventTypes: EventType[],
  tracks: Track[],
  identities: Identity[],
): EventColumns {
  const n = events.length;
  const indexOf = <T>(list: T[]) =>
    new Map<unknown, number>(list.map((x, i) => [x, i]));
  const videoIdx = indexOf(videos);
  const typeIdx = indexOf(eventTypes);
  const trackIdx = indexOf(tracks);
  const identityIdx = indexOf(identities);
  const participant = (p: EventParticipant | null): [number, number] => {
    if (p === null) return [EVENT_PARTICIPANT_NONE, -1];
    const t = trackIdx.get(p);
    if (t !== undefined) return [EVENT_PARTICIPANT_TRACK, t];
    const id = identityIdx.get(p);
    if (id !== undefined) return [EVENT_PARTICIPANT_IDENTITY, id];
    return [EVENT_PARTICIPANT_NONE, -1];
  };

  const cols = {
    video: new Array<number>(n),
    startFrame: new Array<number>(n),
    endFrame: new Array<number>(n),
    type: new Array<number>(n),
    subjectKind: new Array<number>(n),
    targetKind: new Array<number>(n),
    subjectIdx: new Array<number>(n),
    targetIdx: new Array<number>(n),
    isPredicted: new Array<number>(n),
    name: new Array<string>(n),
    source: new Array<string>(n),
  };
  const score = new Array<number>(n).fill(Number.NaN);
  let anyScore = false;
  const scores: number[] = [];
  const scoreOffsets = new Array<number>(n + 1).fill(0);
  const metaOwner: number[] = [];
  const metaKey: string[] = [];
  const metaVal: string[] = [];

  events.forEach((e, i) => {
    cols.video[i] = e.video ? (videoIdx.get(e.video) ?? -1) : -1;
    cols.startFrame[i] = e.startFrame;
    cols.endFrame[i] = e.endFrame;
    cols.type[i] = typeIdx.get(e.type) ?? -1;
    [cols.subjectKind[i], cols.subjectIdx[i]] = participant(e.subject);
    [cols.targetKind[i], cols.targetIdx[i]] = participant(e.target);
    cols.name[i] = e.name;
    cols.source[i] = e.source;
    const predicted = e instanceof PredictedEvent ? e : null;
    cols.isPredicted[i] = predicted ? 1 : 0;
    if (predicted?.score !== null && predicted?.score !== undefined) {
      score[i] = predicted.score;
      anyScore = true;
    }
    if (predicted?.scores) {
      for (let k = 0; k < predicted.scores.length; k++) {
        scores.push(predicted.scores[k]);
      }
    }
    scoreOffsets[i + 1] = scores.length;
    for (const [k, v] of Object.entries(e.metadata)) {
      metaOwner.push(i);
      metaKey.push(String(k));
      metaVal.push(String(v));
    }
  });

  return {
    ...cols,
    score: anyScore ? score : undefined,
    scores: scores.length ? scores : undefined,
    scoreOffsets: scores.length ? scoreOffsets : undefined,
    metaOwner: metaOwner.length ? metaOwner : undefined,
    metaKey: metaOwner.length ? metaKey : undefined,
    metaVal: metaOwner.length ? metaVal : undefined,
  };
}

/**
 * One string per catalog entry and per event, identical exactly when the two hold
 * the same data. Works from the columns, so what is on disk and what is in memory
 * can be compared directly: the in-place saver uses it to tell whether events changed.
 */
export function eventSignatures(
  types: EventTypeColumns | null,
  events: EventColumns | null,
): string[] {
  const meta = (
    n: number,
    owners: ArrayLike<number | bigint> | undefined,
    keys: string[] | undefined,
    vals: string[] | undefined,
  ) =>
    metadataFromEav(n, owners, keys, vals).map((m) =>
      Object.keys(m)
        .sort()
        .map((k) => [k, m[k]]),
    );
  const out: string[] = [];
  if (types) {
    const m = meta(
      types.name.length,
      types.metaOwner,
      types.metaKey,
      types.metaVal,
    );
    types.name.forEach((name, i) => {
      out.push(JSON.stringify(["t", name, types.description?.[i] ?? "", m[i]]));
    });
  }
  if (events) {
    const n = events.video.length;
    const m = meta(n, events.metaOwner, events.metaKey, events.metaVal);
    for (let i = 0; i < n; i++) {
      const kind = (x: ArrayLike<number | bigint>) => Number(x[i]);
      const rawScore = events.score?.[i];
      const score =
        rawScore === undefined || Number.isNaN(Number(rawScore))
          ? null
          : Number(rawScore);
      const lo = Number(events.scoreOffsets?.[i] ?? 0);
      const hi = Number(events.scoreOffsets?.[i + 1] ?? 0);
      const trace =
        events.scores && hi > lo
          ? Array.prototype.slice.call(events.scores, lo, hi).map(Number)
          : null;
      out.push(
        JSON.stringify([
          "e",
          kind(events.video),
          kind(events.startFrame),
          kind(events.endFrame),
          kind(events.type),
          kind(events.subjectKind),
          kind(events.subjectIdx),
          kind(events.targetKind),
          kind(events.targetIdx),
          Boolean(events.isPredicted[i]),
          score,
          trace,
          events.name?.[i] ?? "",
          events.source?.[i] ?? "",
          m[i],
        ]),
      );
    }
  }
  return out;
}
