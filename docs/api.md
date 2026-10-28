# API Reference

This document covers the public API exported from `src/index.ts`. All examples assume:
```ts
import {
  loadSlp,
  saveSlp,
  saveSlpToBytes,
  loadVideo,
  Video,
  Mp4BoxVideoBackend,
  Labels,
  LabeledFrame,
  Instance,
  PredictedInstance,
  Skeleton,
  Track,
  LabelsSet,
  SuggestionFrame,
  ROI,
  UserROI,
  PredictedROI,
  SegmentationMask,
  UserSegmentationMask,
  PredictedSegmentationMask,
  BoundingBox,
  UserBoundingBox,
  PredictedBoundingBox,
  Centroid,
  UserCentroid,
  PredictedCentroid,
  getCentroidSkeleton,
  CENTROID_SKELETON,
  LabelImage,
  UserLabelImage,
  PredictedLabelImage,
  normalizeLabelIds,
  readTrackMateCsv,
  loadTrackMate,
  isTrackMateFile,
  loadUltralytics,
  saveUltralytics,
  readLabels as readUltralytics,
  writeLabels as writeUltralytics,
  readLabelsSet as readUltralyticsSet,
  parseLabelFile,
  detectLineFormat,
  loadJabs,
  JABS_DEFAULT_SKELETON,
  makeJabsDefaultSkeleton,
  predictionToInstance,
  staticObjectToRoi,
  makeSimpleSkeleton,
  Identity,
  Instance3D,
  PredictedInstance3D,
  Camera,
  CameraGroup,
  RecordingSession,
  toDict,
  fromDict,
  toNumpy,
  fromNumpy,
  roisToGeoJSON,
  roisFromGeoJSON,
  writeGeoJSON,
  readGeoJSON,
} from "@talmolab/sleap-io.js";
```

## Core I/O

### `loadSlp(source, options)`
Read `.slp` from a path, URL, `File`, `FileSystemFileHandle`, or byte buffer.

```ts
const labels = await loadSlp("/data/session.slp", {
  openVideos: false,
  h5: { stream: "auto", filenameHint: "session.slp" },
});
```

- `source`: string path/URL, `File`, `FileSystemFileHandle`, `Uint8Array`, or `ArrayBuffer`.
- `options.openVideos` (default `true`): set `false` to skip opening video backends.
- `options.h5.stream`: `"auto" | "range" | "download"` (browser URL streaming).
- `options.h5.filenameHint`: helps name temporary files.
- `options.lazy` (default `false`): use lazy loading for on-demand frame materialization.

### `saveSlp(labels, filename, options)`
Write `.slp` (Node/Electron).

```ts
await saveSlp(labels, "/tmp/roundtrip.slp", {
  embed: false,
  restoreOriginalVideos: true,
});
```

- `options.embed`: which frame images to embed, as Python sleap-io's `embed`:
  - `true` / `"all"`: every labeled frame and every suggestion (`"all+suggestions"` is an alias)
  - `"user"`: user-labeled frames (user instances, negative frames, user centroids/boxes/masks/ROIs)
  - `"suggestions"` / `"user+suggestions"`
  - an explicit list of `[video, frameIdx]` pairs: exactly those frames
  - `"source"`: no images; restore source video paths before writing
  - `false` (default): no embedding (already-embedded images are still kept)
- `options.embedAllVideos` (default `true`, as Python's `embed_all_videos`): when embedding, a video with no frames to embed becomes an empty embedded video referencing the package (keeping its shape and source video), so the package is portable. `false` leaves such videos as external references.
- `options.onProgress(current, total, message)`: called as frames are read (`"Embedding frames"`) and written (`"Writing frames"`). Return `false` to cancel: the save throws `ExportCancelled`.
- `options.restoreOriginalVideos` (default `true`): keep original video paths.

### `saveSlpToBytes(labels, options)`
Serialize Labels to SLP format and return the bytes. Works in both Node.js and browser environments.

```ts
const bytes: Uint8Array = await saveSlpToBytes(labels, { embed: false });
```

- `labels`: `Labels` object to serialize.
- `options`: the same `embed`, `embedAllVideos`, `onProgress` and `restoreOriginalVideos` options as `saveSlp`.
- Returns `Promise<Uint8Array>`.

### `loadVideo(source, options)`
Open a `Video` with an appropriate backend.

```ts
// From file path (Node.js) or URL (browser)
const video = await loadVideo("/data/movie.mp4", { openBackend: true });
const frame0 = await video.getFrame(0);
video.close();

// From File object (browser)
const video = await loadVideo(fileInput.files[0]);
```

- `source`: `string` (path or URL) or `File` (browser).
- `options.dataset`: dataset path for embedded HDF5 sources.
- `options.openBackend` (default `true`): set `false` for deferred open.
- `options.backend`: explicit backend type (`"mp4box"`, `"mediabunny"`, `"media"`, `"hdf5"`).

## Video Backends

### `Video`
`Video` wraps a backend and exposes `shape`, `fps`, `getFrame`, `getFrameTimes`, and `close`.

```ts
const video = await loadVideo("/data/movie.mp4");
const timestamps = await video.getFrameTimes(); // number[] | null

// shape and fps have setters
video.shape = [100, 480, 640, 3];
video.fps = 30;
```

- `shape` and `fps` have both getters and setters.
- `getFrame(index, signal?)` accepts an optional `AbortSignal` to cancel in-flight decodes.
- `getFrameTimes()` returns `null` if the backend does not implement it.

#### Virtual crops

`Video.crop` returns a virtual, on-read cropped view (no pixels are copied or
re-encoded). See [Virtual Cropping](cropping.md) for the full guide.

```ts
const view = full.crop([320, 200, 576, 456]); // [x1, y1, x2, y2], x2/y2 exclusive
view.shape; // [frames, 256, 256, channels]
view.cropRect; // [320, 200, 576, 456]
view.isCropped; // true
view.cropFill; // 0
view.sourceVideo === full; // true

// Region specs (exactly one): explicit rect, bbox, roi, or center+size.
full.crop(null, { bbox: [x1, y1, x2, y2] });
full.crop(null, { roi: myRoi, margin: 8 });
full.crop(null, { center: [cx, cy], size: [128, 128], fill: 0 });

// Coordinate helpers (identity passthrough on an uncropped video):
const local = view.toCropCoords(sourcePts); // subtract (x1, y1)
const back = view.toSourceCoords(local); // add (x1, y1)
```

- `Video.crop(crop?, opts?)` — one region spec across `crop` / `opts.bbox` /
  `opts.roi` / (`opts.center` + `opts.size`); plus `opts.margin`, `opts.fill`
  (out-of-bounds pad value, default `0`), and `opts.shareDecode` (default `true`,
  reuse the source decoder for mosaics).
- `Video.fromCrop(video, crop?, opts?)` — static wrapper around `video.crop(...)`
  (`video` must be a `Video`; path strings are unsupported in the JS port).
- Getters: `isCropped`, `cropRect` (`[x1,y1,x2,y2] | null`), `cropFill`.
- `toCropCoords(pts)` / `toSourceCoords(pts)` — NaN-preserving copies; accept
  `[x,y]` pairs or a flat interleaved buffer.
- Standalone primitives: `cropPoints`, `uncropPoints`, `cropFrame`, and the
  `CropVideoBackend` class are all exported (browser-safe).

### `Mp4BoxVideoBackend`
Browser-only backend for `.mp4` with WebCodecs + mp4box. Supports range requests when the server honors `Range` and falls back to full download.

```ts
// From URL
const backend = new Mp4BoxVideoBackend("/data/movie.mp4");

// From File or Blob (browser)
const backend = new Mp4BoxVideoBackend(file);

const video = new Video({ filename: "/data/movie.mp4", backend });
const t = await video.getFrameTimes();
```

Constructor: `new Mp4BoxVideoBackend(source: string | File | Blob, options?)`

Notes:
- Requires WebCodecs (`VideoDecoder`, `EncodedVideoChunk`).
- Uses a `Range: bytes=0-0` probe to detect range support (no longer requires HEAD).
- Provides precise frame timestamps via `getFrameTimes()`.
- `getFrame()` uses an async queue internally to prevent race conditions, and accepts an optional `AbortSignal`.

### `Hdf5VideoBackend`
Used for `.slp`, `.h5`, or `.hdf5` sources (embedded frames).

- Selected automatically by `loadVideo()` for HDF5 inputs.
- Decodes embedded PNG/JPEG frames in browsers; raw frames require `shape` + `channelOrder`.

### `MediaVideoBackend`
Browser fallback when WebCodecs/mp4box is unavailable.

- Uses `HTMLVideoElement` + canvas.
- No `getFrameTimes()` support.

## Labels & Model Classes

Key types:

- `Labels`, `LabeledFrame`, `Instance`, `PredictedInstance`
- `Skeleton`, `Track`
- `Video`, `SuggestionFrame`, `LabelsSet`
- `Camera`, `CameraGroup`, `RecordingSession`, `InstanceGroup` (camera/3D utilities)
- `Identity` (ground-truth animal identity across sessions)
- `Instance3D`, `PredictedInstance3D` (3D keypoint data)
- `LazyDataStore`, `LazyFrameList` (lazy loading)
- `ROI`, `UserROI`, `PredictedROI` (spatial annotations)
- `SegmentationMask`, `UserSegmentationMask`, `PredictedSegmentationMask` (binary masks)
- `BoundingBox`, `UserBoundingBox`, `PredictedBoundingBox` (detection/tracking)
- `Centroid`, `UserCentroid`, `PredictedCentroid` (point detections/tracking)
- `LabelImage`, `UserLabelImage`, `PredictedLabelImage` (dense instance segmentation)
- `Point` has fields `{ xy, visible, complete, score? }` where `score` is an optional confidence value.
- `LabeledFrame.isNegative` (`boolean`, default `false`): marks negative-annotated frames.
- `SuggestionFrame.group` (`string`, default `"default"`): the suggestion group name.

```ts
const skeleton = new Skeleton(["nose", "tail"]);
const inst = Instance.fromArray([[10, 20], [30, 40]], skeleton);
const video = new Video({ filename: "/data/movie.mp4" });
const frame = new LabeledFrame({ video, frameIdx: 0, instances: [inst] });
const labels = new Labels({ labeledFrames: [frame], skeletons: [skeleton], videos: [video] });

// Add a video with deduplication
labels.addVideo(video); // no-op if already present
```

### Frame and track index lookups

`Labels` builds lazy `Map` indices for O(1) frame and track lookups, replacing linear scans through `labeledFrames`. The frame-aware `find()` / `getCentroids()` / `getBboxes()` / `getMasks()` / `getLabelImages()` / `getRois()` methods use these indices internally when both `video` and `frameIdx` are passed.

```ts
// O(1) frame lookup
const lf = labels.getFrame(video, 42);  // LabeledFrame | null

// O(1) track annotation lookup, sorted by frameIdx
const annotations = labels.getTrackAnnotations(video, track);

// Force index rebuild after external mutations
labels.reindex();
```

Indices are built lazily on first access and auto-invalidated when frames are added or videos replaced. See the [Lazy Loading](#lazy-loading) section below for the caveat that applies when the `Labels` is lazy.

### Deep copy and video remapping

`Labels.copy()` produces a fully independent copy of a `Labels` object with consistent internal references — frame videos point to the copy's video objects, instance skeletons point to the copy's skeletons, etc. Both eager and lazy `Labels` are supported.

`Labels.replaceVideos()` rewrites video references across every collection in one call: `labeledFrames`, `suggestions`, `sessions`, `staticRois`, `temporalRois`, `masks`, `labelImages`, and `centroids`.

```ts
// Deep copy (eager or lazy — matches the original)
const copy = labels.copy();

// Suppress backend auto-opening on the new copy (faster, safer for bulk ops)
const detached = labels.copy({ openVideos: false });

// Replace video references — specify both old and new
labels.replaceVideos({ oldVideos: [oldVid], newVideos: [newVid] });

// Or infer old from current
labels.replaceVideos({ newVideos: [newVid] });

// Or pass a Map
labels.replaceVideos({ videoMap: new Map([[oldVid, newVid]]) });
```

`LazyDataStore.copy()` duplicates the underlying HDF5 column arrays so that two `Labels` wrapping the same source file can diverge safely.

### Frame merging

Merge annotations from one `LabeledFrame` into another with strategy-aware handling. `LabeledFrame.mergeAnnotations(other, strategy?, threshold?)` supports six strategies, applied across all annotation modalities (centroids, bboxes, masks, label images, ROIs). To populate a single `LabeledFrame` in the first place, see [Adding annotations to frames](#adding-annotations-to-frames) below.

| Strategy | Behavior |
|---|---|
| `"keep_original"` | Keep self only, discard everything from `other` |
| `"keep_new"` | Replace self's annotations with (copies of) `other`'s |
| `"keep_both"` *(default)* | Deduplicate by object identity, union both sets (items from `other` are shallow-copied before insertion) |
| `"replace_predictions"` | Keep user annotations from self; drop self's predicted and take all predicted from `other` |
| `"auto"` | Spatially match by centroid distance; user beats predicted; add unmatched |
| `"update_tracks"` | Spatially match and cascade track assignments only |

The `auto` and `update_tracks` strategies compare centroid distances against `threshold` (default `5.0` px). Empty masks and ROIs whose centroid is `null` are skipped by the matcher.

Under `"auto"`, mask merging honors the `fromPredicted` provenance link *before* spatial matching: if a user mask in either frame links to a predicted mask in the other (by object identity), that pair is matched regardless of distance, and only falls back to centroid proximity when no link exists. Other annotation modalities have no `fromPredicted` link, so they are unaffected.

The strategy argument is typed as `MergeStrategy`, re-exported from the package root for type-safe call sites:

```ts
import type { MergeStrategy } from "@talmolab/sleap-io.js";

const strategy: MergeStrategy = "auto";
lfA.mergeAnnotations(lfB, strategy, 3.0);
```

```ts
// Default behavior (keep_both): dedupe by identity, union everything
lfA.mergeAnnotations(lfB);

// Spatial auto merge with a 3 px threshold
lfA.mergeAnnotations(lfB, "auto", 3.0);

// Keep only predicted updates from other, leave user annotations alone
lfA.mergeAnnotations(lfB, "replace_predictions");
```

## Lazy Loading

Load SLP files with on-demand frame materialization for better performance on large datasets.

```ts
// Load lazily - only metadata is parsed initially
const labels = await loadSlp("large_dataset.slp", { lazy: true });
console.log(labels.isLazy);  // true

// Access individual frames on demand
const frame = labels.labeledFrames.at(0);

// Materialize all frames when needed
labels.materialize();
console.log(labels.isLazy);  // false
```

Key classes:

- **`LazyDataStore`**: Holds raw HDF5 column data; supports `materializeFrame(idx)`, `materializeAll()`, `copy()`, and `toNumpy()` fast path.
- **`LazyFrameList`**: Array-like container with `at(idx)`, `length`, `toArray()`, `[Symbol.iterator]()`, and `materializedCount`.

> **Note:** `labels.getFrame()` and `labels.getTrackAnnotations()` **throw** on lazy `Labels`. These O(1) lookups rely on fully materialized frame/track indices and would otherwise silently return stale or partial results. You have two options, both of which trigger full materialization under the hood:
>
> 1. Call `labels.materialize()` explicitly, then use the O(1) lookups.
> 2. Call `labels.find({ video, frameIdx })`, which materializes internally on first call and then uses the same O(1) index. It returns `LabeledFrame[]` — take the first element.
>
> Both paths do the same amount of work on the first call; there's no cheap "lazy-preserving" variant of frame lookup by `(video, frameIdx)`.

```ts
// Option 1: materialize first, then O(1) lookup
labels.materialize();
const lf = labels.getFrame(video, 42);

// Option 2: find() — equivalent, materializes on first call
const [frame] = labels.find({ video, frameIdx: 42 });   // LabeledFrame | undefined
```

## ROI & Segmentation Masks

Spatial annotations stored alongside pose data (SLP format 1.5+).

### `ROI`
Region of interest with GeoJSON-like geometry. `ROI` is abstract; use `UserROI` or `PredictedROI` for construction, or use the static factory methods which return `UserROI`.

```ts
// Create from polygon vertices
const roi = ROI.fromPolygon([[0,0], [100,0], [100,100], [0,100]], {
  category: "region",
});

// Create from multi-polygon
const roi = ROI.fromMultiPolygon([
  [[[0,0], [10,0], [10,10], [0,10], [0,0]]],
  [[[20,20], [30,20], [30,30], [20,30], [20,20]]],
]);

// Predicted ROI with confidence score
const pred = new PredictedROI({
  geometry: { type: "Polygon", coordinates: [[[0,0],[10,0],[10,10],[0,10],[0,0]]] },
  score: 0.95,
});

// Properties
roi.bounds;       // { minX, minY, maxX, maxY }
roi.area;         // polygon area
roi.centroidXy;   // [x, y]
roi.isBbox;       // true if axis-aligned rectangle
roi.isPredicted;  // false for UserROI, true for PredictedROI

// Explode multi-geometries into individual ROIs (preserves subclass)
const parts = roi.explode();  // ROI[]

// Convert to GeoJSON Feature
const feature = roi.toGeoJSON();

// Convert to mask
const mask = roi.toMask(480, 640);
```

> **Note:** `ROI.fromBbox()` and `ROI.fromXyxy()` are deprecated. Use `BoundingBox.fromXywh()` or `BoundingBox.fromXyxy()` instead.

> **Note:** `ROI.video` is retained for static ROIs (arena boundaries, chamber outlines stored on `labels.staticRois`); `ROI.frameIdx` was removed in v0.3.0. Frame-bound ROIs derive their frame from the parent `LabeledFrame` — attach them with `lf.append(roi)`.

### `SegmentationMask`
RLE-encoded binary mask. `SegmentationMask` is abstract; use `UserSegmentationMask` or `PredictedSegmentationMask` for direct construction. The `fromArray()` factory returns `UserSegmentationMask`.

```ts
// Create from 2D boolean array
const mask = SegmentationMask.fromArray(boolArray, 480, 640, {
  category: "segmentation",
});

// Create from array with stride (downsampled prediction)
const mask = SegmentationMask.fromArray(data, 120, 160, {
  stride: 4,  // sets scale to [0.25, 0.25]
});

// Predicted mask with score and optional score map
const pred = new PredictedSegmentationMask({
  rleCounts, height: 480, width: 640,
  score: 0.92,
  scoreMap: new Float32Array(480 * 640),  // optional per-pixel confidence
});

// Properties
mask.data;                // Uint8Array (decoded)
mask.area;                // pixel count
mask.bbox;                // { x, y, width, height }
mask.isPredicted;         // false for User, true for Predicted
mask.scale;               // [number, number] spatial scale (default [1, 1])
mask.offset;              // [number, number] spatial offset (default [0, 0])
mask.hasSpatialTransform; // true if scale != [1,1] or offset != [0,0]
mask.imageExtent;         // { height, width } in image coordinates

// Resample to target size (removes spatial transform)
const fullRes = mask.resampled(480, 640);

// Trace the actual mask boundary as closed polygon rings in image space
// (honors scale/offset). Disjoint blobs and holes each get their own ring;
// an empty mask returns []. Pure data — works in the browser.
const rings = mask.contours();   // number[][][]

// Convert to a polygon ROI tracing the real outline: a Polygon (one blob,
// holes nested as interior rings) or MultiPolygon (disjoint blobs). A
// predicted mask yields a PredictedROI carrying its score.
const roi = mask.toPolygon();

// Convert to BoundingBox object (with metadata)
const bb = mask.toBbox();
```

> **`toPolygon()` traces the boundary** (as of v0.5.0) rather than returning the
> bounding-box rectangle. For the axis-aligned box use `mask.bbox` or
> `mask.toBbox()`. The traced outline is exact and axis-aligned ("staircase");
> consumers wanting smooth curves can post-process the rings (e.g. Chaikin
> subdivision — the demo does this). `get data()` memoizes the decoded raster, so
> repeated `data`/`area`/`bbox`/`contours()` reads decode the RLE once (treat the
> returned buffer as read-only).

**Adopting predictions (human-in-the-loop).** `PredictedSegmentationMask.toUser()`
mirrors `Instance.fromPredicted`: it returns a new `UserSegmentationMask` that
copies the RLE raster and carries over metadata (`name`, `category`, `source`,
`track`, `trackingScore`, `instance`, `scale`, `offset`) while dropping
prediction-only fields (`score`, `scoreMap`). It sets `fromPredicted` on the
returned mask as a provenance link.

```ts
const userMask = pred.toUser();      // userMask.fromPredicted === pred
const detached = pred.toUser(false); // detached.fromPredicted === null
```

> **Note:** like `Instance.fromPredicted`, the mask's `fromPredicted` link IS
> persisted to the SLP format. It is stored as an index into the saved mask list
> and re-linked on read, so it survives a save/load round-trip as long as the
> source prediction is also saved (in the same or another frame). If the source
> prediction is not saved, the link loads as `null`. Recording at least one link
> bumps the SLP `format_id` to 2.4; the column is always written and reads are
> presence-gated, so files written before this column existed load the link as
> `null`.

`LabeledFrame.unusedPredictedMasks` returns the predicted masks in a frame that
no user mask has adopted — the mask analogue of `LabeledFrame.unusedPredictions`.
A prediction is adopted when a user mask in the same frame links to it via
`fromPredicted` (checked first, by identity) or, lacking a link, spatially
overlaps it (bbox-centroid within 5 px). The remaining unclaimed predictions are
returned, e.g. for surfacing unreviewed detections in a proofreading UI.

#### Rendering masks in the browser

`drawMasks` and `drawLabelImage` are **browser-safe** raster compositors (pure
`ImageData` blending — no `skia-canvas`, no Node built-ins). They are exported
from the browser entry, so a UI can composite masks onto a `<canvas>`
client-side without the Node `renderImage` path:

```ts
import { drawMasks } from "@talmolab/sleap-io.js"; // browser build

const ctx = canvas.getContext("2d");
ctx.drawImage(videoFrame, 0, 0);
const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
drawMasks(img, frame.masks, {
  colors: frame.masks.map(maskColorRgb),  // e.g. by track identity
  alpha: 0.45,
});
ctx.putImageData(img, 0, 0);

// Crisp vector outlines on top, from the data model:
for (const mask of frame.masks) {
  for (const ring of mask.contours()) {
    ctx.beginPath();
    ctx.moveTo(ring[0][0], ring[0][1]);
    for (const [x, y] of ring) ctx.lineTo(x, y);
    ctx.closePath();
    ctx.stroke();
  }
}
```

The vector overlays (`drawBboxes`, `drawRois`, `drawCentroids`, `applyOverlay`)
and the full `renderImage`/`renderVideo` pipeline draw through `skia-canvas` and
remain Node-only (main entry). See `demo/` for a complete browser example that
fills masks with `drawMasks` and outlines them with smoothed `contours()`.

`renderImage`/`renderVideo` auto-draw a frame's `centroids` (filled circle
markers, colored by track via the pose palette, scoped to the rendered video),
so a centroid-only frame renders without an explicit overlay. `drawCentroids`
is also exported for standalone use:

```ts
import { drawCentroids } from "@talmolab/sleap-io.js"; // Node (skia-canvas)

drawCentroids(imageData, frame.centroids, {
  colors: frame.centroids.map(centroidColorRgb), // e.g. by track identity
  markerSize: 5,
  alpha: 1,
});
```

### `BoundingBox`
Axis-aligned or rotated bounding box for detection/tracking workflows. `BoundingBox` is abstract; use `UserBoundingBox` or `PredictedBoundingBox` for direct construction.

Bounding boxes use corner coordinates (`x1`, `y1`, `x2`, `y2`) as their primary storage. Center-based properties (`xCenter`, `yCenter`, `width`, `height`) are available as computed getters.

Construct a user-annotated bbox and attach it to a frame:

```ts
const bbox = new UserBoundingBox({
  x1: 0, y1: 20, x2: 100, y2: 100,
  category: "animal",
});

// The parent LabeledFrame provides video + frameIdx context
let lf = labels.getFrame(labels.videos[0], 3);
if (!lf) {
  lf = new LabeledFrame({ video: labels.videos[0], frameIdx: 3 });
  labels.append(lf);
}
lf.append(bbox);
```

Construct a predicted bbox with confidence score:

```ts
const predBbox = new PredictedBoundingBox({
  x1: 0, y1: 20, x2: 100, y2: 100,
  score: 0.95,
});
```

Factory methods (return `UserBoundingBox`):

```ts
const fromCorners = BoundingBox.fromXyxy(10, 20, 110, 100);  // corner coords
const fromTopLeft = BoundingBox.fromXywh(10, 20, 100, 80);   // top-left + size
```

Properties and conversions:

```ts
// Stored properties
bbox.x1; bbox.y1; bbox.x2; bbox.y2;  // corner coordinates
bbox.angle;      // rotation in radians (default 0)

// Computed properties
bbox.xCenter;    // (x1 + x2) / 2
bbox.yCenter;    // (y1 + y2) / 2
bbox.width;      // abs(x2 - x1)
bbox.height;     // abs(y2 - y1)
bbox.xyxy;       // [x1, y1, x2, y2] (axis-aligned bounding box)
bbox.xywh;       // { x, y, width, height }
bbox.corners;    // number[][] (4 corner points, respects rotation)
bbox.bounds;     // { minX, minY, maxX, maxY }
bbox.area;       // width * height
bbox.centroidXy; // [x, y]
bbox.isPredicted; // true for PredictedBoundingBox
bbox.isRotated;  // true if angle != 0

// Conversion
const roi = bbox.toRoi();              // Polygon ROI
const mask = bbox.toMask(480, 640);    // SegmentationMask
```

### `Centroid`
Point representing the center of an object. `Centroid` is abstract; use `UserCentroid` or `PredictedCentroid`.

```ts
// Create user centroid (attach to a frame to provide video + frameIdx context)
const c = new UserCentroid({ x: 100, y: 200, track });
let lf = labels.getFrame(video, 0);
if (!lf) {
  lf = new LabeledFrame({ video, frameIdx: 0 });
  labels.append(lf);
}
lf.append(c);

// Create predicted centroid with confidence
const pc = new PredictedCentroid({
  x: 50, y: 60, z: 3.5,
  score: 0.95,
  trackingScore: 0.8,
  track,
  name: "spot1", source: "trackmate",
});
let lf1 = labels.getFrame(video, 1);
if (!lf1) {
  lf1 = new LabeledFrame({ video, frameIdx: 1 });
  labels.append(lf1);
}
lf1.append(pc);

// Properties
c.xy;           // [x, y]
c.yx;           // [y, x] (row, col)
c.xyz;          // [x, y, z | null]
c.isPredicted;  // false for User, true for Predicted

// Convert to single-node Instance
const inst = pc.toInstance();  // PredictedInstance with centroid skeleton

// Create from Instance (center of mass, bbox center, or anchor node)
const centroid = Centroid.fromInstance(instance);
const centroid = Centroid.fromInstance(instance, { method: "bboxCenter" });
const centroid = Centroid.fromInstance(instance, { method: "anchor", node: "head" });

// Instance convenience method (requires centroid.ts import)
const centroid = instance.toCentroid();
const centroid = instance.toCentroid("anchor", "head");

// Instance centroid shortcut (no Centroid object, just coordinates)
instance.centroidXy;  // [x, y] | null (mean of visible points)
```

#### Centroid Skeleton
```ts
// Shared single-node skeleton for centroid conversions
const skel = getCentroidSkeleton();  // Skeleton(["centroid"])
CENTROID_SKELETON === getCentroidSkeleton();  // true (singleton)
```

### `Labels` Annotation Access

Annotations (`Centroid`, `BoundingBox`, `SegmentationMask`, `LabelImage`, frame-bound `ROI`) live on individual `LabeledFrame` instances, not on `Labels`. The `Labels` getters return flattened read-only views across all frames; mutation always goes through the owning frame.

```ts
// Read-only property getters (flat views across all frames)
labels.rois;           // ROI[]
labels.masks;          // SegmentationMask[]
labels.bboxes;         // BoundingBox[]
labels.centroids;      // Centroid[]
labels.labelImages;    // LabelImage[]
labels.identities;     // Identity[]
labels.staticRois;     // Static (video-level) ROIs, e.g. arena boundaries
labels.temporalRois;   // Frame-bound ROIs (flat view across frames)

// Filtered queries (all support `predicted` filter; frame-aware filters use O(1) indices)
labels.getRois({ video, frameIdx: 0, category: "arena", predicted: false });
labels.getMasks({ video, category: "segmentation", predicted: true });
labels.getBboxes({ video, frameIdx: 0, predicted: true });
labels.getCentroids({ video, frameIdx: 0, track, predicted: true });
labels.getLabelImages({ video, frameIdx: 0, track, category: "cell" });

// Note: getRois({ video }) returns only frame-bound ROIs for that video.
// For static ROIs, use labels.staticRois directly.

// Add a video (deduplicates automatically)
labels.addVideo(video);
```

#### Adding annotations to frames

Use `LabeledFrame.append(annotation)` to add any annotation type. It routes by runtime type to the appropriate per-frame list (`lf.centroids`, `.bboxes`, `.masks`, `.labelImages`, `.rois`) and handles `Instance` and `PredictedInstance` as well. If you know the target list at the call site, you can also push directly.

> **Note:** `Labels` and `LabeledFrame` each have an `append()` method. `labels.append(lf)` adds a `LabeledFrame` to a `Labels` (and also registers its video and tracks). `lf.append(annotation)` adds an annotation to a single `LabeledFrame`. They're different operations at different levels of the hierarchy.

```ts
import {
  Labels,
  LabeledFrame,
  UserCentroid,
  UserBoundingBox,
  UserSegmentationMask,
} from "@talmolab/sleap-io.js";

// Build a frame with mixed annotations
const lf = new LabeledFrame({ video, frameIdx: 5 });
lf.append(new UserCentroid({ x: 100, y: 200, track }));
lf.append(new UserBoundingBox({ x1: 50, y1: 50, x2: 150, y2: 150 }));
lf.append(UserSegmentationMask.fromArray(maskData, 480, 640));

// Or push directly if you know the type
lf.centroids.push(new UserCentroid({ x: 10, y: 20 }));

// Pass frames as labeledFrames when constructing Labels
const labels = new Labels({ labeledFrames: [lf], videos: [video], skeletons: [skeleton] });

// To mutate an existing Labels, look up the target frame first
let existing = labels.getFrame(video, 5);
if (!existing) {
  existing = new LabeledFrame({ video, frameIdx: 5 });
  labels.append(existing);
}
existing.append(anotherBbox);
```

Static ROIs (video-level, e.g. arena boundaries) live on `labels.staticRois`. Use `labels.addStaticRoi(roi)` to add one — the method also registers the ROI's track on `labels.tracks` if present:

```ts
labels.addStaticRoi(arenaRoi);
```

#### SLP format versions

The SLP format version (`format_id`) is set automatically based on content:

| Version | Trigger |
|---------|---------|
| 1.5 | ROIs or masks present |
| 1.6 | ROIs with instance associations |
| 1.7 | Bounding boxes |
| 1.8 | Label images |
| 1.9 | Identities or predicted annotations |
| 2.0 | BBox corner-based format |
| 2.1 | Spatial metadata (scale/offset) |
| 2.2 | Chunked label image storage |
| 2.3 | Virtual video crops present |
| 2.4 | Persisted mask `fromPredicted` provenance link |

##### Virtual Crops (Format 2.3+)

A [virtually-cropped video](cropping.md) stores its crop rect in a dedicated
top-level `/video_crops` dataset (a single JSON string), written **only when at
least one video is cropped**. Its `videos_json` entry describes the **uncropped
source**, and its `source_video` is the uncropped original — so a reader that does
not understand `/video_crops` simply loads the full-frame source (a graceful, lossy
degrade, never an error).

```json
[
  { "video": 2, "crop": [128, 96, 384, 352], "fill": 0 },
  { "video": 5, "crop": [0, 0, 256, 256], "fill": 0 }
]
```

- `video` — integer index into the `videos_json` video list.
- `crop` — `[x1, y1, x2, y2]` in **source** pixel coordinates, `x2`/`y2` exclusive.
- `fill` — out-of-bounds fill value (regions outside the source are padded, not
  clamped).

The presence of any crop bumps `format_id` to `2.3`. Files with no crops are
byte-identical to earlier versions (no `/video_crops` dataset, no version bump).

## Label Images (Dense Segmentation)

Dense integer label images for instance segmentation workflows (Cellpose, StarDist, SAM, Mask R-CNN output). Each pixel stores an object ID (0 = background, positive integers = objects) with per-object metadata (track, category, name).

`LabelImage` is abstract; use `UserLabelImage` or `PredictedLabelImage`, or use the static factory methods which return `UserLabelImage`.

### Creating Label Images

```ts
// From a 2D integer array. Tracks are NOT created by default — pass them
// explicitly, or set `createTracks: true` to auto-create one per label ID.
const li = LabelImage.fromArray(
  new Int32Array([0, 0, 1, 1, 0, 2, 2, 0, 0]),
  3, 3,  // height, width
  { tracks: [trackA, trackB] },
);

// Pure segmentation (no tracking): omit tracks (objects carry track = null).
const segLi = LabelImage.fromArray(maskData, 480, 640);
// Opt in to auto-created tracks named by label ID:
const trackedLi = LabelImage.fromArray(maskData, 480, 640, { createTracks: true });

// From segmentation masks (composites multiple binary masks)
const li2 = LabelImage.fromMasks(masks);

// From per-object binary masks (SAM / Mask R-CNN output)
const li3 = LabelImage.fromBinaryMasks(
  [mask1_2d, mask2_2d, mask3_2d],
  {
    tracks: [trackA, trackB, trackC],
    categories: ["cell", "cell", "debris"],
    scores: [0.95, 0.88, 0.72],  // creates PredictedLabelImage if provided
  },
);

// Batch creation from a stack of 2D arrays (one per frame)
const labelImages = LabelImage.fromStack({
  data: [frame0_2d, frame1_2d, frame2_2d],
  tracks: new Map([[1, trackA], [2, trackB]]),
  createTracks: true,  // auto-create Track objects from label IDs
});

// Attach each label image to its frame
labelImages.forEach((li, i) => {
  let lf = labels.getFrame(video, i);
  if (!lf) {
    lf = new LabeledFrame({ video, frameIdx: i });
    labels.append(lf);
  }
  lf.append(li);
});
```

Frame and video context for label images comes from the parent `LabeledFrame`. Factory methods return plain `UserLabelImage` objects; use `lf.append(li)` to attach them to a frame.

### Properties and Methods

```ts
// Data access
li.data;       // Int32Array (flat, row-major)
li.height;     // number
li.width;      // number
li.objects;    // Map<number, LabelImageObjectInfo>
li.nObjects;   // number of objects
li.labelIds;   // sorted unique non-zero label IDs in data
li.tracks;     // Track[] from object metadata
li.categories; // Set<string> from object metadata

// Spatial transforms (for downsampled predictions)
li.scale;               // [number, number] (default [1, 1])
li.offset;              // [number, number] (default [0, 0])
li.hasSpatialTransform; // boolean
li.imageExtent;         // { height, width } in image coordinates

// Mask extraction
const mask = li.getObjectMask(1);         // Uint8Array for label ID 1
const mask = li.getTrackMask(trackA);     // Uint8Array for all objects with trackA
const mask = li.getCategoryMask("cell");  // Uint8Array for all "cell" objects

// Iteration over objects
for (const [track, category, binaryMask] of li.items()) {
  console.log(track?.name, category, binaryMask.length);
}

// Conversion
const masks = li.toMasks();                       // SegmentationMask[]
const bboxes = li.toBboxes();                     // BoundingBox[]
const fullRes = li.resampled(480, 640);           // new LabelImage at full resolution

// Predicted label image
li.isPredicted;  // false for UserLabelImage, true for PredictedLabelImage
// PredictedLabelImage also has: score, scoreMap, scoreMapScale, scoreMapOffset
```

### `LabelImageObjectInfo`

Per-object metadata stored in `LabelImage.objects`:

```ts
interface LabelImageObjectInfo {
  track: Track | null;
  category: string;
  name: string;
  instance: Instance | null;
  score?: number | null;           // per-object confidence (PredictedLabelImage)
  trackingScore?: number | null;   // per-link tracking cost
}
```

### `normalizeLabelIds()`

Normalize label IDs across multiple frames so each Track or category gets a globally consistent ID. Mutates label images in place.

```ts
// Normalize by track (default) — each Track gets a unique, consistent ID
const trackMap: Map<Track, number> = normalizeLabelIds(labelImages);

// Normalize by category — same category string gets the same ID
const catMap: Map<string, number> = normalizeLabelIds(labelImages, { by: "category" });
```

## Identity & 3D Data Structures

### `Identity`

Ground-truth animal identity that persists across videos, sessions, and experiments. Unlike `Track` (which represents a temporal trajectory within a single video), `Identity` represents the actual animal.

```ts
const id = new Identity({ name: "mouse_A", color: "#ff0000" });

id.name;      // "mouse_A"
id.color;     // "#ff0000"
id.metadata;  // Record<string, unknown>
```

### `Instance3D` and `PredictedInstance3D`

3D keypoint data from multi-camera triangulation.

```ts
// User-annotated 3D instance
const inst3d = new Instance3D({
  points: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  skeleton,
  score: 0.95,
});

inst3d.nVisible;  // number of non-NaN keypoints
inst3d.isEmpty;   // true if all keypoints are NaN/null

// Predicted 3D instance (with per-keypoint scores)
const pred3d = new PredictedInstance3D({
  points: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  skeleton,
  score: 0.95,
  pointScores: [0.99, 0.87],
});
```

### 3D Fields on Existing Types

```ts
// Labels.identities — all ground-truth identities
labels.identities;  // Identity[]

// Camera.size — image dimensions
camera.size;  // [width, height] | undefined

// InstanceGroup — links identity and 3D data to multi-camera groups
group.identity;    // Identity | undefined
group.instance3d;  // Instance3D | undefined
group.points;      // delegates to instance3d.points if available
```

### Lossless, lazy-native session grouping

Cross-camera grouping (`FrameGroup` / `InstanceGroup`) round-trips losslessly
through read → object model → write **without** materializing frames, on all
three read paths (eager, lazy, streaming). Instead of resolving the grouping
against a materialized frame table at read time, the model stores the as-read
index refs (`camcorder_to_lf_and_inst_idx_map` and `labeled_frame_by_camera`)
and resolves them lazily:

```ts
// Typed access — same public API as before, now lazy under the hood.
const fg = labels.sessions[0].frameGroups.get(0)!;
fg.instanceGroups[0].instanceByCamera; // Map<Camera, Instance> (resolved on demand)
fg.labeledFrameByCamera;               // Map<Camera, LabeledFrame> (resolved on demand)
```

Under the lazy reader, resolving one cross-camera ref materializes only that one
frame (via `Labels.frameAt(i)`); an untouched session serializes straight from
the stored refs with **zero** frame materialization. Mutating a group (accessing
its map, then editing/reordering instances) re-derives indices from the live
objects on write, so edits are reflected correctly.

On write, the `sessions_json` shape matches Python `sleap-io`'s `session_to_dict`
byte-for-byte: `calibration` is keyed `cam_0`, `cam_1`, … (with each camera's
`name` kept as a field inside its dict), while `camcorder_to_video_idx_map`,
`camcorder_to_lf_and_inst_idx_map`, and `labeled_frame_by_camera` are keyed by the
bare integer camera index (`"0"`, `"1"`, …). These are two different key spaces on
purpose — both Python and sleap-io.js resolve cameras by their **order** in
`calibration` / positional index, so the calibration key string is cosmetic. This
is a convergence to Python's existing format, **not** a new on-disk feature, so it
does **not** bump `format_id` (that namespace is shared with Python, which already
defines 2.5+). Reading stays tolerant of legacy name-keyed calibration, so loading
an older file and re-saving upgrades the sessions to the canonical shape in place.

### Raw `sessions_json` access (`rawJson` / `rawSessionsJson`) — deprecated

> **Deprecated / transitional, opt-in only.** The typed session model above is
> now a faithful projection of `sessions_json`, so consumers should read typed
> objects rather than this raw dict. `rawJson` is a temporary bridge that will be
> removed once LUCID migrates. It is **`undefined` by default** — pass
> `{ rawSessions: true }` to a read entrypoint to capture it. Capturing it
> deep-clones the whole session payload (doubling session memory in
> `Labels.copy()`), which is why it is off by default.

```ts
// OPT-IN: capture the verbatim parsed sessions_json dict per session.
const labels = await readSlp(path, { rawSessions: true });
labels.sessions[0].rawJson;
//   -> { calibration, camcorder_to_video_idx_map,
//        camcorder_to_lf_and_inst_idx_map, frame_group_dicts, metadata, ... }

// Index-aligned convenience getter (derived from sessions, no stored field)
labels.rawSessionsJson;         // Array<Record<string, unknown> | undefined>
labels.rawSessionsJson[0];      // === labels.sessions[0].rawJson

// Default (no option): rawJson is undefined, no clone, no copy() bloat.
const lean = await readSlp(path);
lean.sessions[0].rawJson;       // undefined
```

`rawSessions` is accepted by `readSlp`, `readSlpLazy`, and `readSlpStreaming`.
When captured, `rawJson` exposes every key, including ones sleap-io.js does not
itself model.

Caveats (when captured):

- **Read-time snapshot.** `rawJson` is captured at read time and is NOT updated
  when you mutate the object model afterwards.
- **Never re-written to disk.** Writes are rebuilt from the object model (the
  single source of truth); `rawJson` is a pure in-memory read surface over the
  already-existing `sessions_json` dataset. It is not stored in `provenance` or
  the `/metadata` blob, so saving does not bloat those.
- **Index-aligned.** `rawSessionsJson[i]` corresponds to `sessions[i]`; entries
  are `undefined` for sessions constructed in-memory or read without
  `rawSessions`.

#### Metadata round-trip (Option 1)

Opaque `metadata` dicts round-trip losslessly through read → object model →
write at every level — `RecordingSession.metadata` (including nested
`metadata.lucid`), `FrameGroup.metadata`, `InstanceGroup.metadata`, and
`CameraGroup.metadata`. Arbitrary unknown keys survive by JSON-equality.

> Note: `CameraGroup.metadata` previously read back as `{}` (silently dropped on
> read); it is now populated from `calibration.metadata`.

#### Object-model write limitations (Option 3)

The write path (`serializeSession`) rebuilds `sessions_json` from a fixed field
set, so it is lossless for everything sleap-io.js structurally models —
per-camera intrinsics/extrinsics (`name`, `rotation`, `translation`, `matrix`,
`distortions`, `size`), `cam_N`-keyed `calibration` with `name` kept as a field,
integer-index-keyed `camcorder_to_video_idx_map` /
`camcorder_to_lf_and_inst_idx_map` / `labeled_frame_by_camera`, and all `metadata`
dicts. `camcorder_to_lf_and_inst_idx_map`
now survives losslessly — it is derived from live objects when the group was
mutated, and otherwise falls back to the as-read index refs (so it is preserved
even in lazy mode). The following remain recoverable only via `rawSessions`:

1. Any unmodeled top-level key NOT under a `metadata` sub-dict (at session /
   per-camera / frame-group / instance-group level) is dropped.
2. Frame groups with zero instance groups are skipped entirely on write.
3. Frame groups are keyed by `frame_idx`; groups sharing (or missing) a
   `frame_idx` collapse to one entry on read (a pre-existing fidelity gap).

> Legacy note: older sleap-io.js builds keyed `calibration` by camera `name`
> when a camera had a distinct name, diverging from Python's `cam_N` convention.
> Reading tolerates both (cameras resolve by order); re-saving canonicalizes to
> `cam_N`.

## SLEAP Analysis CSV export

Export `Labels` to the SLEAP Analysis CSV — one row per instance per frame with
columns `track, frame_idx, instance.score, {node}.score, {node}.x, {node}.y, …`
(node columns sorted alphabetically; missing/invisible points and user-instance
scores are empty cells). `labelsToCsv` is browser-safe (returns the string);
`saveLabelsCsv` writes to disk (Node).

```ts
import { labelsToCsv, saveLabelsCsv } from "@talmolab/sleap-io.js";

const csv = labelsToCsv(labels);               // string (browser + Node)
await saveLabelsCsv(labels, "out.csv");        // Node file write

// Span the whole video: emit NaN rows for unlabeled frames, padded to the video
// length when known (mirrors the Analysis-HDF5 export; sleap-io PR #480).
await saveLabelsCsv(labels, "out.csv", { includeEmpty: true });

// Other options: video (Video | index filter), includeScore (default true),
// startFrame / endFrame (inclusive / exclusive).
```

## GeoJSON I/O

Convert ROIs to/from GeoJSON format.

```ts
// Convert ROIs to GeoJSON FeatureCollection
const geojson = roisToGeoJSON(labels.rois);

// Serialize to JSON string
const json = writeGeoJSON(labels.rois);

// Parse GeoJSON back to ROIs
const rois = roisFromGeoJSON(geojson);
const rois = readGeoJSON(jsonString);
```

## TrackMate CSV I/O

Import TrackMate (ImageJ/Fiji) CSV tracking exports as `PredictedCentroid` objects. Node.js only.

```ts
import { readTrackMateCsv, loadTrackMate, isTrackMateFile } from "@talmolab/sleap-io.js";

// Check if a file is a TrackMate spots CSV
isTrackMateFile("data_spots.csv");  // true/false

// Load spots CSV (auto-detects sibling _edges.csv and .tif video)
const labels = readTrackMateCsv("data_spots.csv");
labels.centroids;  // PredictedCentroid[]
labels.tracks;     // Track[] (from TRACK_ID column)

// With explicit options
const labels = readTrackMateCsv("data_spots.csv", {
  edgesPath: "data_edges.csv",  // optional, auto-detected from _spots suffix
  video: "data.tif",            // optional, auto-detected from _spots suffix
});

// Public API alias
const labels = loadTrackMate("data_spots.csv");
```

TrackMate CSV format:
- `*_spots.csv` — spot detections with `POSITION_X`, `POSITION_Y`, `POSITION_Z`, `FRAME`, `QUALITY`, `TRACK_ID`
- `*_edges.csv` — frame-to-frame linkages; `LINK_COST` is stored as `trackingScore` on target centroids

## Ultralytics YOLO I/O

Read and write [Ultralytics YOLO](https://docs.ultralytics.com/datasets/) datasets — pose, detection, and segmentation. A dataset is a directory of `data.yaml` + per-split `images/` and `labels/`. The format of each label line is auto-detected by value count. Node.js only (directory-based I/O).

```ts
import { loadUltralytics, saveUltralytics } from "@talmolab/sleap-io.js";

// Load a split (auto-detects pose/detection/segmentation per line)
const labels = loadUltralytics("path/to/yolo_dataset", { split: "train" });
labels.labeledFrames;  // pose instances (5 + 3k value lines)
labels.bboxes;         // UserBoundingBox / PredictedBoundingBox (5 / 6 value lines)
labels.rois;           // UserROI from segmentation polygons

// Pointing at data.yaml directly also works
loadUltralytics("path/to/yolo_dataset/data.yaml");

// Write a dataset (default task "pose"; also "detect" / "segment")
await saveUltralytics(labels, "out_dir", { splitRatios: { train: 0.8, val: 0.2 } });
await saveUltralytics(labels, "out_dir", { task: "detect" });   // writes labels.bboxes
await saveUltralytics(labels, "out_dir", { task: "segment" });  // writes labels.rois
```

Lower-level helpers are also exported: `parseLabelFile(path, skeleton, [h, w], { classNames })` returns `{ instances, rois, bboxes }`; `detectLineFormat(parts)` returns `"detection" | "detection_conf" | "segmentation" | "pose"`; plus `writeLabelFile`, `writeBboxLabelFile`, `writeRoiLabelFile`, `readLabelsSet`, `parseDataYaml`, `createDataYaml`, and `createSkeletonFromConfig`.

Label-line formats (coordinates normalized to `[0, 1]`):
- **Detection** — `class_id x_center y_center width height [confidence]` → `UserBoundingBox` (5 values) or `PredictedBoundingBox` (6 values).
- **Segmentation** — `class_id x1 y1 x2 y2 … xn yn` → `UserROI` polygon (multi-polygons are exploded on write; interior holes are dropped with a warning).
- **Pose** — `class_id x_center y_center width height x1 y1 v1 … xn yn vn` → `Instance` (visibility `0` = invisible, `1`/`2` = visible).

> **Image I/O note:** image dimensions are read from the file header (PNG/JPEG/GIF/BMP/TIFF) rather than by decoding pixels. On write, frames backed by an on-disk image file are copied verbatim; frames yielding raw `ImageData` pixels are encoded to PNG; otherwise the frame is skipped with a warning. The `imageFormat`/`imageQuality` options apply only to the raw-pixel PNG path.

## JABS Pose I/O

Read [JABS](https://github.com/KumarLabJax/JABS-behavior-classifier) (Jackson Lab Animal Behavior System) pose files. JABS files are HDF5 on disk, so this reader is Node.js only. Matches Python sleap-io v0.7.0 (PR #371): instances are `PredictedInstance` objects with per-point confidence, and static objects are `UserROI`s rather than synthetic skeletons.

```ts
import { loadJabs } from "@talmolab/sleap-io.js";

// Read a JABS pose file (pose versions 2–6). Async (HDF5 via h5wasm).
const labels = await loadJabs("recording_pose_est_v5.h5");

labels.labeledFrames;   // frames of PredictedInstance poses
labels.tracks;          // one Track per JABS identity (v2 → a single track "1")
labels.staticRois;      // static objects as UserROI (also in labels.rois)
labels.videos[0];       // Video named after the pose file (…_pose_est_vN.h5 → ….avi)

// Optional: override the skeleton (must have one node per keypoint column)
const labels2 = await loadJabs("recording_pose_est_v2.h5", { skeleton: myMouseSkeleton });
```

Static objects become `UserROI`s with `source: "jabs"` and `category: "arena"` for `corners` (or `"anchor"` otherwise); a single-point object (e.g. `lixit`) is a `Point` geometry, multi-point (e.g. `corners`) is a `MultiPoint`. The default `JABS_DEFAULT_SKELETON` is the 12-node "Mouse" skeleton; `makeJabsDefaultSkeleton()`, `predictionToInstance()`, `staticObjectToRoi()`, and `makeSimpleSkeleton()` are also exported.

> JABS coordinates are stored `(y, x)` and flipped to `(x, y)` on read. Static-object coordinates are kept as-stored. The writer (`saveJabs`) is not yet implemented — the common path is a one-time JABS → SLP conversion.

## Skeleton Codecs

### JSON (jsonpickle format)
```ts
import { readSkeletonJson, isSkeletonJson } from "@talmolab/sleap-io.js";

const skeleton = readSkeletonJson(jsonString);
```

### Training Config
```ts
import { readTrainingConfigSkeletons, isTrainingConfig } from "@talmolab/sleap-io.js";

const skeletons = readTrainingConfigSkeletons(jsonString);
```

## Codecs & Numpy Helpers

### Dictionary codec
```ts
const dict = toDict(labels);
const restored = fromDict(dict);
```

### Numpy codec
```ts
const array = toNumpy(labels, { returnConfidence: true });
const rebuilt = fromNumpy(array, { video, skeleton, returnConfidence: true });
```

Also available:
- `Labels.numpy({ video, returnConfidence })`
- `Labels.fromNumpy(data, { video, skeleton, trackNames, firstFrame })`

## Streaming Options

SLP streaming is handled by HDF5 open helpers:

```ts
const labels = await loadSlp("https://example.com/session.slp", {
  openVideos: false,
  h5: { stream: "range", filenameHint: "session.slp" },
});
```

- `stream: "auto"` uses range streaming when available.
- `stream: "download"` forces full file download.
- Node only supports string paths or byte buffers for SLP inputs.
- Browser supports URL, `File`, `FileSystemFileHandle`, or byte buffers.

## Error Handling Notes

Common runtime errors to anticipate:

- SLP parsing:
  - Missing `/metadata` group in invalid `.slp`.
  - Unsupported source type in browser or Node.
  - Failed fetch for remote SLP (bad URL/status).
  - `h5wasm` FS unavailable in unsupported environments.

- MP4/WebCodecs:
  - WebCodecs not supported or non-browser environment.
  - No video tracks or unsupported codec.
  - Failed fetch or no video source available.

- Media backend:
  - Browser-only usage.
  - Video load or seek failures.
