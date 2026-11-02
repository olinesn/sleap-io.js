# sleap-io.js

JavaScript/TypeScript utilities for reading and writing SLEAP `.slp` files with streaming-friendly access patterns and a lightweight data model. This is the JS companion to the Python library at https://github.com/talmolab/sleap-io.

## Intent

- Make SLP parsing available in browsers and serverless runtimes.
- Support streaming-first workflows for large `.slp`/`.pkg.slp` files.
- Provide a minimal data model and codecs that mirror sleap-io behavior.
- Enable client-side visualization and analysis pipelines.

## Features

- SLP read/write with format compatibility (reads format 1.0–3.x, writes the 2.x layout, including embedded frames via HDF5 video datasets).
- Browser-compatible SLP writing via `saveSlpToBytes()`.
- Streaming-friendly file access (URL, `File`, `FileSystemFileHandle`, `Blob`).
- Core data model (`Labels`, `LabeledFrame`, `Instance`, `Skeleton`, `Video`, `Centroid`, `Identity`, `Instance3D`, etc.).
- ROI, segmentation mask, bounding box, and label image annotations with GeoJSON I/O.
- Browser-safe segmentation rendering: `SegmentationMask.contours()` boundary tracing (real outlines, not just the bbox) plus `drawMasks`/`drawLabelImage` raster compositing for client-side UIs.
- Ultralytics YOLO dataset reader/writer (pose, detection, segmentation; Node.js).
- JABS (Jackson Lab) pose-file reader (predicted poses + static-object ROIs; Node.js).
- 3D pose data structures with cross-library interop (Python sleap-io, luc3d).
- Video backends accept `string`, `File`, or `Blob` sources.
- Browser-safe: Node.js-only dependencies (`skia-canvas`, `child_process`) are dynamically imported, so bundlers can tree-shake them.
- Dictionary and numpy codecs for interchange.
- SLEAP Analysis CSV export (`labelsToCsv` / `saveLabelsCsv`), with optional full-video-span padding.
- Demo app for quick inspection.

## Quickstart

```bash
npm install @talmolab/sleap-io.js
```

### Load and save SLP

```ts
import { loadSlp, saveSlp, saveSlpToBytes } from "@talmolab/sleap-io.js";

const labels = await loadSlp("/path/to/session.slp", { openVideos: false });

// Save to file (Node.js/Electron)
await saveSlp(labels, "/tmp/session-roundtrip.slp", { embed: false });

// Save to bytes (works in browsers too)
const bytes: Uint8Array = await saveSlpToBytes(labels);
```

### Load and save Ultralytics YOLO datasets

```ts
import { loadUltralytics, saveUltralytics } from "@talmolab/sleap-io.js";

// Read a split; the per-line format (pose/detection/segmentation) is auto-detected
const labels = loadUltralytics("/path/to/yolo_dataset", { split: "train" });

// Write back out (task: "pose" default, or "detect" / "segment")
await saveUltralytics(labels, "/tmp/yolo_out", { splitRatios: { train: 0.8, val: 0.2 } });
```

Node.js only (directory-based datasets). See [docs/api.md](docs/api.md#ultralytics-yolo-io) for label-line formats and lower-level helpers.

### Load JABS pose files

```ts
import { loadJabs } from "@talmolab/sleap-io.js";

// Read a JABS pose file (HDF5). Returns PredictedInstance objects.
const labels = await loadJabs("/path/to/recording_pose_est_v5.h5");
labels.labeledFrames;  // PredictedInstance poses, one Track per JABS identity
labels.staticRois;     // arena corners / lixit / food hopper as UserROI (source: "jabs")
```

Node.js only (JABS files are HDF5). Pose versions 2–6 are supported.

### Load video

```ts
import { loadVideo, Mp4BoxVideoBackend } from "@talmolab/sleap-io.js";

// From file path (Node.js) or URL (browser)
const video = await loadVideo("/path/to/video.mp4", { openBackend: false });
video.close();

// Mp4BoxVideoBackend also accepts File or Blob (browser)
const backend = new Mp4BoxVideoBackend(file); // File or Blob
```

### Lite mode (Workers-compatible)

For environments that don't support WebAssembly compilation (e.g., Cloudflare Workers), use the `/lite` entry point:

```ts
import { loadSlpMetadata, validateSlpBuffer } from "@talmolab/sleap-io.js/lite";

// Load metadata without pose coordinates
const metadata = await loadSlpMetadata(buffer);
console.log(metadata.skeletons);     // Full skeleton definitions
console.log(metadata.counts);        // { labeledFrames, instances, points, predictedPoints }
console.log(metadata.provenance);    // { sleap_version, ... }

// Quick validation
validateSlpBuffer(buffer);  // throws on invalid
```

The lite module uses [jsfive](https://github.com/usnistgov/jsfive) (pure JavaScript) instead of h5wasm (WebAssembly), enabling use in restricted environments. It can read all metadata but not pose coordinates or video frames.

## Architecture

- **I/O layer**: `loadSlp`/`saveSlp` wrap the HDF5 reader/writer in `src/codecs/slp`.
- **Data model**: `src/model` mirrors sleap-io classes and supports numpy/dict codecs.
- **Backends**: `src/video` provides browser media and embedded HDF5 decoding.
- **Streaming**: `src/codecs/slp/h5.ts` selects URL/File/FS handle strategies.

## Environments and Streaming

- **Env1 (Static SPA)**: Browser-only usage with URL streaming (CORS + Range) or the File System Access API.
- **Env2 (Server/Worker)**: Server-side or worker environments that stream `.slp` from URLs or byte buffers.
- **Env3 (Local Node/Electron)**: Optional local filesystem access for Node/Electron.

Streaming can be tuned with:

```ts
await loadSlp(url, { h5: { stream: "auto", filenameHint: "session.slp" } });
```

## Demo

The demo in `demo/` loads the built package from `dist/`. Build first, then serve the repo with a static server and open `demo/index.html`:

```bash
bun run build
```

## Development

This repo uses [Bun](https://bun.com) (pinned to `bun@1.3.14` via the `packageManager` field) as its package manager, script runner, and test runner. After [installing Bun](https://bun.com/docs/installation):

```bash
bun install          # install dependencies (uses the committed bun.lock)
bun run build        # bundle to dist/ with tsup
bun run lint         # type-check with tsc --noEmit
bun test             # run the unit suite (bun's native test runner)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more detail on the toolchain.

> The published package is plain ESM on the npm registry, so consumers can install it with any package manager (`npm`, `pnpm`, `yarn`, or `bun`) — the `npm install` line above works as-is.

## Release & Publishing

Day-to-day development uses Bun, but releases still publish to the **npm registry** (`bun publish` does not support npm's OIDC trusted publishing / provenance). The first publish must be done manually to unlock the npm package settings:

1. First-time publish (one-time):

```bash
npm login
npm publish --access public
```

2. Enable Trusted Publisher at `https://www.npmjs.com/package/@talmolab/sleap-io.js/access`.

After that, GitHub Releases trigger the publish workflow automatically.

## Links

- Python sleap-io: https://github.com/talmolab/sleap-io
- Docs: https://io.sleap.ai
