# Lite Mode

The `/lite` entry point provides a Workers-compatible SLP reader that uses [jsfive](https://github.com/usnistgov/jsfive) (pure JavaScript) instead of h5wasm (WebAssembly).

## When to use Lite Mode

Use the lite module when:

- Running in **Cloudflare Workers** or other environments that block runtime WebAssembly compilation
- You only need **metadata** (skeletons, counts, video info) without actual pose coordinates
- You want a **smaller bundle** (~50KB vs ~800KB+ for h5wasm)
- You need **quick validation** of SLP files

## Installation

```ts
import { loadSlpMetadata, validateSlpBuffer, isHdf5Buffer } from "@talmolab/sleap-io.js/lite";
```

## API Reference

### `loadSlpMetadata(source, options?)`

Extract metadata from an SLP file without loading pose coordinates.

```ts
const response = await fetch("https://example.com/file.slp");
const buffer = await response.arrayBuffer();
const metadata = await loadSlpMetadata(buffer);

console.log(metadata.version);      // "1.3.4"
console.log(metadata.skeletons);    // Full skeleton definitions
console.log(metadata.tracks);       // Track names
console.log(metadata.videos);       // Video metadata
console.log(metadata.counts);       // { labeledFrames, instances, points, predictedPoints }
console.log(metadata.provenance);   // { sleap_version, ... }
```

**Parameters:**

- `source`: `ArrayBuffer` or `Uint8Array` containing the SLP file
- `options.filename`: Optional filename hint for embedded video paths

**Returns:** `Promise<SlpMetadata>`

**Throws:** `UnsupportedSlpFormatError` if the file is in an SLP format newer than this library can read (3.0 or above)

### `validateSlpBuffer(source)`

Quick structural validation of an SLP file.

```ts
try {
  validateSlpBuffer(buffer);
  console.log("Valid SLP file");
} catch (e) {
  console.error("Invalid:", e.message);
}
```

**Parameters:**

- `source`: `ArrayBuffer` or `Uint8Array` to validate

**Returns:** `true` if valid, throws `Error` with details if invalid, or `UnsupportedSlpFormatError` if the file is in an SLP format newer than this library can read (3.0 or above)

### `isHdf5Buffer(source)`

Check if a buffer starts with the HDF5 magic number.

```ts
if (isHdf5Buffer(buffer)) {
  // Might be an SLP file, do full validation
  const metadata = await loadSlpMetadata(buffer);
}
```

**Parameters:**

- `source`: `ArrayBuffer` or `Uint8Array` to check

**Returns:** `boolean`

## Types

### `SlpMetadata`

```ts
interface SlpMetadata {
  /** SLEAP version that created this file (e.g., "1.3.4") */
  version: string;

  /** HDF5 format ID (e.g., 1.2) */
  formatId: number;

  /** Skeleton definitions with nodes, edges, and symmetries */
  skeletons: Skeleton[];

  /** Track definitions */
  tracks: Track[];

  /** Video metadata (without loaded backends) */
  videos: VideoMetadata[];

  /** Suggestion frame metadata */
  suggestions: SuggestionMetadata[];

  /** Multi-camera recording session metadata */
  sessions: SessionMetadata[];

  /** Dataset counts */
  counts: {
    labeledFrames: number;
    instances: number;
    points: number;
    predictedPoints: number;
  };

  /** Whether any video has embedded image data */
  hasEmbeddedImages: boolean;

  /** Raw provenance data (SLEAP version, build info, etc.) */
  provenance?: Record<string, unknown>;
}
```

### `VideoMetadata`

```ts
interface VideoMetadata {
  filename: string;
  dataset?: string;
  format?: string;
  width?: number;
  height?: number;
  channels?: number;
  fps?: number;
  frameCount?: number;
  channelOrder?: string;
  embedded: boolean;
  sourceVideo?: { filename: string };
}
```

## What's Available vs Not Available

### ✅ Available in Lite Mode

| Metadata | Description |
|----------|-------------|
| Skeletons | Full definitions with nodes, edges, symmetries |
| Tracks | Track names |
| Videos | Filename, dimensions, format, fps |
| Counts | Frame, instance, and point counts |
| Provenance | SLEAP version and build info |
| Sessions | Multi-camera session metadata |
| Suggestions | Suggestion frame info |

### ❌ Not Available in Lite Mode

| Data | Reason |
|------|--------|
| Pose coordinates (x, y) | Requires compound dataset support |
| Point visibility/scores | Requires compound dataset support |
| Instance-frame mapping | Requires compound dataset support |
| Video frame data | Requires VLEN sequence support |

## Full vs Lite Comparison

| Feature | `loadSlp()` | `loadSlpMetadata()` |
|---------|-------------|---------------------|
| HDF5 backend | h5wasm (WASM) | jsfive (pure JS) |
| Bundle size | ~800KB+ | ~50KB |
| Workers compatible | ❌ | ✅ |
| Returns | `Labels` object | `SlpMetadata` object |
| Pose coordinates | ✅ | ❌ |
| Video frame access | ✅ | ❌ |
| Skeleton parsing | ✅ | ✅ |
| Metadata | ✅ | ✅ |

> **Dependencies:** The full entry point pulls in `pako` (^2.1.0) for label image zlib compression, but lite mode does not import it — the `/lite` bundle stays free of `pako` and `h5wasm`.

## Example: File Upload Validation

```ts
import { loadSlpMetadata, validateSlpBuffer, isHdf5Buffer } from "@talmolab/sleap-io.js/lite";

async function handleUpload(file: File) {
  const buffer = await file.arrayBuffer();

  // Quick check
  if (!isHdf5Buffer(buffer)) {
    throw new Error("Not an HDF5 file");
  }

  // Validate structure
  validateSlpBuffer(buffer);

  // Extract metadata
  const metadata = await loadSlpMetadata(buffer, { filename: file.name });

  return {
    valid: true,
    skeletons: metadata.skeletons.map(s => s.name),
    frameCount: metadata.counts.labeledFrames,
    instanceCount: metadata.counts.instances,
    sleapVersion: metadata.provenance?.sleap_version,
  };
}
```
