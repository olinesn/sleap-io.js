// Register Node.js-specific h5wasm and file I/O providers.
// This side-effect import ensures the Node entry uses h5wasm/node and native fs,
// while the browser entry (index.browser.ts) never touches these modules.
import "./codecs/slp/h5-node.js";
// Register the Node `fs`-backed default resolver for merge/matching video
// file-identity checks (kept out of the browser-reachable graph; issue #70).
import "./model/node-fs-resolver.js";
// Register the Node `fs`-backed byte source for reading .seq files from a path.
import "./video/seq-node.js";
// Register the Node `fs`-backed TIFF reader for loadLabelImages() path inputs.
import "./io/label-images-node.js";
// Register the Node `fs`-backed default image-bytes reader for ImageVideoBackend
// (image-sequence videos: a list of image paths, one per frame).
import "./video/node-image-reader.js";

export * from "./model/labels.js";
export * from "./model/labeled-frame.js";
export * from "./model/instance.js";
export * from "./model/skeleton.js";
export * from "./model/video.js";
export * from "./model/suggestions.js";
export * from "./model/labels-set.js";
export * from "./model/camera.js";
export * from "./model/identity.js";
export * from "./model/embedding.js";
export * from "./model/instance3d.js";
export * from "./model/lazy.js";
export * from "./model/roi.js";
export * from "./model/mask.js";
export * from "./model/bbox.js";
export * from "./model/centroid.js";
export * from "./model/label-image.js";
export * from "./video/backend.js";
export * from "./video/mp4box-video.js";
export * from "./video/mp4-decode-core.js";
export * from "./video/mp4box-decode-worker.js";
export * from "./video/worker-mp4-backend.js";
export * from "./video/mediabunny-video.js";
export * from "./video/avi-video.js";
export {
  configureLibavDecoder,
  isLibavDecoderConfigured,
  registerLibavH264Decoder,
  ensureNativeH264Probe,
  nativeH264DecodableSync,
  overrideNativeH264Decodable,
  LibavH264Decoder,
  type LibavDecoderConfig,
} from "./video/libav-h264-decoder.js";
export * from "./video/streaming-hdf5-video.js";
export * from "./video/image-video.js";
export {
  setImageBytesReader,
  getImageBytesReader,
  type ImageBytesReader,
} from "./video/image-source.js";
export {
  resolveVideoSource,
  videoPathCandidates,
  anchorCandidate,
  derivePrefixSwap,
  applyPrefixSwap,
  resolveFirstExisting,
  parsePath,
  formatPath,
  posixDirname,
  posixBasename,
  posixJoin,
  type PosixPath,
  type PrefixSwap,
  type ResolvedVideoSource,
} from "./video/path-resolve.js";
export {
  SeqVideoBackend,
  SeqHeader,
  SeqIndex,
  BlobByteSource,
  type ByteSource,
} from "./video/seq-video.js";
export {
  createVideoBackend,
  UnsupportedVideoFormatError,
  type VideoBackendType,
  type CreateVideoBackendOptions,
} from "./video/factory.js";
// CropRect is re-exported transitively via ./transform/index.js (single source).
export {
  CropVideoBackend,
  type CropWrapOptions,
} from "./video/crop-backend.js";
export {
  GrayscaleVideoBackend,
  type GrayscaleWrapOptions,
} from "./video/grayscale-backend.js";
export * from "./io/main.js";
export * from "./io/remote.js";
export {
  parseGdrive,
  urlFromConfirmation,
  checkDownloadHost,
  openGdrive,
  DEFAULT_MAX_BYTES,
} from "./io/gdrive.js";
export * from "./io/geojson.js";
export * from "./io/trackmate.js";
export * from "./io/ultralytics.js";
export * from "./io/coco.js";
export * from "./io/coco-node.js";
export * from "./io/jabs.js";
export * from "./io/dlc-node.js";
export * from "./codecs/dictionary.js";
export * from "./codecs/numpy.js";
export * from "./codecs/skeleton-yaml.js";
export * from "./codecs/skeleton-json.js";
export * from "./codecs/training-config.js";
export * from "./rendering/index.js";
export * from "./transform/index.js";

// Streaming HDF5 utilities for advanced use cases
export {
  StreamingH5File,
  StreamingH5Writer,
  openStreamingH5,
  openH5Worker,
  isStreamingSupported,
  isRangeSource,
  serviceRangeBridge,
  serviceWriteBridge,
  serviceTruncateBridge,
  type StreamingH5Source,
  type RangeSource,
  type RangeSink,
} from "./codecs/slp/h5-streaming.js";

// Streaming SLP reader (uses Web Worker, recommended for browser)
export { readSlpStreaming } from "./codecs/slp/read-streaming.js";

// Thrown by every SLP reader for a file too new to read (format 3.0 or above).
export { UnsupportedSlpFormatError } from "./codecs/slp/parsers.js";

// Merge/matching public API (issue #90).
// Named re-exports (not `export *`) so the underscore-prefixed internal helpers
// and other module-private utilities in matching.ts stay private.
export {
  // Enums (value + type).
  SkeletonMatchMethod,
  InstanceMatchMethod,
  TrackMatchMethod,
  VideoMatchMethod,
  FrameStrategy,
  ErrorMode,
  // Matcher classes.
  SkeletonMatcher,
  InstanceMatcher,
  TrackMatcher,
  VideoMatcher,
  // Result & error types.
  ConflictResolution,
  MergeError,
  SkeletonMismatchError,
  MergeResult,
  MatchResult,
  MergeProgressBar,
  // Preconfigured matcher singletons.
  STRUCTURE_SKELETON_MATCHER,
  SUBSET_SKELETON_MATCHER,
  OVERLAP_SKELETON_MATCHER,
  DUPLICATE_MATCHER,
  IOU_MATCHER,
  IDENTITY_INSTANCE_MATCHER,
  NAME_TRACK_MATCHER,
  IDENTITY_TRACK_MATCHER,
  AUTO_VIDEO_MATCHER,
  PATH_VIDEO_MATCHER,
  BASENAME_VIDEO_MATCHER,
  IMAGE_DEDUP_VIDEO_MATCHER,
  SHAPE_VIDEO_MATCHER,
  // Filesystem resolver injection (for tests/browser).
  setFsResolver,
  type FsResolver,
} from "./model/matching.js";
