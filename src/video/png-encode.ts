// src/video/png-encode.ts
//
// Pure-JS PNG encoding of raw 8-bit pixels (zlib via `pako`), for browser and
// Node alike: no canvas, no native module. Unlike a canvas, which can only
// write RGBA, this keeps the pixels' own channel count, so a grayscale frame
// becomes a 1-channel (grayscale) PNG, as Python sleap-io writes when it embeds
// one.
//
// By default it encodes the way OpenCV does (what PyQt SLEAP's Python sleap-io
// uses when OpenCV is installed): every row Sub-filtered, zlib at its fastest
// level with the run-length strategy. On video frames that's ~3x faster than
// adaptive filtering at zlib 6 and gives smaller files (a 1024x1024 grayscale
// frame: 13 ms / 393 KB vs 40 ms / 408 KB; OpenCV writes 394 KB).
import { deflate } from "pako";

type ZlibLevel = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

/** PNG color type for each channel count: gray, gray+alpha, RGB, RGBA. */
const COLOR_TYPE: Record<number, number> = { 1: 0, 2: 4, 3: 2, 4: 6 };

/**
 * Encode row-major 8-bit pixels to a PNG byte stream.
 *
 * @param pixels - Interleaved pixels, `width * height * channels` bytes.
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @param channels - 1 (gray), 2 (gray+alpha), 3 (RGB) or 4 (RGBA).
 * @param compressLevel - zlib compression level 0–9. Default: OpenCV's
 *   settings (level 1, run-length strategy).
 */
export function encodePngPixels(
  pixels: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  channels: number,
  compressLevel: number | null = null,
): Uint8Array {
  const colorType = COLOR_TYPE[channels];
  if (colorType === undefined) {
    throw new Error(
      `Cannot PNG-encode a frame with ${channels} channels (expected 1–4).`,
    );
  }
  const stride = width * channels;
  if (pixels.length !== stride * height) {
    throw new Error(
      `Cannot PNG-encode a ${width}x${height} frame with ${channels} ` +
        `channel(s): expected ${stride * height} bytes, got ${pixels.length}.`,
    );
  }

  const raw = subFilterScanlines(pixels, width, height, channels);
  const compressed = deflate(
    raw,
    compressLevel == null
      ? { level: 1, strategy: Z_RLE }
      : {
          level: Math.min(9, Math.max(0, compressLevel)) as ZlibLevel,
        },
  );

  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = colorType;
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const signature = new Uint8Array([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
  ]);
  const chunks = [
    signature,
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", compressed),
    pngChunk("IEND", new Uint8Array(0)),
  ];

  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

/** zlib's run-length strategy (`Z_RLE`), OpenCV's default for PNG. */
const Z_RLE = 3;

/**
 * PNG scanlines with every row Sub-filtered (each byte minus the byte one
 * pixel to its left), as OpenCV writes them. Lossless; cheap to compute and,
 * on video frames, compresses as well as adaptive filtering.
 */
export function subFilterScanlines(
  pixels: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  channels: number,
): Uint8Array {
  const stride = width * channels;
  const out = new Uint8Array((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    const o = y * (stride + 1);
    const r = y * stride;
    out[o] = 1; // filter type Sub
    for (let i = 0; i < channels; i++) out[o + 1 + i] = pixels[r + i];
    for (let i = channels; i < stride; i++) {
      out[o + 1 + i] = (pixels[r + i] - pixels[r + i - channels]) & 0xff;
    }
  }
  return out;
}

/** Build a length-prefixed, CRC32-suffixed PNG chunk. */
function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const typeBytes = new Uint8Array([
    type.charCodeAt(0),
    type.charCodeAt(1),
    type.charCodeAt(2),
    type.charCodeAt(3),
  ]);
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  out.set(typeBytes, 4);
  out.set(data, 8);
  const crcInput = new Uint8Array(4 + data.length);
  crcInput.set(typeBytes, 0);
  crcInput.set(data, 4);
  dv.setUint32(8 + data.length, crc32(crcInput));
  return out;
}

// CRC32 table (lazily built once).
let CRC_TABLE: Uint32Array | null = null;
function crc32(bytes: Uint8Array): number {
  if (CRC_TABLE === null) {
    CRC_TABLE = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) {
        c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      }
      CRC_TABLE[n] = c >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
