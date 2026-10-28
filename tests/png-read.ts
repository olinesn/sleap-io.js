/**
 * Minimal PNG reader for tests: parses an 8-bit, non-interlaced PNG's header,
 * inflates its IDAT data and undoes each row's filter (None/Sub/Up/Average/
 * Paeth), returning the raw pixels. Independent of the encoders under test.
 */
import { inflate } from "pako";

const CHANNELS: Record<number, number> = { 0: 1, 4: 2, 2: 3, 6: 4 };

export function readPng(png: Uint8Array): {
  width: number;
  height: number;
  colorType: number;
  channels: number;
  filters: number[];
  pixels: Uint8Array;
} {
  const dv = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const width = dv.getUint32(16);
  const height = dv.getUint32(20);
  const colorType = png[25];
  const idat: number[] = [];
  let off = 8;
  while (off < png.length) {
    const len = dv.getUint32(off);
    const type = String.fromCharCode(...png.subarray(off + 4, off + 8));
    if (type === "IDAT") idat.push(...png.subarray(off + 8, off + 8 + len));
    off += 12 + len;
  }
  const raw = inflate(Uint8Array.from(idat));
  const channels = CHANNELS[colorType];
  const stride = width * channels;
  const pixels = new Uint8Array(stride * height);
  const filters: number[] = [];
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)];
    filters.push(f);
    const row = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const a = i >= channels ? pixels[y * stride + i - channels] : 0;
      const b = y > 0 ? pixels[(y - 1) * stride + i] : 0;
      const c =
        y > 0 && i >= channels ? pixels[(y - 1) * stride + i - channels] : 0;
      let pred = 0;
      if (f === 1) pred = a;
      else if (f === 2) pred = b;
      else if (f === 3) pred = (a + b) >> 1;
      else if (f === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      } else if (f !== 0) {
        throw new Error(`bad PNG filter type ${f}`);
      }
      pixels[y * stride + i] = (row[i] + pred) & 0xff;
    }
  }
  return { width, height, colorType, channels, filters, pixels };
}
