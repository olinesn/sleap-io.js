import { describe, it, expect, afterEach } from "../bun-test";
import { readPng } from "../png-read";
import { encodePngPixels } from "../../src/video/png-encode.js";
import {
  decodeEncoded,
  encodeImageDataToPng,
} from "../../src/video/image-decode.js";

function ramp(n: number): Uint8Array {
  return Uint8Array.from({ length: n }, (_, i) => (i * 7) % 256);
}

describe("encodePngPixels", () => {
  for (const [channels, colorType] of [
    [1, 0],
    [2, 4],
    [3, 2],
    [4, 6],
  ] as const) {
    it(`writes ${channels}-channel pixels as PNG color type ${colorType}, losslessly`, () => {
      const w = 5;
      const h = 3;
      const pixels = ramp(w * h * channels);
      const png = readPng(encodePngPixels(pixels, w, h, channels));
      expect(png.width).toBe(w);
      expect(png.height).toBe(h);
      expect(png.colorType).toBe(colorType);
      expect([...png.pixels]).toEqual([...pixels]);
    });
  }

  it("writes a PNG a real decoder reads (grayscale expands to equal RGB)", async () => {
    const gray = Uint8Array.from([0, 64, 128, 255]);
    const decoded = await decodeEncoded(encodePngPixels(gray, 2, 2, 1));
    expect(decoded.width).toBe(2);
    expect(decoded.height).toBe(2);
    const rgba = decoded.data as Uint8ClampedArray;
    for (let i = 0; i < 4; i++) {
      expect([rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]]).toEqual([
        gray[i],
        gray[i],
        gray[i],
      ]);
    }
  });

  it("rejects a channel count PNG can't express", () => {
    expect(() => encodePngPixels(new Uint8Array(10), 1, 2, 5)).toThrow(
      "5 channels",
    );
  });

  it("rejects pixel data of the wrong length", () => {
    expect(() => encodePngPixels(new Uint8Array(5), 2, 2, 1)).toThrow(
      "expected 4 bytes, got 5",
    );
  });
});

describe("encodeImageDataToPng", () => {
  const g = globalThis as Record<string, unknown>;
  const saved = { OffscreenCanvas: g.OffscreenCanvas, ImageData: g.ImageData };
  afterEach(() => {
    g.OffscreenCanvas = saved.OffscreenCanvas;
    g.ImageData = saved.ImageData;
  });

  it("encodes a 1-channel RawFrame as a grayscale PNG", async () => {
    const data = Uint8Array.from([10, 20, 30, 40, 50, 60]);
    const png = readPng(
      await encodeImageDataToPng({ data, width: 3, height: 2, channels: 1 }),
    );
    expect(png.colorType).toBe(0);
    expect([...png.pixels]).toEqual([...data]);
  });

  it("infers the channel count from the data length when not given", async () => {
    const data = new Uint8ClampedArray(2 * 2 * 3).fill(9);
    const png = readPng(
      await encodeImageDataToPng({ data, width: 2, height: 2 }),
    );
    expect(png.colorType).toBe(2);
  });

  it("never hands a frame to a browser canvas (WebKit rejects non-ImageData)", async () => {
    // A browser-like canvas whose putImageData, like WebKit's, only accepts a
    // real ImageData: "Argument 1 ('imagedata') to
    // OffscreenCanvasRenderingContext2D.putImageData must be an instance of
    // ImageData". GrayscaleVideoBackend's 1-channel RawFrame used to reach it.
    class FakeImageData {
      constructor(
        public data: Uint8ClampedArray,
        public width: number,
        public height: number,
      ) {}
    }
    let canvasUsed = false;
    class FakeOffscreenCanvas {
      getContext() {
        canvasUsed = true;
        return null;
      }
    }
    g.ImageData = FakeImageData;
    g.OffscreenCanvas = FakeOffscreenCanvas;

    const gray = { data: new Uint8ClampedArray(4 * 3), width: 4, height: 3 };
    expect(readPng(await encodeImageDataToPng(gray)).colorType).toBe(0);
    const rgba = new FakeImageData(new Uint8ClampedArray(4 * 3 * 4), 4, 3);
    await encodeImageDataToPng(rgba as unknown as ImageData);
    expect(canvasUsed).toBe(false);
  });

  it("stores an RGBA ImageData as an RGB PNG, as Python sleap-io (no alpha)", async () => {
    const data = Uint8ClampedArray.from({ length: 3 * 2 * 4 }, (_, i) =>
      i % 4 === 3 ? 255 : i * 5,
    );
    const png = readPng(
      await encodeImageDataToPng({ data, width: 3, height: 2 }),
    );
    expect(png.colorType).toBe(2);
    expect([...png.pixels]).toEqual([...data].filter((_, i) => i % 4 !== 3));
  });

  it("keeps a RawFrame's explicit channel count, even 4", async () => {
    const data = Uint8Array.from({ length: 2 * 2 * 4 }, (_, i) => i);
    const png = readPng(
      await encodeImageDataToPng({ data, width: 2, height: 2, channels: 4 }),
    );
    expect(png.colorType).toBe(6);
    expect([...png.pixels]).toEqual([...data]);
  });
});
