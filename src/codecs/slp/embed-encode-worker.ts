/**
 * Worker half of the streaming NEW-embed path (embedding frames of a
 * continuous video, e.g. an mp4, into a pkg.slp written straight to disk).
 *
 * The main thread reads each frame (the decode itself may already run in the
 * mp4 decode worker) and hands it over, transferred where it can be: an
 * `ImageBitmap`, raw pixels, or already-encoded PNG bytes. These helpers turn
 * that into a PNG and append it to the open destination file's
 * `video{i}/video` dataset row by row, so neither the UI thread nor the wasm
 * heap ever holds the whole embedded video. It writes Python sleap-io's
 * on-disk layout, as the main-thread `writeEmbeddedVideoData` (write.ts) does.
 *
 * Inlined into `H5_WORKER_CODE`, where `writeSourceVideoJsonWorker` comes
 * from. Kept as its own string so tests can evaluate it directly (with that
 * supplied) against h5wasm on Node. The dataset writer here is shared with the
 * worker's raw-copy `appendEmbeddedVideos`.
 *
 * No backticks or `${` in the code below: it sits inside a template literal.
 *
 * @module
 */
export const EMBED_ENCODE_WORKER_CODE = `
// --- streaming new-embed: PNG encode (KEEP IN LOCKSTEP with
// ../../video/png-encode.ts, which the main-thread embed path uses) ---
var PNG_COLOR_TYPE_ENC = { 1: 0, 2: 4, 3: 2, 4: 6 };
var CRC_TABLE_ENC = null;

function crc32Enc(bytes) {
  if (CRC_TABLE_ENC === null) {
    CRC_TABLE_ENC = new Uint32Array(256);
    for (var n = 0; n < 256; n++) {
      var c = n;
      for (var k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      CRC_TABLE_ENC[n] = c >>> 0;
    }
  }
  var crc = 0xffffffff;
  for (var i = 0; i < bytes.length; i++) {
    crc = CRC_TABLE_ENC[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunkEnc(type, data) {
  var out = new Uint8Array(12 + data.length);
  var dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (var i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32Enc(out.subarray(4, 8 + data.length)));
  return out;
}

// A zlib stream of STORED (uncompressed) deflate blocks: a valid PNG IDAT
// where CompressionStream is unavailable. Bigger files, same pixels.
function zlibStoredEnc(raw) {
  var MAX = 65535;
  var blocks = Math.max(1, Math.ceil(raw.length / MAX));
  var out = new Uint8Array(2 + raw.length + blocks * 5 + 4);
  out[0] = 0x78;
  out[1] = 0x01;
  var o = 2;
  for (var b = 0; b < blocks; b++) {
    var start = b * MAX;
    var len = Math.min(MAX, raw.length - start);
    out[o++] = b === blocks - 1 ? 1 : 0;
    out[o++] = len & 0xff;
    out[o++] = (len >>> 8) & 0xff;
    out[o++] = ~len & 0xff;
    out[o++] = (~len >>> 8) & 0xff;
    out.set(raw.subarray(start, start + len), o);
    o += len;
  }
  var a = 1, s = 0;
  for (var i = 0; i < raw.length; i++) {
    a = (a + raw[i]) % 65521;
    s = (s + a) % 65521;
  }
  new DataView(out.buffer).setUint32(o, ((s << 16) | a) >>> 0);
  return out;
}

async function zlibDeflateEnc(raw) {
  if (typeof CompressionStream === 'undefined') return zlibStoredEnc(raw);
  var stream = new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Every row Sub-filtered, as OpenCV (KEEP IN LOCKSTEP with png-encode.ts's
// subFilterScanlines). Compression is CompressionStream's zlib, whose level
// can't be set (the main thread's pako encoder uses OpenCV's level 1 + RLE).
function subFilterScanlinesEnc(pixels, width, height, channels) {
  var stride = width * channels;
  var out = new Uint8Array((stride + 1) * height);
  for (var y = 0; y < height; y++) {
    var o = y * (stride + 1);
    var r = y * stride;
    out[o] = 1;
    for (var c = 0; c < channels; c++) out[o + 1 + c] = pixels[r + c];
    for (var i = channels; i < stride; i++) {
      out[o + 1 + i] = (pixels[r + i] - pixels[r + i - channels]) & 0xff;
    }
  }
  return out;
}

async function encodePngEnc(pixels, width, height, channels) {
  var colorType = PNG_COLOR_TYPE_ENC[channels];
  if (colorType === undefined) {
    throw new Error('Cannot PNG-encode a frame with ' + channels + ' channels (expected 1-4).');
  }
  var stride = width * channels;
  if (pixels.length !== stride * height) {
    throw new Error(
      'Cannot PNG-encode a ' + width + 'x' + height + ' frame with ' + channels +
      ' channel(s): expected ' + stride * height + ' bytes, got ' + pixels.length + '.'
    );
  }
  var raw = subFilterScanlinesEnc(pixels, width, height, channels);
  var ihdr = new Uint8Array(13);
  var dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr[8] = 8;
  ihdr[9] = colorType;
  var chunks = [
    new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunkEnc('IHDR', ihdr),
    pngChunkEnc('IDAT', await zlibDeflateEnc(raw)),
    pngChunkEnc('IEND', new Uint8Array(0))
  ];
  var total = 0;
  for (var c = 0; c < chunks.length; c++) total += chunks[c].length;
  var out = new Uint8Array(total);
  var off = 0;
  for (var d = 0; d < chunks.length; d++) { out.set(chunks[d], off); off += chunks[d].length; }
  return out;
}

// Pixels of a transferred frame, ready to encode. A bitmap (or a pixels payload
// flagged rgba, e.g. an ImageData) is RGBA by construction: its alpha is
// dropped, as video frames have none. collapseToGray keeps channel 0 only,
// exactly as grayscaleFrame (../../transform/frame.ts) - KEEP IN LOCKSTEP.
function framePixelsEnc(frame, collapseToGray) {
  var data, width, height, channels, rgba;
  if (frame.kind === 'bitmap') {
    var bmp = frame.bitmap;
    width = bmp.width;
    height = bmp.height;
    var ctx = new OffscreenCanvas(width, height).getContext('2d');
    if (!ctx) throw new Error('Failed to get a 2D context to read an embedded frame');
    ctx.drawImage(bmp, 0, 0);
    try { bmp.close(); } catch (e) {}
    data = ctx.getImageData(0, 0, width, height).data;
    channels = 4;
    rgba = true;
  } else {
    data = frame.data;
    width = frame.width;
    height = frame.height;
    channels = frame.channels;
    rgba = !!frame.rgba;
  }
  var outCh = collapseToGray ? 1 : (rgba && channels === 4 ? 3 : channels);
  if (outCh === channels) {
    return { pixels: data instanceof Uint8Array ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength), width: width, height: height, channels: channels };
  }
  var n = width * height;
  var out = new Uint8Array(n * outCh);
  for (var p = 0, i = 0, o = 0; p < n; p++, i += channels, o += outCh) {
    for (var ch = 0; ch < outCh; ch++) out[o + ch] = data[i + ch];
  }
  return { pixels: out, width: width, height: height, channels: outCh };
}

// The bytes to store for one transferred frame.
async function encodeFrameEnc(frame, collapseToGray) {
  if (frame.kind === 'png') return frame.bytes;
  var px = framePixelsEnc(frame, collapseToGray);
  return encodePngEnc(px.pixels, px.width, px.height, px.channels);
}

// --- embedded video dataset writer, in Python sleap-io's layout (KEEP IN
// LOCKSTEP with write.ts's writeEmbeddedVideoData). Shared by the raw-copy
// appendEmbeddedVideos and the new-embed messages. '{group}/video' is a 2-D
// int8 dataset, one encoded image per row, zero-padded; its width grows to the
// widest image (chunked, since the widest isn't known until the last frame). ---

// KEEP IN LOCKSTEP with write.ts's paddedChunkWidth.
function paddedChunkWidthEnc(firstFrameBytes) {
  var width = 4096;
  while (width < firstFrameBytes / 8 && width < 262144) width *= 2;
  return width;
}

function beginPaddedVideoEnc(file, group, rows) {
  file.create_group(group);
  return { file: file, group: group, rows: rows, ds: null, width: 0, frameNumbers: [] };
}

function appendPaddedRowEnc(state, frameIdx, bytes) {
  if (!bytes || bytes.length === 0) {
    throw new Error(state.group + ': frame ' + frameIdx + ' has no image bytes');
  }
  var row = state.frameNumbers.length;
  if (row >= state.rows) {
    throw new Error(state.group + ': more frames than the ' + state.rows + ' planned');
  }
  if (!state.ds) {
    state.file.create_dataset({
      name: state.group + '/video',
      data: new Int8Array(0),
      shape: [state.rows, 0],
      maxshape: [state.rows, null],
      chunks: [1, paddedChunkWidthEnc(bytes.length)],
      dtype: '<b'
    });
    state.ds = state.file.get(state.group + '/video');
  }
  if (bytes.length > state.width) {
    state.width = bytes.length;
    state.ds.resize([state.rows, state.width]);
  }
  state.ds.write_slice(
    [[row, row + 1], [0, bytes.length]],
    new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  );
  state.frameNumbers.push(frameIdx);
}

// Attributes exactly as Python sleap-io (KEEP IN LOCKSTEP with write.ts's
// writeEmbeddedVideoAttrs): variable-length UTF-8 strings, int64 dimensions,
// float64 fps.
function writeEmbeddedVideoAttrsEnc(ds, format, channelOrder, attrs) {
  ds.create_attribute('format', format);
  ds.create_attribute('channel_order', channelOrder);
  if (!attrs) return;
  var keys = ['frames', 'height', 'width', 'channels'];
  for (var k = 0; k < keys.length; k++) {
    ds.create_attribute(keys[k], BigInt64Array.from([BigInt(attrs[keys[k]])]), [], '<q');
  }
  if (attrs.fps != null) ds.create_attribute('fps', attrs.fps, [], '<d');
}

// Finish a video: an empty one becomes Python's empty embedded video (a 1-D
// empty dataset); then attributes, frame_numbers (int64, or Python's empty
// float64) and the source_video lineage.
function finishPaddedVideoEnc(state, format, channelOrder, attrs, sourceVideoJson) {
  var file = state.file;
  if (!state.ds) {
    file.create_dataset({ name: state.group + '/video', data: new Int8Array(0), shape: [0], dtype: '<b' });
    state.ds = file.get(state.group + '/video');
  }
  writeEmbeddedVideoAttrsEnc(state.ds, format, channelOrder, attrs);
  var fns = state.frameNumbers;
  if (fns.length > 0) {
    file.create_dataset({
      name: state.group + '/frame_numbers',
      data: BigInt64Array.from(fns, function(f) { return BigInt(f); }),
      shape: [fns.length],
      dtype: '<q'
    });
  } else {
    file.create_dataset({ name: state.group + '/frame_numbers', data: new Float64Array(0), shape: [0], dtype: '<d' });
  }
  if (sourceVideoJson) writeSourceVideoJsonWorker(file, state.group, sourceVideoJson);
  file.flush();
  return { framesWritten: fns.length };
}

// --- streaming new-embed messages ---
function beginEncodedVideoEnc(file, entry) {
  var state = beginPaddedVideoEnc(file, 'video' + entry.videoIndex, entry.frameNumbers.length);
  state.entry = entry;
  return state;
}

function appendEncodedBytesEnc(state, frameIdx, bytes) {
  appendPaddedRowEnc(state, frameIdx, bytes);
}

function endEncodedVideoEnc(file, state) {
  var entry = state.entry;
  return finishPaddedVideoEnc(state, entry.format, entry.channelOrder, entry.videoAttrs, entry.sourceVideoJson);
}
`;
