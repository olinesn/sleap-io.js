import { describe, it, expect } from "../bun-test";
import {
  WorkerMp4BoxBackend,
  type WorkerLike,
} from "../../src/video/worker-mp4-backend";
import type {
  Mp4ParseResult,
  Mp4Sample,
} from "../../src/video/mp4-decode-core";
import type { ByteSourceDescriptor } from "../../src/video/mp4box-decode-worker";

/** A controllable stand-in for the decode Worker. */
class FakeWorker implements WorkerLike {
  posted: Array<Record<string, unknown>> = [];
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  terminated = false;

  postMessage(message: unknown): void {
    this.posted.push(message as Record<string, unknown>);
  }
  terminate(): void {
    this.terminated = true;
  }
  /** Simulate a worker→main message. */
  emit(message: unknown): void {
    this.onmessage?.({ data: message });
  }
  lastOfType(type: string): Record<string, unknown> | undefined {
    for (let i = this.posted.length - 1; i >= 0; i -= 1) {
      if (this.posted[i].type === type) return this.posted[i];
    }
    return undefined;
  }
  countOfType(type: string): number {
    return this.posted.filter((m) => m.type === type).length;
  }
}

interface FakeBitmap {
  id: string;
  closed: boolean;
  close(): void;
}
function fakeBitmap(id: string): ImageBitmap {
  const b: FakeBitmap = {
    id,
    closed: false,
    close() {
      this.closed = true;
    },
  };
  return b as unknown as ImageBitmap;
}

function makeParseResult(count: number, kfStep = 30): Mp4ParseResult {
  const samples: Mp4Sample[] = [];
  const keyframeIndices: number[] = [];
  for (let i = 0; i < count; i += 1) {
    const isKeyframe = i % kfStep === 0;
    if (isKeyframe) keyframeIndices.push(i);
    samples.push({
      offset: i * 10,
      size: 10,
      timestamp: i * 1000,
      duration: 1000,
      isKeyframe,
      cts: i,
      decodeIndex: i,
    });
  }
  return {
    samples,
    keyframeIndices,
    config: { codec: "avc1.42E01E", codedWidth: 64, codedHeight: 64 },
    shape: [count, 64, 64, 3],
    fps: 30,
    fileSize: count * 10,
  };
}

const BYTE_SOURCE: ByteSourceDescriptor = {
  kind: "tauri",
  url: "ipc://localhost/read_range",
  headers: { "Tauri-Invoke-Key": "k" },
  path: "/x.mp4",
  size: 1000,
};

const tick = () => new Promise((r) => setTimeout(r, 0));

function makeBackend(parseResult = makeParseResult(1000)) {
  const fake = new FakeWorker();
  const backend = new WorkerMp4BoxBackend({
    parseResult,
    byteSource: BYTE_SOURCE,
    filename: "x.mp4",
    createWorker: () => fake,
  });
  return { fake, backend };
}

async function makeReadyBackend(parseResult = makeParseResult(1000)) {
  const { fake, backend } = makeBackend(parseResult);
  fake.emit({ type: "ready" });
  await backend.ready;
  return { fake, backend };
}

describe("WorkerMp4BoxBackend init", () => {
  it("posts init and resolves ready on the worker's ready message", async () => {
    const { fake, backend } = makeBackend();
    expect(fake.posted[0].type).toBe("init");
    expect((fake.posted[0] as { byteSource: unknown }).byteSource).toEqual(
      BYTE_SOURCE,
    );
    fake.emit({ type: "ready" });
    await expect(backend.ready).resolves.toBeUndefined();
    expect(backend.shape).toEqual([1000, 64, 64, 3]);
    expect(backend.fps).toBe(30);
  });

  it("create() rejects (and cleans up) when the worker is unsupported", async () => {
    const fake = new FakeWorker();
    const p = WorkerMp4BoxBackend.create({
      parseResult: makeParseResult(10),
      byteSource: BYTE_SOURCE,
      filename: "x.mp4",
      createWorker: () => fake,
    });
    fake.emit({ type: "unsupported", reason: "no WebCodecs in worker" });
    await expect(p).rejects.toThrow(/unsupported/);
    expect(fake.terminated).toBe(true);
  });
});

describe("WorkerMp4BoxBackend getFrame", () => {
  it("cache miss → posts a decode with the planned range, resolves on target bitmap", async () => {
    const { fake, backend } = await makeReadyBackend();
    const gp = backend.getFrame(50);
    await tick();
    const decode = fake.lastOfType("decode");
    expect(decode).toMatchObject({
      type: "decode",
      target: 50,
      start: 30, // keyframe before 50
      end: 110, // min(50 + lookahead(60), 999)
      cacheStart: 30,
      cacheEnd: 110,
    });
    const reqId = decode?.reqId as number;
    const bmp = fakeBitmap("f50");
    fake.emit({ type: "bitmap", reqId, frame: 50, bitmap: bmp });
    await expect(gp).resolves.toBe(bmp);
  });

  it("caches neighbor frames pushed during the same decode", async () => {
    const { fake, backend } = await makeReadyBackend();
    const gp = backend.getFrame(50);
    await tick();
    const reqId = fake.lastOfType("decode")?.reqId as number;
    fake.emit({ type: "bitmap", reqId, frame: 51, bitmap: fakeBitmap("f51") });
    fake.emit({ type: "bitmap", reqId, frame: 50, bitmap: fakeBitmap("f50") });
    await gp;
    // Frame 51 is now a cache hit — no new decode posted.
    const before = fake.countOfType("decode");
    const hit = await backend.getFrame(51);
    expect((hit as unknown as FakeBitmap).id).toBe("f51");
    expect(fake.countOfType("decode")).toBe(before);
  });

  it("resolves with the cached frame on decodeDone if the target never arrived", async () => {
    const { fake, backend } = await makeReadyBackend();
    const gp = backend.getFrame(50);
    await tick();
    const reqId = fake.lastOfType("decode")?.reqId as number;
    // decodeDone without a bitmap for 50 → resolves null (nothing cached).
    fake.emit({ type: "decodeDone", reqId, aborted: false });
    await expect(gp).resolves.toBeNull();
  });

  it("drop-stale: a newer demand aborts older in-flight demands + settles them", async () => {
    const { fake, backend } = await makeReadyBackend();
    const gp1 = backend.getFrame(50);
    await tick();
    const reqId1 = fake.lastOfType("decode")?.reqId as number;

    // A newer demand supersedes the first (fast-scrub case).
    const gp2 = backend.getFrame(500);
    await tick();

    // The stale request was aborted in the worker so its serial queue can't back up,
    const abort = fake.lastOfType("abort");
    expect(abort).toMatchObject({ type: "abort", reqId: reqId1 });
    // ...and its promise settled NOW (null — nothing cached) instead of hanging
    // until the worker eventually decodes a long-superseded frame.
    await expect(gp1).resolves.toBeNull();

    // The new demand gets its own decode and resolves normally.
    const decode = fake.lastOfType("decode");
    expect(decode?.target).toBe(500);
    expect(decode?.reqId).not.toBe(reqId1);
    const bmp = fakeBitmap("f500");
    fake.emit({
      type: "bitmap",
      reqId: decode?.reqId,
      frame: 500,
      bitmap: bmp,
    });
    await expect(gp2).resolves.toBe(bmp);
  });
});

describe("WorkerMp4BoxBackend pinned reads (embedding)", () => {
  it("a newer demand does not cancel a pinned read", async () => {
    const { fake, backend } = await makeReadyBackend();
    const pinned = backend.getFrame(50, { pinned: true });
    await tick();
    const pinnedReq = fake.lastOfType("decode")?.reqId as number;

    const ui = backend.getFrame(500); // e.g. the app showing another frame
    await tick();

    expect(
      fake.posted.some((m) => m.type === "abort" && m.reqId === pinnedReq),
    ).toBe(false);
    const bmp = fakeBitmap("f50");
    fake.emit({ type: "bitmap", reqId: pinnedReq, frame: 50, bitmap: bmp });
    await expect(pinned).resolves.toBe(bmp);

    const uiReq = fake.lastOfType("decode")?.reqId as number;
    const bmp500 = fakeBitmap("f500");
    fake.emit({ type: "bitmap", reqId: uiReq, frame: 500, bitmap: bmp500 });
    await expect(ui).resolves.toBe(bmp500);
  });

  it("a pinned read does not cancel an older demand either", async () => {
    const { fake, backend } = await makeReadyBackend();
    const ui = backend.getFrame(50);
    await tick();
    const uiReq = fake.lastOfType("decode")?.reqId as number;

    backend.getFrame(500, { pinned: true });
    await tick();

    expect(
      fake.posted.some((m) => m.type === "abort" && m.reqId === uiReq),
    ).toBe(false);
    const bmp = fakeBitmap("f50");
    fake.emit({ type: "bitmap", reqId: uiReq, frame: 50, bitmap: bmp });
    await expect(ui).resolves.toBe(bmp);
  });
});

describe("WorkerMp4BoxBackend abort", () => {
  it("posts abort and resolves when the caller's signal fires", async () => {
    const { fake, backend } = await makeReadyBackend();
    const controller = new AbortController();
    const gp = backend.getFrame(50, { signal: controller.signal });
    await tick();
    const reqId = fake.lastOfType("decode")?.reqId as number;
    controller.abort();
    const abort = fake.lastOfType("abort");
    expect(abort).toMatchObject({ type: "abort", reqId });
    await expect(gp).resolves.toBeNull();
  });

  it("never issues an already-aborted read", async () => {
    const { fake, backend } = await makeReadyBackend();
    const controller = new AbortController();
    controller.abort();
    const before = fake.countOfType("decode");
    const result = await backend.getFrame(50, { signal: controller.signal });
    expect(result).toBeNull();
    expect(fake.countOfType("decode")).toBe(before);
  });
});

describe("WorkerMp4BoxBackend decodeAhead", () => {
  it("posts a decode, caches pushed aheadFrames, and coalesces", async () => {
    const { fake, backend } = await makeReadyBackend();
    backend.decodeAhead(100);
    const decode = fake.lastOfType("decode");
    expect(decode).toMatchObject({ type: "decode", target: 100, start: 90 });
    const reqId = decode?.reqId as number;

    // Second call while in flight → coalesced (no new decode).
    backend.decodeAhead(100);
    expect(fake.countOfType("decode")).toBe(1);

    // Pushed frame gets cached; then decodeDone frees the ahead slot.
    fake.emit({
      type: "bitmap",
      reqId,
      frame: 105,
      bitmap: fakeBitmap("a105"),
    });
    fake.emit({ type: "decodeDone", reqId, aborted: false });

    const hit = await backend.getFrame(105);
    expect((hit as unknown as FakeBitmap).id).toBe("a105");

    // After the ahead completed, a new decodeAhead can post again.
    backend.decodeAhead(200);
    expect(fake.countOfType("decode")).toBeGreaterThan(1);
  });

  it("a demand getFrame preempts in-flight decode-ahead", async () => {
    const { fake, backend } = await makeReadyBackend();
    backend.decodeAhead(100);
    const aheadReqId = fake.lastOfType("decode")?.reqId as number;
    backend.getFrame(500);
    await tick();
    const abort = fake.lastOfType("abort");
    expect(abort).toMatchObject({ type: "abort", reqId: aheadReqId });
  });
});

describe("WorkerMp4BoxBackend nearestKeyframe", () => {
  it("answers synchronously from the keyframe table", async () => {
    const { backend } = await makeReadyBackend();
    expect(backend.nearestKeyframe(75)).toBe(60);
    expect(backend.nearestKeyframe(30)).toBe(30);
    expect(backend.nearestKeyframe(5)).toBe(0);
    expect(backend.nearestKeyframe(100000)).toBe(990); // clamped
  });
});

describe("WorkerMp4BoxBackend close", () => {
  it("terminates the worker, closes cached bitmaps, and resolves pending", async () => {
    const { fake, backend } = await makeReadyBackend();
    const gp = backend.getFrame(50);
    await tick();
    const reqId = fake.lastOfType("decode")?.reqId as number;
    const cached = fakeBitmap("c");
    fake.emit({ type: "bitmap", reqId, frame: 51, bitmap: cached });

    // Pending read for 50 hasn't resolved yet.
    const pending = backend.getFrame(52);
    await tick();

    backend.close();
    expect(fake.terminated).toBe(true);
    expect((cached as unknown as FakeBitmap).closed).toBe(true);
    await expect(gp).resolves.toBeNull();
    await expect(pending).resolves.toBeNull();
  });
});
