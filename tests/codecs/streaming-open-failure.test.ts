/**
 * A worker that fails to open its file is ended. Nothing else holds it, so it would
 * otherwise stay alive for the life of the page with h5wasm loaded in it. The Worker
 * is faked: the Node suite has none that can run the real code.
 */
import { describe, it, expect, afterEach, vi } from "../bun-test";
import { openH5Worker } from "../../src/codecs/slp/h5-streaming.js";

afterEach(() => vi.unstubAllGlobals());

describe("openH5Worker", () => {
  it("ends the worker when the file won't open, and throws what went wrong", async () => {
    const terminate = vi.fn();
    vi.stubGlobal(
      "Worker",
      class {
        onmessage?: (e: unknown) => void;
        terminate = terminate;
        addEventListener() {}
        postMessage(msg: { id: number; type: string }) {
          const ok = msg.type === "init";
          queueMicrotask(() =>
            this.onmessage?.({
              data: ok
                ? { id: msg.id, success: true }
                : { id: msg.id, success: false, error: "could not open" },
            }),
          );
        }
      },
    );
    const file = new File([new Uint8Array(8)], "bad.slp");
    await expect(openH5Worker(file)).rejects.toThrow("could not open");
    expect(terminate).toHaveBeenCalledTimes(1);
  });

  it("leaves it running when the file opens", async () => {
    const terminate = vi.fn();
    vi.stubGlobal(
      "Worker",
      class {
        onmessage?: (e: unknown) => void;
        terminate = terminate;
        addEventListener() {}
        postMessage(msg: { id: number }) {
          queueMicrotask(() =>
            this.onmessage?.({ data: { id: msg.id, success: true, keys: [] } }),
          );
        }
      },
    );
    await openH5Worker(new File([new Uint8Array(8)], "ok.slp"));
    expect(terminate).not.toHaveBeenCalled();
  });
});
