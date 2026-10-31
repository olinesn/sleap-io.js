/**
 * Any read the main-thread reader makes that HDF5 says failed is an error.
 *
 * This fails them where HDF5 would, underneath `guardH5Status`.
 */
import { describe, expect, it, setDefaultTimeout, vi } from "./bun-test";
import { fileURLToPath } from "node:url";
import { Module } from "h5wasm/node";
import { getH5Module, guardH5Status } from "../src/codecs/slp/h5.js";
import { readSlp } from "../src/codecs/slp/read.js";
import { poses } from "./worker-file";

setDefaultTimeout(120_000);

const path = (name: string): string =>
  fileURLToPath(new URL(`./data/slp/${name}`, import.meta.url));

const reads = new Set<string>();
let failing: string | undefined;

/** What HDF5 says of a read that `get_dataset_data` or `get_attribute_data` makes. */
function reportingFailure(name: string, key: (args: unknown[]) => string) {
  const calls = Module as unknown as Record<
    string,
    (...a: unknown[]) => number
  >;
  const read = calls[name];
  calls[name] = function (this: unknown, ...args) {
    const which = key(args);
    reads.add(which);
    return which === failing ? -1 : read.apply(this, args);
  };
}

const prepared = (async () => {
  const module = await getH5Module();
  reportingFailure("get_dataset_data", (args) => String(args[1]));
  reportingFailure("get_attribute_data", (args) => `${args[1]} @${args[2]}`);
  guardH5Status(module);
})();

const WITHOUT_VIDEOS = [
  "py_written_30_twin_legacy.slp",
  "py_written_30_twin.slp",
  "py_written_25_identity.slp",
  "py_written_27_category.slp",
  "py_written_28_multiview3d.slp",
  "py_written_annotations.slp",
  "predictions_1.2.7_provenance_and_tracking.slp",
  "js_flat_annotations.slp",
  "multiview.slp",
  "imgvideo.slp",
  "typical.slp",
];
const WITH_VIDEOS = [
  "minimal_instance.pkg.slp",
  "spilled_source_video.pkg.slp",
  "cropped_format_2_3.pkg.slp",
  "py_written_30_embedded.pkg.slp",
  "py_written_30_embedded_legacy.pkg.slp",
];

describe("any read the main-thread reader makes", () => {
  for (const [file, openVideos] of [
    ...WITHOUT_VIDEOS.map((file) => [file, false] as const),
    ...WITH_VIDEOS.map((file) => [file, true] as const),
  ]) {
    it(`${file}: HDF5 failing any of them is an error`, async () => {
      await prepared;
      vi.spyOn(console, "warn").mockImplementation(() => {});

      reads.clear();
      failing = undefined;
      const clean = poses(await readSlp(path(file), { openVideos }));
      const made = [...reads];
      expect(made.length).toBeGreaterThan(0);

      try {
        for (failing of made) {
          const outcome = await readSlp(path(file), { openVideos }).then(
            () => "it opened",
            (error: Error) => error.message,
          );
          expect(`${failing}: ${outcome}`).toMatch(/: Could not read /);
        }
      } finally {
        failing = undefined;
        vi.restoreAllMocks();
      }

      // Nothing was freed that shouldn't have been, or this would abort or read garbage.
      expect(poses(await readSlp(path(file), { openVideos }))).toEqual(clean);
    });
  }
});
