import { describe, it, expect, beforeEach, afterEach, vi } from "../bun-test";
import {
  isUrl,
  isGdriveUrl,
  resolveUrl,
  redactUrl,
  redactUrlsInText,
  redactedCauseSummary,
  RemoteIOError,
  statusToMessage,
  identityHeaders,
  stripCrossOriginHeaders,
  withRetries,
  headOrRangeProbe,
  maskUrlCredentials,
  maskUrlCredentialsDeep,
  urlBasename,
} from "../../src/io/remote.js";

describe("isUrl", () => {
  it("recognizes remote schemes", () => {
    expect(isUrl("http://h/x")).toBe(true);
    expect(isUrl("https://h/x")).toBe(true);
    expect(isUrl("gs://b/o")).toBe(true);
    expect(isUrl("s3://b/o")).toBe(true);
    expect(isUrl("gcs://b/o")).toBe(true);
    expect(isUrl("az://b/o")).toBe(true);
    expect(isUrl("abfs://b/o")).toBe(true);
  });

  it("rejects non-URLs (drive-letter guard etc.)", () => {
    expect(isUrl("")).toBe(false);
    expect(isUrl(123 as unknown as string)).toBe(false);
    expect(isUrl(null as unknown as string)).toBe(false);
    // Single-letter scheme (Windows drive letter) must NOT be a URL.
    expect(isUrl("C:\\path\\file.slp")).toBe(false);
    expect(isUrl("relative/path.slp")).toBe(false);
    expect(isUrl("/abs/path.slp")).toBe(false);
    // Unknown scheme.
    expect(isUrl("ftp://h/x")).toBe(false);
  });
});

describe("isGdriveUrl", () => {
  it("matches Drive hosts case-insensitively", () => {
    expect(isGdriveUrl("https://drive.google.com/file/d/ABC/view")).toBe(true);
    expect(isGdriveUrl("https://DOCS.GOOGLE.COM/uc?id=ABC")).toBe(true);
  });

  it("rejects other hosts and non-URLs", () => {
    expect(isGdriveUrl("https://example.com/x")).toBe(false);
    expect(isGdriveUrl("not a url")).toBe(false);
    expect(isGdriveUrl("C:\\path")).toBe(false);
  });
});

describe("resolveUrl", () => {
  it("maps gs:// to storage.googleapis.com preserving nested path + query", () => {
    expect(resolveUrl("gs://my-bucket/path/to/obj.slp")).toEqual({
      url: "https://storage.googleapis.com/my-bucket/path/to/obj.slp",
      gdrive: false,
    });
    expect(resolveUrl("gs://b/a/b/c.slp?x=1")).toEqual({
      url: "https://storage.googleapis.com/b/a/b/c.slp?x=1",
      gdrive: false,
    });
  });

  it("maps gcs:// to storage.googleapis.com", () => {
    expect(resolveUrl("gcs://b/o")).toEqual({
      url: "https://storage.googleapis.com/b/o",
      gdrive: false,
    });
  });

  it("passes through http(s) and flags Drive", () => {
    expect(resolveUrl("https://h/x.slp")).toEqual({
      url: "https://h/x.slp",
      gdrive: false,
    });
    expect(resolveUrl("https://drive.google.com/file/d/ABC/view")).toEqual({
      url: "https://drive.google.com/file/d/ABC/view",
      gdrive: true,
    });
  });

  it("throws RemoteIOError pointing to presigned https for cloud schemes", () => {
    for (const u of ["s3://b/o", "az://b/o", "abfs://c@a.dfs/x"]) {
      let err: unknown;
      try {
        resolveUrl(u);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(RemoteIOError);
      expect((err as Error).message).toContain("presigned https");
    }
  });

  it("throws for non-URL input", () => {
    expect(() => resolveUrl("/local/path.slp")).toThrow(RemoteIOError);
  });
});

// The redacted value of a sensitive param is either `***` or its percent-encoded
// form `%2A%2A%2A` depending on the URL implementation; assert it is masked
// (some form present) and the original secret is gone.
function maskedTokenPresent(out: string, param: string): boolean {
  return out.includes(`${param}=%2A%2A%2A`) || out.includes(`${param}=***`);
}

describe("redactUrl", () => {
  it("masks userinfo and sensitive query values, keeps others", () => {
    const out = redactUrl("https://u:p@host/x?token=secret&keep=1");
    expect(out).toContain("***:***@host");
    expect(maskedTokenPresent(out, "token")).toBe(true);
    expect(out).toContain("keep=1");
    expect(out).not.toContain("secret");
  });

  it("masks sensitive params case-insensitively", () => {
    const out = redactUrl("https://h/x?Access_Token=abc");
    expect(maskedTokenPresent(out, "Access_Token")).toBe(true);
    expect(out).not.toContain("abc");
  });

  it("masks each known sensitive param", () => {
    for (const p of [
      "token",
      "access_token",
      "x-amz-security-token",
      "sas",
      "sig",
    ]) {
      const out = redactUrl(`https://h/x?${p}=zzz`);
      expect(out).not.toContain("zzz");
    }
  });

  it("returns malformed URLs unchanged", () => {
    expect(redactUrl("not a url")).toBe("not a url");
  });

  it("leaves non-sensitive params untouched", () => {
    expect(redactUrl("https://h/x?a=1&b=2")).toBe("https://h/x?a=1&b=2");
  });

  it("masks a sensitive param in a fragment", () => {
    expect(redactUrl("https://h/x#token=abc")).toBe("https://h/x#token=***");
    expect(redactUrl("https://h/x#a=1&SIG=abc&b=2")).toBe(
      "https://h/x#a=1&SIG=***&b=2",
    );
  });

  it("leaves any other fragment alone", () => {
    for (const url of [
      "https://h/x#t=1/2",
      "https://h/x?a=1#b",
      "https://h/x",
    ]) {
      expect(redactUrl(url)).toBe(url);
    }
  });
});

describe("redactedCauseSummary", () => {
  it("scrubs URL tokens out of error text", () => {
    const out = redactedCauseSummary(
      new Error("fetch failed for https://h/x?token=abc"),
    );
    expect(out.startsWith("Error:")).toBe(true);
    expect(out.includes("token=***") || out.includes("token=%2A%2A%2A")).toBe(
      true,
    );
    expect(out).not.toContain("abc");
  });
});

describe("redactUrlsInText", () => {
  const masked = (url: string) => redactUrl(url);

  it("masks every URL in the text and leaves the rest", () => {
    const a = "https://h/x?token=abc";
    const b = "gs://b/y?sig=def&keep=1";
    const out = redactUrlsInText(`first ${a} then '${b}' and "${a}"`);
    expect(out).toBe(
      `first ${masked(a)} then '${masked(b)}' and "${masked(a)}"`,
    );
    expect(out).not.toContain("abc");
    expect(out).not.toContain("def");
  });

  it("masks a password", () => {
    const url = "https://u:abc@h/x";
    expect(redactUrlsInText(`at ${url}.`)).toBe(`at ${masked(url)}.`);
  });

  it("leaves a sentence's punctuation where it was", () => {
    for (const end of [".", ",", ";", ":", "!", "?", ")", "..."]) {
      const url = "https://h/x?token=abc";
      expect(redactUrlsInText(`(see ${url}${end} next`)).toBe(
        `(see ${masked(url)}${end} next`,
      );
    }
  });

  it("masks what's after the punctuation inside a URL", () => {
    const url = "https://h/x?a=1.&token=abc";
    expect(redactUrlsInText(`at ${url}.`)).toBe(`at ${masked(url)}.`);
  });

  it("leaves text with nothing sensitive alone", () => {
    for (const text of [
      "",
      "no url here",
      "token=abc",
      "C:\\a\\b",
      "x://y",
      "a b#1?%.slp",
      "/Users/me/y.slp",
      "?mytoken=abc",
      "?tokens=abc",
      "?a=token",
    ]) {
      expect(redactUrlsInText(text)).toBe(text);
    }
  });

  it("masks a token after a space or a quote ends the URL", () => {
    for (const url of [
      "https://h/x?name=a b&token=abc",
      "https://h/x?name=o'b&token=abc",
      'https://h/x?name=a"b&token=abc',
      "https://h/x?name=a<b&token=abc",
      "https://h/o'b/x?token=abc",
      "https://h/a b/x?token=abc",
    ]) {
      const out = redactUrlsInText(`at ${url}. Done`);
      expect(out).not.toContain("abc");
      expect(out).toEndWith(". Done");
    }
  });

  it("masks what's left of a URL", () => {
    for (const [text, kept] of [
      ["y.slp?token=abc", "y.slp?token=***"],
      ["y.slp&token=abc", "y.slp&token=***"],
      ["'y.slp?token=abc'", "'y.slp?token=***'"],
      ["http://h/aaa...bbb&token=abc", "token="],
    ]) {
      const out = redactUrlsInText(text);
      expect(out).toContain(kept);
      expect(out).not.toContain("abc");
    }
  });

  it("masks each sensitive name, in any case", () => {
    for (const name of [
      "token",
      "TOKEN",
      "Access_Token",
      "x-amz-security-token",
      "SAS",
      "sig",
    ]) {
      const out = redactUrlsInText(`y.slp?${name}=abc&keep=1`);
      expect(out).toBe(`y.slp?${name}=***&keep=1`);
    }
  });

  it("masks a value holding what a URL can hold", () => {
    for (const value of ["a/b", "a.b.c", "a+b==", "a:b", "a%2Fb", "a,b;c"]) {
      expect(redactUrlsInText(`y.slp?token=${value} next`)).toBe(
        "y.slp?token=*** next",
      );
    }
  });

  it("masks a token in a fragment", () => {
    const out = redactUrlsInText("at https://h/x#a=1&token=abc. Done");
    expect(out).toBe("at https://h/x#a=1&token=***. Done");
  });
});

describe("urlBasename", () => {
  it("is the last part of a URL's path", () => {
    expect(urlBasename("https://x.test/y.slp")).toBe("y.slp");
    expect(urlBasename("https://x.test/a/b/y.slp")).toBe("y.slp");
    expect(urlBasename("HTTPS://X.TEST/A/Y.SLP")).toBe("Y.SLP");
    expect(urlBasename("gs://bucket/a/y.slp")).toBe("y.slp");
  });

  it("isn't decoded or rewritten", () => {
    expect(urlBasename("https://x.test/a%20b.slp")).toBe("a%20b.slp");
    expect(urlBasename("https://x.test/a b.slp")).toBe("a b.slp");
    expect(urlBasename("https://x.test/日本語.slp")).toBe("日本語.slp");
  });

  it("leaves out the query and fragment, slashes and all", () => {
    for (const url of [
      "https://x.test/y.slp?token=abc",
      "https://x.test/y.slp?f=a/b.slp&token=abc",
      "https://x.test/y.slp?token=a/bc",
      "https://x.test/y.slp#a/bc",
      "https://x.test/y.slp?a=1#b/c",
      "https://x.test/y.slp#a?b=1/c",
    ]) {
      expect(urlBasename(url)).toBe("y.slp");
    }
  });

  it("is empty for a URL with no name of its own", () => {
    for (const url of [
      "https://x.test",
      "https://x.test/",
      "https://x.test/a/",
      "https://x.test/a/?c=1",
      "https://x.test?c=1/2",
      "https://x.test#a/b",
    ]) {
      expect(urlBasename(url)).toBe("");
    }
  });

  it("never takes what's before the path", () => {
    for (const url of [
      "https://me:abc@x.test",
      "https://me:abc@x.test/",
      "https://me:abc@x.test?c=1",
      "https://me:abc@x.test#c",
      "https:/me:abc@x.test",
      "https:me:abc@x.test",
      "https:\\\\me:abc@x.test",
    ]) {
      expect(urlBasename(url)).toBe("");
    }
    expect(urlBasename("https://me:abc@x.test/y.slp")).toBe("y.slp");
  });

  it("cuts anything else at its first ?", () => {
    expect(urlBasename("y.slp")).toBe("y.slp");
    expect(urlBasename("/Users/me/y.slp")).toBe("y.slp");
    expect(urlBasename("y.slp?token=abc")).toBe("y.slp");
    expect(urlBasename("/api/get?f=a/y.slp&token=abc")).toBe("get");
    expect(urlBasename("a#1.slp")).toBe("a#1.slp");
    expect(urlBasename("C:\\me\\y.slp")).toBe("C:\\me\\y.slp");
    expect(urlBasename("")).toBe("");
  });
});

describe("maskUrlCredentials", () => {
  it.each([
    ["https://h/y.slp?token=SECRET", "https://h/y.slp?token=***"],
    ["https://h/y.slp?c=12&token=SECRET", "https://h/y.slp?c=12&token=***"],
    ["https://h/y.slp?TOKEN=SECRET", "https://h/y.slp?TOKEN=***"],
    ["https://h/y.slp?%74oken=SECRET", "https://h/y.slp?token=***"],
    ["https://h/y.slp?token=SEC/RET", "https://h/y.slp?token=***"],
    ["https://h/y.slp?token=SEC%2FRET", "https://h/y.slp?token=***"],
    ["https://h/y.slp?token=SEC RET", "https://h/y.slp?token=***"],
    ["https://h/y.slp?token=SEC'RET", "https://h/y.slp?token=***"],
    ["https://h/y.slp#token=SECRET", "https://h/y.slp#token=***"],
    ["https://me:SECRET@h/y.slp", "https://***:***@h/y.slp"],
    ["https://me:SEC RET@h/y.slp", "https://***:***@h/y.slp"],
    ["https://me:SECRET@h", "https://***:***@h/"],
    ["gs://me:SECRET@b/y.slp?sig=SECRET", "gs://***:***@b/y.slp?sig=***"],
    [
      "s3://b/y.slp?x-amz-security-token=SECRET",
      "s3://b/y.slp?x-amz-security-token=***",
    ],
    ["az://c/y.slp?sas=SECRET", "az://c/y.slp?sas=***"],
  ])("a URL as a whole: %s", (url, masked) => {
    expect(maskUrlCredentials(url)).toBe(masked);
  });

  it.each([
    ["https://me:SECRET@h:notaport/y.slp", "https://***:***@h:notaport/y.slp"],
    ["https://me:SEC/RET@h:notaport/y.slp", "https://***:***@h:notaport/y.slp"],
    [
      "https://me:SECRET@h:notaport/y.slp?token=SECRET",
      "https://***:***@h:notaport/y.slp?token=***",
    ],
    ["https://h:notaport/a@b.slp", "https://***:***@b.slp"],
    [
      "https://me:SECRET@h:notaport/y.slp?email=a@b",
      "https://***:***@h:notaport/y.slp?email=a@b",
    ],
    [
      "https://h:notaport/y.slp?email=a@b",
      "https://h:notaport/y.slp?email=a@b",
    ],
  ])("a URL nothing can read still says what it says: %s", (url, masked) => {
    expect(maskUrlCredentials(url)).toBe(masked);
  });

  it.each([
    ["y.slp?token=SECRET", "y.slp?token=***"],
    ["/some/dir/y.slp?a=1&token=SECRET", "/some/dir/y.slp?a=1&token=***"],
    ["y.slp&token=SECRET", "y.slp&token=***"],
    ["y.slp#token=SECRET", "y.slp#token=***"],
    [
      "track https://h/v.mp4?token=SECRET --flag",
      "track https://h/v.mp4?token=*** --flag",
    ],
    [
      "track https://me:SECRET@h/v.mp4 --flag",
      "track https://***:***@h/v.mp4 --flag",
    ],
    [
      "https://h/a.mp4 https://me:SECRET@h/w.mp4",
      "https://h/a.mp4 https://***:***@h/w.mp4",
    ],
    [
      "https://p/?url=https://h/y.slp?token=SECRET",
      "https://p/?url=https://h/y.slp?token=***",
    ],
  ])("what's left of a URL, and a URL in a line of text: %s", (text, masked) => {
    expect(maskUrlCredentials(text)).toBe(masked);
  });

  it.each([
    "https://h",
    "HTTP://H/a b.mp4?x=1 2",
    "https://h/a%20b.mp4?x=é#frag",
    "https://h/y.slp?token=***",
    "https://h/y.slp?token",
    "https://***:***@h/y.slp",
    "https://h/a@b.mp4",
    "gs://b/a b.mp4",
    "/Users/me/a b#1?%.mp4",
    "/Users/me/me@host/y.mp4",
    "C:\\Users\\me\\b.mp4",
    "tests/data/videos/centered_pair_low_quality.mp4",
    "file:///Users/me/x.mp4",
    "mailto:me@h",
    "日本語.slp",
    ".",
    "",
  ])("nothing to mask comes back as it was: %s", (value) => {
    expect(maskUrlCredentials(value)).toBe(value);
  });

  it("masking twice changes nothing more", () => {
    for (const url of [
      "https://h/y.slp?a=b c&token=SECRET",
      "https://me:SECRET@h/y.slp#token=SECRET",
      "https://me:SECRET@h:notaport/y.slp",
      "y.slp?token=SECRET",
    ]) {
      const once = maskUrlCredentials(url);
      expect(once).not.toContain("SECRET");
      expect(maskUrlCredentials(once)).toBe(once);
    }
  });
});

describe("maskUrlCredentialsDeep", () => {
  const url = "https://h/y.slp?token=SECRET";
  const masked = "https://h/y.slp?token=***";

  it("every string, at any depth, keys too", () => {
    expect(
      maskUrlCredentialsDeep({
        filename: url,
        list: [url, { deeper: [url] }],
        [url]: 1,
      }),
    ).toEqual({
      filename: masked,
      list: [masked, { deeper: [masked] }],
      [masked]: 1,
    });
    expect(maskUrlCredentialsDeep(url)).toBe(masked);
    expect(maskUrlCredentialsDeep([url])).toEqual([masked]);
  });

  it("leaves what it was given alone", () => {
    const given = { filename: url, list: [url] };
    maskUrlCredentialsDeep(given);
    expect(given).toEqual({ filename: url, list: [url] });
  });

  it("reads a value as JSON.stringify would", () => {
    class Named {
      filename = url;
    }
    const value = {
      when: new Date(0),
      custom: { toJSON: () => ({ filename: url }) },
      named: new Named(),
      n: 1,
      nothing: null,
      missing: undefined,
      yes: true,
      numbers: new Float64Array([1, 2]),
    };
    const once = JSON.stringify(maskUrlCredentialsDeep(value));
    expect(once).toBe(JSON.stringify(value).replaceAll(url, masked));
    expect(once).not.toContain("SECRET");
  });

  it("a JSON tree with no URL in it writes the same bytes", () => {
    const value = {
      "2": "b",
      "1": "a",
      shape: [1, 384, 384, 1],
      fps: null,
      __proto__x: { é: "日本語" },
    };
    expect(JSON.stringify(maskUrlCredentialsDeep(value))).toBe(
      JSON.stringify(value),
    );
  });
});

describe("RemoteIOError", () => {
  it("redacts url, sets status, and does NOT chain the raw cause", () => {
    const raw = new Error("boom https://h/x?token=SECRET");
    const err = new RemoteIOError({
      message: "file not found",
      url: "https://u:p@h/x?token=SECRET",
      status: 404,
      cause: raw,
    });
    expect(err.status).toBe(404);
    expect(err.url).toContain("***:***@h");
    expect(maskedTokenPresent(err.url, "token")).toBe(true);
    expect(err.message).toContain("status=404");
    expect(err.message).toContain("url=");
    // The raw transport error must NOT be chained.
    expect((err as { cause?: unknown }).cause).toBeUndefined();
    // No secret anywhere.
    expect(err.message).not.toContain("SECRET");
    expect(err.url).not.toContain("SECRET");
  });
});

describe("statusToMessage", () => {
  it("maps known statuses", () => {
    expect(statusToMessage(404)).toBe("file not found");
    expect(statusToMessage(416)).toBe("range past end of file");
    expect(statusToMessage(412)).toBe(
      "file changed since cached (ETag mismatch)",
    );
    expect(statusToMessage(503)).toBe("HTTP 503");
  });
});

describe("identityHeaders", () => {
  it("forces Accept-Encoding: identity and drops any case", () => {
    const out = identityHeaders({
      "accept-encoding": "gzip",
      Authorization: "Bearer T",
    });
    expect(out["Accept-Encoding"]).toBe("identity");
    expect(out["accept-encoding"]).toBeUndefined();
    expect(out.Authorization).toBe("Bearer T");
  });
});

describe("stripCrossOriginHeaders", () => {
  it("keeps headers same-origin", () => {
    const h = { Authorization: "Bearer T", "X-Other": "1" };
    expect(stripCrossOriginHeaders(h, "https://a/x", "https://a/y")).toEqual(h);
  });

  it("drops sensitive headers cross-origin (any case)", () => {
    const out = stripCrossOriginHeaders(
      { authorization: "Bearer T", Cookie: "c", "X-Other": "1" },
      "https://a/x",
      "https://b/y",
    );
    expect(out.authorization).toBeUndefined();
    expect(out.Cookie).toBeUndefined();
    expect(out["X-Other"]).toBe("1");
  });
});

describe("withRetries", () => {
  it("retries retryable 503 then succeeds", async () => {
    let calls = 0;
    const result = await withRetries(
      async () => {
        calls++;
        if (calls < 3) {
          throw new RemoteIOError({
            message: "HTTP 503",
            url: "https://h/x",
            status: 503,
          });
        }
        return "ok";
      },
      { retries: 3 },
    );
    expect(result).toBe("ok");
    expect(calls).toBe(3);
  });

  it("does not retry a non-retryable 404", async () => {
    let calls = 0;
    let err: unknown;
    try {
      await withRetries(
        async () => {
          calls++;
          throw new RemoteIOError({
            message: "file not found",
            url: "https://h/x",
            status: 404,
          });
        },
        { retries: 3 },
      );
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(RemoteIOError);
    expect(calls).toBe(1);
  });
});

describe("headOrRangeProbe", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("falls back to Range GET on 405 and forwards Authorization on both", async () => {
    const mock = vi.fn(async (_url: string, init?: RequestInit) => {
      if (init?.method === "HEAD") {
        return { ok: false, status: 405 } as Response;
      }
      return { ok: false, status: 206 } as Response;
    });
    globalThis.fetch = mock as unknown as typeof fetch;

    const ok = await headOrRangeProbe("https://h/x", {
      headers: { Authorization: "Bearer T" },
    });
    expect(ok).toBe(true);
    const calls = (mock as unknown as { mock: { calls: unknown[][] } }).mock
      .calls;
    expect(calls).toHaveLength(2);
    for (const c of calls) {
      const init = c[1] as RequestInit & { headers: Record<string, string> };
      expect(init.headers.Authorization).toBe("Bearer T");
    }
  });

  it("returns false (never throws) on a thrown TypeError", async () => {
    globalThis.fetch = (async () => {
      throw new TypeError("network down");
    }) as unknown as typeof fetch;
    expect(await headOrRangeProbe("https://h/x")).toBe(false);
  });

  it("returns false for unsupported cloud scheme", async () => {
    expect(await headOrRangeProbe("s3://b/o")).toBe(false);
  });
});
