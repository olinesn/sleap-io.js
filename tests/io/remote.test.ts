import { describe, it, expect, beforeEach, afterEach, vi } from "../bun-test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  isUrl,
  isGdriveUrl,
  resolveUrl,
  redactUrl,
  redactUrlsInText,
  redactedCauseSummary,
  RemoteIOError,
  SENSITIVE_QUERY_PARAMS,
  statusToMessage,
  identityHeaders,
  stripCrossOriginHeaders,
  withRetries,
  headOrRangeProbe,
  urlBasename,
} from "../../src/io/remote.js";
import { SIGNED_QUERIES, shownPart, signedQuery } from "../signed-url-fixtures";

/** Every name on the list. Spelled out, so taking one off fails a test. */
const SENSITIVE_NAMES = [
  "token",
  "access_token",
  "x-amz-security-token",
  "sas",
  "sig",
  "x-amz-signature",
  "x-amz-credential",
  "awsaccesskeyid",
  "x-goog-signature",
  "x-goog-credential",
  "googleaccessid",
  "signature",
  "authorization",
  "key",
  "api_key",
  "apikey",
  "api-key",
];

/** Params that only look like credentials, or say what a failure needs said. */
const ORDINARY_NAMES = [
  "monkey",
  "keys",
  "keyframe",
  "key_frame",
  "primary_key",
  "signatures",
  "signature_version",
  "mysig",
  "X-Amz-Algorithm",
  "X-Amz-Date",
  "X-Amz-Expires",
  "X-Amz-SignedHeaders",
  "X-Goog-Algorithm",
  "X-Goog-Date",
  "X-Goog-Expires",
  "X-Goog-SignedHeaders",
  "Expires",
  "Key-Pair-Id",
  "Hash-Algorithm",
  "se",
  "sp",
  "sv",
  "sr",
  "alt",
];

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
    for (const p of SENSITIVE_NAMES) {
      const out = redactUrl(`https://h/x?${p}=zzz`);
      expect(out).not.toContain("zzz");
    }
  });

  it("masks them in any case, in a query or a fragment", () => {
    for (const p of SENSITIVE_NAMES) {
      for (const name of [p, p.toUpperCase()]) {
        const url = `https://h/x?keep=1&${name}=zzz&also=2#a=1&${name}=zzz&b=2`;
        const out = redactUrl(url);
        expect(out).not.toContain("zzz");
        expect(out).toContain("keep=1");
        expect(out).toContain("also=2");
        expect(out).toContain(`#a=1&${name}=***&b=2`);
      }
    }
  });

  it("masks every credential in a signed URL, and leaves the rest", () => {
    for (const scheme of ["https://h/x?", "s3://h/x?", "gs://h/x?"]) {
      for (const { query, secrets, kept } of SIGNED_QUERIES) {
        const out = redactUrl(`${scheme}${query}`);
        expect(shownPart(out, secrets)).toBeNull();
        for (const param of kept) expect(out).toContain(param);
      }
    }
  });

  it("leaves a param that only looks like a credential", () => {
    for (const name of ORDINARY_NAMES) {
      const url = `https://h/x?${name}=zzz#${name}=zzz`;
      expect(redactUrl(url)).toBe(url);
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
      "key=abc",
      "?key",
      "?a=key",
      ...ORDINARY_NAMES.map((name) => `y.slp?${name}=abc`),
      ...ORDINARY_NAMES.map(
        (name) => `https://h/x?a=1&${name}=abc#${name}=abc`,
      ),
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
      ...SENSITIVE_NAMES,
      ...SENSITIVE_NAMES.map((n) => n.toUpperCase()),
      "Access_Token",
      "X-Amz-Signature",
      "X-Goog-Credential",
      "GoogleAccessId",
      "AWSAccessKeyId",
      "Signature",
      "Key",
    ]) {
      const out = redactUrlsInText(`y.slp?${name}=abc&keep=1`);
      expect(out).toBe(`y.slp?${name}=***&keep=1`);
    }
  });

  it("masks every credential in a signed URL it quotes, and leaves the rest", () => {
    for (const { query, secrets, kept } of SIGNED_QUERIES) {
      for (const text of [
        `Couldn't load https://h/x?${query}. Status: 403`,
        `Failed to load 'https://h/x?${query}'.`,
        `name=a b&${query}`,
        `y.slp?${query}`,
      ]) {
        const out = redactUrlsInText(text);
        expect(shownPart(out, secrets)).toBeNull();
        for (const param of kept) expect(out).toContain(param);
      }
      expect(
        redactUrlsInText(`Couldn't load https://h/x?${query}. Status: 403`),
      ).toEndWith(". Status: 403");
    }
  });

  describe("a URL a browser shortened", () => {
    /** As Chromium quotes a URL over 1 KB. */
    const shorten = (url: string) =>
      `${url.slice(0, 511)}...${url.slice(-510)}`;
    const cloudStorage = signedQuery("Cloud Storage");

    it("masks the end of a value whose name went with the middle", () => {
      const url = `https://h/${"a".repeat(600)}/y.slp?${cloudStorage.query}`;
      const quoted = shorten(url);
      expect(quoted).not.toContain("X-Goog-Signature");
      expect(shownPart(quoted, cloudStorage.secrets)).not.toBeNull();

      const out = redactUrlsInText(`Failed to load '${quoted}'.`);
      expect(shownPart(out, cloudStorage.secrets)).toBeNull();
      expect(out).toEndWith("...***'.");
      expect(out).toStartWith(`Failed to load 'https://h/${"a".repeat(400)}`);
    });

    it("whichever part of the URL the cut falls in", () => {
      for (const { query, secrets } of SIGNED_QUERIES) {
        // Every place the cut can fall, around every param.
        for (let padding = 0; padding < 1500; padding += 7) {
          const url = `https://h/${"a".repeat(padding)}?${query}`;
          if (url.length <= 1024) continue;
          const out = redactUrlsInText(`Failed to load '${shorten(url)}'.`);
          expect(shownPart(out, secrets)).toBeNull();
          // What's left of the last param may have gained an `=` from redactUrl.
          expect(out).toMatch(/\*\*\*=?'\.$/);
        }
      }
    });

    it("still masks what's before the cut", () => {
      const url = `https://me:abc@h/y.slp?sig=abc&a=${"b".repeat(1500)}`;
      const out = redactUrlsInText(`Failed to load '${shorten(url)}'.`);
      expect(out).not.toContain("abc");
      expect(out).toContain("sig=***");
      expect(out).toContain("https://***:***@h/y.slp");
    });

    it("needs a run too long to be anything but the end of a URL", () => {
      const end = (length: number) => "x".repeat(length);
      for (const text of [`a...${end(255)}`, `a... ${end(1000)}`, "a...b"]) {
        expect(redactUrlsInText(text)).toBe(text);
      }
      expect(redactUrlsInText(`a...${end(256)}`)).toBe("a...***");
      expect(redactUrlsInText(`a...${end(510)}`)).toBe("a...***");
    });

    it("every one in the text", () => {
      const quoted = `https://h/a...${"x".repeat(510)}`;
      expect(redactUrlsInText(`'${quoted}' and '${quoted}'`)).toBe(
        "'https://h/a...***' and 'https://h/a...***'",
      );
    });

    it("leaves what ends the sentence", () => {
      const quoted = `https://h/a...${"x".repeat(510)}`;
      for (const [before, after] of [
        ["'", "'."],
        ['"', '".'],
        ["(", ")."],
        ["at ", "."],
        ["at ", ", then"],
        ["at ", " then"],
      ]) {
        expect(redactUrlsInText(`${before}${quoted}${after}`)).toBe(
          `${before}https://h/a...***${after}`,
        );
      }
    });
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

describe("SENSITIVE_QUERY_PARAMS", () => {
  it("holds the names, lowercased, that its users expect", () => {
    expect([...SENSITIVE_QUERY_PARAMS].sort()).toEqual(
      [...SENSITIVE_NAMES].sort(),
    );
    for (const name of SENSITIVE_QUERY_PARAMS) {
      expect(name).toBe(name.toLowerCase());
    }
  });

  it("every name on it counts everywhere", () => {
    for (const name of SENSITIVE_QUERY_PARAMS) {
      expect(redactUrl(`https://h/x?${name}=zzz`)).not.toContain("zzz");
      expect(redactUrl(`https://h/x#${name}=zzz`)).toBe(
        `https://h/x#${name}=***`,
      );
      expect(redactUrlsInText(`a b?${name}=zzz c`)).toBe(`a b?${name}=*** c`);
      expect(redactUrlsInText(`a b&${name}=zzz c`)).toBe(`a b&${name}=*** c`);
      expect(redactUrlsInText(`a b#${name}=zzz c`)).toBe(`a b#${name}=*** c`);
    }
  });

  // The pattern is built as the file loads, so this loads a copy of the file with these
  // names on its list.
  it("a name is plain text, whatever it holds", async () => {
    const names = [
      "a.b",
      "c(d",
      "e)f",
      "g+h",
      "i*j",
      "k?l",
      "m|n",
      "[o",
      "p]",
      "{q}",
      "r^s",
      "$t",
      "u\\v",
    ];
    const source = readFileSync(
      fileURLToPath(new URL("../../src/io/remote.ts", import.meta.url)),
      "utf8",
    );
    const list = "SENSITIVE_QUERY_PARAMS = new Set([";
    expect(source).toContain(list);
    const dir = mkdtempSync(join(tmpdir(), "remote-list-"));
    try {
      const copy = join(dir, "remote.ts");
      writeFileSync(
        copy,
        source.replace(
          list,
          () => `${list}${JSON.stringify(names).slice(1, -1)},`,
        ),
      );
      const loaded = await import(copy);
      for (const name of names) {
        expect(loaded.redactUrlsInText(`a b&${name}=zzz c`)).toBe(
          `a b&${name}=*** c`,
        );
        expect(
          loaded.redactUrl(`https://h/x?${encodeURIComponent(name)}=zzz`),
        ).not.toContain("zzz");
      }
      // What those would mean in a pattern doesn't count.
      for (const text of [
        "?aXb=zzz",
        "?gggh=zzz",
        "?j=zzz",
        "?l=zzz",
        "?m=zzz",
        "?n=zzz",
        "?o=zzz",
      ]) {
        expect(loaded.redactUrlsInText(text)).toBe(text);
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
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
