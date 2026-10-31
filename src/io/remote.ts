/**
 * Remote-loading helpers: URL/scheme resolution, credential redaction, a typed
 * {@link RemoteIOError}, retry/backoff, header transforms, and a non-throwing
 * existence probe.
 *
 * This is the browser+Node-portable subset of Python sleap-io's `_remote.py`
 * (PRs #439/#445). It threads auth headers end-to-end and guarantees that no
 * thrown error or log line leaks credentials (userinfo / sensitive query
 * params). See {@link redactUrl} and {@link RemoteIOError}. Nor does a saved file
 * (see {@link maskUrlCredentials}), which is why a URL with credentials doesn't
 * open from one.
 *
 * @module
 */

/** URL schemes recognized as remote (vs. a local path). Port of Python `URL_SCHEMES`. */
export const URL_SCHEMES = new Set([
  "http",
  "https",
  "s3",
  "gs",
  "gcs",
  "az",
  "abfs",
]);

/** Cloud schemes that need a provider SDK (not available in the browser). */
export const CLOUD_SCHEMES = new Set(["s3", "gs", "gcs", "az", "abfs"]);

/** Hosts handled by the Google Drive resolver. */
export const GDRIVE_HOSTS = new Set(["drive.google.com", "docs.google.com"]);

/**
 * Headers dropped on a cross-origin redirect and never sent to Google Drive
 * hosts. Lowercase for case-insensitive matching.
 */
export const SENSITIVE_HEADERS = new Set([
  "authorization",
  "cookie",
  "proxy-authorization",
]);

/**
 * Query-param names (lowercased) whose VALUE is masked to `***` in
 * {@link redactUrl}. Other params are left untouched.
 */
export const SENSITIVE_QUERY_PARAMS = new Set([
  "token",
  "access_token",
  "x-amz-security-token",
  "sas",
  "sig",
]);

/** HTTP status codes that {@link withRetries} treats as retryable. */
export const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

/**
 * Matches URL tokens embedded in arbitrary error text for scrubbing. Punctuation at
 * the end belongs to the sentence, not the URL.
 */
const URL_IN_TEXT_PATTERN =
  /(?:http|https|s3|gs|gcs|az|abfs):\/\/[^\s"'<>]*[^\s"'<>.,;:!?)]/gi;

/**
 * Matches a sensitive query param and its value anywhere in text, URL or no URL: a
 * space or quote ends what {@link URL_IN_TEXT_PATTERN} can see of a URL, a browser
 * shortens a long one, and a name cut from one (`y.slp?token=...`) no longer looks
 * like one. The `#` is for the same param in a fragment.
 */
const SENSITIVE_PARAM_PATTERN = new RegExp(
  `([?&#](?:${[...SENSITIVE_QUERY_PARAMS].join("|")})=)[^\\s"'<>&#]*[^\\s"'<>&#.,;:!?)]`,
  "gi",
);

/**
 * Whether `value` looks like a remote URL (one of {@link URL_SCHEMES}).
 *
 * Port of Python `_is_url`. The `scheme.length > 1` guard is mandatory so a
 * Windows drive letter (`C:\...`) is not misread as a URL.
 */
export function isUrl(value: unknown): boolean {
  if (typeof value !== "string" || value.length === 0) return false;
  const colon = value.indexOf(":");
  if (colon <= 1) return false; // length-1 scheme (drive letter) or no scheme
  const scheme = value.slice(0, colon).toLowerCase();
  return scheme.length > 1 && URL_SCHEMES.has(scheme);
}

/**
 * Whether `value` is a Google Drive share URL. Port of Python `_is_gdrive_url`.
 * Never throws on malformed input.
 */
export function isGdriveUrl(value: unknown): boolean {
  if (!isUrl(value)) return false;
  try {
    return GDRIVE_HOSTS.has(new URL(value as string).hostname.toLowerCase());
  } catch {
    return false;
  }
}

/**
 * The name at the end of a URL's path, undecoded, or `""` if the path has none. Only
 * the path counts: a slash in the query doesn't start a new name, and what comes
 * before the path can hold a password. A plain path or file name is cut at its first
 * `?` the same way.
 */
export function urlBasename(url: string): string {
  const path = isUrl(url)
    ? url.replace(/^[^:]+:[/\\]*[^/\\?#]*/, "").split(/[?#]/, 1)[0]
    : url.split("?", 1)[0];
  return path.split("/").pop() ?? "";
}

/**
 * Mask credentials in a URL: userinfo becomes `***:***@host`, and the VALUES of
 * sensitive query params ({@link SENSITIVE_QUERY_PARAMS}) become `***`
 * (serialized percent-encoded as `%2A%2A%2A`, matching Python), in a fragment too.
 * Other params are left intact. On parse failure the input is returned unchanged.
 *
 * Port of Python `_redact_url`, which leaves the fragment alone.
 */
export function redactUrl(url: string): string {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }
  if (parsed.username || parsed.password) {
    parsed.username = "***";
    parsed.password = "***";
  }
  for (const key of [...parsed.searchParams.keys()]) {
    if (SENSITIVE_QUERY_PARAMS.has(key.toLowerCase())) {
      parsed.searchParams.set(key, "***");
    }
  }
  parsed.hash = parsed.hash.replace(SENSITIVE_PARAM_PATTERN, "$1***");
  return parsed.toString();
}

/**
 * `text` with credentials masked in every URL it quotes (see {@link redactUrl}), and
 * in whatever is left of a URL it can't recognize as one.
 */
export function redactUrlsInText(text: string): string {
  return text
    .replace(SENSITIVE_PARAM_PATTERN, "$1***")
    .replace(URL_IN_TEXT_PATTERN, (m) => redactUrl(m));
}

/** `url` as {@link redactUrl} masks it, or exactly as given if that finds nothing. */
function maskIfCredentialed(url: string): string {
  let asParsed: string;
  try {
    asParsed = new URL(url).href;
  } catch {
    // `redactUrl` can't read it either, and what comes before an `@` is still a password.
    return url.replace(/^([^:]+:\/\/)[^?#]*@/, "$1***:***@");
  }
  const masked = redactUrl(url);
  return masked === asParsed ? url : masked;
}

/**
 * `value`, a path, a URL or a line of text holding one, fit to go in a file: a URL's
 * credentials are masked and every other character stays as it was. That's the
 * difference from {@link redactUrl}, which writes any URL back as it parses it, so a
 * public URL or a local path saved through this comes back byte for byte.
 *
 * A value that is a URL as a whole is read as one, so a space or a quote can't end it
 * early, which is what the text patterns of {@link redactUrlsInText} can't see past.
 */
export function maskUrlCredentials(value: string): string {
  // Neither a param nor a password without these, and an image sequence is many names.
  if (!value.includes("=") && !value.includes("@")) return value;
  const whole = isUrl(value) ? maskIfCredentialed(value) : value;
  return whole
    .replace(SENSITIVE_PARAM_PATTERN, "$1***")
    .replace(URL_IN_TEXT_PATTERN, maskIfCredentialed);
}

/**
 * {@link maskUrlCredentials} on every string in what is about to be written as JSON,
 * object keys included, reading `value` as `JSON.stringify` would. Returns a copy.
 */
export function maskUrlCredentialsDeep<T>(value: T): T;
export function maskUrlCredentialsDeep(value: unknown): unknown {
  if (typeof value === "string") return maskUrlCredentials(value);
  if (value === null || typeof value !== "object") return value;
  const { toJSON } = value as { toJSON?: unknown };
  if (typeof toJSON === "function") {
    return maskUrlCredentialsDeep(toJSON.call(value));
  }
  if (Array.isArray(value)) return value.map((v) => maskUrlCredentialsDeep(v));
  return Object.fromEntries(
    Object.entries(value).map(([key, v]) => [
      maskUrlCredentials(key),
      maskUrlCredentialsDeep(v),
    ]),
  );
}

/**
 * A short, credential-scrubbed one-line summary of an arbitrary error, suitable
 * for logging and for the `cause=` segment of {@link RemoteIOError}. Any URL
 * tokens embedded in the message are run through {@link redactUrl}.
 *
 * Port of Python `_redacted_cause_summary`.
 */
export function redactedCauseSummary(e: unknown): string {
  const typeName =
    (e as { constructor?: { name?: string } } | null)?.constructor?.name ??
    "Error";
  const rawMsg = String(
    (e as { message?: unknown } | null)?.message ?? (e as unknown),
  );
  const redactedMsg = redactUrlsInText(rawMsg)
    .replace(/\s*\n\s*/g, " ")
    .trim();
  return `${typeName}: ${redactedMsg}`;
}

/**
 * Typed error for every remote-loading failure. The `url` is ALWAYS stored
 * redacted (redaction happens in the constructor from the RAW url passed in),
 * and the raw transport error is NEVER chained as `.cause` — only a redacted
 * {@link redactedCauseSummary} is kept, so credentials cannot leak through a
 * re-throw or `.cause` inspection.
 *
 * Port of Python `RemoteIOError`.
 */
export class RemoteIOError extends Error {
  readonly status: number | null;
  /** ALWAYS redacted. */
  readonly url: string;
  readonly causeSummary?: string;
  /**
   * Delay (ms) hinted by a `Retry-After` response header, threaded to
   * {@link withRetries}. Not part of Python; a JS-side channel so the retry
   * loop can honor server backoff without re-issuing the failed request.
   */
  retryAfterMs?: number;

  constructor(opts: {
    message: string;
    url: string;
    status?: number | null;
    cause?: unknown;
  }) {
    const redactedUrl = redactUrl(opts.url);
    const causeSummary =
      opts.cause !== undefined ? redactedCauseSummary(opts.cause) : undefined;
    const parts = [
      opts.message,
      `status=${opts.status ?? "null"}`,
      `url=${redactedUrl}`,
    ];
    if (causeSummary) parts.push(`cause=${causeSummary}`);
    super(parts.join("; "));
    this.name = "RemoteIOError";
    this.status = opts.status ?? null;
    this.url = redactedUrl;
    this.causeSummary = causeSummary;
    // CRITICAL: do NOT set this.cause = opts.cause. The raw fetch/transport
    // error string can embed ?token= / userinfo and would leak through .cause
    // or a re-throw. We keep only the redacted causeSummary.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** Result of {@link resolveUrl}. */
export interface ResolvedUrl {
  /** Fetchable HTTPS URL (passthrough for http(s); mapped for gs/gcs). */
  url: string;
  /** When true, the caller must route through the Google Drive resolver. */
  gdrive: boolean;
}

/**
 * Turn any user-supplied URL into a fetchable HTTPS URL (or flag it for Google
 * Drive). The single public scheme gate.
 *
 * - `http(s)://` (non-Drive host): passthrough, `gdrive: false`.
 * - Google Drive host: `gdrive: true` (caller routes to `openGdrive`).
 * - `gs://<bucket>/<obj>` / `gcs://...`: mapped to
 *   `https://storage.googleapis.com/<bucket>/<obj>` (object path + query
 *   preserved verbatim). NOTE: this only resolves PUBLIC objects without auth;
 *   private buckets still need a presigned HTTPS URL — no signing is attempted.
 * - `s3://` / `az://` / `abfs://`: throws {@link RemoteIOError} (no in-browser
 *   SDK) directing the user to a presigned `https://` URL.
 * - non-URL input: throws {@link RemoteIOError} (`resolveUrl` only acts on URLs).
 *
 * Port of Python's scheme handling.
 */
export function resolveUrl(url: string): ResolvedUrl {
  if (!isUrl(url)) {
    throw new RemoteIOError({ message: "Not a URL", url, status: null });
  }
  if (isGdriveUrl(url)) {
    return { url, gdrive: true };
  }
  const colon = url.indexOf(":");
  const scheme = url.slice(0, colon).toLowerCase();
  if (scheme === "http" || scheme === "https") {
    return { url, gdrive: false };
  }
  if (scheme === "gs" || scheme === "gcs") {
    // Split once on "://", then once on "/" -> bucket + rest. Preserve the
    // object path (including extra slashes) and any query string verbatim.
    const rest = url.slice(url.indexOf("://") + 3);
    const slash = rest.indexOf("/");
    const bucket = slash === -1 ? rest : rest.slice(0, slash);
    const objectPath = slash === -1 ? "" : rest.slice(slash + 1);
    return {
      url: `https://storage.googleapis.com/${bucket}/${objectPath}`,
      gdrive: false,
    };
  }
  // s3 / az / abfs — no in-browser SDK.
  throw new RemoteIOError({
    message: `Cloud scheme '${scheme}' is not supported in the browser. Pass a presigned https:// URL instead.`,
    url,
    status: null,
  });
}

/**
 * Map an HTTP status to a short human message. Port of Python's `_raise_remote`
 * table.
 */
export function statusToMessage(status: number): string {
  switch (status) {
    case 404:
      return "file not found";
    case 416:
      return "range past end of file";
    case 412:
      return "file changed since cached (ETag mismatch)";
    default:
      return `HTTP ${status}`;
  }
}

/**
 * Build and throw a {@link RemoteIOError} from a transport-level failure (no
 * `response`). Classifies fetch network errors / aborts. Port of Python's
 * fetch-level classification in `_raise_remote`.
 *
 * @param url Raw URL (redacted by the error constructor).
 * @param e The thrown transport error.
 * @param status Optional HTTP status when a response was received.
 */
export function raiseRemote(
  url: string,
  e: unknown,
  status?: number | null,
): never {
  let message: string;
  if (status != null) {
    message = statusToMessage(status);
  } else if (
    e instanceof Error &&
    (e.name === "AbortError" ||
      (typeof DOMException !== "undefined" &&
        e instanceof DOMException &&
        e.name === "AbortError"))
  ) {
    message = "timeout";
  } else if (e instanceof TypeError) {
    message = "connection error";
  } else {
    const typeName =
      (e as { constructor?: { name?: string } } | null)?.constructor?.name ??
      "Error";
    message = `unexpected error: ${typeName}`;
  }
  throw new RemoteIOError({ message, url, status: status ?? null, cause: e });
}

/**
 * Copy `headers`, force `Accept-Encoding: identity`, and drop any user-supplied
 * `Accept-Encoding` (case-insensitive) so it cannot be overridden. Apply to
 * every ranged request. Port of Python `_identity_headers`.
 */
export function identityHeaders(
  headers?: Record<string, string>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers ?? {})) {
    if (k.toLowerCase() === "accept-encoding") continue;
    out[k] = v;
  }
  out["Accept-Encoding"] = "identity";
  return out;
}

/**
 * Drop sensitive headers ({@link SENSITIVE_HEADERS}) when `toUrl` is a different
 * origin than `fromUrl`; otherwise return `headers` unchanged. Port of Python
 * `_strip_cross_origin_headers`. Invoked only where WE follow redirects
 * manually (Node range reader, Drive); the browser strips `Authorization`
 * cross-origin natively.
 */
export function stripCrossOriginHeaders(
  headers: Record<string, string>,
  fromUrl: string,
  toUrl: string,
): Record<string, string> {
  let sameOrigin = false;
  try {
    sameOrigin = new URL(toUrl).origin === new URL(fromUrl).origin;
  } catch {
    sameOrigin = false;
  }
  if (sameOrigin) return headers;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) {
    if (SENSITIVE_HEADERS.has(k.toLowerCase())) continue;
    out[k] = v;
  }
  return out;
}

/**
 * Run `fn`, retrying on retryable {@link RemoteIOError}s (retryable status or a
 * connection-error classification) with exponential backoff, honoring a
 * `Retry-After` hint when present.
 *
 * Backoff: `min(200 * 2**attempt, 30000)` ms (attempt 0-indexed). Port of
 * Python `_open_with_retries` / `_retry_sleep_seconds`.
 */
export async function withRetries<T>(
  fn: () => Promise<T>,
  options?: { retries?: number },
): Promise<T> {
  const retries = options?.retries ?? 3;
  let attempt = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      return await fn();
    } catch (e) {
      const retryable =
        e instanceof RemoteIOError &&
        (e.status === null || RETRYABLE_STATUSES.has(e.status ?? -1));
      if (!retryable || attempt >= retries) {
        throw e;
      }
      const retryAfterMs = (e as RemoteIOError).retryAfterMs;
      const delayMs =
        retryAfterMs != null
          ? Math.min(retryAfterMs, 30000)
          : Math.min(200 * 2 ** attempt, 30000);
      attempt += 1;
      await new Promise((r) => setTimeout(r, delayMs));
    }
  }
}

/**
 * Parse a `Retry-After` header into milliseconds. Only the integer-seconds form
 * is honored; the HTTP-date form is ignored (returns undefined → computed
 * backoff). Used by fetch wrappers to attach `retryAfterMs` to a thrown error.
 */
export function parseRetryAfterMs(value: string | null): number | undefined {
  if (!value) return undefined;
  const secs = Number.parseInt(value, 10);
  if (!Number.isFinite(secs) || String(secs) !== value.trim()) return undefined;
  return Math.min(secs * 1000, 30000);
}

/**
 * `fetch` wrapped in {@link withRetries}: every remote byte fetch goes through
 * here so transient failures are retried with backoff (mirrors Python's
 * `_open_with_retries`, which wraps every remote open).
 *
 * Retries on:
 * - transient network errors (`raiseRemote` classifies a `TypeError` →
 *   "connection error" / an `AbortError` → "timeout" as a retryable
 *   {@link RemoteIOError} with `status === null`), and
 * - retryable HTTP statuses ({@link RETRYABLE_STATUSES}: 429/500/502/503/504),
 *   honoring a `Retry-After` header via {@link parseRetryAfterMs}.
 *
 * For a NON-retryable status (e.g. 206/404/redirect) the `Response` is returned
 * unchanged so the caller applies its own handling. Every thrown error is the
 * redacted typed {@link RemoteIOError}; the raw transport error never escapes.
 *
 * @param url Resolved fetch URL (raw; redacted by the error constructor).
 * @param init `RequestInit` (headers, method, Range, etc.).
 * @param options `retries` forwarded to {@link withRetries}.
 */
export async function fetchRetrying(
  url: string,
  init?: RequestInit,
  options?: { retries?: number },
): Promise<Response> {
  return withRetries(async () => {
    let response: Response;
    try {
      response = await fetch(url, init);
    } catch (e) {
      // Throws a redacted RemoteIOError; network errors (TypeError/AbortError)
      // are classified with status === null, which withRetries treats as
      // retryable.
      raiseRemote(url, e);
    }
    if (RETRYABLE_STATUSES.has(response.status)) {
      const err = new RemoteIOError({
        message: statusToMessage(response.status),
        url,
        status: response.status,
      });
      err.retryAfterMs = parseRetryAfterMs(
        response.headers?.get?.("Retry-After") ?? null,
      );
      throw err;
    }
    return response;
  }, options);
}

/**
 * Non-throwing existence probe for a URL. Tries HEAD, falling back to a
 * `Range: bytes=0-0` GET when HEAD is unavailable. ALWAYS returns a boolean
 * (any thrown error → `false`). Port of Python `_head_or_range_probe`.
 *
 * For Google Drive, HEAD is rejected by Google, so success is approximated by
 * whether a file id can be parsed from the URL (no network).
 */
export async function headOrRangeProbe(
  url: string,
  options?: { headers?: Record<string, string> },
): Promise<boolean> {
  let resolved: ResolvedUrl;
  try {
    resolved = resolveUrl(url);
  } catch {
    return false;
  }
  if (resolved.gdrive) {
    // Drive rejects HEAD; treat a parseable file id as "exists" without network.
    try {
      const { parseGdrive } = await import("./gdrive.js");
      parseGdrive(url);
      return true;
    } catch {
      return false;
    }
  }
  const headers = options?.headers ?? {};
  try {
    const head = await fetch(resolved.url, { method: "HEAD", headers });
    if (head.ok) return true;
    // Fall through to a ranged GET for 405 (and any other non-ok).
  } catch {
    // HEAD itself threw (network error / method blocked) — try GET below.
  }
  try {
    const ranged = await fetch(resolved.url, {
      method: "GET",
      headers: { ...headers, Range: "bytes=0-0" },
    });
    return ranged.ok || ranged.status === 206;
  } catch {
    return false;
  }
}
