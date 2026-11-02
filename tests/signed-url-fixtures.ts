/**
 * Signed URLs and API keys in the shapes their services give them: what boto3,
 * google-cloud-storage and azure-storage-blob produce, down to the order of the params
 * and the length of each value, and what CloudFront, Backblaze and Google document.
 * The values are made up.
 */

const repeatTo = (text: string, length: number) =>
  text.repeat(Math.ceil(length / text.length)).slice(0, length);

/** AWS's own example key ID. */
const AWS_KEY_ID = "AKIAIOSFODNN7EXAMPLE";
const SERVICE_ACCOUNT = "reader%40project.iam.gserviceaccount.com";
const HMAC_SHA256_HEX = repeatTo("0123456789abcdef", 64);
const RSA_2048_HEX = repeatTo("fedcba9876543210", 512);
const RSA_2048_BASE64 = repeatTo("u3Ii7a729J0NcaQU5Brsl3N%2Fcyht", 344);
const HMAC_SHA1_BASE64 = "Jrotie1L6y7WzLpFmnOPJxBX0BU%3D";
const SESSION_TOKEN = repeatTo("IQoJb3JpZ2luX2VjEJr%2F", 1100);
const API_KEY = "not-a-real-api-key-0123456789";

export interface SignedQuery {
  name: string;
  /** What follows the `?`. */
  query: string;
  /** What no message may show. */
  secrets: string[];
  /** What a message quoting the whole query should still show. */
  kept: string[];
}

export const SIGNED_QUERIES: SignedQuery[] = [
  {
    name: "S3",
    query: `X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=${AWS_KEY_ID}%2F20261031%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20261031T092209Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=${HMAC_SHA256_HEX}`,
    secrets: [AWS_KEY_ID, HMAC_SHA256_HEX],
    kept: [
      "X-Amz-Algorithm=AWS4-HMAC-SHA256",
      "X-Amz-Date=20261031T092209Z",
      "X-Amz-Expires=3600",
      "X-Amz-SignedHeaders=host",
    ],
  },
  {
    name: "S3 with a session token",
    query: `X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=${AWS_KEY_ID}%2F20261031%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20261031T092209Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Security-Token=${SESSION_TOKEN}&X-Amz-Signature=${HMAC_SHA256_HEX}`,
    secrets: [AWS_KEY_ID, SESSION_TOKEN, HMAC_SHA256_HEX],
    kept: ["X-Amz-Date=20261031T092209Z", "X-Amz-Expires=3600"],
  },
  {
    name: "S3's older form",
    query: `AWSAccessKeyId=${AWS_KEY_ID}&Signature=${HMAC_SHA1_BASE64}&Expires=1793442129`,
    secrets: [AWS_KEY_ID, HMAC_SHA1_BASE64],
    kept: ["Expires=1793442129"],
  },
  {
    name: "S3's older form with a session token",
    query: `AWSAccessKeyId=${AWS_KEY_ID}&Signature=${HMAC_SHA1_BASE64}&x-amz-security-token=${SESSION_TOKEN}&Expires=1793442129`,
    secrets: [AWS_KEY_ID, HMAC_SHA1_BASE64, SESSION_TOKEN],
    kept: ["Expires=1793442129"],
  },
  {
    name: "Cloud Storage",
    query: `X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=${SERVICE_ACCOUNT}%2F20261031%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20261031T092209Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=${RSA_2048_HEX}`,
    secrets: [SERVICE_ACCOUNT, RSA_2048_HEX],
    kept: [
      "X-Goog-Algorithm=GOOG4-RSA-SHA256",
      "X-Goog-Date=20261031T092209Z",
      "X-Goog-Expires=3600",
      "X-Goog-SignedHeaders=host",
    ],
  },
  {
    name: "Cloud Storage's older form",
    query: `Expires=1793442129&GoogleAccessId=${SERVICE_ACCOUNT}&Signature=${RSA_2048_BASE64}`,
    secrets: [SERVICE_ACCOUNT, RSA_2048_BASE64],
    kept: ["Expires=1793442129"],
  },
  {
    name: "Azure",
    query: `se=2026-11-01T00%3A00%3A00Z&sp=r&sv=2026-10-06&sr=b&sig=${HMAC_SHA1_BASE64}`,
    secrets: [HMAC_SHA1_BASE64],
    kept: ["se=2026-11-01T00%3A00%3A00Z", "sp=r", "sv=2026-10-06", "sr=b"],
  },
  {
    name: "CloudFront",
    query: `Expires=1767290400&Signature=${RSA_2048_BASE64}&Key-Pair-Id=K2JCJMDEHXQW5F&Hash-Algorithm=SHA256`,
    secrets: [RSA_2048_BASE64],
    kept: [
      "Expires=1767290400",
      "Key-Pair-Id=K2JCJMDEHXQW5F",
      "Hash-Algorithm=SHA256",
    ],
  },
  {
    name: "Backblaze B2",
    query: `Authorization=${API_KEY}`,
    secrets: [API_KEY],
    kept: [],
  },
  {
    name: "a Google API key",
    query: `alt=media&key=${API_KEY}`,
    secrets: [API_KEY],
    kept: ["alt=media"],
  },
  {
    name: "an API key, first",
    query: `api_key=${API_KEY}&v=2`,
    secrets: [API_KEY],
    kept: ["v=2"],
  },
  {
    name: "an API key, last",
    query: `v=2&apikey=${API_KEY}`,
    secrets: [API_KEY],
    kept: ["v=2"],
  },
  {
    name: "an API key, with a hyphen",
    query: `api-key=${API_KEY}&v=2`,
    secrets: [API_KEY],
    kept: ["v=2"],
  },
];

export function signedQuery(name: string): SignedQuery {
  const found = SIGNED_QUERIES.find((signed) => signed.name === name);
  if (!found) throw new Error(`No signed query called ${name}`);
  return found;
}

/**
 * The first run of 12 characters from any secret that `text` shows, or null. Part of a
 * signature counts, since a browser can cut one short.
 */
export function shownPart(text: string, secrets: string[]): string | null {
  for (const secret of secrets) {
    for (let i = 0; i + 12 <= secret.length; i++) {
      const part = secret.slice(i, i + 12);
      if (text.includes(part)) return part;
    }
  }
  return null;
}
