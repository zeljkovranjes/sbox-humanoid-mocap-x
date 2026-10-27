import { HttpError } from './error.mjs';

/** Reads a small JSON object body (at most 4 KiB, application/json). Throws HttpError otherwise. */
export async function readJson(request) {
  if (!/^application\/json\b/i.test(request.headers['content-type'] ?? '')) throw new HttpError(415, 'json_required', 'Send application/json.');
  // Compressed bodies could expand past the size limit after decoding; only identity is accepted.
  const encoding = request.headers['content-encoding'];
  if (encoding && encoding.toLowerCase() !== 'identity') throw new HttpError(415, 'encoding_not_allowed', 'Send uncompressed request bodies.');
  const chunks = []; let size = 0;
  // Small bodies have no reason to trickle in; the server's long timeout exists for uploads.
  const timer = setTimeout(() => request.destroy(new HttpError(408, 'body_timeout', 'The request body took too long.')), 10_000);
  try {
    for await (const chunk of request) {
      size += chunk.length;
      if (size > 4096) throw new HttpError(413, 'body_too_large', 'Request exceeds 4 KiB.');
      chunks.push(chunk);
    }
  } finally { clearTimeout(timer); }
  let value;
  try { value = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}'); }
  catch { throw new HttpError(400, 'invalid_json', 'The request body is not valid JSON.'); }
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new HttpError(400, 'invalid_json', 'Send a JSON object.');
  return value;
}
