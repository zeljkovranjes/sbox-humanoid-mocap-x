import http from 'node:http';
import https from 'node:https';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { createHmac } from 'node:crypto';
import { HttpError } from './error.mjs';

const maximumResultBytes = 128 * 1024 * 1024;

/** Accepts only the documented capture routes. Job IDs never grant ownership. */
export function captureRoute(method, path) {
  if (method === 'GET' && path === '/v1/capabilities') return true;
  if (['GET', 'POST'].includes(method) && path === '/v1/jobs') return true;
  const job = /^\/v1\/jobs\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(\/video|\/result)?$/.exec(path);
  return Boolean(job && (job[1] === '/video' ? method === 'PUT' : job[1] === '/result' ? method === 'GET' : ['GET', 'DELETE'].includes(method)));
}

function bounded(maximum) {
  let bytes = 0;
  return new Transform({ transform(chunk, encoding, done) {
    bytes += chunk.length;
    done(bytes > maximum ? new HttpError(413, 'body_too_large', 'Transfer exceeded its byte limit.') : null, chunk);
  } });
}

/** The processing host's key for one owner: signed with the shared secret, so the host needs no
 * per-user configuration and each user's jobs stay separate. */
export function ownerKey(secret, owner) {
  return `${owner}.${createHmac('sha256', secret).update(owner).digest('base64url')}`;
}

/** The host's operator key for /v1/settings, derived from the same secret. */
export function adminKey(secret) {
  return `admin.${createHmac('sha256', secret).update('admin').digest('base64url')}`;
}

// Editors trim by frame times, so allow the same rounding the host does.
const clipTolerance = 0.01;

/** The job body forwarded to the host: only the known capture fields plus the account's policy,
 * so callers cannot set their own priority or limits. Rejects ranges past the clip limit. */
export function jobRequest(body, limits) {
  const { model, cameraInterval, start, end = null, name = null } = body;
  if (typeof start !== 'number' || (end !== null && typeof end !== 'number'))
    throw new HttpError(400, 'invalid_job', 'start and end are video times in seconds.');
  if (limits.maximumClipSeconds !== null && end !== null && end - start > limits.maximumClipSeconds + clipTolerance)
    throw new HttpError(413, 'clip_too_long', `Clips are limited to ${limits.maximumClipSeconds} seconds for your account. Trim the clip and try again.`);
  return Buffer.from(JSON.stringify({ model, cameraInterval, start, end, name, policy: { priority: limits.priority,
    maximumClipSeconds: limits.maximumClipSeconds, maximumUploadBytes: limits.maximumUploadBytes } }));
}

/** Small authenticated JSON request to the host, for capabilities, status and settings. */
export async function hostJson(config, path, key, { method = 'GET', body } = {}) {
  let response;
  try {
    response = await fetch(new URL(path, config.processingUrl), { method, redirect: 'error', signal: AbortSignal.timeout(10_000),
      headers: { authorization: `Bearer ${key}`, ...(body ? { 'content-type': 'application/json' } : {}) },
      body: body ? JSON.stringify(body) : undefined });
  } catch { throw new HttpError(502, 'backend_unavailable', 'The processing host is not reachable.'); }
  const value = await response.json().catch(() => null);
  if (!response.ok) throw new HttpError(response.status === 401 ? 502 : response.status, 'backend_rejected',
    value?.detail ?? `The processing host answered HTTP ${response.status}.`);
  return value;
}

/** The host's capabilities as one account sees them: its own upload and clip limits and priority. */
export async function capabilities(config, owner, limits) {
  const host = await hostJson(config, 'v1/capabilities', ownerKey(config.keySecret, owner));
  const clip = Math.min(host.maximumClipSeconds ?? Infinity, limits.maximumClipSeconds ?? Infinity);
  return { ...host, maximumUploadBytes: Math.min(host.maximumUploadBytes, limits.maximumUploadBytes),
    maximumClipSeconds: Number.isFinite(clip) ? clip : null, priority: limits.priority };
}

/** Streams the authorized owner's request to the existing C# queue, including Retry-After.
 * No redirects, automatic retries or disk copies; the backend key is derived per owner.
 * `body` replaces the request body (already read and checked); `maximumUploadBytes` is the
 * account's upload limit. */
export async function proxy(request, response, config, owner, path, { body, maximumUploadBytes = config.maximumUploadBytes } = {}) {
  const upload = request.method === 'PUT';
  const maximum = upload ? maximumUploadBytes : 16_384;
  // Compressed bodies could expand past every byte limit downstream, so only identity is forwarded.
  const encoding = request.headers['content-encoding'];
  if (encoding && encoding.toLowerCase() !== 'identity') throw new HttpError(415, 'encoding_not_allowed', 'Send uncompressed request bodies.');
  const length = request.headers['content-length'];
  if (length && (!/^\d+$/.test(length) || Number(length) > maximum))
    throw new HttpError(413, 'body_too_large', upload
      ? `The video is larger than the ${Math.floor(maximum / 1048576)} MiB this account may upload.` : 'The request body is too large.');
  const url = new URL(path.slice(1), config.processingUrl);
  const headers = { authorization: `Bearer ${ownerKey(config.keySecret, owner)}` };
  if (body) Object.assign(headers, { 'content-type': 'application/json', 'content-length': String(body.length) });
  else for (const name of ['content-type', 'content-length', 'x-video-extension'])
    if (request.headers[name]) headers[name] = request.headers[name];
  const cancellation = new AbortController();
  const disconnected = () => { if (!response.writableFinished) cancellation.abort(); };
  response.once('close', disconnected);
  const signal = AbortSignal.any([cancellation.signal, AbortSignal.timeout(30 * 60 * 1000)]);
  const outgoing = (url.protocol === 'https:' ? https : http).request(url, { method: request.method, headers, signal });
  const received = new Promise((resolve, reject) => { outgoing.once('response', resolve); outgoing.once('error', reject); });
  const writing = pipeline(body ? Readable.from([body]) : request, bounded(maximum), outgoing, { signal });
  try {
    await Promise.all([writing, (async () => {
      const incoming = await received;
      if (incoming.statusCode >= 300 && incoming.statusCode < 400) {
        incoming.destroy(); throw new HttpError(502, 'backend_redirect', 'The processing host unexpectedly redirected a request.');
      }
      // A host 401/403 means the website's own key was refused, not the user's sign-in.
      if (incoming.statusCode === 401 || incoming.statusCode === 403) {
        incoming.destroy(); throw new HttpError(502, 'backend_rejected', 'The processing host refused this website. Check MOCAP_KEY_SECRET.');
      }
      response.statusCode = incoming.statusCode;
      for (const name of ['content-type', 'content-length', 'content-disposition', 'retry-after'])
        if (incoming.headers[name]) response.setHeader(name, incoming.headers[name]);
      const limit = path.endsWith('/result') ? maximumResultBytes : path === '/v1/jobs' ? 262_144 : 16_384;
      await pipeline(incoming, bounded(limit), response, { signal });
    })()]);
  } catch (error) {
    cancellation.abort();
    if (error instanceof HttpError) throw error;
    throw new HttpError(502, 'backend_unavailable', 'The processing host connection failed. Check job status before retrying.');
  } finally { response.off('close', disconnected); }
}
