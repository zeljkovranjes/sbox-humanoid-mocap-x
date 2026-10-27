import { timingSafeEqual, createHash } from 'node:crypto';
import { HttpError, json } from './error.mjs';
import { proxy, ownerKey, adminKey, hostJson } from './proxy.mjs';
import { readJson } from './body.mjs';
import { accountPattern } from './config.mjs';
import { parseRole, effectivePolicy } from './roles.mjs';

const hash = value => createHash('sha256').update(value).digest();
const jobPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** True when the bearer token is the operator's ADMIN_API_KEY (constant-time comparison). */
export function isAdminKey(config, authorization) {
  if (!config.adminKey || !authorization?.startsWith('Bearer ')) return false;
  return timingSafeEqual(hash(authorization.slice(7)), hash(config.adminKey));
}

/** Admin API for operators and Discord bots: accounts, whitelist, roles, sign-ins, an account's
 * jobs, sign-out and host settings. Accounts are "<provider>:<account ID>". The caller has
 * already been authorized as an admin. Throws HttpError on bad input. */
export async function adminRoute(request, response, url, { config, store, sessions }) {
  let parts;
  try { parts = url.pathname.split('/').filter(Boolean).slice(1).map(decodeURIComponent); }
  catch { throw new HttpError(400, 'invalid_path', 'The path is not valid URL encoding.'); }
  const method = request.method;
  const number = (name, fallback, maximum) => {
    const value = url.searchParams.get(name);
    if (value === null) return fallback;
    if (!/^\d{1,6}$/.test(value) || Number(value) > maximum) throw new HttpError(400, 'invalid_query', `${name} must be 0–${maximum}.`);
    return Number(value);
  };
  const id = parts[1];
  if (['users', 'whitelist'].includes(parts[0]) && id !== undefined && !accountPattern.test(id))
    throw new HttpError(400, 'invalid_account', 'Accounts are <provider>:<account ID>, for example discord:123.');

  if (method === 'GET' && parts.length === 1 && parts[0] === 'status') {
    json(response, 200, { ...await store.counts(), browserSessions: sessions.sessions.size, editors: sessions.tokens.size,
      host: await hostStatus(config) });
    return;
  }
  if (parts[0] === 'settings' && parts.length === 1 && ['GET', 'PUT'].includes(method)) {
    const body = method === 'PUT' ? await readJson(request) : undefined;
    if (body && !Number.isInteger(body.concurrentJobs)) throw new HttpError(400, 'invalid_settings', 'concurrentJobs must be a whole number.');
    json(response, 200, await hostJson(config, 'v1/settings', adminKey(config.keySecret),
      { method, body: body && { concurrentJobs: body.concurrentJobs } }));
    return;
  }
  if (method === 'GET' && parts.length === 1 && parts[0] === 'users') {
    const search = url.searchParams.get('search') ?? '';
    if (search.length > 80) throw new HttpError(400, 'invalid_query', 'search is at most 80 characters.');
    json(response, 200, { users: await store.listUsers({ search, limit: number('limit', 50, 500), offset: number('offset', 0, 1_000_000) }) });
    return;
  }
  if (parts[0] === 'users' && parts.length === 2 && method === 'GET') {
    const user = await store.user(id);
    if (!user) throw new HttpError(404, 'not_found', 'No account with this ID has signed in.');
    const policy = effectivePolicy(await store.roles(), { assigned: user.role, discordRoles: user.discordRoles, guildOwner: user.guildOwner }, config);
    json(response, 200, { user: { ...user, roles: policy.roles, limits: { priority: policy.priority,
      maximumClipSeconds: policy.maximumClipSeconds, maximumUploadBytes: policy.maximumUploadBytes } } });
    return;
  }
  if (parts[0] === 'users' && parts[2] === 'sign-out' && parts.length === 3 && method === 'POST') {
    json(response, 200, { revoked: sessions.revokeUser(id) }); return;
  }
  if (parts[0] === 'users' && parts[2] === 'jobs' && parts.length === 3 && method === 'GET') {
    await proxy(request, response, config, id, '/v1/jobs'); return;
  }
  if (parts[0] === 'users' && parts[2] === 'jobs' && parts.length === 4 && method === 'DELETE') {
    if (!jobPattern.test(parts[3])) throw new HttpError(400, 'invalid_job', 'Job IDs are lowercase UUIDs.');
    await proxy(request, response, config, id, `/v1/jobs/${parts[3]}`); return;
  }
  if (parts[0] === 'whitelist' && parts.length === 1 && method === 'GET') {
    json(response, 200, { whitelist: (await store.listWhitelist()).map(row => ({ id: row.account_id, note: row.note, role: row.role ?? null,
      addedAt: row.added_at, username: row.username ?? null, name: row.display_name ?? null })) });
    return;
  }
  if (parts[0] === 'whitelist' && parts.length === 2 && method === 'PUT') {
    const body = await readJson(request);
    if (body.role !== undefined && body.role !== null && typeof body.role !== 'string') throw new HttpError(400, 'unknown_role', 'role must be a role name or null.');
    await store.whitelist(id, body.note ?? '', body.role ?? null);
    json(response, 200, { id, whitelisted: true, role: body.role ?? null }); return;
  }
  if (parts[0] === 'whitelist' && parts.length === 2 && method === 'DELETE') {
    if (!await store.unwhitelist(id)) throw new HttpError(404, 'not_found', 'This account is not whitelisted.');
    response.writeHead(204); response.end(); return;
  }
  if (parts[0] === 'roles' && parts.length === 1 && method === 'GET') {
    json(response, 200, { roles: await store.roles() }); return;
  }
  if (parts[0] === 'roles' && parts.length === 2 && method === 'PUT') {
    const role = parseRole(parts[1], await readJson(request));
    await store.saveRole(role); json(response, 200, { role }); return;
  }
  if (parts[0] === 'roles' && parts.length === 2 && method === 'DELETE') {
    if (!await store.deleteRole(parts[1])) throw new HttpError(404, 'not_found', 'There is no role with this name.');
    response.writeHead(204); response.end(); return;
  }
  if (parts[0] === 'sign-ins' && parts.length === 1 && method === 'GET') {
    const filter = url.searchParams.get('account') ?? undefined;
    if (filter !== undefined && !accountPattern.test(filter)) throw new HttpError(400, 'invalid_account', 'Accounts are <provider>:<account ID>.');
    json(response, 200, { signIns: await store.signIns({ id: filter, limit: number('limit', 50, 1000) }) }); return;
  }
  throw new HttpError(404, 'not_found', 'Route not found.');
}

// Capabilities need any valid key; host:status is never a real account ("host" is a reserved
// provider name) and owns no jobs.
async function hostStatus(config) {
  try { return await hostJson(config, 'v1/capabilities', ownerKey(config.keySecret, 'host:status')); }
  catch (error) { return { reachable: false, status: error.status }; }
}
