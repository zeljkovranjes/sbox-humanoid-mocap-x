import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createHash, randomUUID } from 'node:crypto';
import { configuration } from '../src/config.mjs';
import { openDatabase } from '../src/database.mjs';
import { Store } from '../src/store.mjs';
import { Sessions } from '../src/sessions.mjs';
import { application } from '../src/app.mjs';
import { ownerKey, adminKey as hostAdminKey } from '../src/proxy.mjs';

async function listen(server) {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${server.address().port}`;
}

const secret = 'k'.repeat(40); const adminKey = 'a'.repeat(40);

test('live HTTP: sign-in, editor loopback flow, access rules, signed proxy keys and admin API', async context => {
  const jobs = new Map(); const observed = []; let processingAvailable = true; let redirect = false; let concurrentJobs = 1; let refuseKeys = false;
  const backend = createServer(async (request, response) => {
    const chunks = []; for await (const chunk of request) chunks.push(chunk);
    const bytes = Buffer.concat(chunks); const owner = request.headers.authorization;
    observed.push({ url: request.url, method: request.method, owner, bytes, headers: request.headers });
    response.setHeader('content-type', 'application/json');
    const send = (code, data) => { response.statusCode = code; response.end(JSON.stringify(data)); };
    if (redirect) { response.writeHead(302, { location: 'https://example.com' }); response.end(); return; }
    if (refuseKeys) { send(401, {}); return; }
    if (request.url === '/v1/capabilities') { send(200, { protocol: 1, processingAvailable, maximumUploadBytes: 4 * 1024 * 1024, maximumActiveJobs: 1,
      queueLength: 0, processing: 0, concurrentJobs, maximumClipSeconds: 600 }); return; }
    if (request.url === '/v1/settings') {
      if (owner !== `Bearer ${hostAdminKey(secret)}`) { send(401, {}); return; }
      if (request.method === 'PUT') { const wanted = JSON.parse(bytes).concurrentJobs; if (wanted > 4) { send(400, { detail: 'Allow 1 to 4 jobs processing at once.' }); return; } concurrentJobs = wanted; }
      send(200, { concurrentJobs, maximumConcurrentJobs: 4 }); return;
    }
    if (request.url === '/v1/jobs' && request.method === 'GET') { send(200, [...jobs.values()].filter(job => job.owner === owner)); return; }
    if (request.method === 'POST') {
      if (!processingAvailable) { send(503, { detail: 'Processor unavailable' }); return; }
      const body = JSON.parse(bytes);
      const job = { id: randomUUID(), owner, state: 'awaiting-upload', message: 'Upload video', name: body.name ?? null, priority: body.policy.priority };
      jobs.set(job.id, job); send(201, job); return;
    }
    const id = request.url.split('/')[3]; const job = jobs.get(id);
    if (!job || job.owner !== owner) { send(404, {}); return; }
    if (request.method === 'PUT') { job.state = 'queued'; send(202, {}); return; }
    if (request.method === 'DELETE') { jobs.delete(id); response.writeHead(204); response.end(); return; }
    if (request.url.endsWith('/result')) { send(job.state === 'completed' ? 200 : 409, { fixture: 'motion' }); return; }
    response.setHeader('retry-after', '2'); send(200, { ...job, queuePosition: job.state === 'queued' ? 1 : null });
  });
  const backendUrl = await listen(backend);
  const config = configuration({ PROCESSING_URL: backendUrl, MOCAP_KEY_SECRET: secret, ADMIN_API_KEY: adminKey, DISCORD_GUILD_ID: '777', MAX_UPLOAD_MB: '1' });
  let account = { provider: 'discord', id: 'discord:123', username: 'person', name: 'Person', avatar: null, inGuild: true, guildOwner: false };
  const fake = { name: 'discord', label: 'Discord', authorize: async () => ({ url: 'https://discord.com/oauth2/authorize?x=1', state: 'state', verifier: 'v' }),
    identify: async () => account };
  const providers = new Map([['discord', fake]]);
  const database = await openDatabase('sqlite::memory:'); const store = new Store(database);
  await store.ensureDefaultRole(30);
  const sessions = new Sessions(config, providers, store);
  const server = createServer(await application({ config, sessions, store, providers }));
  const base = await listen(server); config.publicUrl = new URL(base);
  context.after(async () => { server.closeAllConnections(); backend.closeAllConnections();
    await Promise.all([new Promise(r => server.close(r)), new Promise(r => backend.close(r))]); await database.destroy(); });
  const manual = { redirect: 'manual' };
  const loginCookie = response => response.headers.get('set-cookie').match(/mocap_login=([A-Za-z0-9_-]+)/)[1];
  async function browserLogin() {
    let response = await fetch(base + '/auth/login/discord', manual);
    assert.equal(response.status, 302); assert.equal(response.headers.get('location'), 'https://discord.com/oauth2/authorize?x=1');
    response = await fetch(base + '/auth/callback/discord?code=c&state=state', { ...manual, headers: { cookie: `mocap_login=${loginCookie(response)}` } });
    assert.equal(response.status, 303);
    return response.headers.get('set-cookie').match(/mocap_session=([A-Za-z0-9_-]{43})/)[1];
  }

  let response = await fetch(base + '/'); assert.equal(response.status, 200);
  response = await fetch(base + '/auth/providers'); assert.deepEqual(await response.json(), { providers: [{ name: 'discord', label: 'Discord' }] });
  assert.equal((await fetch(base + '/auth/login/unknown', manual)).status, 404);
  assert.deepEqual(await (await fetch(base + '/auth/me')).json(), { signedIn: false });

  // Browser session: signed in but not whitelisted yet.
  const cookie = `mocap_session=${await browserLogin()}`;
  const headers = { cookie, origin: base, 'content-type': 'application/json' };
  const me = await (await fetch(base + '/auth/me', { headers: { cookie } })).json();
  assert.deepEqual({ signedIn: me.signedIn, id: me.user.id, whitelisted: me.whitelisted, canProcess: me.canProcess }, { signedIn: true, id: 'discord:123', whitelisted: false, canProcess: false });
  response = await fetch(base + '/v1/capabilities', { headers: { cookie } }); assert.equal(response.status, 403);
  assert.equal((await response.json()).error.code, 'not_whitelisted');
  assert.equal((await fetch(base + '/v1/capabilities')).status, 401);

  // Admin API with the operator key: whitelist the account.
  const admin = { authorization: `Bearer ${adminKey}`, 'content-type': 'application/json' };
  assert.equal((await fetch(base + '/admin/users')).status, 401);
  assert.equal((await fetch(base + '/admin/users', { headers: { authorization: 'Bearer ' + 'x'.repeat(40) } })).status, 403);
  assert.equal((await fetch(base + '/admin/users', { headers: { cookie } })).status, 403, 'regular accounts are not admins');
  response = await fetch(base + '/admin/whitelist/discord:123', { method: 'PUT', headers: admin, body: JSON.stringify({ note: 'friend' }) });
  assert.equal(response.status, 200);
  assert.equal((await fetch(base + '/admin/whitelist/not-an-account', { method: 'PUT', headers: admin, body: '{}' })).status, 400);
  assert.equal((await fetch(base + '/admin/whitelist/discord:5', { method: 'PUT', headers: { authorization: admin.authorization }, body: '{}' })).status, 415);
  const listed = await (await fetch(base + '/admin/whitelist', { headers: admin })).json();
  assert.deepEqual(listed.whitelist.map(row => [row.id, row.note, row.name]), [['discord:123', 'friend', 'Person']]);
  const users = await (await fetch(base + '/admin/users?search=Pers', { headers: admin })).json();
  assert.deepEqual(users.users.map(user => [user.id, user.whitelisted, user.inGuild]), [['discord:123', true, true]]);
  assert.equal((await (await fetch(base + '/admin/users/discord%3A123', { headers: admin })).json()).user.note, 'friend');
  assert.equal((await fetch(base + '/admin/users/discord:404', { headers: admin })).status, 404);
  assert.equal((await fetch(base + '/admin/users?limit=999999', { headers: admin })).status, 400);
  const signIns = await (await fetch(base + '/admin/sign-ins?account=discord:123', { headers: admin })).json();
  assert.equal(signIns.signIns.length, 1); assert.equal(signIns.signIns[0].client, 'browser');
  const status = await (await fetch(base + '/admin/status', { headers: admin })).json();
  assert.deepEqual([status.users, status.whitelisted, status.host.processingAvailable], [1, 1, true]);

  // Roles, Discord role mapping and host settings through the admin API.
  assert.deepEqual((await (await fetch(base + '/admin/roles', { headers: admin })).json()).roles,
    [{ name: 'default', priority: false, maxClipSeconds: 30, maxUploadMb: null, discordRoleId: null }]);
  response = await fetch(base + '/admin/roles/vip', { method: 'PUT', headers: admin, body: JSON.stringify({ priority: true, maxClipSeconds: 90 }) });
  assert.equal(response.status, 200);
  assert.equal((await fetch(base + '/admin/roles/Bad%20Name', { method: 'PUT', headers: admin, body: '{}' })).status, 400);
  assert.equal((await fetch(base + '/admin/roles/vip', { method: 'PUT', headers: admin, body: JSON.stringify({ maxClipSeconds: 0 }) })).status, 400);
  assert.equal((await fetch(base + '/admin/roles/default', { method: 'DELETE', headers: admin })).status, 400);
  assert.equal((await fetch(base + '/admin/whitelist/discord:123', { method: 'PUT', headers: admin, body: JSON.stringify({ note: 'friend', role: 'nobody' }) })).status, 400);
  assert.equal((await fetch(base + '/admin/whitelist/discord:123', { method: 'PUT', headers: admin, body: JSON.stringify({ note: 'friend', role: 'vip' }) })).status, 200);
  const vip = (await (await fetch(base + '/admin/users/discord:123', { headers: admin })).json()).user;
  assert.deepEqual([vip.role, vip.roles, vip.limits.priority, vip.limits.maximumClipSeconds], ['vip', ['default', 'vip'], true, 90]);
  store.roleCache = null; store.cache.clear();
  assert.equal((await (await fetch(base + '/auth/me', { headers: { cookie } })).json()).limits.priority, true);
  assert.equal((await fetch(base + '/admin/whitelist/discord:123', { method: 'PUT', headers: admin, body: JSON.stringify({ note: 'friend' }) })).status, 200);
  assert.equal((await fetch(base + '/admin/roles/vip', { method: 'DELETE', headers: admin })).status, 204);
  store.roleCache = null; store.cache.clear();
  assert.deepEqual(await (await fetch(base + '/admin/settings', { headers: admin })).json(), { concurrentJobs: 1, maximumConcurrentJobs: 4 });
  response = await fetch(base + '/admin/settings', { method: 'PUT', headers: admin, body: JSON.stringify({ concurrentJobs: 2 }) });
  assert.deepEqual([response.status, concurrentJobs], [200, 2]);
  response = await fetch(base + '/admin/settings', { method: 'PUT', headers: admin, body: JSON.stringify({ concurrentJobs: 9 }) });
  assert.deepEqual([response.status, (await response.json()).error.message], [400, 'Allow 1 to 4 jobs processing at once.']);
  assert.equal((await fetch(base + '/admin/settings', { method: 'PUT', headers: admin, body: JSON.stringify({ concurrentJobs: '2' }) })).status, 400);
  assert.equal((await fetch(base + '/admin/settings', { headers: { cookie } })).status, 403);

  // Processing through the proxy with the owner's signed key and the account's policy.
  const capabilities = await (await fetch(base + '/v1/capabilities', { headers: { cookie } })).json();
  assert.deepEqual([capabilities.maximumUploadBytes, capabilities.maximumClipSeconds, capabilities.priority], [1024 * 1024, 30, false],
    'capabilities report the account\'s own limits');
  const before = observed.length;
  response = await fetch(base + '/v1/jobs', { method: 'POST', headers, body: JSON.stringify({ model: 'Latest', cameraInterval: 10, start: 0, end: 31 }) });
  assert.deepEqual([response.status, (await response.json()).error.code, observed.length], [413, 'clip_too_long', before], 'long clips stop before the host');
  response = await fetch(base + '/v1/jobs', { method: 'POST', headers,
    body: JSON.stringify({ model: 'Latest', cameraInterval: 10, start: 0, end: 5, name: 'walk.mp4', policy: { priority: true }, extra: 1 }) });
  assert.equal(response.status, 201); const job = await response.json();
  assert.equal(observed.at(-1).owner, `Bearer ${ownerKey(secret, 'discord:123')}`); assert.equal(observed.at(-1).headers.cookie, undefined);
  assert.deepEqual(JSON.parse(observed.at(-1).bytes), { model: 'Latest', cameraInterval: 10, start: 0, end: 5, name: 'walk.mp4',
    policy: { priority: false, maximumClipSeconds: 30, maximumUploadBytes: 1024 * 1024 } }, 'only known fields and the server-side policy reach the host');
  assert.equal((await fetch(base + '/v1/jobs', { method: 'POST', headers: { cookie, 'content-type': 'application/json' }, body: '{}' })).status, 403, 'browser writes need Origin');
  const forwarded = observed.length;
  response = await fetch(base + '/v1/jobs', { method: 'POST', headers: { ...headers, 'content-encoding': 'gzip' }, body: '{}' });
  assert.deepEqual([response.status, observed.length], [415, forwarded], 'compressed JSON is refused before the host');
  response = await fetch(base + `/v1/jobs/${job.id}/video`, { method: 'PUT', headers: { ...headers, 'x-video-extension': '.mp4' }, body: Buffer.alloc(1024, 7) });
  assert.equal(response.status, 202);
  response = await fetch(base + `/v1/jobs/${job.id}/video`, { method: 'PUT', headers: { ...headers, 'x-video-extension': '.mp4', 'content-encoding': 'gzip' }, body: Buffer.alloc(16) });
  assert.equal(response.status, 415, 'compressed uploads are refused');
  response = await fetch(base + `/v1/jobs/${job.id}/video`, { method: 'PUT', headers: { ...headers, 'x-video-extension': '.mp4' }, body: Buffer.alloc(1024 * 1024 + 1) });
  assert.equal(response.status, 413, 'uploads over MAX_UPLOAD_MB are refused before forwarding');
  const mine = await (await fetch(base + '/v1/jobs', { headers: { cookie } })).json();
  assert.deepEqual(mine.map(entry => [entry.id, entry.name, entry.state]), [[job.id, 'walk.mp4', 'queued']]);
  const viaAdmin = await (await fetch(base + '/admin/users/discord:123/jobs', { headers: admin })).json();
  assert.deepEqual(viaAdmin.map(entry => entry.id), [job.id]);

  // Editor: straight redirect to the provider, loopback code, PKCE exchange, bearer access.
  const verifier = 'e'.repeat(64); const challenge = createHash('sha256').update(verifier).digest('base64url');
  const editorUrl = `${base}/auth/editor?provider=discord&port=50123&state=${'s'.repeat(20)}&code_challenge=${challenge}&code_challenge_method=S256`;
  assert.equal((await fetch(editorUrl.replace('S256', 'plain'), manual)).status, 400);
  response = await fetch(editorUrl, manual);
  assert.equal(response.status, 302); assert.match(response.headers.get('location'), /^https:\/\/discord\.com\//);
  response = await fetch(base + '/auth/callback/discord?code=c&state=state', { ...manual, headers: { cookie: `mocap_login=${loginCookie(response)}` } });
  const loopback = new URL(response.headers.get('location'));
  assert.equal(loopback.origin, 'http://127.0.0.1:50123'); assert.equal(loopback.searchParams.get('state'), 's'.repeat(20));
  const exchange = body => fetch(base + '/auth/editor/token', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  response = await exchange({ code: loopback.searchParams.get('code'), verifier });
  assert.equal(response.status, 200); const editor = await response.json();
  assert.deepEqual([editor.user.id, editor.whitelisted, editor.canProcess, typeof editor.expiresAt], ['discord:123', true, true, 'string']);
  assert.equal((await exchange({ code: loopback.searchParams.get('code'), verifier })).status, 400, 'codes are single use');
  const bearer = { authorization: `Bearer ${editor.token}` };
  assert.equal((await (await fetch(base + '/auth/me', { headers: bearer })).json()).user.id, 'discord:123');
  assert.equal((await fetch(base + `/v1/jobs/${job.id}`, { headers: bearer })).status, 200);
  assert.equal((await fetch(base + `/v1/jobs/${job.id}`, { method: 'DELETE', headers: bearer })).status, 204);

  // Removing the account from the whitelist takes effect immediately (after the short cache).
  assert.equal((await fetch(base + '/admin/whitelist/discord:123', { method: 'DELETE', headers: admin })).status, 204);
  store.cache.clear();
  assert.equal((await fetch(base + '/v1/capabilities', { headers: bearer })).status, 403);
  assert.equal((await fetch(base + '/admin/whitelist/discord:123', { method: 'DELETE', headers: admin })).status, 404);

  // The Discord server owner signs in and gets full admin access without any key.
  account = { ...account, id: 'discord:1', name: 'Owner', guildOwner: true };
  const ownerCookie = `mocap_session=${await browserLogin()}`;
  const ownerMe = await (await fetch(base + '/auth/me', { headers: { cookie: ownerCookie } })).json();
  assert.deepEqual([ownerMe.admin, ownerMe.canProcess, ownerMe.limits.priority], [true, true, true]);
  response = await fetch(base + '/v1/jobs', { method: 'POST', headers: { cookie: ownerCookie, origin: base, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'Latest', cameraInterval: 10, start: 0, end: 5 }) });
  assert.equal((await response.json()).priority, true, 'the server owner\'s jobs use the priority queue');
  assert.equal((await fetch(base + '/admin/users', { headers: { cookie: ownerCookie } })).status, 200);
  assert.equal((await fetch(base + '/admin/whitelist/discord:2', { method: 'PUT', headers: { cookie: ownerCookie, 'content-type': 'application/json' }, body: '{}' })).status, 403,
    'cookie-authenticated admin writes need Origin');
  assert.equal((await fetch(base + '/admin/whitelist/discord:2', { method: 'PUT', headers: { cookie: ownerCookie, origin: base, 'content-type': 'application/json' }, body: '{}' })).status, 200);

  // Admin sign-out revokes every session and editor of the account.
  const revoked = await (await fetch(base + '/admin/users/discord:123/sign-out', { method: 'POST', headers: admin })).json();
  assert.equal(revoked.revoked, 2);
  assert.equal((await fetch(base + '/v1/capabilities', { headers: bearer })).status, 401);

  // Host failures surface as 502 without leaking backend details.
  redirect = true; assert.equal((await fetch(base + '/v1/capabilities', { headers: { cookie: ownerCookie } })).status, 502); redirect = false;
  refuseKeys = true;
  response = await fetch(base + '/v1/jobs', { headers: { cookie: ownerCookie } });
  assert.deepEqual([response.status, (await response.json()).error.code], [502, 'backend_rejected'], 'a host 401 must not look like an expired sign-in');
  refuseKeys = false;
  processingAvailable = false;
  assert.equal((await fetch(base + '/v1/jobs', { method: 'POST', headers: { cookie: ownerCookie, origin: base, 'content-type': 'application/json' }, body: '{}' })).status, 400);
  assert.equal((await fetch(base + '/v1/jobs', { method: 'POST', headers: { cookie: ownerCookie, origin: base, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'Latest', cameraInterval: 10, start: 0, end: 5 }) })).status, 503);
  assert.equal((await fetch(base + '/v1/jobs?redirect=https://example.com', { headers: { cookie: ownerCookie } })).status, 404);
  response = await fetch(base + '/auth/logout', { method: 'POST', headers: { cookie: ownerCookie, origin: base } }); assert.equal(response.status, 204);
  assert.deepEqual(await (await fetch(base + '/auth/me', { headers: { cookie: ownerCookie } })).json(), { signedIn: false });
});

test('HTTPS deployments use __Host- cookies and HSTS', async context => {
  const config = configuration({ MOCAP_KEY_SECRET: secret, PUBLIC_URL: 'https://mocap.example' });
  const fake = { name: 'discord', label: 'Discord', authorize: async () => ({ url: 'https://discord.com/oauth2/authorize', state: 'state', verifier: 'v' }),
    identify: async () => ({ provider: 'discord', id: 'discord:5', username: null, name: null, avatar: null, inGuild: null, guildOwner: false }) };
  const providers = new Map([['discord', fake]]);
  const database = await openDatabase('sqlite::memory:'); const store = new Store(database);
  const server = createServer(await application({ config, sessions: new Sessions(config, providers, store), store, providers }));
  const base = await listen(server);
  context.after(async () => { server.closeAllConnections(); await new Promise(r => server.close(r)); await database.destroy(); });
  let response = await fetch(base + '/auth/login/discord', { redirect: 'manual' });
  assert.equal(response.headers.get('strict-transport-security'), 'max-age=31536000');
  const login = response.headers.get('set-cookie');
  assert.match(login, /^__Host-mocap_login=[A-Za-z0-9_-]+; Path=\/; HttpOnly; SameSite=Lax; Max-Age=300; Secure$/);
  response = await fetch(base + '/auth/callback/discord?code=c&state=state', { redirect: 'manual',
    headers: { cookie: `mocap_login=${login.split(';')[0].split('=')[1]}` } });
  assert.equal(response.status, 400, 'the unprefixed name is ignored over HTTPS');
  response = await fetch(base + '/auth/callback/discord?code=c&state=state', { redirect: 'manual', headers: { cookie: login.split(';')[0] } });
  assert.equal(response.status, 303);
  const cookies = response.headers.getSetCookie();
  assert.deepEqual(cookies.map(value => value.split('=')[0]), ['__Host-mocap_login', '__Host-mocap_session'], 'clearing the login keeps the session cookie');
});
