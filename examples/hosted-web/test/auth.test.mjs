import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { configuration } from '../src/config.mjs';
import { providers, discordProfile, oidcProfile } from '../src/identity.mjs';
import { openDatabase } from '../src/database.mjs';
import { Store } from '../src/store.mjs';
import { Sessions } from '../src/sessions.mjs';

const secret = 's'.repeat(40);
const base = { MOCAP_KEY_SECRET: secret, DISCORD_GUILD_ID: '777' };
const settings = (extra = {}) => configuration({ ...base, DISCORD_CLIENT_ID: '456', DISCORD_CLIENT_SECRET: 'example-client-secret', ...extra });
const challenge = verifier => createHash('sha256').update(verifier).digest('base64url');

test('Discord sign-in uses code + PKCE + state and reads server membership and ownership', async () => {
  const config = settings(); const requests = []; let guilds = [{ id: '777', owner: true }, { id: '1' }]; let scope = 'identify guilds guilds.members.read';
  const fetcher = async (url, options) => {
    requests.push({ url: String(url), options });
    if (String(url).endsWith('/oauth2/token')) return Response.json({ access_token: 'access', token_type: 'Bearer', expires_in: 3600, scope });
    if (String(url).endsWith('/users/@me')) return Response.json({ id: '123', username: 'mocap-user', global_name: 'Mocap User', avatar: null });
    if (String(url).endsWith('/guilds/777/member')) return Response.json({ roles: ['555', 'bad role'] });
    return Response.json(guilds);
  };
  const discord = (await providers(config, fetcher)).get('discord');
  const pending = await discord.authorize(); const url = new URL(pending.url);
  assert.equal(url.searchParams.get('scope'), 'identify guilds guilds.members.read');
  assert.equal(url.searchParams.get('code_challenge_method'), 'S256');
  assert.equal(url.searchParams.get('redirect_uri'), 'http://127.0.0.1:5081/auth/callback/discord');
  const callback = () => { const value = new URL('/auth/callback/discord', config.publicUrl);
    value.search = new URLSearchParams({ state: pending.state, code: 'code', iss: 'https://discord.com' }); return value; };
  const owner = await discord.identify(callback(), pending);
  assert.deepEqual({ id: owner.id, name: owner.name, inGuild: owner.inGuild, guildOwner: owner.guildOwner, discordRoles: owner.discordRoles },
    { id: 'discord:123', name: 'Mocap User', inGuild: true, guildOwner: true, discordRoles: ['555'] });
  assert.equal(new URLSearchParams(requests[0].options.body).get('code_verifier'), pending.verifier);
  guilds = [{ id: '1' }];
  assert.deepEqual((({ inGuild, guildOwner }) => ({ inGuild, guildOwner }))(await discord.identify(callback(), pending)), { inGuild: false, guildOwner: false });
  scope = 'identify guilds'; guilds = [{ id: '777' }];
  assert.deepEqual((await discord.identify(callback(), pending)).discordRoles, [], 'member roles are optional');
  scope = 'identify';
  await assert.rejects(discord.identify(callback(), pending), { status: 403 });
  const wrong = callback(); wrong.searchParams.set('state', 'wrong'); const before = requests.length;
  await assert.rejects(discord.identify(wrong, pending));
  assert.equal(requests.length, before, 'state mismatch must fail before exchanging credentials');
});

test('OpenID Connect providers discover endpoints and allow missing names and pictures', async () => {
  const issuer = 'https://login.example';
  const config = configuration({ ...base, OIDC_PROVIDERS: 'corp', OIDC_CORP_ISSUER: issuer, OIDC_CORP_CLIENT_ID: 'client', OIDC_CORP_CLIENT_SECRET: 'secret', OIDC_CORP_LABEL: 'Corp' });
  const now = Math.floor(Date.now() / 1000);
  const part = value => Buffer.from(JSON.stringify(value)).toString('base64url');
  const idToken = `${part({ alg: 'RS256', typ: 'JWT' })}.${part({ iss: issuer, aud: 'client', sub: 'abc-123', iat: now, exp: now + 300 })}.c2lnbmF0dXJl`;
  const fetcher = async url => {
    if (String(url).endsWith('/.well-known/openid-configuration'))
      return Response.json({ issuer, authorization_endpoint: issuer + '/authorize', token_endpoint: issuer + '/token' });
    return Response.json({ access_token: 'access', token_type: 'Bearer', expires_in: 3600, id_token: idToken });
  };
  const corp = (await providers(config, fetcher)).get('corp');
  assert.equal(corp.label, 'Corp');
  const pending = await corp.authorize();
  assert.equal(new URL(pending.url).searchParams.get('scope'), 'openid profile');
  const callback = new URL('/auth/callback/corp', config.publicUrl); callback.search = new URLSearchParams({ state: pending.state, code: 'code' });
  const account = await corp.identify(callback, pending);
  assert.deepEqual(account, { provider: 'corp', id: 'corp:abc-123', username: null, name: null, avatar: null, inGuild: null, guildOwner: false });
});

test('profiles keep only validated fields', () => {
  const discord = discordProfile({ id: '123', username: '<script>', global_name: null, avatar: 'a_' + 'a'.repeat(32), email: 'hidden' });
  assert.equal(discord.name, '<script>'); assert.equal(discord.email, undefined);
  assert.match(discord.avatar, /^https:\/\/cdn.discordapp.com\/avatars\/123\//);
  assert.throws(() => discordProfile({ id: '../secret', username: 'name' }));
  assert.throws(() => discordProfile({ id: '123', username: 'name', avatar: '../path' }));
  assert.equal(oidcProfile('google', '42', { picture: 'http://insecure/avatar.png', name: 'G' }).avatar, null);
  assert.equal(oidcProfile('google', '42', { picture: 'https://lh3.googleusercontent.com/a/x=s96' }).avatar, 'https://lh3.googleusercontent.com/a/x=s96');
  assert.equal(oidcProfile('corp', '42', { picture: 'https://intranet.example/avatar.png' }).avatar, null, 'unknown avatar hosts are dropped');
  assert.throws(() => oidcProfile('google', 'bad subject', {}));
});

test('editor sign-in exchanges a one-time code with PKCE; access follows whitelist, server and admin rules', async () => {
  let now = 1000; let account = { provider: 'discord', id: 'discord:123', username: 'person', name: 'Person', avatar: null, inGuild: true, guildOwner: false };
  const fake = { name: 'discord', label: 'Discord', authorize: async () => ({ url: 'https://discord.com/', state: 'state', verifier: 'verifier' }),
    identify: async () => account };
  const config = settings({ ADMIN_ACCOUNTS: 'google:root' });
  const database = await openDatabase('sqlite::memory:'); const store = new Store(database, () => now);
  await store.ensureDefaultRole(30);
  const sessions = new Sessions(config, new Map([['discord', fake], ['google', { ...fake, name: 'google' }]]), store, () => now);
  const verifier = 'v'.repeat(43);
  const editor = { port: 52345, state: 'editor-state-0000', challenge: challenge(verifier) };
  await assert.rejects(sessions.begin('missing'), { status: 404 });
  await assert.rejects(sessions.begin('discord', { ...editor, port: 80 }), { status: 400 });
  const started = await sessions.begin('discord', editor);
  await assert.rejects(sessions.complete('google', started.token, new URL('https://example.test/')), { status: 400 });
  const tampered = started.token.slice(0, -2) + (started.token.at(-2) === 'A' ? 'B' : 'A') + started.token.at(-1);
  await assert.rejects(sessions.complete('discord', tampered, new URL('https://example.test/')), { status: 400 }, 'login cookies are authenticated');
  now += 5 * 60 * 1000;
  await assert.rejects(sessions.complete('discord', started.token, new URL('https://example.test/')), { status: 400 }, 'login cookies expire');
  const second = await sessions.begin('discord', editor);
  const { editorRedirect } = await sessions.complete('discord', second.token, new URL('https://example.test/'));
  const redirect = new URL(editorRedirect);
  assert.equal(redirect.origin, 'http://127.0.0.1:52345'); assert.equal(redirect.searchParams.get('state'), editor.state);
  const code = redirect.searchParams.get('code');
  assert.throws(() => sessions.exchange(code, 'w'.repeat(43)), { status: 400 });
  assert.throws(() => sessions.exchange(code, verifier), { status: 400 }, 'a failed attempt consumes the code');
  const third = await sessions.begin('discord', editor);
  const again = new URL((await sessions.complete('discord', third.token, new URL('https://example.test/'))).editorRedirect).searchParams.get('code');
  const issued = sessions.exchange(again, verifier);
  const token = sessions.bearer(issued.token);
  assert.equal(token.owner, 'discord:123');
  assert.equal((await sessions.profile(token)).canProcess, false, 'not whitelisted yet');
  await store.whitelist('discord:123', 'friend'); store.cache.clear();
  const defaults = { priority: false, maximumClipSeconds: 30, maximumUploadBytes: 512 * 1024 * 1024 };
  assert.deepEqual(await sessions.profile(token), { user: token.user, inGuild: true, guildOwner: false, whitelisted: true, canProcess: true,
    admin: false, roles: ['default'], limits: defaults });

  // Roles: the most generous of default, the assigned role and Discord-mapped roles applies.
  await store.saveRole({ name: 'supporter', priority: true, maxClipSeconds: 120, maxUploadMb: 64, discordRoleId: '555' });
  await store.saveRole({ name: 'trial', priority: false, maxClipSeconds: 10, maxUploadMb: null, discordRoleId: null });
  await store.whitelist('discord:123', 'friend', 'trial'); store.cache.clear(); store.roleCache = null;
  assert.deepEqual((await sessions.profile(token)).limits, defaults, 'a stricter role never lowers the default');
  assert.deepEqual((await sessions.profile({ ...token, discordRoles: ['555'] })),
    { ...await sessions.profile(token), roles: ['default', 'supporter', 'trial'], limits: { priority: true, maximumClipSeconds: 120, maximumUploadBytes: 512 * 1024 * 1024 } });
  await assert.rejects(store.whitelist('discord:123', '', 'missing'), { status: 400 });
  await assert.rejects(store.deleteRole('default'), { status: 400 });
  assert.equal(await store.deleteRole('trial'), true); store.cache.clear();
  assert.equal((await store.access('discord:123')).role, null, 'deleting a role unassigns it');
  assert.equal(sessions.revokeUser('discord:123'), 1); assert.equal(sessions.bearer(issued.token), undefined);

  account = { ...account, id: 'discord:9', inGuild: false };
  const outsider = await sessions.begin('discord'); await store.whitelist('discord:9');
  const outsiderSession = sessions.browser((await sessions.complete('discord', outsider.token, new URL('https://example.test/'))).session);
  assert.equal((await sessions.profile(outsiderSession)).canProcess, false, 'Discord accounts must be in the server');

  account = { ...account, id: 'discord:1', inGuild: true, guildOwner: true };
  const owner = await sessions.begin('discord');
  const ownerSession = sessions.browser((await sessions.complete('discord', owner.token, new URL('https://example.test/'))).session);
  assert.deepEqual((({ admin, canProcess, whitelisted, limits }) => ({ admin, canProcess, whitelisted, priority: limits.priority }))(await sessions.profile(ownerSession)),
    { admin: true, canProcess: true, whitelisted: true, priority: true }, 'the Discord server owner is an admin with priority');

  // One account cannot fill the session table: its oldest sessions make room.
  for (let index = 0; index < 10; index++) await sessions.complete('discord', (await sessions.begin('discord')).token, new URL('https://example.test/'));
  assert.equal([...sessions.sessions.values()].filter(value => value.owner === 'discord:1').length, 8);

  account = { provider: 'google', id: 'google:root', username: null, name: null, avatar: null, inGuild: null, guildOwner: false };
  const listed = await sessions.begin('google');
  const listedSession = sessions.browser((await sessions.complete('google', listed.token, new URL('https://example.test/'))).session);
  assert.equal((await sessions.profile(listedSession)).admin, true, 'ADMIN_ACCOUNTS grants admin to other providers');
  // A full session table evicts accounts that were not whitelisted, never whitelisted ones.
  for (let index = 0; sessions.sessions.size < 4096; index++) sessions.sessions.set('filler' + index, { owner: 'discord:x', trusted: false, expires: now + 1000 });
  const before = sessions.sessions.size;
  await sessions.complete('discord', (await sessions.begin('discord')).token, new URL('https://example.test/'));
  assert.equal(sessions.sessions.size, before, 'a throwaway session made room');
  for (const [key, value] of sessions.sessions) if (!value.trusted) sessions.sessions.set(key, { ...value, trusted: true });
  await assert.rejects(sessions.complete('google', (await sessions.begin('google')).token, new URL('https://example.test/')), { status: 429 });
  for (const key of [...sessions.sessions.keys()]) if (key.startsWith('filler')) sessions.sessions.delete(key);
  assert.equal((await store.signIns()).length, 17);
  assert.equal((await store.counts()).users, 4);

  now += 8 * 60 * 60 * 1000;
  assert.equal(sessions.browser(sessions.sessions.keys().next().value), undefined);
  await database.destroy();
});

test('configuration validates secrets, providers, admins and URLs', () => {
  assert.throws(() => configuration({}), /MOCAP_KEY_SECRET/);
  assert.throws(() => configuration({ ...base, ADMIN_API_KEY: 'short' }), /ADMIN_API_KEY/);
  assert.throws(() => configuration({ ...base, ADMIN_ACCOUNTS: 'not an account' }), /ADMIN_ACCOUNTS/);
  assert.throws(() => configuration({ ...base, DISCORD_CLIENT_ID: 'only-id' }), /DISCORD_CLIENT_SECRET/);
  assert.throws(() => configuration({ ...base, OIDC_PROVIDERS: 'corp' }), /OIDC_CORP_ISSUER/);
  assert.throws(() => configuration({ ...base, OIDC_PROVIDERS: 'discord', OIDC_DISCORD_ISSUER: 'https://evil.example' }), /other than/);
  assert.throws(() => configuration({ ...base, DEFAULT_MAX_CLIP_SECONDS: '0' }), /DEFAULT_MAX_CLIP_SECONDS/);
  assert.throws(() => configuration({ ...base, DISCORD_GUILD_ID: 'name' }), /DISCORD_GUILD_ID/);
  for (const publicUrl of ['http://example.com', 'https://user:password@example.com', 'https://example.com/path'])
    assert.throws(() => configuration({ ...base, PUBLIC_URL: publicUrl }));
  const config = configuration({ ...base, GOOGLE_CLIENT_ID: 'g', GOOGLE_CLIENT_SECRET: 'gs', MICROSOFT_CLIENT_ID: 'm', MICROSOFT_CLIENT_SECRET: 'ms' });
  assert.deepEqual(config.providers.map(provider => provider.issuer),
    ['https://accounts.google.com', 'https://login.microsoftonline.com/9188040d-6c67-4c5b-b112-36a304b66dad/v2.0']);
});
