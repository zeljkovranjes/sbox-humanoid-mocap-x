import { createCipheriv, createDecipheriv, createHash, hkdfSync, randomBytes } from 'node:crypto';
import { HttpError } from './error.mjs';
import { effectivePolicy } from './roles.mjs';

const sessionMilliseconds = 8 * 60 * 60 * 1000;
const loginMilliseconds = 5 * 60 * 1000;
const codeMilliseconds = 2 * 60 * 1000;
// Per account, the oldest entry makes room for a new one; the totals bound memory.
const sessionsPerAccount = 8, codesPerAccount = 4;
const maximumSessions = 4096, maximumCodes = 1024;
const digest = value => createHash('sha256').update(value).digest('hex');
const random = () => randomBytes(32).toString('base64url');
const challengeOf = verifier => createHash('sha256').update(verifier).digest('base64url');

/** Bounded, process-local credentials: browser sessions and editor tokens. Pending logins live in
 * an encrypted cookie, so anonymous visitors hold no server memory. Restart signs everyone out.
 * Access, roles and limits are re-checked on every use, so admin changes apply immediately;
 * Discord membership, ownership and roles are as of sign-in. */
export class Sessions {
  /** `providers` maps provider names to sign-in adapters (see identity.mjs). */
  constructor(config, providers, store, clock = Date.now) {
    this.config = config; this.providers = providers; this.store = store; this.clock = clock;
    this.sessions = new Map(); this.codes = new Map(); this.tokens = new Map();
    this.loginKey = Buffer.from(hkdfSync('sha256', config.keySecret, '', 'mocap-login-cookie', 32));
  }

  // At most once a second: every request calls this, and each pass walks every map.
  prune() {
    const now = this.clock();
    if (now - (this.pruned ?? -Infinity) < 1000) return;
    this.pruned = now;
    for (const entries of [this.sessions, this.codes, this.tokens])
      for (const [key, value] of entries) if (value.expires <= now) entries.delete(key);
  }

  /** Starts sign-in with `provider`. `editor` ({ port, state, challenge }) makes the callback
   * hand a one-time code to the editor's loopback listener instead of opening a browser session.
   * Returns the provider URL and the sealed login cookie value. */
  async begin(provider, editor = null) {
    const adapter = this.providers.get(provider);
    if (!adapter) throw new HttpError(404, 'unknown_provider', 'This host does not offer that sign-in provider.');
    if (editor) validateEditor(editor);
    const { url, state, verifier } = await adapter.authorize();
    return { token: this.seal({ provider, state, verifier, editor, expires: this.clock() + loginMilliseconds }), url };
  }

  /** Finishes sign-in and records it. Returns a browser session token, or the loopback URL
   * that delivers the editor's one-time code. */
  async complete(provider, token, url) {
    this.prune();
    const pending = this.open(token);
    if (!pending || pending.provider !== provider || pending.expires <= this.clock())
      throw new HttpError(400, 'expired_login', 'This login expired or was already used. Start again.');
    const user = await this.providers.get(provider).identify(url, pending);
    const admin = user.guildOwner || this.config.admins.has(user.id);
    const access = { inGuild: user.inGuild, guildOwner: user.guildOwner, whitelisted: admin || await this.store.isWhitelisted(user.id) };
    await this.store.recordSignIn(user, access, pending.editor ? 'editor' : 'browser');
    const account = { owner: user.id, user: { id: user.id, provider: user.provider, username: user.username, name: user.name, avatar: user.avatar },
      inGuild: user.inGuild, guildOwner: user.guildOwner, discordRoles: user.discordRoles ?? [], admin, trusted: access.whitelisted };
    if (pending.editor) {
      admit(this.codes, account.owner, codesPerAccount, maximumCodes, 'Too many pending editor logins. Try again shortly.');
      const code = random();
      this.codes.set(digest(code), { ...account, challenge: pending.editor.challenge, expires: this.clock() + codeMilliseconds });
      const callback = new URL(`http://127.0.0.1:${pending.editor.port}/callback`);
      callback.search = new URLSearchParams({ code, state: pending.editor.state }).toString();
      return { editorRedirect: callback.href };
    }
    admit(this.sessions, account.owner, sessionsPerAccount, maximumSessions, 'The host session limit has been reached.');
    const sessionToken = random();
    this.sessions.set(digest(sessionToken), { ...account, expires: this.clock() + sessionMilliseconds });
    return { session: sessionToken };
  }

  /** Exchanges an editor's one-time code and PKCE verifier for its bearer token. Signing in
   * again replaces that account's previous editor token. */
  exchange(code, verifier) {
    this.prune();
    const key = digest(String(code)); const pending = this.codes.get(key);
    this.codes.delete(key);
    if (!pending || typeof verifier !== 'string' || !/^[A-Za-z0-9._~-]{43,128}$/.test(verifier) || challengeOf(verifier) !== pending.challenge)
      throw new HttpError(400, 'invalid_grant', 'This sign-in code is invalid, expired or already used. Sign in again.');
    for (const [existing, value] of this.tokens) if (value.owner === pending.owner) this.tokens.delete(existing);
    admit(this.tokens, pending.owner, 1, maximumSessions, 'The host session limit has been reached.');
    const token = random(); const expires = this.clock() + sessionMilliseconds;
    const { challenge, ...account } = pending;
    this.tokens.set(digest(token), { ...account, expires });
    return { token, expiresAt: new Date(expires).toISOString() };
  }

  browser(token) { return this.live(this.sessions, token); }
  bearer(token) { return this.live(this.tokens, token); }

  live(entries, token) {
    this.prune();
    const value = token ? entries.get(digest(token)) : undefined;
    return value && value.expires > this.clock() ? value : undefined;
  }

  /** The account's profile, current access, roles and limits; whitelist and roles are read live.
   * Server membership is required only for Discord accounts when the host names a server. */
  async profile(account) {
    const access = await this.store.access(account.user.id);
    const whitelisted = account.admin || access.whitelisted;
    const policy = effectivePolicy(await this.store.roles(),
      { assigned: access.role, discordRoles: account.discordRoles, guildOwner: account.guildOwner }, this.config);
    return { user: account.user, inGuild: account.inGuild, guildOwner: account.guildOwner, whitelisted,
      canProcess: account.inGuild !== false && whitelisted, admin: account.admin, roles: policy.roles,
      limits: { priority: policy.priority, maximumClipSeconds: policy.maximumClipSeconds, maximumUploadBytes: policy.maximumUploadBytes } };
  }

  logout(token) { this.sessions.delete(digest(token)); }
  revokeEditor(token) { this.tokens.delete(digest(token)); }

  /** Signs a user out of every browser session and editor, including codes not yet exchanged;
   * returns how many were revoked. */
  revokeUser(id) {
    let count = 0;
    for (const entries of [this.sessions, this.tokens, this.codes])
      for (const [key, value] of entries) if (value.user.id === id) { entries.delete(key); count++; }
    return count;
  }

  // AES-256-GCM with a key derived from MOCAP_KEY_SECRET: the browser holds the pending login but
  // cannot read or change it. The provider's single-use code stops a replayed callback.
  seal(value) {
    const iv = randomBytes(12); const cipher = createCipheriv('aes-256-gcm', this.loginKey, iv);
    const data = Buffer.concat([cipher.update(JSON.stringify(value), 'utf8'), cipher.final()]);
    return Buffer.concat([iv, cipher.getAuthTag(), data]).toString('base64url');
  }

  open(token) {
    if (typeof token !== 'string' || !/^[A-Za-z0-9_-]{40,2048}$/.test(token)) return null;
    try {
      const bytes = Buffer.from(token, 'base64url');
      const decipher = createDecipheriv('aes-256-gcm', this.loginKey, bytes.subarray(0, 12));
      decipher.setAuthTag(bytes.subarray(12, 28));
      return JSON.parse(Buffer.concat([decipher.update(bytes.subarray(28)), decipher.final()]).toString('utf8'));
    } catch { return null; }
  }
}

// Makes room for one more entry: evicts the account's oldest beyond its share; when the map is
// full, evicts the oldest entry of an account that was not whitelisted at sign-in, so throwaway
// accounts cannot lock out whitelisted ones. Refuses only when every entry is trusted.
function admit(entries, owner, perAccount, maximum, message) {
  const own = [...entries].filter(([, value]) => value.owner === owner).sort((a, b) => a[1].expires - b[1].expires);
  for (const [key] of own.slice(0, Math.max(0, own.length - perAccount + 1))) entries.delete(key);
  if (entries.size < maximum) return;
  let oldest;
  for (const entry of entries) if (!entry[1].trusted && (!oldest || entry[1].expires < oldest[1].expires)) oldest = entry;
  if (!oldest) throw new HttpError(429, 'login_capacity', message);
  entries.delete(oldest[0]);
}

function validateEditor({ port, state, challenge }) {
  if (!Number.isInteger(port) || port < 1024 || port > 65535 || typeof state !== 'string' || !/^[A-Za-z0-9_-]{16,128}$/.test(state)
      || typeof challenge !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(challenge))
    throw new HttpError(400, 'invalid_editor_login', 'Editor sign-in needs a loopback port, a state and an S256 code challenge.');
}
