import * as oauth from 'oauth4webapi';
import { HttpError } from './error.mjs';

const timeout = () => AbortSignal.timeout(15_000);

/** Sign-in providers built on oauth4webapi: Discord (with server membership and ownership) and
 * any OpenID Connect provider such as Google or Microsoft. Each returns an account with
 * `id` "<provider>:<subject>"; username, name and avatar are null when not provided. */
export async function providers(config, fetcher = fetch) {
  const enabled = new Map();
  for (const settings of config.providers) {
    const provider = settings.name === 'discord'
      ? new DiscordProvider(settings, config, fetcher)
      : await OidcProvider.discover(settings, config, fetcher);
    enabled.set(settings.name, provider);
  }
  return enabled;
}

class Provider {
  constructor(settings, config, fetcher) {
    this.name = settings.name; this.label = settings.label; this.fetcher = fetcher;
    this.client = { client_id: settings.clientId };
    this.clientAuth = oauth.ClientSecretPost(settings.clientSecret);
    this.callback = new URL(`/auth/callback/${settings.name}`, config.publicUrl).href;
  }

  /** Authorization URL with state and S256 PKCE; the caller keeps the returned secrets. */
  async authorize() {
    const state = oauth.generateRandomState();
    const verifier = oauth.generateRandomCodeVerifier();
    const url = new URL(this.issuer.authorization_endpoint);
    url.search = new URLSearchParams({ client_id: this.client.client_id, redirect_uri: this.callback, response_type: 'code',
      scope: this.scopes.join(' '), state, code_challenge_method: 'S256', code_challenge: await oauth.calculatePKCECodeChallenge(verifier) }).toString();
    return { url: url.href, state, verifier };
  }

  async exchange(url, pending) {
    const parameters = oauth.validateAuthResponse(this.issuer, this.client, url, pending.state);
    const response = await oauth.authorizationCodeGrantRequest(this.issuer, this.client, this.clientAuth, parameters, this.callback,
      pending.verifier, { [oauth.customFetch]: this.fetcher, signal: timeout() });
    return oauth.processAuthorizationCodeResponse(this.issuer, this.client, response, this.responseOptions);
  }

  async get(accessToken, url, maximumBytes) {
    const response = await oauth.protectedResourceRequest(accessToken, 'GET', new URL(url), undefined, undefined,
      { [oauth.customFetch]: this.fetcher, signal: timeout() });
    if (!response.ok) throw new HttpError(502, 'identity_unavailable', `${this.label} could not return the signed-in account.`);
    const chunks = []; let size = 0;
    for await (const chunk of response.body) {
      size += chunk.length;
      if (size > maximumBytes) throw new HttpError(502, 'invalid_identity', `${this.label} returned an oversized response.`);
      chunks.push(chunk);
    }
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  }
}

/** Discord: identify for the profile, guilds for membership and ownership of DISCORD_GUILD_ID,
 * and guilds.members.read for the member's Discord roles, which roles can map to limits. */
class DiscordProvider extends Provider {
  constructor(settings, config, fetcher) {
    super(settings, config, fetcher);
    this.guildId = config.guildId;
    this.scopes = config.guildId ? ['identify', 'guilds', 'guilds.members.read'] : ['identify', 'guilds'];
    this.issuer = { issuer: 'https://discord.com', authorization_endpoint: 'https://discord.com/api/oauth2/authorize',
      token_endpoint: 'https://discord.com/api/oauth2/token', authorization_response_iss_parameter_supported: true };
  }

  async identify(url, pending) {
    const tokens = await this.exchange(url, pending);
    const granted = tokens.scope?.split(' ') ?? [];
    if (!['identify', 'guilds'].every(scope => granted.includes(scope)))
      throw new HttpError(403, 'missing_scope', 'Discord did not grant the identify and guilds scopes.');
    const user = discordProfile(await this.get(tokens.access_token, 'https://discord.com/api/v10/users/@me', 16_384));
    if (!this.guildId) return { ...user, inGuild: null, guildOwner: false, discordRoles: [] };
    // 200 guilds per page is Discord's per-user membership maximum, so one page is complete.
    const guilds = await this.get(tokens.access_token, 'https://discord.com/api/v10/users/@me/guilds?limit=200', 1_048_576);
    if (!Array.isArray(guilds)) throw new HttpError(502, 'invalid_identity', 'Discord returned an invalid server list.');
    const guild = guilds.find(entry => entry?.id === this.guildId);
    const discordRoles = guild && granted.includes('guilds.members.read') ? await this.memberRoles(tokens.access_token) : [];
    return { ...user, inGuild: Boolean(guild), guildOwner: guild?.owner === true, discordRoles };
  }

  // Roles only raise limits, so a Discord hiccup here signs the member in with the default role.
  async memberRoles(accessToken) {
    try {
      const member = await this.get(accessToken, `https://discord.com/api/v10/users/@me/guilds/${this.guildId}/member`, 262_144);
      return Array.isArray(member?.roles) ? member.roles.filter(id => typeof id === 'string' && /^\d{1,20}$/.test(id)).slice(0, 250) : [];
    } catch (error) {
      console.warn('Discord member roles unavailable; using the default role', { type: error.name });
      return [];
    }
  }
}

/** Any OpenID Connect provider found through its issuer's discovery document. */
class OidcProvider extends Provider {
  static async discover(settings, config, fetcher) {
    const provider = new OidcProvider(settings, config, fetcher);
    const issuer = new URL(settings.issuer);
    const response = await oauth.discoveryRequest(issuer, { [oauth.customFetch]: fetcher, signal: timeout() });
    provider.issuer = await oauth.processDiscoveryResponse(issuer, response);
    return provider;
  }

  constructor(settings, config, fetcher) {
    super(settings, config, fetcher);
    this.scopes = ['openid', 'profile']; this.responseOptions = { expectedNonce: oauth.expectNoNonce };
  }

  async identify(url, pending) {
    const tokens = await this.exchange(url, pending);
    const claims = oauth.getValidatedIdTokenClaims(tokens);
    let details = claims;
    if (this.issuer.userinfo_endpoint && (claims.name === undefined || claims.picture === undefined)) {
      const userinfo = await this.get(tokens.access_token, this.issuer.userinfo_endpoint, 65_536);
      if (userinfo?.sub === claims.sub) details = { ...userinfo, ...claims };
    }
    return { ...oidcProfile(this.name, claims.sub, details), inGuild: null, guildOwner: false };
  }
}

const text = (value, maximum) => typeof value === 'string' && value.length > 0 && value.length <= maximum ? value : null;

/** Validated Discord profile fields; rejects malformed identities. */
export function discordProfile(user) {
  if (typeof user?.id !== 'string' || !/^\d{1,20}$/.test(user.id) || typeof user.username !== 'string' || !user.username.length || user.username.length > 80
      || (user.global_name != null && (typeof user.global_name !== 'string' || user.global_name.length > 128))
      || (user.avatar != null && (typeof user.avatar !== 'string' || !/^(a_)?[a-f0-9]{32}$/.test(user.avatar))))
    throw new HttpError(502, 'invalid_identity', 'Discord returned an invalid profile.');
  const avatar = user.avatar
    ? `https://cdn.discordapp.com/avatars/${user.id}/${user.avatar}.webp?size=128`
    : `https://cdn.discordapp.com/embed/avatars/${Number((BigInt(user.id) >> 22n) % 6n)}.png`;
  return { provider: 'discord', id: `discord:${user.id}`, username: user.username, name: user.global_name || user.username, avatar };
}

/** Validated OpenID Connect fields. Only the subject is required; others become null. */
export function oidcProfile(provider, subject, claims) {
  if (typeof subject !== 'string' || !/^[A-Za-z0-9._-]{1,128}$/.test(subject))
    throw new HttpError(502, 'invalid_identity', 'The provider returned an unusable account ID.');
  return { provider, id: `${provider}:${subject}`, username: text(claims.preferred_username, 80) ?? text(claims.email, 80),
    name: text(claims.name, 128), avatar: avatarUrl(text(claims.picture, 500)) };
}

// Editors download avatars, so only known image CDNs pass; anything else shows as not provided.
const avatarHosts = /^(cdn\.discordapp\.com|[a-z0-9-]+\.googleusercontent\.com)$/;
function avatarUrl(value) {
  if (!value) return null;
  try { const url = new URL(value); return url.protocol === 'https:' && !url.port && avatarHosts.test(url.hostname) ? url.href : null; }
  catch { return null; }
}
