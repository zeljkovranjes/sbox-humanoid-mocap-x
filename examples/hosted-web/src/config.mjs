function endpoint(value, name) {
  const url = new URL(value);
  const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if ((url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) || url.username || url.password || url.search || url.hash)
    throw new Error(`${name} must use HTTPS, or HTTP on loopback, without credentials or query parameters.`);
  return url;
}

const accountPattern = /^[a-z][a-z0-9]{0,15}:[A-Za-z0-9._-]{1,128}$/;
const reservedProviders = ['discord', 'google', 'microsoft', 'admin', 'host'];
// Microsoft's personal-account tenant; its discovery issuer only validates with the tenant ID.
const microsoftConsumers = '9188040d-6c67-4c5b-b112-36a304b66dad';

/** Reads operator configuration; fails at startup for invalid URLs, secrets, IDs or providers. */
export function configuration(env) {
  const publicUrl = endpoint(env.PUBLIC_URL ?? 'http://127.0.0.1:5081', 'PUBLIC_URL');
  if (publicUrl.pathname !== '/') throw new Error('PUBLIC_URL must be an origin without a path prefix.');
  const processingUrl = endpoint(env.PROCESSING_URL ?? 'http://127.0.0.1:5080', 'PROCESSING_URL');
  processingUrl.pathname = processingUrl.pathname.replace(/\/$/, '') + '/';
  const keySecret = env.MOCAP_KEY_SECRET ?? '';
  if (!/^[\x21-\x7e]{32,256}$/.test(keySecret))
    throw new Error('MOCAP_KEY_SECRET must be the processing host\'s 32–256 character printable secret.');
  const adminKey = env.ADMIN_API_KEY || null;
  if (adminKey && !/^[\x21-\x7e]{32,256}$/.test(adminKey)) throw new Error('ADMIN_API_KEY must be 32–256 printable characters.');
  const admins = new Set((env.ADMIN_ACCOUNTS ?? '').split(',').map(entry => entry.trim()).filter(Boolean));
  for (const account of admins) if (!accountPattern.test(account)) throw new Error(`ADMIN_ACCOUNTS entry "${account}" is not <provider>:<account ID>.`);
  const guildId = env.DISCORD_GUILD_ID || null;
  if (guildId && !/^\d{1,20}$/.test(guildId)) throw new Error('DISCORD_GUILD_ID must be a numeric Discord server ID.');
  const uploadMegabytes = Number(env.MAX_UPLOAD_MB ?? 512);
  if (!Number.isInteger(uploadMegabytes) || uploadMegabytes < 1 || uploadMegabytes > 65536)
    throw new Error('MAX_UPLOAD_MB must be 1–65536 and match the processing host\'s MOCAP_MAX_UPLOAD_MB.');
  const defaultClipSeconds = Number(env.DEFAULT_MAX_CLIP_SECONDS ?? 60);
  if (!Number.isInteger(defaultClipSeconds) || defaultClipSeconds < 1 || defaultClipSeconds > 86400)
    throw new Error('DEFAULT_MAX_CLIP_SECONDS must be 1–86400.');
  const port = Number(env.PORT ?? 5081);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT must be 1–65535.');
  return { publicUrl, processingUrl, keySecret, adminKey, admins, guildId, port, host: env.LISTEN_HOST ?? '127.0.0.1',
    maximumUploadBytes: uploadMegabytes * 1024 * 1024, defaultClipSeconds,
    databaseUrl: env.DATABASE_URL ?? 'sqlite:./mocap.db', providers: providerSettings(env) };
}

/** Enabled sign-in providers. Each needs both client credentials; OIDC ones also an issuer. */
function providerSettings(env) {
  const providers = [];
  const add = (name, label, prefix, issuer) => {
    const clientId = env[prefix + '_CLIENT_ID']; const clientSecret = env[prefix + '_CLIENT_SECRET'];
    if (!clientId && !clientSecret) return;
    if (!clientId || !clientSecret) throw new Error(`Set both ${prefix}_CLIENT_ID and ${prefix}_CLIENT_SECRET.`);
    if (issuer !== undefined) endpoint(issuer, prefix + '_ISSUER');
    providers.push({ name, label, clientId, clientSecret, issuer });
  };
  add('discord', 'Discord', 'DISCORD');
  add('google', 'Google', 'GOOGLE', 'https://accounts.google.com');
  const tenant = env.MICROSOFT_TENANT || microsoftConsumers;
  if (!/^[0-9a-f-]{36}$/i.test(tenant)) throw new Error('MICROSOFT_TENANT must be a tenant ID (GUID).');
  add('microsoft', 'Microsoft', 'MICROSOFT', `https://login.microsoftonline.com/${tenant}/v2.0`);
  for (const name of (env.OIDC_PROVIDERS ?? '').split(',').map(entry => entry.trim().toLowerCase()).filter(Boolean)) {
    // Built-in names stay reserved even when unused, so an OIDC subject can never pose as a Discord ID.
    if (!/^[a-z][a-z0-9]{0,15}$/.test(name) || reservedProviders.includes(name) || providers.some(provider => provider.name === name))
      throw new Error(`OIDC_PROVIDERS entry "${name}" must be a unique lowercase name other than ${reservedProviders.join(', ')}.`);
    const prefix = 'OIDC_' + name.toUpperCase();
    if (!env[prefix + '_ISSUER']) throw new Error(`Set ${prefix}_ISSUER for the ${name} provider.`);
    add(name, env[prefix + '_LABEL'] || name, prefix, env[prefix + '_ISSUER']);
  }
  return providers;
}

export { accountPattern };
