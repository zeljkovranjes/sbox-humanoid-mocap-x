import { readFile } from 'node:fs/promises';
import { AuthorizationResponseError, OperationProcessingError, ResponseBodyError } from 'oauth4webapi';
import { HttpError, json } from './error.mjs';
import { captureRoute, proxy, capabilities, jobRequest } from './proxy.mjs';
import { adminRoute, isAdminKey } from './admin.mjs';
import { readJson } from './body.mjs';

const assets = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/polling.js', ['polling.js', 'text/javascript; charset=utf-8']],
  ['/style.css', ['style.css', 'text/css; charset=utf-8']],
]);

// Over HTTPS the __Host- prefix pins cookies to this exact origin: Secure, no Domain, Path=/.
const cookieName = (config, name) => (config.publicUrl.protocol === 'https:' ? '__Host-' : '') + name;

function cookie(request, config, name, pattern = /^[A-Za-z0-9_-]{43}$/) {
  const prefix = cookieName(config, name) + '=';
  const entry = (request.headers.cookie ?? '').split(';').map(part => part.trim()).find(part => part.startsWith(prefix));
  const value = entry?.slice(prefix.length) ?? '';
  return pattern.test(value) ? value : '';
}

// Appends, so one response can clear the login cookie and set the session cookie.
function setCookie(response, config, name, value, seconds) {
  const secure = config.publicUrl.protocol === 'https:' ? '; Secure' : '';
  response.setHeader('set-cookie', [response.getHeader('set-cookie') ?? []].flat()
    .concat(`${cookieName(config, name)}=${value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${seconds}${secure}`));
}

/** HTTP entry point. Sign-in, account and admin decisions live in their own modules; processing is
 * proxied to the C# host. `context` holds { config, sessions, store, providers }. */
export async function application(context) {
  const { config, sessions } = context;
  const files = new Map();
  for (const [path, [name, type]] of assets)
    files.set(path, { bytes: await readFile(new URL('../public/' + name, import.meta.url)), type });
  return async (request, response) => {
    response.setHeader('cache-control', 'no-store');
    response.setHeader('x-content-type-options', 'nosniff');
    response.setHeader('referrer-policy', 'no-referrer');
    if (config.publicUrl.protocol === 'https:') response.setHeader('strict-transport-security', 'max-age=31536000');
    response.setHeader('content-security-policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' https:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    try {
      const url = new URL(request.url, config.publicUrl);
      const path = url.pathname; const method = request.method;
      if (url.origin !== config.publicUrl.origin) throw new HttpError(400, 'invalid_url', 'Invalid request origin.');
      if (method === 'GET' && files.has(path)) {
        const file = files.get(path); response.writeHead(200, { 'content-type': file.type }); response.end(file.bytes); return;
      }
      if (method === 'GET' && path === '/health') { json(response, 200, { status: 'ok' }); return; }
      if (method === 'GET' && path === '/auth/providers') {
        json(response, 200, { providers: [...context.providers.values()].map(({ name, label }) => ({ name, label })) }); return;
      }
      const login = /^\/auth\/login\/([a-z][a-z0-9]{0,15})$/.exec(path);
      if (method === 'GET' && (login || path === '/auth/editor')) {
        const editor = login ? null : { port: Number(url.searchParams.get('port')), state: url.searchParams.get('state'),
          challenge: url.searchParams.get('code_challenge') };
        if (editor && url.searchParams.get('code_challenge_method') !== 'S256')
          throw new HttpError(400, 'invalid_editor_login', 'Editor sign-in requires code_challenge_method=S256.');
        const started = await sessions.begin(login ? login[1] : url.searchParams.get('provider') ?? '', editor);
        setCookie(response, config, 'mocap_login', started.token, 300);
        response.writeHead(302, { location: started.url }); response.end(); return;
      }
      const callback = /^\/auth\/callback\/([a-z][a-z0-9]{0,15})$/.exec(path);
      if (method === 'GET' && callback) {
        const done = await sessions.complete(callback[1], cookie(request, config, 'mocap_login', /^[A-Za-z0-9_-]{40,2048}$/), url);
        setCookie(response, config, 'mocap_login', '', 0);
        if (done.editorRedirect) { response.writeHead(302, { location: done.editorRedirect }); response.end(); return; }
        setCookie(response, config, 'mocap_session', done.session, 8 * 60 * 60);
        response.writeHead(303, { location: '/' }); response.end(); return;
      }
      if (path === '/auth/editor/token') {
        if (method !== 'POST') throw new HttpError(405, 'method_not_allowed', 'Use POST.');
        const body = await readJson(request);
        const issued = sessions.exchange(body.code, body.verifier);
        json(response, 200, { ...issued, ...await sessions.profile(sessions.bearer(issued.token)) }); return;
      }

      const authorization = request.headers.authorization;
      const bearer = authorization?.match(/^Bearer ([A-Za-z0-9_-]{43})$/)?.[1];
      const sessionToken = cookie(request, config, 'mocap_session');
      const account = authorization ? sessions.bearer(bearer) : sessions.browser(sessionToken);
      const browserWrite = !authorization && !['GET', 'HEAD'].includes(method);
      if (path === '/auth/me') {
        if (method !== 'GET') throw new HttpError(405, 'method_not_allowed', 'Use GET.');
        json(response, 200, account ? { signedIn: true, ...await sessions.profile(account) } : { signedIn: false }); return;
      }
      if (path === '/auth/logout') {
        if (method !== 'POST') throw new HttpError(405, 'method_not_allowed', 'Use POST.');
        if (browserWrite) sameOrigin(request, config);
        if (bearer) sessions.revokeEditor(bearer);
        else { sessions.logout(sessionToken); setCookie(response, config, 'mocap_session', '', 0); }
        response.writeHead(204); response.end(); return;
      }
      if (path === '/admin' || path.startsWith('/admin/')) {
        const keyed = isAdminKey(config, authorization);
        if (!keyed && !(account && (await sessions.profile(account)).admin))
          throw new HttpError(authorization || account ? 403 : 401, 'admin_required', 'Use the host\'s admin key or an admin account.');
        if (!keyed && browserWrite) sameOrigin(request, config);
        await adminRoute(request, response, url, context); return;
      }
      if (!captureRoute(method, path) || url.search) throw new HttpError(404, 'not_found', 'Route not found.');
      if (!account) throw new HttpError(401, 'invalid_token', 'Sign in first.');
      if (browserWrite) sameOrigin(request, config);
      const profile = await sessions.profile(account);
      if (!profile.canProcess)
        throw new HttpError(403, profile.inGuild === false ? 'not_in_server' : 'not_whitelisted', profile.inGuild === false
          ? 'Join the host\'s Discord server to use processing.' : 'This account is not whitelisted on this host yet.');
      if (path === '/v1/capabilities') { json(response, 200, await capabilities(config, account.owner, profile.limits)); return; }
      if (method === 'POST' && path === '/v1/jobs') {
        await proxy(request, response, config, account.owner, path, { body: jobRequest(await readJson(request), profile.limits) }); return;
      }
      await proxy(request, response, config, account.owner, path, { maximumUploadBytes: profile.limits.maximumUploadBytes });
    } catch (error) {
      if (response.headersSent || response.destroyed) { response.destroy(); return; }
      const known = error instanceof HttpError;
      const invalidInput = error instanceof RangeError;
      const oauthFailure = error instanceof AuthorizationResponseError || error instanceof OperationProcessingError || error instanceof ResponseBodyError;
      if (!known && !oauthFailure && !invalidInput) console.error('Host request failed', { type: error.name, method: request.method, path: request.url.split('?')[0] });
      if (known && error.status === 429) response.setHeader('retry-after', '30');
      // An unread body could otherwise trickle in for the upload timeout; close the connection.
      if (!request.complete) response.setHeader('connection', 'close');
      const status = known ? error.status : invalidInput || oauthFailure ? 400 : 500;
      json(response, status, { error: { code: known ? error.code : invalidInput ? 'invalid_input' : oauthFailure ? 'invalid_oauth_response' : 'internal_error',
        message: known || invalidInput ? error.message : oauthFailure ? 'Sign-in could not be validated. Start sign-in again.' : 'The host could not complete this request.' } });
    }
  };
}

function sameOrigin(request, config) {
  if (request.headers.origin !== config.publicUrl.origin)
    throw new HttpError(403, 'origin_required', 'Browser writes must come from this application.');
}
