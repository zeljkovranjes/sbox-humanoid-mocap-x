import { retryDelay, waitForCapture } from './polling.js';

const element = id => document.getElementById(id);
let job = null;
let active = null;
let removing = null;
let ready = false;
let maximumUpload = 0;
let maximumClip = Infinity;
const status = message => { element('status').textContent = message; };

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: 'same-origin', ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const error = new Error(body?.error?.message ?? body?.detail ?? `Host returned HTTP ${response.status}.`);
    error.status = response.status;
    error.delay = retryDelay(response.headers.get('retry-after'));
    throw error;
  }
  return response;
}

const placeholder = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><circle cx="24" cy="24" r="24" fill="#2b3a2f"/></svg>');

async function initialize() {
  const me = await (await api('/auth/me')).json();
  element('account').hidden = !me.signedIn;
  if (!me.signedIn) {
    const { providers } = await (await api('/auth/providers')).json();
    for (const provider of providers) {
      const link = document.createElement('a');
      link.className = 'button'; link.href = `/auth/login/${encodeURIComponent(provider.name)}`; link.textContent = `Sign in with ${provider.label}`;
      element('providers').append(link);
    }
    status(providers.length ? 'Sign in to use this processing host.' : 'The operator has not configured a sign-in provider.');
    return;
  }
  const user = me.user;
  element('avatar').src = user.avatar ?? placeholder;
  element('name').textContent = user.name ?? user.username ?? 'Name not provided';
  element('account-id').textContent = user.id;
  element('access').textContent = (me.admin ? 'Admin · full access' : me.canProcess ? 'Whitelisted'
    : me.inGuild === false ? 'Join the host\'s Discord server to process videos.' : 'Not whitelisted yet. Ask the host to add your account ID.')
    + (me.canProcess && me.limits?.priority ? ' · Priority queue' : '');
  if (!me.canProcess) { status('Signed in. Processing is not enabled for this account.'); return; }
  const capabilities = await (await api('/v1/capabilities')).json();
  ready = capabilities.protocol === 1 && capabilities.processingAvailable;
  maximumUpload = Number.isSafeInteger(capabilities.maximumUploadBytes) ? capabilities.maximumUploadBytes : 0;
  maximumClip = Number.isFinite(capabilities.maximumClipSeconds) ? capabilities.maximumClipSeconds : Infinity;
  element('capture-fields').disabled = !ready;
  status(ready ? 'Ready for third-person video.' : 'The operator has not connected a working video processor.');
}

// A job that is already processing cannot be removed, so signing out never waits on it.
element('logout').onclick = async () => {
  try { if (job) await remove(); } catch { /* the job expires on the host */ }
  try { await api('/auth/logout', { method: 'POST' }); location.reload(); }
  catch (error) { status(error.message); }
};

element('capture').onsubmit = async event => {
  event.preventDefault(); if (active) return;
  const video = element('video').files[0];
  const start = Number(element('start').value); const end = Number(element('end').value);
  if (!video || video.size < 1 || video.size > maximumUpload || !Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    status(`Select a nonempty video up to ${Math.floor(maximumUpload / 1048576)} MiB and an end time after the start.`); return;
  }
  if (end - start > maximumClip + 0.01) { status(`Clips are limited to ${maximumClip} seconds for your account. Shorten the range.`); return; }
  if (job) { status('Download the previous result and remove its job before starting another.'); return; }
  active = new AbortController(); const signal = active.signal;
  element('capture-fields').disabled = true; element('download').hidden = true;
  try {
    const created = await (await api('/v1/jobs', { method: 'POST', headers: { 'content-type': 'application/json' }, signal,
      body: JSON.stringify({ model: element('model').value, cameraInterval: Number(element('interval').value), start, end, name: video.name.slice(0, 120) }) })).json();
    job = created.id; element('job-id').textContent = 'Job: ' + job; element('cancel').disabled = false;
    status('Uploading video…');
    await api(`/v1/jobs/${job}/video`, { method: 'PUT', body: video, signal,
      headers: { 'content-type': 'application/octet-stream', 'x-video-extension': '.' + video.name.split('.').at(-1).toLowerCase() } });
    await waitForCapture(job, (id, signal) => api(`/v1/jobs/${id}`, { signal }), status, signal);
    element('download').href = `/v1/jobs/${job}/result`; element('download').hidden = false;
  } catch (error) { if (!signal.aborted) status(error.message + (job ? ' You can remove this job or ask the host about its ID.' : '')); }
  finally { active = null; element('capture-fields').disabled = !ready; }
};

function remove() {
  return removing ??= removeJob().finally(() => { removing = null; });
}

async function removeJob() {
  active?.abort();
  if (job) {
    try { await api(`/v1/jobs/${job}`, { method: 'DELETE' }); }
    catch (error) { if (error.status !== 404) throw error; }
  }
  job = null; element('job-id').textContent = ''; element('cancel').disabled = true; element('download').hidden = true; status('Job removed.');
}
element('cancel').onclick = () => remove().catch(error => status(error.message));
initialize().catch(error => status(error.message));
