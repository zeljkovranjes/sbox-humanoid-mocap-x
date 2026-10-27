import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

test('overlapping removals cannot clear a later capture', async context => {
  const nodes = new Map();
  const element = id => {
    if (!nodes.has(id)) nodes.set(id, { value: '', hidden: false, disabled: false, textContent: '' });
    return nodes.get(id);
  };
  const previousDocument = globalThis.document;
  globalThis.document = { getElementById: element };
  context.after(() => {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  });
  element('video').files = [{ size: 20, name: 'clip.mp4' }];
  element('start').value = '0'; element('end').value = '5';
  element('model').value = 'Latest'; element('interval').value = '10';
  const ids = ['11111111-1111-4111-8111-111111111111', '22222222-2222-4222-8222-222222222222'];
  const deletions = []; let created = 0;
  context.mock.method(globalThis, 'fetch', async (path, options) => {
    if (path === '/auth/me') return Response.json({ signedIn: true, canProcess: true, admin: false, inGuild: true,
      user: { id: 'discord:1', name: 'Person', username: 'person', avatar: null } });
    if (path === '/v1/capabilities') return Response.json({ protocol: 1, processingAvailable: true, maximumUploadBytes: 512 * 1024 * 1024 });
    if (path === '/v1/jobs') return Response.json({ id: ids[created++] }, { status: 201 });
    if (options.method === 'DELETE') {
      const pending = deferred(); deletions.push({ path, ...pending }); return pending.promise;
    }
    if (options.method === 'PUT') return new Response(null, { status: 202 });
    return Response.json({ id: path.split('/').at(-1), state: 'completed', message: 'Done' });
  });
  await import('../public/app.js');
  await setImmediate();
  const submit = () => element('capture').onsubmit({ preventDefault() {} });
  await submit();
  assert.equal(element('download').hidden, false);
  const first = element('cancel').onclick();
  const second = element('cancel').onclick();
  assert.equal(deletions[0].path, `/v1/jobs/${ids[0]}`);
  deletions[0].resolve(new Response(null, { status: 204 }));
  await first;
  await submit();
  assert.equal(created, 2);
  for (const late of deletions.slice(1)) late.resolve(new Response(null, { status: 204 }));
  await second;
  assert.equal(element('job-id').textContent, `Job: ${ids[1]}`);
  assert.equal(element('download').href, `/v1/jobs/${ids[1]}/result`);
  assert.equal(element('download').hidden, false);
  assert.equal(deletions.length, 1);
  const failed = element('cancel').onclick();
  deletions[1].resolve(Response.json({ detail: 'Host temporarily unavailable' }, { status: 503 }));
  await failed;
  assert.equal(element('job-id').textContent, `Job: ${ids[1]}`);
  assert.equal(element('status').textContent, 'Host temporarily unavailable');
  const retried = element('cancel').onclick();
  deletions[2].resolve(new Response(null, { status: 204 }));
  await retried;
  assert.equal(element('job-id').textContent, '');
  assert.equal(element('download').hidden, true);
});
