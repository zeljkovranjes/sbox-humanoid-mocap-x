import test from 'node:test';
import assert from 'node:assert/strict';
import { describe, retryDelay, waitForCapture } from '../public/polling.js';

test('Retry-After accepts seconds and dates, with finite bounded fallbacks', () => {
  const now = Date.parse('2026-10-27T12:00:00Z');
  assert.equal(retryDelay(null, now), 1000); assert.equal(retryDelay('0', now), 1000);
  assert.equal(retryDelay('7', now), 7000); assert.equal(retryDelay('9999', now), 30_000);
  assert.equal(retryDelay('Tue, 27 Oct 2026 12:00:07 GMT', now), 7000);
  assert.equal(retryDelay('Tue, 27 Oct 2026 11:59:00 GMT', now), 1000);
  assert.equal(retryDelay('invalid', now), 1000);
});

const id = 'capture';
const response = (state, queuePosition, retry = '7') => Response.json({ id, state, queuePosition, message: state }, { headers: { 'retry-after': retry } });
const failure = status => Object.assign(new Error('fixture host failure'), { status, delay: 3000 });

test('status polling retries transport failures, displays queue positions, and resets its retry count', async () => {
  const sequence = [new TypeError('offline'), failure(503), response('queued', 2),
    ...Array.from({ length: 6 }, () => failure(429)), response('processing', null, new Date(Date.now() + 120_000).toUTCString()), response('completed')];
  const delays = []; const messages = []; const controller = new AbortController();
  await waitForCapture(id, async () => { const item = sequence.shift(); if (item instanceof Error) throw item; return item; },
    message => messages.push(message), controller.signal, async delay => delays.push(delay));
  assert.equal(sequence.length, 0); assert.deepEqual(delays, [2000, 3000, 7000, ...Array(6).fill(3000), 30_000]);
  assert(messages.includes('queued · Position 2'));
});

test('status lines show place in line, progress and estimated times', () => {
  assert.equal(describe({ state: 'queued', message: 'Waiting', queuePosition: 2, queueLength: 5, secondsUntilStart: 300, secondsRemaining: 640 }),
    'Waiting · Position 2 of 5 · starts in 5 min · about 11 min left');
  assert.equal(describe({ state: 'processing', message: 'Reconstructing', progress: 0.42, secondsRemaining: 45 }),
    'Reconstructing · 42% · about 45 s left');
  assert.equal(describe({ state: 'completed', message: 'Done', secondsRemaining: 5 }), 'Done');
});

test('status polling bounds failures and rejects malformed states without retrying', async () => {
  const controller = new AbortController(); let attempts = 0; let waits = 0;
  await assert.rejects(waitForCapture(id, async () => { attempts++; throw failure(503); }, () => {}, controller.signal,
    async () => { waits++; }), /fixture host failure/);
  assert.equal(attempts, 7); assert.equal(waits, 6);
  await assert.rejects(waitForCapture(id, async () => { throw failure(401); }, () => {}, controller.signal, async () => { throw new Error('unexpected retry'); }), /fixture host failure/);
  for (const item of [response('queued', 0), response('completed', 1), response('failed'), Response.json({ id: 'wrong', state: 'completed', message: 'Done' })])
    await assert.rejects(waitForCapture(id, async () => item, () => {}, controller.signal), /invalid job status|Capture failed/);
  const pending = waitForCapture(id, async () => response('queued', 1), () => {}, controller.signal);
  controller.abort(); await assert.rejects(pending, { name: 'AbortError' });
});

test('status-body transport failures retry within the same bounded budget', async () => {
  const controller = new AbortController(); const delays = []; let attempts = 0;
  const interrupted = { json: async () => { throw new TypeError('connection closed during status body'); } };
  await waitForCapture(id, async () => ++attempts === 1 ? interrupted : response('completed'), () => {},
    controller.signal, async delay => delays.push(delay));
  assert.equal(attempts, 2); assert.deepEqual(delays, [2000]);

  attempts = 0; delays.length = 0;
  await assert.rejects(waitForCapture(id, async () => { attempts++; return interrupted; }, () => {},
    controller.signal, async delay => delays.push(delay)), /connection closed during status body/);
  assert.equal(attempts, 7); assert.equal(delays.length, 6);

  await assert.rejects(waitForCapture(id, async () => new Response('{invalid'), () => {}, controller.signal,
    async () => assert.fail('Malformed JSON must not be retried')), SyntaxError);
});
