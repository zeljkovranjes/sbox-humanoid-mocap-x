/** Converts Retry-After delta-seconds or HTTP dates to a bounded polling delay. */
export function retryDelay(value, now = Date.now()) {
  const milliseconds = value == null ? 1000 : /^\d+$/.test(value) ? Number(value) * 1000 : Date.parse(value) - now;
  return Number.isFinite(milliseconds) ? Math.max(1000, Math.min(30_000, milliseconds)) : 1000;
}

/** Polls an existing job only. Creation/upload are never retried here. */
export async function waitForCapture(id, read, report, signal, wait = delay) {
  let failures = 0;
  for (;;) {
    signal.throwIfAborted();
    let response, current;
    try { response = await read(id, signal); current = await response.json(); failures = 0; }
    catch (error) {
      if (signal.aborted || !(error instanceof TypeError || [429, 502, 503, 504].includes(error.status)) || ++failures > 6) throw error;
      report('Waiting for the host to respond…'); await wait(error.delay ?? 2000, signal); continue;
    }
    if (current.id !== id || typeof current.message !== 'string' || current.message.length > 200
      || current.queuePosition != null && (!Number.isInteger(current.queuePosition) || current.queuePosition < 1
        || current.queuePosition > 1_000_000 || current.state !== 'queued')) throw new Error('The host returned an invalid job status.');
    report(describe(current));
    if (current.state === 'completed') return;
    if (!['queued', 'processing'].includes(current.state)) throw new Error(`Capture ${current.state}.`);
    await wait(retryDelay(response.headers.get('retry-after')), signal);
  }
}

/** One status line: message, place in line and estimated times (estimates come from the host). */
export function describe(job) {
  const parts = [job.message];
  if (job.queuePosition) parts.push(`Position ${job.queuePosition}` + (Number.isInteger(job.queueLength) && job.queueLength >= job.queuePosition ? ` of ${job.queueLength}` : ''));
  if (job.state === 'processing' && typeof job.progress === 'number' && job.progress >= 0 && job.progress <= 1) parts.push(`${Math.round(job.progress * 100)}%`);
  if (job.state === 'queued' && Number.isInteger(job.secondsUntilStart)) parts.push(`starts in ${duration(job.secondsUntilStart)}`);
  if (Number.isInteger(job.secondsRemaining) && ['queued', 'processing'].includes(job.state)) parts.push(`about ${duration(job.secondsRemaining)} left`);
  return parts.join(' · ');
}

function duration(seconds) {
  seconds = Math.max(0, seconds);
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.round(seconds / 60);
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

function delay(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(signal.reason); return; }
    const cancelled = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', cancelled); resolve(); }, milliseconds);
    signal.addEventListener('abort', cancelled, { once: true });
  });
}
