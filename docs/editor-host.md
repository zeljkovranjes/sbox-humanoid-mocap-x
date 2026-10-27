# In-editor processing host

The library can serve the capture API from the s&box editor. Jobs call the same
`LocalCapture` and C#/HLSL pipeline as local capture. Install no inference worker,
Python environment or process bridge.

The public provider website handles HTTPS, OAuth, accounts and roles. It sends
signed account requests to this loopback listener. See the [provider guide],
[HTTP API] and [OAuth protocol] for that side of the deployment.

[provider guide]: provider-guide.md
[HTTP API]: hosted-api.md
[OAuth protocol]: hosted-oauth.md

## Requirements

- A running s&box editor with the library and its tensor-inference dependency
  installed. Windows decodes uploads with Media Foundation; Linux needs FFmpeg
  (`ffmpeg` and `ffprobe` on `PATH`).
- A GPU with a reported memory budget. Kernels use s&box graphics APIs on AMD
  and NVIDIA; native NVIDIA acceptance remains to be performed.
- Both FlowHMR releases and all perception/rig weights installed through
  **Mocap adjustments → Processing → Setup → Local models**.
  Downloads require a click. Startup
  verifies the pinned checksums and never downloads missing files.
- Enough disk space for uploaded videos and decoded frames. Each job is capped
  at 5,000 frames at 30 fps and a 16 GiB decoded cache, independently of upload
  size. Decoder dimensions are capped at 8K with checked strides and pixel area.

On Linux the FFmpeg decode of the five-second validation clip is byte-identical
to the predecoded frames the GPU checks use.

## Configure and start

Set environment variables before launching the editor. `.env.example` lists
them; the library does not load `.env` files. A PowerShell example:

```powershell
$env:MOCAP_KEY_SECRET = '<same random secret as the provider website>'
$env:MOCAP_PORT = '5080'
$env:MOCAP_MAX_CLIP_SECONDS = '60'
$env:MOCAP_MAX_UPLOAD_MB = '512'
$env:MOCAP_CONCURRENT_JOBS = '1'
# Launch the editor from this shell so it inherits these variables.
```

Open the library's project. In the editor console, run:

```console
mocap_x_host_start
```

The command verifies Base and Latest weights before opening the listener. It
prints the loopback address when ready. Keep that editor open for processing.
Point the example website's `PROCESSING_URL` at `http://127.0.0.1:5080` and give
the website the same `MOCAP_KEY_SECRET`. Keep this secret on the provider's
machines; clients receive their own website session tokens.

If Windows reports a URL reservation access error, have the administrator grant
the exact `http://127.0.0.1:<port>/` URL to the editor account. The listener does
not accept wildcard or public-network prefixes. Put the website/reverse proxy
on the public HTTPS address; do not expose the processing listener directly.

| Variable | Default | Accepted values |
| --- | --- | --- |
| `MOCAP_KEY_SECRET` | unset | 32–256 printable non-space characters shared with the website |
| `MOCAP_API_KEYS` | unset | Optional JSON map of owner names to random 32–256 character keys; owner names cannot contain `:` |
| `MOCAP_PORT` | `5080` | 1–65535, loopback HTTP only |
| `MOCAP_MAX_UPLOAD_MB` | `512` | 1–65536 MiB per upload |
| `MOCAP_MAX_CLIP_SECONDS` | `5000 / 30` | 1–166.66666666666666 seconds; per-role limits may lower this |
| `MOCAP_MAX_JOBS_PER_USER` | `2` | 1–16 unfinished jobs per account |
| `MOCAP_CONCURRENT_JOBS` | `1` | 1–4; saved admin settings override the environment on later starts |
| `MOCAP_LANES` | `0` | Auto (`0`) or 1–6 frames at once within each capture |
| `MOCAP_CACHE_DIRECTORY` | OS local application data + `HumanoidMocapX` | Root of the existing editor cache |
| `MOCAP_MODELS_DIRECTORY` | `<cache>/models` | Installed model directory |
| `MOCAP_DATA_DIRECTORY` | `<cache>/host` | Private job files and saved concurrency settings |

At least one signing secret or owner-map key is required. Signed provider
accounts may carry trusted role limits and priority. Plain owner-map keys cannot
set their own priority or limits. Only the derived admin key changes
`/v1/settings`; ordinary account keys cannot reach it.

## Queue and memory admission

After upload, the host inspects the video's actual duration and caps the selected
end at EOF. An unspecified end also stops at the account's configured limit.
The resulting range drives both processing and ETA estimates.

Priority jobs wait before normal jobs; each group follows upload-completion
order. A job starts only below the configured concurrency cap and when the
memory check admits it. Increasing concurrency does not force extra jobs onto
a full GPU. The check reserves each active job's estimated peak, including
work that has not allocated its tensors yet.

Auto reserves for up to six frames at once within each capture. Actual processing
still caps the lane count to the reported GPU memory budget. A higher per-capture
reservation can admit fewer simultaneous jobs. Set `MOCAP_LANES=4` to retain
the four-lane bound.

Reservations use conservative bounds for the shipped weights and buffer shapes:
Omega grows with sampled camera frames, Flow with total frames, and SAM with
the selected lane count. The editor samples public GPU memory counters once
per second. Stale readings and temporary memory pressure leave jobs queued and
cancelable. Jobs whose settings exceed physical memory or engine buffer limits
become `failed` with a readable reason. The host never silently reduces camera
sampling or changes models to fit a job.

These bounds include headroom, but another application can allocate memory
after a reading. Use a dedicated provider editor/GPU and test the intended clip
lengths, resolutions and concurrency before admitting users.

## Progress, failures and shutdown

Processing reports an overall fraction based on completed work: person tracking
0–10%, camera reconstruction 10–20%, body/hands 20–97%, then motion generation
and saving 97–100%. A loading stage can hold its last fraction while work is in
progress. ETAs are estimates learned from completed jobs, not deadlines.

The [HTTP API] defines the request/response bodies and status codes. In
particular, upload success is `202`, oversized requests are `413`, unsupported
content encoding is `415`, queue limits are `429`, and cancelling an already
processing job is `409`. A GPU admission failure appears in the job's `failed`
state after upload; polling that job still returns `200`. Storage IO failures
return `500` and are logged in the provider editor without exposing local paths
in the HTTP error. Downloaded motion also omits provider source-video paths.

If temporary-file deletion fails after processing, the job becomes `failed` and
the provider log records the exception. The queue releases that job's processing
slot and continues with other jobs. Cancelling/deleting the failed job retries
cleanup; its normal expiry or host shutdown also attempts removal. A filesystem
error can therefore leave files on disk until cleanup succeeds.

Stop explicitly before editing or updating the library:

```console
mocap_x_host_stop
```

Stop cancels startup, uploads and active processing, waits for GPU work to finish
unwinding, then releases the listener and deletes this run's temporary files and
results. Operator shutdown is distinct from a user's job cancellation request.
Repeated stop calls wait for the same shutdown. Hotload requests a stop and
requires an explicit restart; it does not resume inference in changed code.

While running, completed results expire after 24 hours. A restart invalidates
old job IDs and removes abandoned run directories that have the host's marker
and no live lock. Saved concurrency settings remain. A crash or powered-off
machine cannot perform timed deletion; cleanup runs at the next start.

## Validation

The engine-independent suite covers real loopback HTTP, owner isolation, role
limits, priority/concurrency, memory reservations, uploaded-range inspection,
motion validation, provider-path removal, storage errors and draining shutdown.
Injected cleanup failures check that failed uploads release ownership, later
captures still run, and an unexpected worker/reporting failure cancels and drains
the remaining workers instead of leaving a stalled dispatcher.
The editor assembly also builds without warnings.

Native Windows uploaded H.264 → GPU → downloaded motion, NVIDIA hardware, and
live host hotload shutdown still require acceptance tests. The existing AMD
five-second GPU fixture uses predecoded frames and is documented separately in
[video validation].

[video validation]: video-validation.md
