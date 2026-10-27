Humanoid Mocap X hosting
========================

The s&box library can send videos to a processing host and import the returned
full-body motion. Preview, Human/Citizen/custom-armature retargeting, scale,
foot locking and FBX export stay in the existing Humanoid Mocap workspace.
The [Node web example] adds OAuth sign-in (Discord, Google, Microsoft or any
OpenID Connect provider), a whitelist, roles and an admin API. Users sign in
through the browser; nobody handles API keys. The C# service keeps every
account's jobs separate.

[Node web example]: examples/hosted-web/README.md

## Current state

The client and service protocol are implemented. The service starter rejects
capture requests until its operator registers a working video processor.
The local C#/HLSL model sequence has processed a five-second predecoded clip
and exported Human/Citizen animation. Its Windows decoder integration and
complete motion-quality acceptance are still unverified; Linux local video
capture is unavailable. Neither production path has passed a complete
video-to-animation acceptance test yet.

## Configure the s&box library

Load these local projects while the packages are unpublished:

```text
sbox/libraries/humanoid-mocap/humanoid-mocap.sbproj
sbox/shaders/compute/tensor-inference/tensor-inference.sbproj
sbox/libraries/humanoid-mocap-x/humanoid-mocap-x.sbproj
```

1. Open **View > Humanoid Mocap X**. The workspace opens straight away. Until
   you choose where videos are processed, the drop area reads **Currently
   disabled** and the settings button (top right) has a red dot.
2. Open settings, then **Processing > Setup**. Under **Runs on**, choose
   **This computer** (download the models, about 10 GB, under **Local
   models**) or **A provider**: enter its address, for example
   `mocap.example.com`, and press **Connect** to list its sign-in options.
3. Choose one, such as **Sign in with Discord**. The browser opens the
   provider's sign-in page directly; after signing in, the editor continues
   by itself and the workspace unlocks.
4. With a provider, the bar above the workspace shows your picture, name,
   account ID and status. **Processing > Account** lists everything the
   provider reports: whitelist, Discord server, priority queue, longest clip
   and largest upload ("Name not provided" when the provider does not share
   it; providers share no email address). Sign out or switch accounts there.
5. Drop or open a video and start capture. The status area shows upload,
   place in line, progress and estimated time. **Cancel** cancels the job
   while it waits; a job that is already processing cannot be cancelled.
6. The **Queue** panel lists your jobs on the provider with the same details.
   **Add videos** queues several at once; when the provider allows only one
   job at a time, the rest wait in the panel and upload as jobs finish.
   Finished jobs open into the workspace with the open button.

**Mocap adjustments > Setup** shows where videos are processed, has **Change
setup** to switch between your GPU and a provider, and sets **Base** or
**Latest** and camera sampling at **1**, **10** or **30** frames (Latest and
10 are the defaults).

The entire selected video file is uploaded, together with the requested time
range. The host is responsible for decoding and processing that range.
Returned motion is validated before it enters the workspace. Human remains
the initial target, foot locking is enabled, and FBX uses ModelDoc import
scale 1. Hosted processing does not require users to install a local worker
or download the host's model weights.

The address, provider and processing choice are remembered. The sign-in
lasts up to eight hours and is kept only in memory: it is never written to
editor cookies or project files, so restarting the editor asks you to sign in
again with one click. Sign-in uses a one-time code delivered to a temporary
`127.0.0.1` listener, protected by PKCE ([RFC 8252]). HTTPS is required except
on loopback addresses for local development. API redirects are rejected.

[RFC 8252]: https://www.rfc-editor.org/rfc/rfc8252

## Download local models

In **Mocap adjustments > Setup**, choose **Base** or **Latest**. Expand
**Local models**, then click **Download** beside each required model. The
**FlowHMR · selected model** row downloads the chosen variant. SAM 3D Body
DINOv3 downloads its checkpoint, MHR asset, configuration and license;
VGGT-Omega downloads the 1B-512 checkpoint and license. YOLOX-L supplies
person detection; the separate SMPL-H neutral asset supplies FlowHMR's
skeleton and shape directions. The perception
packages use pinned public mirrors listed in [model sources].

[model sources]: docs/model-sources.md

Only a click starts a transfer. The panel shows progress and **Cancel
download** stops it. Closing the workspace also cancels. Completed files
are retained and verified on reuse; an unfinished file is removed and starts
again on the next click. Files are saved under the operating system's local
application-data directory, in `HumanoidMocapX/models`.
Use **Open folder** to view those files.
Click a section heading to collapse or expand it; each section remembers
its previous state.

On Windows, after choosing **This computer**, open a video. Capture verifies installed
model hashes without contacting a download server. It reuses Humanoid
Mocap's Windows Media Foundation decoder, including H.264, and samples the
displayed frames at 30 fps from their presentation timestamps. Missing or
corrupt models produce an error naming the file and its download control.

Temporary decoded pixels retain their original resolution and are removed
on success, failure or cancellation. Each capture is limited to 5,000 output
frames and 16 GiB of decoded pixels; select a shorter time range if needed.
The cache is under `HumanoidMocapX/captures` in local application data. Saved
`.hmotion` captures remain there for reopening and export.

Windows production capture still needs a live acceptance run. On Linux,
use hosted capture or open saved motion; WebM preview remains available.
The Linux preview API returned stale/incomplete pixels in frame-extraction
tests and is not used for inference. See [GPU validation] for the checks that
have actually passed.

[GPU validation]: docs/gpu-validation.md

## Run the host starter

Hosting for other people? Follow the [provider guide] first; it links back
here for details.

[provider guide]: docs/provider-guide.md

For sign-in, roles and the admin API, run `examples/hosted-web` alongside
this service. It streams the same API to the C# host, preserving its queue
and ownership checks. It depends on `oauth4webapi` and `kysely` and has no
inference subprocess. See [OAuth setup] for providers and credentials, and
the [API reference] for every endpoint, including roles, the priority queue,
concurrent jobs and the admin API for Discord bots.

[OAuth setup]: docs/hosted-oauth.md
[API reference]: docs/hosted-api.md

The separate repository is at `services/humanoid-mocap-host`. It targets
.NET 10.0.403 and references the engine-free shared motion projects in their
workspace locations. Keep those checkouts when building or publishing it.
It has no NuGet package dependencies or local inference subprocess.

```sh
cd services/humanoid-mocap-host
dotnet build HumanoidMocapHost.csproj
dotnet run --project tests/Host.Tests.csproj
dotnet run --project HumanoidMocapHost.csproj
```

Set these process environment variables before starting the server. The
`.env.example` file documents them; the server does not load `.env` itself.

| Variable | Value |
| --- | --- |
| `MOCAP_PORT` | Loopback port; defaults to `5080`. The host never listens on other interfaces |
| `MOCAP_KEY_SECRET` | 32–256 printable characters shared with the website, which derives each account's key from it |
| `MOCAP_API_KEYS` | Optional JSON mapping user names (no `:`) to separate random keys, for clients without the website |
| `MOCAP_DATA_DIRECTORY` | Directory for uploads, results and saved settings; defaults to `./data` |
| `MOCAP_MAX_JOBS_PER_USER` | Unfinished jobs per account, 1–16; defaults to 2 |
| `MOCAP_MAX_UPLOAD_MB` | Largest upload; defaults to 512 |
| `MOCAP_MAX_CLIP_SECONDS` | Longest processed range; defaults to 600 |
| `MOCAP_CONCURRENT_JOBS` | Jobs processed at once, 1–4, on first start; defaults to 1. Admins change it with `PUT /admin/settings`, which is saved and kept across restarts |

Without the website, the key mapping has this shape; replace the placeholder:

```json
{"your-user":"replace-with-a-random-key-of-at-least-32-characters"}
```

Generate each key using a cryptographic random generator. This C# expression
produces a 256-bit key:

```csharp
Convert.ToHexString(System.Security.Cryptography.RandomNumberGenerator.GetBytes(32))
```

Give each user a different owner/key entry. Job IDs alone cannot authorize
uploads, status requests, downloads or cancellation. Keys can be replaced
in the host environment and the service restarted; a restart also loses
in-memory job metadata.

Use the website or a trusted HTTPS reverse proxy in front; the host listens
on loopback only. The starter is for one service instance. It permits 64
unfinished jobs globally, `MOCAP_MAX_JOBS_PER_USER` per owner and
`MOCAP_CONCURRENT_JOBS` processing at once, and keeps each owner's 16 newest
jobs. Priority jobs wait ahead of the others. Uploads, clip length and
results are bounded (`MOCAP_MAX_UPLOAD_MB`, `MOCAP_MAX_CLIP_SECONDS`,
128 MiB); the website can lower the first two per role. Uploads time out
after thirty minutes; processing has a cooperative one-hour cancellation
deadline.

## Connect the processing implementation

Implement [`IClipProcessor`] and replace the `UnconfiguredProcessor`
registration in `src/main.cs`. Set `IsAvailable` to true only after the
processor and its weights are ready. Unconfigured hosts advertise false in
capabilities and return HTTP 503 when asked to create a capture.

[`IClipProcessor`]: ../../../services/humanoid-mocap-host/src/processor.cs

`ProcessAsync` receives validated model/range settings, the original video
extension, a readable video stream, a writable result stream, a progress
callback and cancellation. Streams belong to the host. Report short status
messages, honor cancellation and propagate processing failures.

Write a valid `.hmotion` JSON document. Use the shared `FlowMotion.ToDocument`
adapter when your backend produces FlowHMR's 52-joint SMPL-H motion. The
document must use metres, right-handed X-right/Y-up axes, normalized XYZW
quaternions, parents before children and increasing frame times. It must
contain a Hips bone role. Preserve the contact probabilities so the shared
retargeter can perform foot locking. The host validates the document before
advertising completion; the client validates it again before import.
The [provider-output reference] specifies array shapes, coordinate and
rotation conventions, contact probabilities, finger roles and lifecycle.

[provider-output reference]: docs/provider-output.md

The library's local FlowHMR ports do not yet supply a complete video
processor. The hosting operator must provide detection, camera/body feature
extraction and reconstruction, and arrange the corresponding model assets.
No placeholder in this starter fabricates a successful capture.

## HTTP protocol

All `/v1` endpoints require `Authorization: Bearer <key>`: an account key the
website derives from `MOCAP_KEY_SECRET`, or one from `MOCAP_API_KEYS`.
`/v1/settings` takes only the website's derived admin key. The website adds
sign-in, roles and admin routes on top; the [API reference] lists them all.
Requests and responses use JSON with camel-case properties, except the raw
video upload and motion download. A base URL can include a path prefix if
the reverse proxy routes it consistently.

| Method and path | Request | Response |
| --- | --- | --- |
| `GET /health` | None; no key required | `{"status":"ok"}` |
| `GET /v1/capabilities` | None | Protocol 1, readiness, upload and clip limits, queue length, jobs processing and processed at once |
| `POST /v1/jobs` | Capture settings below | HTTP 201 and an `awaiting-upload` job |
| `PUT /v1/jobs/{id}/video` | Raw bytes; `X-Video-Extension: .mp4` | HTTP 202 |
| `GET /v1/jobs/{id}` | None | Job ID, state, progress message and optional queue position |
| `GET /v1/jobs/{id}/result` | None | Completed `.hmotion` JSON |
| `DELETE /v1/jobs/{id}` | None | HTTP 204; cancels a waiting job or removes a finished one. Processing jobs return 409 |
| `GET` / `PUT /v1/settings` | `{"concurrentJobs":2}` for PUT | `{"concurrentJobs","maximumConcurrentJobs"}`; admin key only |

```json
{"model":"Latest","cameraInterval":10,"start":0,"end":8}
```

`end` can be null to process up to the clip limit. A website also sends a
`policy` with the account's priority and limits; the host ignores it from
`MOCAP_API_KEYS` clients. Supported extensions are
`.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv` and `.webm`. The processor must still
validate video contents; an extension is not evidence of a valid video.

States are `awaiting-upload`, `uploading`, `queued`, `processing`,
`cancelling`, `cancelled`, `completed` and `failed`. The host sends
`Retry-After: 2` while queued/processing. Providers can omit this header;
both clients otherwise poll once per second. The editor and Node example
honor delta-seconds and HTTP-date values, bounded to 1–30 seconds. Status
GETs retry transport failures and HTTP 429/502/503/504 up to six times;
successful status responses reset that retry count. Cancellation interrupts
the wait. Uploads and job creation are never retried automatically.
Never retry a job-creation POST automatically: an interrupted response may
already have created the job. Unknown jobs and other users' jobs both return
404. Invalid keys return 401; malformed input returns 400; an invalid state
returns 409; queue/owner capacity returns 429 with `Retry-After: 30`; an
unconfigured processor returns 503. The optional `queuePosition` is one-based
among waiting jobs and null outside `queued`. It is advisory and can change
after other jobs finish or are cancelled. A queued job is not processing yet.

```json
{"id":"5b655d79-9ef2-48ed-ae6f-3db3f87b3d06","state":"queued","message":"Waiting for processing","queuePosition":2}
```

Do not expose other owners' job IDs, filenames or profiles in a queue
response. Job IDs are opaque UUIDs, never credentials. Providers may add
optional fields without changing protocol 1; clients must reject unknown
required states/protocol versions rather than treat them as completed.

| Status | Meaning and client action |
| --- | --- |
| 200 | Capabilities, status or a complete motion result |
| 201 | Job created; persist the returned ID before uploading |
| 202 | Upload accepted; poll the job until terminal |
| 204 | Deletion/cancellation accepted; body is empty |
| 400 | Invalid settings, extension, empty input or malformed request; fix the input |
| 401 | Missing/expired/rejected credential; authenticate again |
| 403 | Account not enabled or unsafe browser request in the OAuth frontend |
| 404 | Unknown, expired, deleted or other-owner job; do not enumerate IDs |
| 405 | Unsupported method |
| 409 | Job state does not permit this action; for an early result request, poll first |
| 413 | Upload or clip limit exceeded; the message says which. Trim the clip or reduce the file size |
| 429 | Host capacity/rate limit; honor `Retry-After`, then let the user retry creation |
| 500 | Server error; report the job ID to the operator |
| 502 / 504 | Proxy/backend failure; check status before repeating a mutation |
| 503 | Processor unavailable; do not upload until capabilities reports ready |

Error bodies may be empty or `application/problem+json`; the OAuth frontend
also has its documented JSON error envelope. Use the status code for control
flow, and show only bounded messages. Do not log bearer credentials, raw
video bodies or OAuth callback query strings. Every API response uses
`Cache-Control: no-store`.

The client deletes remote jobs after downloading, on failure and on
cancellation. If the host is unreachable during cleanup, the client reports
the job ID so the operator can remove it. Uploaded videos are deleted when
processing ends. Completed results are deleted 24 hours after processing;
failed, cancelled and abandoned pre-upload jobs after one hour, checked once
per minute. Active processing must finish cancellation
before its files are removed.

The editor bounds a hosted capture to two hours including upload and queue
time, each HTTP request to thirty minutes, and remote cleanup to ten seconds.
The example service bounds upload to thirty minutes and active processing
to one hour; expiry cleanup runs every minute. Processors must observe
cancellation and release their input/output streams before returning.
Providers with longer queues should expose their position and delay through
status, and account for the editor deadline. The browser example has no
separate job deadline, but its credentials expire after eight hours.

Job metadata is not durable. Each server lifetime owns a locked run folder in
the data directory; the next start deletes folders no running host holds, so
results never outlive their 24 hours. Replace the storage/queue adapters with
durable infrastructure before relying on restart recovery or running multiple
server instances.

## Acceptance tests

Final acceptance requires downloaded real-person clips measuring 5–10 seconds
each. Record the source URL, measured duration, Base/Latest selection, camera
interval, processing GPU, target rig and exported FBX/ModelDoc result. Exercise
both AMD and NVIDIA backends. Synthetic test streams and motion fixtures do
not establish model quality or satisfy the video requirement.

The current environment has downloaded clips/models and an accessible AMD
GPU/editor session. Full video inference and live UI/export acceptance
remain outstanding. The test projects cover transport, ownership,
cancellation, bounded input, motion validation and shared CPU logic.
These tests do not establish the quality of reconstructed motion.
