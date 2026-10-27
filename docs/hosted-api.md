Hosted API
==========

Everything the hosted website ([`examples/hosted-web`]) exposes: sign-in, the
capture API that browsers and Humanoid Mocap X use, and the admin API for
operators and Discord bots. All bodies are JSON unless stated. Errors return
`{ "error": { "code": "...", "message": "..." } }` with a matching HTTP status;
`message` is written for end users and safe to show.

[`examples/hosted-web`]: ../examples/hosted-web/README.md

Accounts are written `<provider>:<account ID>`, for example `discord:80351110224678912`
or `google:104729384756123456789`. Use that form wherever an `{account}` appears;
`:` may be sent as-is or percent-encoded.

<br>

## Sign-in

| Method and path | Purpose |
| --- | --- |
| `GET /auth/providers` | `{ providers: [{ name, label }] }`, the enabled sign-in providers. |
| `GET /auth/login/{provider}` | Browser sign-in. Redirects straight to the provider; returns to `/`. |
| `GET /auth/editor?provider=&port=&state=&code_challenge=&code_challenge_method=S256` | Editor sign-in (below). |
| `GET /auth/callback/{provider}` | The provider's redirect URI. Register `PUBLIC_URL/auth/callback/{provider}`. |
| `POST /auth/editor/token` | `{ code, verifier }` → `{ token, expiresAt, user, ... }` (below). |
| `GET /auth/me` | `{ signedIn: false }` or the profile below, for a cookie or `Authorization: Bearer <editor token>`. |
| `POST /auth/logout` | Ends the browser session or revokes the editor token. `204`. |

Profile (from `/auth/me` and `/auth/editor/token`):

```json
{
  "signedIn": true,
  "user": { "id": "discord:123", "provider": "discord", "username": "person", "name": "Person", "avatar": "https://..." },
  "inGuild": true,
  "guildOwner": false,
  "whitelisted": true,
  "canProcess": true,
  "admin": false,
  "roles": ["default", "supporter"],
  "limits": { "priority": true, "maximumClipSeconds": 120, "maximumUploadBytes": 268435456 }
}
```

`username`, `name` and `avatar` are `null` when the provider does not supply them.
`inGuild` is `null` for non-Discord accounts or when no server is configured.
`canProcess` requires the whitelist (admins are always whitelisted) and, for
Discord accounts when `DISCORD_GUILD_ID` is set, membership of that server.
The server's owner and accounts in `ADMIN_ACCOUNTS` are admins. `roles` and
`limits` are described under [Roles](#roles); a `null` clip limit means the
processing host's own (`MOCAP_MAX_CLIP_SECONDS`).

Editor sign-in follows [RFC 8252] for native apps. The editor listens on
`127.0.0.1:<port>` (1024–65535), creates a random `state` (16–128 URL-safe
characters) and a PKCE verifier, and opens `/auth/editor` in the browser. The
site redirects straight to the provider. After sign-in the browser is sent to
`http://127.0.0.1:<port>/callback?code=...&state=...`; the editor checks `state`
and posts the code with its verifier to `/auth/editor/token`. Codes are single
use and expire after two minutes. Editor tokens and browser sessions last eight
hours; signing in again replaces the account's previous editor token. One
account keeps at most eight browser sessions; older ones are signed out.

Discord membership, server ownership and Discord roles are read at sign-in and
kept for the session. The whitelist, assigned roles and role limits are read on
every request, so admin changes apply within five seconds.

[RFC 8252]: https://www.rfc-editor.org/rfc/rfc8252

<br>

## Capture API

Authenticated with the browser cookie (browser writes also need a matching
`Origin`) or `Authorization: Bearer <editor token>`. Accounts that cannot
process receive `403` with code `not_whitelisted` or `not_in_server`.

| Method and path | Result |
| --- | --- |
| `GET /v1/capabilities` | `{ protocol, processingAvailable, maximumUploadBytes, maximumActiveJobs, queueLength, processing, concurrentJobs, maximumClipSeconds, priority }`, with this account's limits. |
| `GET /v1/jobs` | The account's jobs, oldest first. |
| `POST /v1/jobs` | `{ model: "Latest"\|"Base", cameraInterval: 1\|10\|30, start, end, name? }` → `201` job |
| `PUT /v1/jobs/{id}/video` | Raw video bytes, header `X-Video-Extension: .mp4` (also `.mov .m4v .avi .mkv .webm`) → `202` |
| `GET /v1/jobs/{id}` | The job. `Retry-After` suggests a polling delay while queued or processing. |
| `GET /v1/jobs/{id}/result` | The `.hmotion` motion once `completed`, otherwise `409`. |
| `DELETE /v1/jobs/{id}` | Cancels a waiting job or deletes a finished one → `204`. A job that is `processing` cannot be cancelled → `409`. |

Job:

```json
{
  "id": "6f1c…", "state": "queued", "message": "Waiting for processing", "name": "walk.mp4",
  "createdAt": "2026-10-27T15:00:00+00:00", "queuePosition": 2, "queueLength": 4,
  "progress": null, "secondsUntilStart": 310, "secondsRemaining": 620, "expiresAt": null,
  "priority": false
}
```

States: `awaiting-upload`, `uploading`, `queued`, `processing`, `completed`,
`failed`, `cancelled`. `queuePosition` (1-based, out of `queueLength` waiting
jobs across all users) is set while queued; `progress` (0–1) while processing.
`secondsUntilStart` and `secondsRemaining` are estimates learned from the
host's finished jobs, spread over `concurrentJobs` processing slots. Each
account may have `maximumActiveJobs` unfinished jobs (the operator's
`MOCAP_MAX_JOBS_PER_USER`; `1` means one at a time); more return `429` with
`Retry-After`. Results are deleted 24 hours after a job completes
(`expiresAt`); jobs that never receive a video expire after an hour. Each
account keeps its 16 newest jobs; creating more deletes its oldest finished job.

Limits:

- `end - start` may be at most `maximumClipSeconds` (0.01 s rounding allowed);
  longer ranges return `413` with code `clip_too_long` before anything is
  queued. A job without `end` processes up to `start + maximumClipSeconds`.
- Uploads larger than `maximumUploadBytes` return `413` with code
  `body_too_large`, before forwarding when `Content-Length` says so and
  otherwise as soon as the limit is passed.
- Compressed bodies (`Content-Encoding` other than `identity`) return `415`.
- Jobs with `priority` wait ahead of all other jobs; within each group the
  first finished upload starts first. A running job is never interrupted.

<br>

## Roles

Roles set queue priority and limits per account. Every account has the
`default` role, which cannot be deleted; its clip limit starts at
`DEFAULT_MAX_CLIP_SECONDS`. An account also has:

- the role assigned on its whitelist entry, and
- every role whose `discordRoleId` is one of its Discord roles in
  `DISCORD_GUILD_ID` (read at sign-in with the `guilds.members.read` scope).

The most generous value of each setting across those roles applies, so roles
only add allowances: `priority` if any role has it, the largest
`maxClipSeconds` and the largest `maxUploadMb`. `null` means the processing
host's own limit. Uploads never exceed `MAX_UPLOAD_MB`, and clips never exceed
the host's `MOCAP_MAX_CLIP_SECONDS`. The Discord server owner always has
priority.

Role:

```json
{ "name": "supporter", "priority": true, "maxClipSeconds": 120, "maxUploadMb": 256, "discordRoleId": "112233445566778899" }
```

Names are 1–32 lowercase letters, digits or hyphens, starting with a letter.
`maxClipSeconds` is 1–86400 and `maxUploadMb` 1–65536, each a whole number or
`null`. To restrict someone below the default, lower the `default` role and
give everyone else a role.

<br>

## Admin API

For operators and Discord bots. Send `Authorization: Bearer <ADMIN_API_KEY>`,
or call it as a signed-in admin account (browser writes need a matching
`Origin`). Missing credentials return `401`, others `403`.

| Method and path | Result |
| --- | --- |
| `GET /admin/status` | `{ users, whitelisted, browserSessions, editors, host }`; `host` is the processing host's capabilities (including `concurrentJobs`) or `{ reachable: false }`. |
| `GET /admin/settings` | `{ concurrentJobs, maximumConcurrentJobs }` from the processing host. |
| `PUT /admin/settings` | Body `{ concurrentJobs }` (1 to `maximumConcurrentJobs`) → the new settings. Applies at once and survives host restarts. Lowering it lets running jobs finish. A job starts only when the GPU has memory for it, whatever this is set to. |
| `GET /admin/roles` | `{ roles: [role] }`, `default` first. |
| `PUT /admin/roles/{name}` | Creates or replaces a role. Body: the role's fields without `name`; omitted limits become `null`, omitted `priority` `false`. → `200 { role }` |
| `DELETE /admin/roles/{name}` | `204`; the role is removed from whitelist entries. `default` → `400`, unknown → `404`. |
| `GET /admin/users?search=&limit=&offset=` | Accounts seen at sign-in, newest first. `search` matches the exact account or part of a username/name (≤ 80 chars); `limit` ≤ 500. |
| `GET /admin/users/{account}` | `{ user }` with its roles and effective limits, or `404` if the account never signed in. |
| `GET /admin/users/{account}/jobs` | That account's jobs (as `GET /v1/jobs`). |
| `DELETE /admin/users/{account}/jobs/{id}` | Cancels or deletes that account's job (same rules as above). |
| `POST /admin/users/{account}/sign-out` | `{ revoked }`: ends every browser session, editor token and pending editor sign-in of the account. |
| `GET /admin/whitelist` | `{ whitelist: [{ id, note, role, addedAt, username, name }] }` |
| `PUT /admin/whitelist/{account}` | Body `{ note?, role? }` (note ≤ 200 chars; role an existing role or `null`) → `200 { id, whitelisted: true, role }`. Replaces the note and role. |
| `DELETE /admin/whitelist/{account}` | `204`, or `404` if not whitelisted. |
| `GET /admin/sign-ins?account=&limit=` | Recent sign-in attempts `{ signIns: [{ id, account, at, inGuild, whitelisted, client }] }`, `limit` ≤ 1000. The last 50 per account are kept. |

User (from `GET /admin/users/{account}`; the list omits `roles` and `limits`):

```json
{ "id": "discord:123", "provider": "discord", "username": "person", "name": "Person", "avatar": null,
  "inGuild": true, "guildOwner": false, "discordRoles": ["112233445566778899"],
  "whitelisted": true, "note": "friend", "role": "supporter",
  "firstSeen": "2026-10-27T15:00:00.000Z", "lastSeen": "2026-10-27T16:00:00.000Z",
  "roles": ["default", "supporter"],
  "limits": { "priority": true, "maximumClipSeconds": 120, "maximumUploadBytes": 268435456 } }
```

`role` is the role assigned on the whitelist entry; `roles` also includes
`default` and Discord-mapped roles, using the Discord roles from the last
sign-in.

<br>

## Examples

```sh
HOST=https://mocap.example.com
ADMIN="Authorization: Bearer $ADMIN_API_KEY"
JSON='content-type: application/json'

# A Discord bot's /whitelist command
curl -X PUT "$HOST/admin/whitelist/discord:80351110224678912" -H "$ADMIN" -H "$JSON" \
  -d '{"note":"added by a moderator"}'

# Default users: 30-second clips, 256 MiB uploads
curl -X PUT "$HOST/admin/roles/default" -H "$ADMIN" -H "$JSON" \
  -d '{"maxClipSeconds":30,"maxUploadMb":256}'

# Members with the Discord "Supporter" role: priority queue, 2-minute clips
curl -X PUT "$HOST/admin/roles/supporter" -H "$ADMIN" -H "$JSON" \
  -d '{"priority":true,"maxClipSeconds":120,"maxUploadMb":512,"discordRoleId":"112233445566778899"}'

# Or give one account a role directly
curl -X PUT "$HOST/admin/whitelist/discord:80351110224678912" -H "$ADMIN" -H "$JSON" \
  -d '{"note":"friend","role":"supporter"}'

# Process two jobs at once
curl -X PUT "$HOST/admin/settings" -H "$ADMIN" -H "$JSON" -d '{"concurrentJobs":2}'

# Who signed in recently, and whether they were let in
curl "$HOST/admin/sign-ins?limit=20" -H "$ADMIN"

# A user's queue, and cancelling one of their waiting jobs
curl "$HOST/admin/users/discord:80351110224678912/jobs" -H "$ADMIN"
curl -X DELETE "$HOST/admin/users/discord:80351110224678912/jobs/6f1c…" -H "$ADMIN"

# Remove access and sign the account out everywhere
curl -X DELETE "$HOST/admin/whitelist/discord:80351110224678912" -H "$ADMIN"
curl -X POST "$HOST/admin/users/discord:80351110224678912/sign-out" -H "$ADMIN"
```

The whitelist is also available from the host machine without HTTP:

```console
$ npm run whitelist -- add discord:80351110224678912 friend
$ npm run whitelist -- list
$ npm run whitelist -- remove discord:80351110224678912
```

<br>

## Processing host

The website talks to the processing host (the s&box editor host or the C#
service) on a loopback or private address; users never reach it. Each request
carries a key derived from `MOCAP_KEY_SECRET`: `<account>.<HMAC-SHA256>` for
an account's jobs, and `admin.<HMAC of "admin">` for `GET`/`PUT /v1/settings`.
Only signed account keys may send a job's `policy` (priority and limits); keys
from `MOCAP_API_KEYS` always get the host's own limits. The host listens on
loopback only.
