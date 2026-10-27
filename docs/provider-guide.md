Provider guide
==============

How to host Humanoid Mocap X processing for other people, from an empty
machine to a verified public address. Users only enter your address in the
editor and sign in; they never handle keys.

```text
Humanoid Mocap X / browser ──HTTPS──▶ reverse proxy ──▶ website (Node, :5081)
                                                           │ signed per-account keys
                                                           ▼
                                             processing host (s&box editor, :5080)
```

Both services listen on `127.0.0.1` only. The reverse proxy is the only public
port.

<br>

## 1. Machine

- Windows 10/11 with an s&box editor install, Node 24 or later, and a reverse
  proxy that terminates HTTPS (Caddy, nginx or IIS). Uploaded videos are
  decoded with Windows Media Foundation, so the processing host needs Windows;
  the website can run anywhere Node runs.
- A GPU with enough free memory for at least one job. The host reserves each
  job's estimated peak, which grows with the clip, before starting it; see
  [Queue and memory admission]. Processing has been measured on an AMD RX 9070
  (16 GB); other cards are untested. Jobs start only when the GPU has room, so
  "jobs at once" can be set higher than the card allows; extra jobs simply wait.
- Disk: about 11.6 GB for the models the host needs (both FlowHMR releases),
  plus room for a partial download of the largest file (about 4.6 GB), uploads
  (`MAX_UPLOAD_MB` per job in flight) and up to 16 GiB of decoded frames per
  active job. Results are deleted 24 hours after processing.

[Queue and memory admission]: editor-host.md#queue-and-memory-admission

<br>

## 2. Shared secret

Generate one random secret and give it to both services as
`MOCAP_KEY_SECRET`. The website derives each account's host key and its admin
key from it, so it must match exactly.

```sh
node -e "console.log(require('node:crypto').randomBytes(32).toString('base64url'))"
```

Generate a second, different value for the website's `ADMIN_API_KEY` if a
Discord bot or script will manage the host.

<br>

## 3. Processing host

The processing host runs inside the s&box editor with the Humanoid Mocap X
project open, using the same GPU code as local capture. Download the models
once (settings > **Processing > Setup > Local models**), set the variables
below, launch the editor from that shell and run `mocap_x_host_start` in the
editor console.
[In-editor processing host] has the full steps, limits and troubleshooting.

| Variable | Typical value |
| --- | --- |
| `MOCAP_KEY_SECRET` | the shared secret |
| `MOCAP_PORT` | `5080` |
| `MOCAP_MAX_JOBS_PER_USER` | `1` to let each person queue one capture at a time |
| `MOCAP_MAX_UPLOAD_MB` | `512` (match the website's `MAX_UPLOAD_MB`) |
| `MOCAP_MAX_CLIP_SECONDS` | `120`; roles on the website can lower it |
| `MOCAP_CONCURRENT_JOBS` | `1`; change it later from the admin API |

[In-editor processing host]: editor-host.md

<br>

## 4. Website

```sh
cd examples/hosted-web
cp .env.example .env      # then edit it
npm ci
npm test
npm start
```

Set at least `PUBLIC_URL` (your public HTTPS origin), `PROCESSING_URL`
(`http://127.0.0.1:5080`), `MOCAP_KEY_SECRET`, `DATABASE_URL` and one sign-in
provider. `.env.example` explains every variable. SQLite needs nothing else;
for PostgreSQL or MySQL run `npm install pg` or `npm install mysql2` and set
`DATABASE_URL` accordingly.

Register `PUBLIC_URL/auth/callback/<provider>` as the redirect URI with each
provider you enable, for example
`https://mocap.example.com/auth/callback/discord`. [Hosted sign-in] lists the
scopes and settings for Discord, Google, Microsoft and other OpenID Connect
providers.

[Hosted sign-in]: hosted-oauth.md

For a Discord community, set `DISCORD_GUILD_ID`. Members must be in that
server, and its owner automatically becomes an admin with the priority queue.

<br>

## 5. HTTPS

Point your domain at the machine and proxy it to the website. Allow uploads up
to `MAX_UPLOAD_MB` and keep the request streaming. With Caddy:

```text
mocap.example.com {
	request_body {
		max_size 512MB
	}
	reverse_proxy 127.0.0.1:5081
}
```

Never expose ports 5080 or 5081 directly. Add per-client rate limits at the
proxy if your proxy supports them.

<br>

## 6. Access, roles and capacity

Whitelist people by account ID; they can copy it from their profile card in
the editor or from the website after signing in.

```sh
npm run whitelist -- add discord:80351110224678912 "first tester"
```

Everything else goes through the admin API with `ADMIN_API_KEY`, or as a
signed-in admin in the browser. The [API reference] has every route with curl
examples; the usual first steps are:

```sh
HOST=https://mocap.example.com
ADMIN="Authorization: Bearer $ADMIN_API_KEY"
JSON='content-type: application/json'

# Everyone: 30-second clips, 256 MiB uploads
curl -X PUT "$HOST/admin/roles/default" -H "$ADMIN" -H "$JSON" -d '{"maxClipSeconds":30,"maxUploadMb":256}'

# Discord "Supporter" role: priority queue and 2-minute clips
curl -X PUT "$HOST/admin/roles/supporter" -H "$ADMIN" -H "$JSON" \
  -d '{"priority":true,"maxClipSeconds":120,"discordRoleId":"112233445566778899"}'

# Process two jobs at once (kept across restarts)
curl -X PUT "$HOST/admin/settings" -H "$ADMIN" -H "$JSON" -d '{"concurrentJobs":2}'
```

[API reference]: hosted-api.md

A Discord bot can expose the same operations as slash commands: whitelist an
account, assign a role, show someone's queue, cancel a waiting job, sign
someone out and change how many jobs run at once. Every response is JSON,
and every error message is safe to show to users.

<br>

## 7. Check it works

```sh
curl -s $HOST/health                                # {"status":"ok"}
curl -s $HOST/auth/providers                        # your sign-in options
curl -s $HOST/admin/status -H "$ADMIN"              # host reachable, processingAvailable: true
curl -s -o /dev/null -w '%{http_code}\n' $HOST/v1/capabilities   # 401 without sign-in
```

Then, in Humanoid Mocap X, choose **Use a provider**, enter your address,
sign in, and check the profile card shows your account as an admin with your
limits. Drop a short video into the workspace and confirm it uploads, shows
its place in line, processes and opens.

<br>

## Data and retention

- Videos are deleted as soon as processing ends; motion results 24 hours
  after processing; failed, cancelled and abandoned jobs after one hour.
  Restarting the host deletes everything from its previous run.
- The website stores account IDs, the names and avatars providers share, the
  last 50 sign-in attempts per account, the whitelist and roles. Provider
  access tokens are discarded after sign-in.
- Sessions last eight hours and live in memory; restarting the website signs
  everyone out. Discord membership, ownership and roles are checked at sign-in.
