Mocap hosted web
================

This application provides the sign-in website and API for a Humanoid Mocap X
processing host. People sign in with Discord, Google, Microsoft or any OpenID
Connect provider; Humanoid Mocap X signs in the same way, with no keys to
copy. The site checks the whitelist (and Discord server membership), records
every sign-in in SQL through [Kysely], and forwards captures to the processing host.

[Kysely]: https://kysely.dev

```sh
cp .env.example .env
# Set PUBLIC_URL, PROCESSING_URL, MOCAP_KEY_SECRET and at least one provider.
npm ci
npm test
npm start
npm run whitelist -- add discord:<your Discord user ID> owner
```

<br>

## Example

```text
Browser / Humanoid Mocap X → this site (HTTPS) → in-editor processing host
```

Requires Node 24 or later and a processing host: normally the [in-editor host]
running in an s&box editor on Windows, or the [standalone starter] with your
own processor. Give both the same
`MOCAP_KEY_SECRET`. Register `PUBLIC_URL/auth/callback/<provider>` with each
provider you enable. In Humanoid Mocap X, enter `PUBLIC_URL` as the provider
address and choose **Sign in**; the browser goes straight to the provider and
the editor picks up the sign-in by itself.

[in-editor host]: ../../docs/editor-host.md
[standalone starter]: ../../../../../services/humanoid-mocap-host

<br>

## Details

- The [provider guide] goes from an empty machine to a verified public
  address: secret, processing host, website, HTTPS, roles and checks.
- [Hosted sign-in] covers providers, who may process, the editor's RFC 8252
  loopback sign-in and how the host credential is derived.
- The [API reference] lists every route: sign-in, the capture API with queue
  positions and time estimates, and the admin API for operators and Discord
  bots (`ADMIN_API_KEY`).
- The owner of `DISCORD_GUILD_ID` and accounts in `ADMIN_ACCOUNTS` are admins;
  the owner's jobs use the priority queue.
- [Roles] set priority, the longest clip and the largest upload per account,
  assigned on the whitelist or mapped from Discord roles. Admins change them,
  and how many jobs process at once, through the admin API.
- Results are deleted 24 hours after processing; a job that is already
  processing cannot be cancelled.
- Use a trusted HTTPS reverse proxy. Keep both internal ports private,
  preserve request streaming, allow 512 MiB uploads with long timeouts and
  apply per-client rate limits there. This is one instance with in-memory
  sessions; restarting it signs everyone out.

[provider guide]: ../../docs/provider-guide.md
[Hosted sign-in]: ../../docs/hosted-oauth.md
[API reference]: ../../docs/hosted-api.md
[Roles]: ../../docs/hosted-api.md#roles

<br>

## Validation

`npm test` covers provider sign-in with mocked Discord and OpenID Connect
responses (PKCE, state, server membership and ownership, missing profile
fields), the editor's loopback code exchange, access rules, the SQL store, and
a live loopback HTTP run through the proxy and every admin route. Real
provider consent requires the operator's own credentials.

<br>

#### License

<sup>
Licensed under the <a href="../../LICENSE-MIT">MIT license</a>.
</sup>
