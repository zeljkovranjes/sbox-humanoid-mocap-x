Hosted sign-in
==============

The [hosted website] signs people in with an OAuth provider and forwards their
captures to the processing host. Nobody copies or pastes keys: the browser and
Humanoid Mocap X both sign in with the provider, and the website derives the
host credential itself. Every route is listed in the [API reference].

[hosted website]: ../examples/hosted-web/README.md
[API reference]: hosted-api.md

## Providers

Sign-in uses [`oauth4webapi`] (authorization-code grant, S256 PKCE and state)
for every provider. Register `PUBLIC_URL/auth/callback/<name>` as the redirect
URI and put each provider's client ID and secret in the website's environment.

[`oauth4webapi`]: https://github.com/panva/oauth4webapi

| Provider | Environment | Scopes | Notes |
| --- | --- | --- | --- |
| Discord | `DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET`, optional `DISCORD_GUILD_ID` | `identify guilds`, plus `guilds.members.read` when a server is set | `guilds` shows whether the account is in the configured server and owns it; `guilds.members.read` returns its Discord roles there, which [roles] can map to limits and priority. |
| Google | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | `openid profile` | OpenID Connect discovery from `https://accounts.google.com`. |
| Microsoft | `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET`, optional `MICROSOFT_TENANT` | `openid profile` | Use your directory (tenant) ID; the default allows personal accounts. |
| Any OIDC provider | `OIDC_PROVIDERS=name`, `OIDC_<NAME>_ISSUER`, `_CLIENT_ID`, `_CLIENT_SECRET`, `_LABEL` | `openid profile` | Discovered from the issuer. |

Accounts are stored as `<provider>:<account ID>`. Only that ID is required;
username, display name and avatar are kept when the provider returns them and
shown as not provided otherwise. Provider access tokens are used once during
sign-in and discarded.

## Who can process

- A **whitelisted** account can process. Operators manage the whitelist with
  the admin API, a Discord bot built on it, or `npm run whitelist`.
- When `DISCORD_GUILD_ID` is set, **Discord** accounts must also be members of
  that server. Other providers rely on the whitelist alone.
- The **owner of that Discord server** and accounts listed in `ADMIN_ACCOUNTS`
  are admins: always allowed to process and allowed to use `/admin/*`.

Anyone who completes sign-in gets a session so the editor can show their
profile and status, but only accounts allowed above reach the processing host.
The server owner also gets the priority queue.
Access is checked on every request, so removing someone from the whitelist
takes effect within five seconds. Every sign-in attempt is recorded with the
account, time, server membership, whitelist status and client.

## Editor sign-in

Humanoid Mocap X uses the native-app flow from [RFC 8252]. It listens briefly on
a random `127.0.0.1` port and opens `/auth/editor` in the browser, which
redirects straight to the chosen provider. After sign-in the browser returns
to the editor's loopback address with a single-use code; the editor exchanges
it with its PKCE verifier for an eight-hour token. The code is delivered only
to `127.0.0.1` on the computer that signs in. Without a confirmation page, a
sign-in link made by someone else still signs you in for *their* editor; if
nothing listens on its port, the browser's error page shows the code. Never
open sign-in links other people send you, and never paste such a page's address
to anyone.

[RFC 8252]: https://www.rfc-editor.org/rfc/rfc8252

## Processing host credential

The website and the processing host share `MOCAP_KEY_SECRET`. For each
request the website sends `Authorization: Bearer <account>.<signature>`, where
the signature is HMAC-SHA256 of the account with that secret. The host checks
it in constant time and keeps every account's jobs separate, so adding a user
needs no host configuration. Browsers and the editor never see this key.
The website also sends each job's priority and limits from the account's
[roles]; the host accepts them only with a signed key. An `admin.<signature>`
key derived the same way lets the website change the host's concurrent jobs.

[roles]: hosted-api.md#roles

## Data and limits

The website stores accounts, sign-in attempts and the whitelist through
[Kysely]. `DATABASE_URL` selects SQLite (built in), PostgreSQL (`npm install pg`)
or MySQL (`npm install mysql2`); other Kysely dialects can be added in
`src/database.mjs`. Sessions and editor tokens live in memory, last eight
hours and end when the website restarts. A sign-in in progress is kept in an
encrypted, five-minute cookie rather than in memory, and over HTTPS every
cookie uses the `__Host-` prefix and the site sends HSTS. The last 50 sign-in
attempts per account are kept.

Discord membership, ownership and roles are checked when the user signs in and
hold for that session (at most eight hours); the website does not keep Discord
tokens to re-check them. Sign a user out (`POST /admin/users/{account}/sign-out`)
to apply a Discord change at once. Editor sign-in redirects straight to the
provider without a confirmation page, as requested; the loopback code and
PKCE verifier mean only the editor that started a sign-in can finish it, so a
remaining risk needs other software already running on the user's computer.

[Kysely]: https://kysely.dev

Run one website instance behind an HTTPS reverse proxy, keep its port and the
processing host's port private, allow uploads of `MAX_UPLOAD_MB` with long
timeouts, and apply per-client rate limits at the proxy.
