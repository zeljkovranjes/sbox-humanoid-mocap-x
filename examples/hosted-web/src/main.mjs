import { createServer } from 'node:http';
import { configuration } from './config.mjs';
import { providers } from './identity.mjs';
import { openDatabase } from './database.mjs';
import { Store } from './store.mjs';
import { Sessions } from './sessions.mjs';
import { application } from './app.mjs';

const config = configuration(process.env);
const enabled = await providers(config);
if (!enabled.size) console.warn('No sign-in provider is configured; set DISCORD_*, GOOGLE_*, MICROSOFT_* or OIDC_* credentials.');
const database = await openDatabase(config.databaseUrl);
const store = new Store(database);
await store.ensureDefaultRole(config.defaultClipSeconds);
const sessions = new Sessions(config, enabled, store);
const server = createServer(await application({ config, sessions, store, providers: enabled }));
server.requestTimeout = 30 * 60 * 1000;
server.headersTimeout = 30_000;
server.maxHeadersCount = 40;
server.maxConnections = 1024;
server.listen(config.port, config.host, () => console.info(`Mocap web listening on ${config.publicUrl.origin}`));
for (const event of ['SIGINT', 'SIGTERM']) process.once(event, () => {
  server.close(); server.closeIdleConnections();
  setTimeout(() => server.closeAllConnections(), 10_000).unref();
  database.destroy();
});
