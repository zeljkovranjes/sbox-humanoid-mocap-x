import { openDatabase } from './database.mjs';
import { Store } from './store.mjs';

// Operator command line: npm run whitelist -- add <provider:id> [note] | remove <provider:id> | list
const [command, account, ...note] = process.argv.slice(2);
const database = await openDatabase(process.env.DATABASE_URL ?? 'sqlite:./mocap.db');
const store = new Store(database);
try {
  if (command === 'add' && account) { await store.whitelist(account, note.join(' ')); console.info(`Whitelisted ${account}`); }
  else if (command === 'remove' && account) console.info(await store.unwhitelist(account) ? `Removed ${account}` : `${account} was not whitelisted`);
  else if (command === 'list') for (const row of await store.listWhitelist())
    console.info(`${row.account_id}\t${row.display_name ?? row.username ?? 'not signed in yet'}\t${row.note}`);
  else { console.error('Usage: npm run whitelist -- add <provider:id> [note] | remove <provider:id> | list'); process.exitCode = 2; }
} catch (error) { console.error(error.message); process.exitCode = 1; }
finally { await database.destroy(); }
