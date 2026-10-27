import { Kysely, SqliteDialect, PostgresDialect, MysqlDialect } from 'kysely';

/** Opens the configured SQL database through Kysely and creates the tables when missing.
 * DATABASE_URL selects the dialect: sqlite:<file> (built in), postgres://… (install `pg`) or
 * mysql://… (install `mysql2`). Any other Kysely dialect can be returned from `dialect` below. */
export async function openDatabase(url) {
  const database = new Kysely({ dialect: await dialect(url) });
  await createTables(database);
  return database;
}

async function dialect(url) {
  if (url.startsWith('sqlite:')) {
    const { DatabaseSync } = await import('node:sqlite');
    const file = url.slice('sqlite:'.length) || ':memory:';
    return new SqliteDialect({ database: sqlite(new DatabaseSync(file)) });
  }
  if (/^postgres(ql)?:\/\//.test(url)) {
    const { default: pg } = await driver('pg');
    return new PostgresDialect({ pool: new pg.Pool({ connectionString: url }) });
  }
  if (url.startsWith('mysql://')) {
    const { createPool } = await driver('mysql2');
    return new MysqlDialect({ pool: createPool(url) });
  }
  throw new Error('DATABASE_URL must start with sqlite:, postgres:// or mysql://.');
}

async function driver(name) {
  try { return await import(name); }
  catch { throw new Error(`DATABASE_URL needs the "${name}" package. Run: npm install ${name}`); }
}

// Kysely's SQLite dialect expects better-sqlite3's statement shape; Node's built-in module
// differs only in spread parameters and booleans, so no native dependency is needed.
function sqlite(database) {
  const bind = parameters => parameters.map(value => typeof value === 'boolean' ? Number(value) : value);
  return {
    close: () => database.close(),
    prepare(sql) {
      const statement = database.prepare(sql);
      return {
        reader: statement.columns().length > 0,
        all: parameters => statement.all(...bind(parameters)),
        run: parameters => statement.run(...bind(parameters)),
        iterate: parameters => statement.iterate(...bind(parameters)),
      };
    },
  };
}

// Portable column types only, so every dialect accepts the same schema. Flags are integers 0/1
// (in_guild is null for providers without servers), times are ISO-8601 text, accounts are
// "<provider>:<account ID>", discord_roles is a comma-separated ID list and null limits mean the
// processing host's own.
async function createTables(database) {
  await database.schema.createTable('mocap_users').ifNotExists()
    .addColumn('account_id', 'varchar(160)', column => column.primaryKey())
    .addColumn('provider', 'varchar(16)', column => column.notNull())
    .addColumn('username', 'varchar(80)')
    .addColumn('display_name', 'varchar(128)')
    .addColumn('avatar', 'varchar(500)')
    .addColumn('in_guild', 'integer')
    .addColumn('guild_owner', 'integer', column => column.notNull())
    .addColumn('discord_roles', 'varchar(5300)')
    .addColumn('first_seen', 'varchar(32)', column => column.notNull())
    .addColumn('last_seen', 'varchar(32)', column => column.notNull())
    .execute();
  await database.schema.createTable('mocap_sign_ins').ifNotExists()
    .addColumn('id', 'varchar(36)', column => column.primaryKey())
    .addColumn('account_id', 'varchar(160)', column => column.notNull())
    .addColumn('at', 'varchar(32)', column => column.notNull())
    .addColumn('in_guild', 'integer')
    .addColumn('whitelisted', 'integer', column => column.notNull())
    .addColumn('client', 'varchar(16)', column => column.notNull())
    .execute();
  await database.schema.createIndex('mocap_sign_ins_user').ifNotExists().on('mocap_sign_ins').column('account_id').execute();
  await database.schema.createTable('mocap_whitelist').ifNotExists()
    .addColumn('account_id', 'varchar(160)', column => column.primaryKey())
    .addColumn('note', 'varchar(200)', column => column.notNull())
    .addColumn('role', 'varchar(32)')
    .addColumn('added_at', 'varchar(32)', column => column.notNull())
    .execute();
  await database.schema.createTable('mocap_roles').ifNotExists()
    .addColumn('name', 'varchar(32)', column => column.primaryKey())
    .addColumn('priority', 'integer', column => column.notNull())
    .addColumn('max_clip_seconds', 'integer')
    .addColumn('max_upload_mb', 'integer')
    .addColumn('discord_role_id', 'varchar(20)')
    .execute();
}
