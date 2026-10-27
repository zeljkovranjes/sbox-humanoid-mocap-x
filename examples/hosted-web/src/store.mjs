import { randomUUID } from 'node:crypto';
import { accountPattern } from './config.mjs';
import { HttpError } from './error.mjs';
import { defaultRole } from './roles.mjs';

const cacheSeconds = 5;
const maximumCached = 10_000;
// Sign-in attempts kept per account; older ones are deleted as new ones arrive.
const signInHistory = 50;
const flag = value => (value ? 1 : 0);
const optionalFlag = value => (value === null || value === undefined ? null : flag(value));

/** Accounts, sign-in attempts, the whitelist and roles, in any SQL database Kysely supports.
 * Accounts are "<provider>:<account ID>". */
export class Store {
  constructor(database, clock = Date.now) {
    this.database = database; this.clock = clock; this.cache = new Map(); this.roleCache = null;
  }

  /** Creates the default role with the operator's starting clip limit when it is missing. */
  async ensureDefaultRole(maxClipSeconds) {
    if (!await this.database.selectFrom('mocap_roles').select('name').where('name', '=', defaultRole).executeTakeFirst())
      await this.database.insertInto('mocap_roles').values({ name: defaultRole, priority: 0, max_clip_seconds: maxClipSeconds,
        max_upload_mb: null, discord_role_id: null }).execute();
    this.roleCache = null;
  }

  now() { return new Date(this.clock()).toISOString(); }

  /** Saves the latest profile and appends the attempt, whether or not access is granted. */
  async recordSignIn(user, access, client) {
    const at = this.now();
    const values = { provider: user.provider, username: user.username, display_name: user.name, avatar: user.avatar,
      in_guild: optionalFlag(access.inGuild), guild_owner: flag(access.guildOwner),
      discord_roles: user.discordRoles?.length ? user.discordRoles.join(',') : null, last_seen: at };
    await this.database.transaction().execute(async transaction => {
      const updated = await transaction.updateTable('mocap_users').set(values).where('account_id', '=', user.id).executeTakeFirst();
      if (!Number(updated.numUpdatedRows))
        await transaction.insertInto('mocap_users').values({ account_id: user.id, first_seen: at, ...values }).execute();
      await transaction.insertInto('mocap_sign_ins').values({ id: randomUUID(), account_id: user.id, at,
        in_guild: optionalFlag(access.inGuild), whitelisted: flag(access.whitelisted), client }).execute();
      const oldest = await transaction.selectFrom('mocap_sign_ins').select('at').where('account_id', '=', user.id)
        .orderBy('at', 'desc').limit(1).offset(signInHistory).executeTakeFirst();
      if (oldest) await transaction.deleteFrom('mocap_sign_ins').where('account_id', '=', user.id).where('at', '<=', oldest.at).execute();
    });
  }

  /** { whitelisted, role } for the ID. Cached briefly because every proxied request asks. */
  async access(id) {
    const cached = this.cache.get(id);
    if (cached && cached.expires > this.clock()) return cached.value;
    const row = await this.database.selectFrom('mocap_whitelist').select('role').where('account_id', '=', id).executeTakeFirst();
    const value = { whitelisted: Boolean(row), role: row?.role ?? null };
    if (this.cache.size >= maximumCached) this.cache.clear();
    this.cache.set(id, { value, expires: this.clock() + cacheSeconds * 1000 });
    return value;
  }

  async isWhitelisted(id) { return (await this.access(id)).whitelisted; }

  /** Whitelists the ID with an optional note and role (null for only the default role). */
  async whitelist(id, note = '', role = null) {
    validId(id);
    if (typeof note !== 'string' || note.length > 200) throw new RangeError('Whitelist notes are at most 200 characters.');
    if (role !== null && !(await this.roles()).some(entry => entry.name === role))
      throw new HttpError(400, 'unknown_role', `There is no role named "${String(role).slice(0, 32)}". Create it first.`);
    const existing = await this.database.updateTable('mocap_whitelist').set({ note, role }).where('account_id', '=', id).executeTakeFirst();
    if (!Number(existing.numUpdatedRows))
      await this.database.insertInto('mocap_whitelist').values({ account_id: id, note, role, added_at: this.now() }).execute();
    this.cache.delete(id);
  }

  /** All roles, default first. Cached briefly because every proxied request asks. */
  async roles() {
    if (this.roleCache && this.roleCache.expires > this.clock()) return this.roleCache.value;
    const rows = await this.database.selectFrom('mocap_roles').selectAll().orderBy('name').execute();
    const value = rows.map(row => ({ name: row.name, priority: Boolean(row.priority), maxClipSeconds: row.max_clip_seconds,
      maxUploadMb: row.max_upload_mb, discordRoleId: row.discord_role_id }))
      .sort((a, b) => Number(b.name === defaultRole) - Number(a.name === defaultRole));
    this.roleCache = { value, expires: this.clock() + cacheSeconds * 1000 };
    return value;
  }

  /** Creates or replaces a role validated by parseRole. */
  async saveRole(role) {
    const values = { priority: flag(role.priority), max_clip_seconds: role.maxClipSeconds, max_upload_mb: role.maxUploadMb,
      discord_role_id: role.discordRoleId };
    const existing = await this.database.updateTable('mocap_roles').set(values).where('name', '=', role.name).executeTakeFirst();
    if (!Number(existing.numUpdatedRows)) await this.database.insertInto('mocap_roles').values({ name: role.name, ...values }).execute();
    this.roleCache = null;
  }

  /** Deletes a role and unassigns it; returns false when it did not exist. The default role stays. */
  async deleteRole(name) {
    if (name === defaultRole) throw new HttpError(400, 'default_role', 'The default role cannot be deleted; change its limits instead.');
    const removed = await this.database.transaction().execute(async transaction => {
      await transaction.updateTable('mocap_whitelist').set({ role: null }).where('role', '=', name).execute();
      return transaction.deleteFrom('mocap_roles').where('name', '=', name).executeTakeFirst();
    });
    this.roleCache = null; this.cache.clear();
    return Number(removed.numDeletedRows) > 0;
  }

  /** Row counts for the admin status, counted in SQL. */
  async counts() {
    const count = table => this.database.selectFrom(table).select(eb => eb.fn.countAll().as('count')).executeTakeFirstOrThrow();
    const [users, whitelisted] = await Promise.all([count('mocap_users'), count('mocap_whitelist')]);
    return { users: Number(users.count), whitelisted: Number(whitelisted.count) };
  }

  /** Removes the ID; returns false when it was not whitelisted. */
  async unwhitelist(id) {
    validId(id);
    const removed = await this.database.deleteFrom('mocap_whitelist').where('account_id', '=', id).executeTakeFirst();
    this.cache.delete(id);
    return Number(removed.numDeletedRows) > 0;
  }

  listWhitelist() {
    return this.database.selectFrom('mocap_whitelist as w').leftJoin('mocap_users as u', 'u.account_id', 'w.account_id')
      .select(['w.account_id', 'w.note', 'w.role', 'w.added_at', 'u.username', 'u.display_name']).orderBy('w.added_at').execute();
  }

  /** Users seen at sign-in, newest first, optionally filtered by ID or name fragment. */
  async listUsers({ search = '', limit = 50, offset = 0 } = {}) {
    let query = this.database.selectFrom('mocap_users as u').leftJoin('mocap_whitelist as w', 'w.account_id', 'u.account_id')
      .select(['u.account_id', 'u.provider', 'u.username', 'u.display_name', 'u.avatar', 'u.in_guild', 'u.guild_owner', 'u.discord_roles',
        'u.first_seen', 'u.last_seen', 'w.note', 'w.role'])
      .orderBy('u.last_seen', 'desc').limit(limit).offset(offset);
    if (search) query = query.where(eb => eb.or([eb('u.account_id', '=', search), eb('u.username', 'like', `%${search}%`),
      eb('u.display_name', 'like', `%${search}%`)]));
    return (await query.execute()).map(user);
  }

  async user(id) {
    validId(id);
    const row = await this.database.selectFrom('mocap_users as u').leftJoin('mocap_whitelist as w', 'w.account_id', 'u.account_id')
      .select(['u.account_id', 'u.provider', 'u.username', 'u.display_name', 'u.avatar', 'u.in_guild', 'u.guild_owner', 'u.discord_roles',
        'u.first_seen', 'u.last_seen', 'w.note', 'w.role'])
      .where('u.account_id', '=', id).executeTakeFirst();
    return row ? user(row) : null;
  }

  signIns({ id, limit = 50 } = {}) {
    let query = this.database.selectFrom('mocap_sign_ins').selectAll().orderBy('at', 'desc').limit(limit);
    if (id) { validId(id); query = query.where('account_id', '=', id); }
    return query.execute().then(rows => rows.map(row => ({ id: row.id, account: row.account_id, at: row.at,
      inGuild: row.in_guild === null ? null : Boolean(row.in_guild), whitelisted: Boolean(row.whitelisted), client: row.client })));
  }
}

function user(row) {
  return { id: row.account_id, provider: row.provider, username: row.username, name: row.display_name, avatar: row.avatar,
    inGuild: row.in_guild === null ? null : Boolean(row.in_guild),
    guildOwner: Boolean(row.guild_owner), discordRoles: row.discord_roles ? row.discord_roles.split(',') : [],
    whitelisted: row.note !== null && row.note !== undefined, note: row.note ?? null, role: row.role ?? null,
    firstSeen: row.first_seen, lastSeen: row.last_seen };
}

function validId(id) {
  if (typeof id !== 'string' || !accountPattern.test(id)) throw new RangeError('Accounts are <provider>:<account ID>, for example discord:123.');
}
