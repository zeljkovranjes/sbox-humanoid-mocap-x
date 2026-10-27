import { HttpError } from './error.mjs';

/** Every account has this role; it cannot be deleted. */
export const defaultRole = 'default';
const namePattern = /^[a-z][a-z0-9-]{0,31}$/;
const maximumClipSeconds = 86_400;
const maximumUploadMb = 65_536;

/** Validates an admin's role definition. Limits are whole numbers or null for the processing
 * host's own limit; discordRoleId optionally grants the role to members holding that Discord role. */
export function parseRole(name, body) {
  if (!namePattern.test(name)) throw new HttpError(400, 'invalid_role', 'Role names are 1–32 lowercase letters, digits or hyphens, starting with a letter.');
  if (body.priority !== undefined && typeof body.priority !== 'boolean') throw new HttpError(400, 'invalid_role', 'priority must be true or false.');
  const limit = (key, maximum) => {
    const value = body[key] ?? null;
    if (value !== null && (!Number.isInteger(value) || value < 1 || value > maximum))
      throw new HttpError(400, 'invalid_role', `${key} must be a whole number from 1 to ${maximum}, or null for the host's limit.`);
    return value;
  };
  const discordRoleId = body.discordRoleId ?? null;
  if (discordRoleId !== null && (typeof discordRoleId !== 'string' || !/^\d{1,20}$/.test(discordRoleId)))
    throw new HttpError(400, 'invalid_role', 'discordRoleId must be a numeric Discord role ID or null.');
  return { name, priority: body.priority === true, maxClipSeconds: limit('maxClipSeconds', maximumClipSeconds),
    maxUploadMb: limit('maxUploadMb', maximumUploadMb), discordRoleId };
}

/** An account's queue priority and limits: the most generous of the default role, its assigned
 * role and the roles mapped to its Discord roles. The Discord server owner always has priority.
 * A null clip limit means the processing host's own; uploads never exceed MAX_UPLOAD_MB. */
export function effectivePolicy(roles, { assigned = null, discordRoles = [], guildOwner = false }, config) {
  const matched = roles.filter(role => role.name === defaultRole || role.name === assigned
    || (role.discordRoleId !== null && discordRoles.includes(role.discordRoleId)));
  const most = values => values.length === 0 || values.includes(null) ? null : Math.max(...values);
  const uploadMb = most(matched.map(role => role.maxUploadMb));
  return { roles: matched.map(role => role.name), priority: guildOwner || matched.some(role => role.priority),
    maximumClipSeconds: most(matched.map(role => role.maxClipSeconds)),
    maximumUploadBytes: Math.min(config.maximumUploadBytes, uploadMb === null ? Infinity : uploadMb * 1024 * 1024) };
}
