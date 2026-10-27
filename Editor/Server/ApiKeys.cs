#nullable enable annotations

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Buffers.Text;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace HumanoidMocapHost;

/// <summary>Maps bearer keys to owners. Keys come from an explicit owner map, or are signed by a
/// trusted front end with a shared secret as <c>&lt;provider&gt;:&lt;account&gt;.&lt;HMAC-SHA256&gt;</c>
/// (for example <c>discord:123…</c> or <c>google:104…</c>), so the front end can serve any
/// whitelisted account without per-user host configuration. Signed owners always contain ':' and
/// owner-map names never do, so map keys cannot reach a signed account's jobs or job policy.</summary>
internal sealed class ApiKeys
{
    const int MinimumSecret = 32, MaximumSecret = 256;
    // A plain Regex, not [GeneratedRegex]: the s&box editor compiler may not run source generators.
    static readonly Regex SignedOwnerPattern = new(@"^[a-z][a-z0-9]{0,15}:[A-Za-z0-9._-]{1,128}$", RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100));
    readonly Dictionary<string, string> owners = new(StringComparer.Ordinal);
    readonly byte[]? secret;

    /// <summary>Reads the optional owner map (owner name to random key) and the optional signing
    /// secret. Throws when neither is configured or either is malformed.</summary>
    internal ApiKeys(string? json, string? secret = null)
    {
        if (!string.IsNullOrWhiteSpace(json))
        {
            var keys = JsonSerializer.Deserialize<Dictionary<string, string>>(json)
                ?? throw new ArgumentException("MOCAP_API_KEYS must map owner names to random API keys.");
            foreach (var pair in keys)
            {
                if (string.IsNullOrWhiteSpace(pair.Key) || pair.Key.Length > 64 || pair.Key.Contains(':') || pair.Value is null || pair.Value.Length is < 32 or > 256
                    || pair.Value.Any(char.IsWhiteSpace) || !owners.TryAdd(Hash(pair.Value), pair.Key))
                    throw new ArgumentException("MOCAP_API_KEYS contains an invalid owner (names cannot contain ':'), duplicate key or key shorter than 32 characters.");
            }
        }
        if (!string.IsNullOrEmpty(secret))
        {
            if (secret.Length is < MinimumSecret or > MaximumSecret || secret.Any(c => c is < '!' or > '~'))
                throw new ArgumentException("MOCAP_KEY_SECRET must be 32–256 printable characters without spaces.");
            this.secret = Encoding.UTF8.GetBytes(secret);
        }
        if (owners.Count == 0 && this.secret is null)
            throw new ArgumentException("Configure MOCAP_KEY_SECRET or at least one MOCAP_API_KEYS entry.");
    }

    internal string? Owner(string authorization)
    {
        if (!authorization.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) || authorization.Length is < 39 or > 263) return null;
        var key = authorization[7..];
        if (owners.TryGetValue(Hash(key), out var owner)) return owner;
        return secret is null ? null : SignedOwner(key);
    }

    /// <summary>True for accounts a front end signed, the only callers whose job policy is trusted.</summary>
    internal static bool IsSignedAccount(string owner) => owner.Contains(':');

    /// <summary>The key a front end holding the same secret sends for <paramref name="owner"/>.</summary>
    internal string Sign(string owner)
    {
        if (!SignedOwnerPattern.IsMatch(owner)) throw new ArgumentException("Signed owners are <provider>:<account ID>.", nameof(owner));
        return owner + "." + Mac(owner);
    }

    /// <summary>The front end's operator key for host settings: "admin." plus a MAC of "admin",
    /// which no account can sign because accounts contain ':'.</summary>
    internal string AdminKey() => "admin." + Mac("admin");

    /// <summary>True when the bearer key is <see cref="AdminKey"/>; always false without a secret.</summary>
    internal bool IsAdmin(string authorization)
    {
        if (secret is null || !authorization.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) || authorization.Length > 263) return false;
        return CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(authorization[7..]), Encoding.ASCII.GetBytes(AdminKey()));
    }

    string Mac(string message)
    {
        if (secret is null) throw new InvalidOperationException("No signing secret is configured.");
        return Base64Url.EncodeToString(HMACSHA256.HashData(secret, Encoding.UTF8.GetBytes(message)));
    }

    string? SignedOwner(string key)
    {
        var dot = key.LastIndexOf('.');
        if (dot < 1 || !SignedOwnerPattern.IsMatch(key.AsSpan(0, dot))) return null;
        var owner = key[..dot];
        return CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(key), Encoding.ASCII.GetBytes(Sign(owner))) ? owner : null;
    }

    static string Hash(string key) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key)));
}
