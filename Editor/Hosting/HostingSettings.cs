using System;
using System.Threading;

namespace HumanoidMocapX.EditorTools;

/// <summary>The provider choice and the signed-in session. The token and profile live only in
/// memory: closing the editor signs out, and nothing secret is written to disk. Any thread may
/// sign in or out; the session is swapped as one value.</summary>
internal sealed class HostingSettings
{
    internal bool Enabled;
    internal string ServiceUrl = "";
    HostSession session;

    /// <summary>Raised after sign-in, sign-out or a profile refresh, through <see cref="Dispatch"/>.</summary>
    internal event Action Changed;

    /// <summary>Runs <see cref="Changed"/>; the window sets it to queue onto the editor's main thread.</summary>
    internal Action<Action> Dispatch = handler => handler();

    internal string Token => Volatile.Read(ref session)?.Token ?? "";
    internal HostProfile Profile => Volatile.Read(ref session)?.Profile;
    internal DateTimeOffset ExpiresAt => Volatile.Read(ref session)?.ExpiresAt ?? default;
    internal bool SignedIn => Volatile.Read(ref session) is { } current && current.ExpiresAt > DateTimeOffset.UtcNow;

    /// <summary>The provider's base URL. Accepts a bare domain (HTTPS is assumed); plain HTTP is
    /// allowed only on loopback for local testing. Rejects credentials, queries and fragments.</summary>
    internal Uri Endpoint()
    {
        var text = ServiceUrl.Trim();
        if (text.Length > 0 && !text.Contains("://", StringComparison.Ordinal)) text = "https://" + text;
        if (!Uri.TryCreate(text, UriKind.Absolute, out var uri)
            || !(uri.Scheme == Uri.UriSchemeHttps || uri.Scheme == Uri.UriSchemeHttp && uri.IsLoopback)
            || uri.UserInfo.Length != 0 || uri.Query.Length != 0 || uri.Fragment.Length != 0)
            throw new ArgumentException("Enter the provider's address, for example mocap.example.com.");
        return new Uri(uri.AbsoluteUri.TrimEnd('/') + "/");
    }

    /// <summary>Throws when there is no current sign-in, so callers can prompt for one.</summary>
    internal void RequireSignedIn()
    {
        if (!SignedIn) throw new ArgumentException("Sign in to your processing provider first.");
    }

    internal void SignIn(string token, DateTimeOffset expiresAt, HostProfile profile)
    {
        Volatile.Write(ref session, new(token, expiresAt, profile)); Notify();
    }

    /// <summary>Replaces the profile of the session that used <paramref name="token"/>.</summary>
    internal void Update(string token, HostProfile profile)
    {
        var current = Volatile.Read(ref session);
        if (current?.Token == token && Interlocked.CompareExchange(ref session, current with { Profile = profile }, current) == current) Notify();
    }

    internal void SignOut()
    {
        if (Interlocked.Exchange(ref session, null) is not null) Notify();
    }

    /// <summary>Signs out only if <paramref name="token"/> is still the current one, so a late 401
    /// for an old sign-in cannot end a newer one.</summary>
    internal void Expire(string token)
    {
        var current = Volatile.Read(ref session);
        if (current?.Token == token && Interlocked.CompareExchange(ref session, null, current) == current) Notify();
    }

    void Notify() { if (Changed is { } handlers) Dispatch(handlers); }

    sealed record HostSession(string Token, DateTimeOffset ExpiresAt, HostProfile Profile);
}

/// <summary>The signed-in account. Username, name and avatar are null when the provider does not
/// share them. InGuild is null when no Discord server applies. Limits are null from providers
/// that do not report them.</summary>
internal sealed record HostProfile(HostUser User, bool? InGuild, bool GuildOwner, bool Whitelisted, bool CanProcess, bool Admin,
    HostLimits Limits = null);

/// <summary>The account's queue priority, longest clip in seconds (null: the host's own) and
/// largest upload in bytes.</summary>
internal sealed record HostLimits(bool Priority, double? MaximumClipSeconds, long MaximumUploadBytes);

internal sealed record HostUser(string Id, string Provider, string Username, string Name, string Avatar);

/// <summary>One sign-in option offered by the provider's site.</summary>
internal sealed record HostProvider(string Name, string Label);
