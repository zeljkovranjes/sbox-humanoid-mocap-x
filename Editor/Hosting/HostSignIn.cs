using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Web;

namespace HumanoidMocapX.EditorTools;

/// <summary>Signs the editor in to a provider's site without keys: the browser goes straight to
/// the provider, and the result returns to a one-shot listener on 127.0.0.1 (RFC 8252) as a
/// single-use code that only this editor's PKCE verifier can redeem.</summary>
internal sealed class HostSignIn : IDisposable
{
    static readonly TimeSpan SignInTimeout = TimeSpan.FromMinutes(5), RequestTimeout = TimeSpan.FromSeconds(10);
    const int MaximumRequestBytes = 8192;
    readonly HttpClient http;
    readonly JsonSerializerOptions json = new(JsonSerializerDefaults.Web);

    internal HostSignIn(HttpMessageHandler handler = null)
    {
        http = new(handler ?? new HttpClientHandler { AllowAutoRedirect = false }) { Timeout = TimeSpan.FromSeconds(30) };
    }

    /// <summary>The sign-in options the provider's site offers. Throws when it is not a Humanoid
    /// Mocap X provider or cannot be reached.</summary>
    internal async Task<IReadOnlyList<HostProvider>> ProvidersAsync(Uri endpoint, CancellationToken cancellation)
    {
        using var response = await http.GetAsync(new Uri(endpoint, "auth/providers"), cancellation);
        if (!response.IsSuccessStatusCode) throw new HttpRequestException($"The address answered HTTP {(int)response.StatusCode}; it does not look like a Humanoid Mocap X provider.");
        var body = await Read<ProviderList>(response, cancellation);
        if (body.Providers is not { Count: > 0 }) throw new InvalidDataException("This provider has not enabled any sign-in method yet.");
        return body.Providers;
    }

    /// <summary>Opens the browser at the chosen provider and waits up to five minutes for the
    /// sign-in to return, then applies the session to <paramref name="settings"/>.</summary>
    internal async Task SignInAsync(HostingSettings settings, string provider, Action<string> openBrowser, CancellationToken cancellation)
    {
        var endpoint = settings.Endpoint();
        var verifier = Base64Url(RandomNumberGenerator.GetBytes(48));
        var challenge = Base64Url(SHA256.HashData(Encoding.ASCII.GetBytes(verifier)));
        var state = Base64Url(RandomNumberGenerator.GetBytes(24));
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        var query = $"provider={Uri.EscapeDataString(provider)}&port={port}&state={state}&code_challenge={challenge}&code_challenge_method=S256";
        openBrowser(new Uri(endpoint, "auth/editor?" + query).AbsoluteUri);

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        timeout.CancelAfter(SignInTimeout);
        var code = await Task.Run(() => ReceiveCode(listener, state, timeout.Token), timeout.Token);

        using var exchange = new HttpRequestMessage(HttpMethod.Post, new Uri(endpoint, "auth/editor/token"))
        {
            Content = new StringContent(JsonSerializer.Serialize(new { code, verifier }), Encoding.UTF8, "application/json")
        };
        using var response = await http.SendAsync(exchange, cancellation);
        if (!response.IsSuccessStatusCode) throw new HttpRequestException("The provider did not accept the sign-in. Try again.");
        var result = await Read<TokenResponse>(response, cancellation);
        if (result.Token is not { Length: 43 } token || result.User?.Id is null) throw new InvalidDataException("The provider returned an invalid sign-in.");
        settings.SignIn(token, result.ExpiresAt, result.Profile());
    }

    /// <summary>Re-reads the profile (whitelist changes, expiry). Signs out on 401.</summary>
    internal async Task RefreshAsync(HostingSettings settings, CancellationToken cancellation)
    {
        if (!settings.SignedIn) { settings.SignOut(); return; }
        var token = settings.Token;
        using var request = new HttpRequestMessage(HttpMethod.Get, new Uri(settings.Endpoint(), "auth/me"));
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        using var response = await http.SendAsync(request, cancellation);
        var me = response.IsSuccessStatusCode ? await Read<TokenResponse>(response, cancellation) : null;
        if (me is { SignedIn: true, User: not null }) settings.Update(token, me.Profile());
        else if (response.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.OK) settings.Expire(token);
        else throw new HttpRequestException($"The provider returned HTTP {(int)response.StatusCode} while checking your sign-in.");
    }

    /// <summary>Revokes the editor token on the site (best effort) and forgets it locally.</summary>
    internal async Task SignOutAsync(HostingSettings settings)
    {
        try
        {
            if (settings.SignedIn)
            {
                using var request = new HttpRequestMessage(HttpMethod.Post, new Uri(settings.Endpoint(), "auth/logout"));
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", settings.Token);
                using var cancel = new CancellationTokenSource(RequestTimeout);
                using var response = await http.SendAsync(request, cancel.Token);
            }
        }
        // Signing out locally must not depend on the provider being reachable.
        catch (Exception error) when (error is HttpRequestException or OperationCanceledException or ArgumentException) { }
        settings.SignOut();
    }

    // Serves the single browser redirect; other requests (favicons, probes) get 404 and are ignored.
    static async Task<string> ReceiveCode(TcpListener listener, string state, CancellationToken cancellation)
    {
        while (true)
        {
            using var client = await listener.AcceptTcpClientAsync(cancellation);
            using var stream = client.GetStream();
            using var perRequest = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            perRequest.CancelAfter(RequestTimeout);
            string target;
            try { target = await RequestTarget(stream, perRequest.Token); }
            catch (Exception error) when (error is IOException or InvalidDataException || error is OperationCanceledException && !cancellation.IsCancellationRequested) { continue; }
            if (target is null || !target.StartsWith("/callback?", StringComparison.Ordinal))
            { await Respond(stream, 404, "Not found."); continue; }
            var query = HttpUtility.ParseQueryString(target["/callback".Length..]);
            var returned = query["state"] ?? "";
            if (!CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(returned), Encoding.ASCII.GetBytes(state)) || query["code"] is not { Length: 43 } code)
            { await Respond(stream, 400, "This sign-in link does not belong to this editor. Start signing in again from Humanoid Mocap X."); continue; }
            await Respond(stream, 200, "You are signed in to Humanoid Mocap X. You can close this tab and return to s&box.");
            return code;
        }
    }

    // Reads the request line and headers (bounded), returning the GET target or null.
    static async Task<string> RequestTarget(NetworkStream stream, CancellationToken cancellation)
    {
        var buffer = new byte[MaximumRequestBytes]; var length = 0;
        while (length < buffer.Length)
        {
            var read = await stream.ReadAsync(buffer.AsMemory(length), cancellation);
            if (read == 0) break;
            length += read;
            if (Encoding.ASCII.GetString(buffer, 0, length).Contains("\r\n\r\n", StringComparison.Ordinal)) break;
        }
        var line = Encoding.ASCII.GetString(buffer, 0, length).Split("\r\n", 2)[0].Split(' ');
        if (line.Length != 3 || line[0] != "GET" || !line[2].StartsWith("HTTP/1.", StringComparison.Ordinal)) throw new InvalidDataException("Not an HTTP GET request.");
        return line[1];
    }

    static async Task Respond(NetworkStream stream, int status, string message)
    {
        var html = "<!doctype html><meta charset=utf-8><title>Humanoid Mocap X</title>"
            + "<body style=\"font:16px system-ui,sans-serif;background:#1b1f1c;color:#e4ebe6;display:grid;place-items:center;height:100vh;margin:0\">"
            + "<p>" + WebUtility.HtmlEncode(message) + "</p>";
        var body = Encoding.UTF8.GetBytes(html);
        var head = $"HTTP/1.1 {status} {(status == 200 ? "OK" : status == 404 ? "Not Found" : "Bad Request")}\r\nContent-Type: text/html; charset=utf-8\r\n"
            + $"Content-Length: {body.Length}\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n\r\n";
        await stream.WriteAsync(Encoding.ASCII.GetBytes(head));
        await stream.WriteAsync(body);
    }

    async Task<T> Read<T>(HttpResponseMessage response, CancellationToken cancellation)
    {
        await response.Content.LoadIntoBufferAsync(65536, cancellation);
        return JsonSerializer.Deserialize<T>(await response.Content.ReadAsStringAsync(cancellation), json)
            ?? throw new InvalidDataException("The provider returned an empty response.");
    }

    static string Base64Url(byte[] bytes) => Convert.ToBase64String(bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_');

    public void Dispose() => http.Dispose();

    sealed record ProviderList(List<HostProvider> Providers);

    sealed record TokenResponse(string Token, DateTimeOffset ExpiresAt, bool SignedIn, HostUser User, bool? InGuild, bool GuildOwner,
        bool Whitelisted, bool CanProcess, bool Admin, HostLimits Limits)
    {
        internal HostProfile Profile() => new(User, InGuild, GuildOwner, Whitelisted, CanProcess, Admin, Limits);
    }
}
