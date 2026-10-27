using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization;
using HumanoidMocapX;

namespace HumanoidMocapHost;

/// <summary>Serves the capture API over the BCL <see cref="HttpListener"/>, for hosts that cannot
/// run ASP.NET, such as an s&amp;box editor. Routes, authentication, limits and status codes match
/// the ASP.NET entry in main.cs. Listens on loopback only: terminate HTTPS at a reverse proxy.
/// /v1/settings needs <see cref="ApiKeys.AdminKey"/>; every other /v1 route needs an owner key.</summary>
internal sealed class HttpHost : IAsyncDisposable
{
    /// <summary>The JSON conventions the X client and the ASP.NET entry use.</summary>
    internal static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { Converters = { new JsonStringEnumConverter() } };
    const long MaximumOptionsBytes = 64 * 1024;
    // Small JSON bodies have no reason to trickle in; uploads keep the queue's 30-minute limit.
    static readonly TimeSpan BodyTimeout = TimeSpan.FromSeconds(15);
    readonly HttpListener listener = new();
    readonly ApiKeys keys;
    readonly CaptureQueue queue;
    readonly Action<Exception> reportError;
    readonly CancellationTokenSource stopping = new();
    readonly HashSet<Task> handlers = new();
    readonly Task accepting;

    /// <summary>Starts listening on <paramref name="prefix"/>, for example http://127.0.0.1:5080/.
    /// Unexpected request failures go to <paramref name="reportError"/>. Rejects non-loopback
    /// prefixes; throws <see cref="HttpListenerException"/> when the port is unavailable.</summary>
    internal HttpHost(string prefix, ApiKeys keys, CaptureQueue queue, Action<Exception> reportError)
    {
        if (!Uri.TryCreate(prefix, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttp || !uri.IsLoopback
            || uri.AbsolutePath != "/" || !prefix.EndsWith('/'))
            throw new ArgumentException("The host listens on a loopback http://host:port/ prefix; publish it through an HTTPS reverse proxy.", nameof(prefix));
        this.keys = keys; this.queue = queue; this.reportError = reportError;
        listener.Prefixes.Add(prefix);
        listener.Start();
        accepting = AcceptAsync();
    }

    async Task AcceptAsync()
    {
        while (!stopping.IsCancellationRequested)
        {
            HttpListenerContext context;
            try { context = await listener.GetContextAsync(); }
            catch (Exception) when (stopping.IsCancellationRequested) { return; }
            catch (Exception error) { reportError(error); return; }
            var handler = HandleAsync(context);
            lock (handlers) handlers.Add(handler);
            _ = handler.ContinueWith(done => { lock (handlers) handlers.Remove(done); }, TaskScheduler.Default);
        }
    }

    // Never faults: a client that disconnects mid-response only loses its own reply.
    async Task HandleAsync(HttpListenerContext context)
    {
        var response = context.Response;
        try { await RespondAsync(context.Request, response); }
        catch (Exception error) when (error is HttpListenerException or IOException or ObjectDisposedException
            || error is OperationCanceledException && stopping.IsCancellationRequested) { }
        catch (Exception error)
        {
            reportError(error);
            try { response.StatusCode = 500; } catch (InvalidOperationException) { }
        }
        finally
        {
            try { response.Close(); } catch (Exception error) when (error is HttpListenerException or ObjectDisposedException) { }
        }
    }

    async Task RespondAsync(HttpListenerRequest request, HttpListenerResponse response)
    {
        response.Headers["Cache-Control"] = "no-store";
        try
        {
            await RouteAsync(request, response);
        }
        catch (KeyNotFoundException) { response.StatusCode = 404; }
        catch (LimitExceededException error) { await Problem(response, 413, error.Message); }
        catch (JsonException) { await Problem(response, 400, "The request body is not valid JSON for this route."); }
        catch (Exception error) when (error is ArgumentException or InvalidDataException) { await Problem(response, 400, error.Message); }
        catch (IOException error)
        {
            reportError(error);
            await Problem(response, 500, "The processing host could not read or write job data. Contact the provider with the job ID.");
        }
        catch (QueueCapacityException error)
        {
            response.Headers["Retry-After"] = "30";
            await Problem(response, 429, error.Message);
        }
        catch (InvalidOperationException error) when (error is not ObjectDisposedException) { await Problem(response, 409, error.Message); }
        catch (OperationCanceledException) when (!stopping.IsCancellationRequested)
        {
            await Problem(response, 409, "The capture was cancelled during the upload.");
        }
    }

    async Task RouteAsync(HttpListenerRequest request, HttpListenerResponse response)
    {
        var path = request.Url!.AbsolutePath; var method = request.HttpMethod;
        if (path == "/health")
        {
            if (method != "GET") { response.StatusCode = 405; return; }
            await Write(response, 200, new { status = "ok" }); return;
        }
        if (path != "/v1" && !path.StartsWith("/v1/", StringComparison.Ordinal)) { response.StatusCode = 404; return; }
        // Compressed bodies could expand past every byte limit, so only identity encoding is accepted.
        if (request.Headers["Content-Encoding"] is { Length: > 0 } encoding && !encoding.Equals("identity", StringComparison.OrdinalIgnoreCase))
        { response.StatusCode = 415; return; }
        var authorization = request.Headers["Authorization"] ?? "";
        if (path == "/v1/settings")
        {
            if (!keys.IsAdmin(authorization)) { response.StatusCode = 401; return; }
            if (method == "PUT") queue.SetConcurrentJobs((await ReadBody<HostSettings>(request)).ConcurrentJobs);
            else if (method != "GET") { response.StatusCode = 405; return; }
            await Write(response, 200, Settings(queue));
            return;
        }
        if (keys.Owner(authorization) is not { } owner) { response.StatusCode = 401; return; }
        var segments = path.Split('/', StringSplitOptions.RemoveEmptyEntries);
        switch (segments)
        {
            case ["v1", "capabilities"]:
                if (method != "GET") { response.StatusCode = 405; return; }
                await Write(response, 200, queue.Capabilities());
                return;
            case ["v1", "jobs"]:
                if (method == "GET") { await Write(response, 200, queue.List(owner)); return; }
                if (method != "POST") { response.StatusCode = 405; return; }
                var job = await ReadBody<HostJobRequest>(request);
                var options = job.Options; var name = job.DisplayName();
                if (!queue.IsAvailable) { await Problem(response, 503, "The host's video processor is not configured."); return; }
                await Write(response, 201, queue.Create(owner, options, name, ApiKeys.IsSignedAccount(owner) ? job.Policy : null));
                return;
            case ["v1", "jobs", var text] when Guid.TryParse(text, out var id):
                if (method == "GET")
                {
                    var status = queue.Status(owner, id);
                    if (status.State is "queued" or "processing") response.Headers["Retry-After"] = "2";
                    await Write(response, 200, status);
                }
                else if (method == "DELETE") { queue.Cancel(owner, id); response.StatusCode = 204; }
                else response.StatusCode = 405;
                return;
            case ["v1", "jobs", var text, "video"] when Guid.TryParse(text, out var id):
                if (method != "PUT") { response.StatusCode = 405; return; }
                if (request.ContentLength64 > queue.MaximumUploadBytes) { response.StatusCode = 413; return; }
                await queue.UploadAsync(owner, id, request.Headers["X-Video-Extension"] ?? "", request.InputStream, stopping.Token);
                response.StatusCode = 202;
                return;
            case ["v1", "jobs", var text, "result"] when Guid.TryParse(text, out var id):
                if (method != "GET") { response.StatusCode = 405; return; }
                await using (var result = queue.OpenResult(owner, id))
                {
                    response.StatusCode = 200; response.ContentType = "application/json";
                    response.Headers["Content-Disposition"] = "attachment; filename=capture.hmotion";
                    response.ContentLength64 = result.Length;
                    await result.CopyToAsync(response.OutputStream, stopping.Token);
                }
                return;
            default:
                response.StatusCode = 404;
                return;
        }
    }

    /// <summary>The operator view of host settings, shared with the ASP.NET entry.</summary>
    internal static object Settings(CaptureQueue queue) => new { concurrentJobs = queue.ConcurrentJobs, maximumConcurrentJobs = CaptureQueue.MaximumConcurrentJobs };

    async Task<T> ReadBody<T>(HttpListenerRequest request)
    {
        using var body = new MemoryStream();
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(stopping.Token);
        timeout.CancelAfter(BodyTimeout);
        try { await CaptureQueue.CopyBounded(request.InputStream, body, MaximumOptionsBytes, timeout.Token); }
        catch (OperationCanceledException) when (!stopping.IsCancellationRequested)
        { throw new InvalidDataException("The request body took too long to arrive."); }
        body.Position = 0;
        return await JsonSerializer.DeserializeAsync<T>(body, Json) ?? throw new InvalidDataException("The request body is empty.");
    }

    static async Task Write<T>(HttpListenerResponse response, int status, T value)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(value, Json);
        response.StatusCode = status; response.ContentType = "application/json; charset=utf-8"; response.ContentLength64 = bytes.Length;
        await response.OutputStream.WriteAsync(bytes);
    }

    static Task Problem(HttpListenerResponse response, int status, string detail)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(new { title = ((HttpStatusCode)status).ToString(), status, detail }, Json);
        response.StatusCode = status; response.ContentType = "application/problem+json"; response.ContentLength64 = bytes.Length;
        return response.OutputStream.WriteAsync(bytes).AsTask();
    }

    /// <summary>Stops accepting requests, cancels in-flight uploads and waits for every handler.</summary>
    public async ValueTask DisposeAsync()
    {
        if (stopping.IsCancellationRequested) return;
        await stopping.CancelAsync();
        listener.Stop();
        await accepting;
        Task[] running;
        lock (handlers) running = handlers.ToArray();
        await Task.WhenAll(running);
        listener.Close(); stopping.Dispose();
    }
}
