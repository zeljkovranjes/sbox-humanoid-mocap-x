using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocap.Core.Mapping;
using HumanoidMocap.Core.Motion;

namespace HumanoidMocapX.EditorTools;

/// <summary>Talks to the signed-in provider: streams captures, lists and cancels the account's
/// queued jobs, and validates returned motion before import.</summary>
internal sealed class HostedCapture : IBodyCapture, IDisposable
{
    const long MaximumResultBytes = 128L * 1024 * 1024;
    // Editors trim by frame times, so allow the same rounding the provider does.
    const double ClipTolerance = 0.01;
    readonly HostingSettings settings;
    readonly CaptureOptions options;
    readonly string cache;
    readonly HttpClient http;
    readonly Func<TimeSpan, CancellationToken, Task> delay;
    readonly JsonSerializerOptions json = new(JsonSerializerDefaults.Web) { Converters = { new JsonStringEnumConverter() } };
    readonly CancellationTokenSource lifetime = new();
    readonly object gate = new();
    bool active, disposed;

    internal HostedCapture(HostingSettings settings, CaptureOptions options, string cache, HttpMessageHandler handler = null,
        Func<TimeSpan, CancellationToken, Task> delay = null)
    {
        this.settings = settings; this.options = options; this.cache = cache;
        http = new(handler ?? new HttpClientHandler { AllowAutoRedirect = false }) { Timeout = TimeSpan.FromMinutes(30) };
        this.delay = delay ?? Task.Delay;
    }

    public string UnavailableReason
    {
        get
        {
            try { settings.Endpoint(); settings.RequireSignedIn(); }
            catch (ArgumentException error) { return error.Message; }
            return settings.Profile is { CanProcess: false } profile ? NotAllowed(profile) : null;
        }
    }

    internal static string NotAllowed(HostProfile profile) => profile.InGuild == false
        ? "Join the provider's Discord server to process videos." : "This account is not whitelisted on this provider yet.";

    /// <summary>The provider's limits and queue size.</summary>
    internal Task<HostedCapabilities> CapabilitiesAsync(CancellationToken token)
        => Read<HostedCapabilities>(Endpoint(), settings.Token, "v1/capabilities", token);

    /// <summary>The account's jobs on the provider, oldest first, with queue positions and estimates.</summary>
    internal async Task<HostedJob[]> ListAsync(CancellationToken token)
    {
        using var request = Message(Endpoint(), settings.Token, HttpMethod.Get, "v1/jobs");
        using var response = await Send(request, token);
        await response.Content.LoadIntoBufferAsync(262144, token);
        return JsonSerializer.Deserialize<HostedJob[]>(await response.Content.ReadAsStringAsync(token), json)
            ?? throw new InvalidDataException("The host returned an invalid job list.");
    }

    /// <summary>Creates a job and uploads the video; processing continues on the provider. Fails
    /// with the provider's message when the account's job limit is reached.</summary>
    internal async Task<HostedJob> SubmitAsync(string video, double start, double? end, CancellationToken token)
    {
        var endpoint = Endpoint(); var key = settings.Token;
        await using var input = new FileStream(video, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, true);
        var job = await Create(endpoint, key, video, input.Length, start, end, token);
        await Upload(endpoint, key, job.Id, video, input, token);
        return job;
    }

    /// <summary>Cancels a waiting job or removes a finished one. The provider refuses to cancel a job
    /// that is already processing.</summary>
    internal async Task CancelAsync(Guid job, CancellationToken token)
    {
        using var request = Message(Endpoint(), settings.Token, HttpMethod.Delete, "v1/jobs/" + job.ToString("D"));
        using var response = await Send(request, token);
    }

    /// <summary>Downloads and validates a completed job's motion; returns the saved .hmotion path.</summary>
    internal Task<string> DownloadAsync(Guid job, string video, CancellationToken token)
        => Task.Run(() => Download(Endpoint(), settings.Token, "v1/jobs/" + job.ToString("D") + "/result", video, token), token);

    Uri Endpoint() { var endpoint = settings.Endpoint(); settings.RequireSignedIn(); return endpoint; }

    public async Task<string> CaptureAsync(string video, double start, double? end,
        Action<string> progress, CancellationToken cancellationToken)
    {
        var endpoint = Endpoint(); var key = settings.Token;
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (active) throw new InvalidOperationException("Wait for the current hosted capture to finish.");
            active = true;
        }
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, lifetime.Token);
        linked.CancelAfter(TimeSpan.FromHours(2));
        var token = linked.Token; Guid? job = null;
        try
        {
            await using var input = new FileStream(video, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, true);
            job = (await Create(endpoint, key, video, input.Length, start, end, token)).Id;
            var path = "v1/jobs/" + job.Value.ToString("D");
            progress?.Invoke("Uploading video to " + endpoint.Host + "…");
            await Upload(endpoint, key, job.Value, video, input, token);
            while (true)
            {
                var (status, interval) = await ReadStatus(endpoint, key, path, progress, token);
                if (status.Id != job || status.Message is null || status.Message.Length > 200
                    || status.QueuePosition is < 1 or > 1_000_000 || status.QueuePosition.HasValue && status.State != "queued")
                    throw new InvalidDataException("The host returned an invalid job status.");
                progress?.Invoke(Describe(status));
                if (status.State == "completed") break;
                if (status.State is "failed" or "cancelled" or "cancelling")
                    throw new InvalidOperationException("Hosted capture " + status.State + ". Job: " + job);
                if (status.State is not ("queued" or "processing")) throw new InvalidDataException("Unexpected hosted capture state: " + status.State);
                await delay(interval, token);
            }
            return await Task.Run(() => Download(endpoint, key, path + "/result", video, token), token);
        }
        finally
        {
            try { if (job.HasValue) await RemoveJob(endpoint, key, job.Value, progress); }
            finally { lock (gate) { active = false; if (disposed) Release(); } }
        }
    }

    /// <summary>One status line: message, place in line, progress and the provider's time estimates.</summary>
    internal static string Describe(HostedJob job)
    {
        var text = job.Message;
        if (job.QueuePosition is { } position) text += $" · {position}" + (job.QueueLength >= position ? $" of {job.QueueLength} in line" : " in line")
            + (job.Priority ? " (priority)" : "");
        if (job.State == "processing" && job.Progress is >= 0 and <= 1) text += $" · {Math.Round(job.Progress.Value * 100)}%";
        if (job.State == "queued" && job.SecondsUntilStart is { } start) text += " · starts in " + Duration(start);
        if (job.State is "queued" or "processing" && job.SecondsRemaining is { } left) text += " · about " + Duration(left) + " left";
        return text;
    }

    /// <summary>Human-friendly durations for estimates: "45 s", "12 min", "1 h 5 min".</summary>
    internal static string Duration(int seconds)
    {
        seconds = Math.Max(0, seconds);
        if (seconds < 60) return seconds + " s";
        var minutes = (int)Math.Round(seconds / 60.0);
        return minutes < 60 ? minutes + " min" : $"{minutes / 60} h {minutes % 60} min";
    }

    async Task<HostedJob> Create(Uri endpoint, string key, string video, long bytes, double start, double? end, CancellationToken token)
    {
        var request = new HostedCaptureOptions(options.Model, options.CameraInterval, start, end); request.Validate();
        var extension = Path.GetExtension(video).ToLowerInvariant();
        if (extension is not (".mp4" or ".mov" or ".m4v" or ".avi" or ".mkv" or ".webm"))
            throw new ArgumentException("Select a supported video file.", nameof(video));
        var capabilities = await Read<HostedCapabilities>(endpoint, key, "v1/capabilities", token);
        if (capabilities.Protocol != 1 || !capabilities.ProcessingAvailable)
            throw new InvalidOperationException("The provider's video processor is not ready or uses an unsupported protocol.");
        var maximum = capabilities.MaximumUploadBytes;
        if (maximum < 1 || bytes < 1 || bytes > maximum)
            throw new InvalidDataException($"The video is empty or larger than the {maximum / (1024 * 1024)} MiB this account may upload.");
        if (capabilities.MaximumClipSeconds is { } clip && end - start > clip + ClipTolerance)
            throw new InvalidDataException($"Clips are limited to {clip:0.##} seconds for this account. Shorten the range and try again.");
        using var create = Message(endpoint, key, HttpMethod.Post, "v1/jobs");
        var name = Path.GetFileName(video); if (name.Length > 120) name = name[..120];
        create.Content = new StringContent(JsonSerializer.Serialize(new { request.Model, request.CameraInterval, request.Start, request.End, name }, json),
            Encoding.UTF8, "application/json");
        using var response = await Send(create, token);
        var created = await Json<HostedJob>(response, token);
        if (created.Id == Guid.Empty || created.State != "awaiting-upload") throw new InvalidDataException("The host returned an invalid capture job.");
        return created;
    }

    async Task Upload(Uri endpoint, string key, Guid job, string video, Stream input, CancellationToken token)
    {
        using var upload = Message(endpoint, key, HttpMethod.Put, "v1/jobs/" + job.ToString("D") + "/video");
        upload.Headers.Add("X-Video-Extension", Path.GetExtension(video).ToLowerInvariant());
        upload.Content = new StreamContent(input, 65536);
        upload.Content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        using var uploaded = await Send(upload, token);
    }

    async Task RemoveJob(Uri endpoint, string key, Guid job, Action<string> progress)
    {
        using var cleanup = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        try
        {
            using var delete = Message(endpoint, key, HttpMethod.Delete, "v1/jobs/" + job.ToString("D"));
            using var response = await http.SendAsync(delete, cleanup.Token);
            if (response.StatusCode != HttpStatusCode.NotFound) response.EnsureSuccessStatusCode();
        }
        catch (Exception error) when (error is HttpRequestException or OperationCanceledException or IOException)
        { progress?.Invoke("Could not remove remote job " + job + ". The host may retain it until expiry."); }
    }

    async Task<string> Download(Uri endpoint, string key, string relative, string video, CancellationToken token)
    {
        Directory.CreateDirectory(cache);
        var final = Path.Combine(cache, Guid.NewGuid().ToString("N") + ".hmotion"); var partial = final + ".partial";
        try
        {
            using var request = Message(endpoint, key, HttpMethod.Get, relative);
            using var response = await Send(request, token);
            if (response.Content.Headers.ContentLength is > MaximumResultBytes) throw new InvalidDataException("The motion download exceeds the size limit.");
            await using (var input = await response.Content.ReadAsStreamAsync(token))
            await using (var output = new FileStream(partial, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, true))
            {
                var buffer = new byte[65536]; long total = 0;
                while (await input.ReadAsync(buffer, token) is var read && read > 0)
                {
                    total += read;
                    if (total > MaximumResultBytes) throw new InvalidDataException("The motion download exceeds the size limit.");
                    await output.WriteAsync(buffer.AsMemory(0, read), token);
                }
            }
            MotionDocument motion;
            await using (var stream = new FileStream(partial, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, true))
                motion = await JsonSerializer.DeserializeAsync<MotionDocument>(stream, MotionDocument.JsonOptions, token)
                    ?? throw new InvalidDataException("The host returned an empty motion document.");
            motion.Validate();
            if (!motion.Bones.Any(bone => bone.Role == BoneRole.Hips)) throw new InvalidDataException("Hosted capture must return full-body motion.");
            // Remote metadata cannot select a local sidecar file or preview source.
            motion.OriginalReconstruction = null; motion.SourceVideo = video;
            await using (var stream = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, 65536, true))
                await JsonSerializer.SerializeAsync(stream, motion, MotionDocument.JsonOptions, token);
            token.ThrowIfCancellationRequested(); File.Move(partial, final);
            return final;
        }
        finally { if (File.Exists(partial)) File.Delete(partial); }
    }

    async Task<T> Read<T>(Uri endpoint, string key, string path, CancellationToken token)
    {
        using var request = Message(endpoint, key, HttpMethod.Get, path);
        using var response = await Send(request, token);
        return await Json<T>(response, token);
    }

    async Task<(HostedJob Job, TimeSpan Interval)> ReadStatus(Uri endpoint, string key, string path,
        Action<string> progress, CancellationToken token)
    {
        const int maximumRetries = 6;
        for (var retry = 0; ; retry++)
        {
            using var request = Message(endpoint, key, HttpMethod.Get, path);
            HttpResponseMessage response;
            try { response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, token); }
            catch (HttpRequestException) when (retry < maximumRetries)
            {
                progress?.Invoke("Waiting for the processing host to respond…");
                await delay(TimeSpan.FromSeconds(2), token); continue;
            }
            using (response)
            {
                var interval = PollDelay(response.Headers.RetryAfter, DateTimeOffset.UtcNow);
                if (response.IsSuccessStatusCode) return (await Json<HostedJob>(response, token), interval);
                if (retry >= maximumRetries || response.StatusCode is not (HttpStatusCode.TooManyRequests
                    or HttpStatusCode.BadGateway or HttpStatusCode.ServiceUnavailable or HttpStatusCode.GatewayTimeout))
                    throw Failure(request, response.StatusCode);
                progress?.Invoke("The processing host is busy; waiting to check this job again…");
                await delay(interval, token);
            }
        }
    }

    internal static TimeSpan PollDelay(RetryConditionHeaderValue retry, DateTimeOffset now)
    {
        var seconds = retry?.Delta?.TotalSeconds ?? (retry?.Date is { } date ? (date - now).TotalSeconds : 1);
        return TimeSpan.FromSeconds(Math.Clamp(seconds, 1, 30));
    }

    async Task<T> Json<T>(HttpResponseMessage response, CancellationToken token)
    {
        await response.Content.LoadIntoBufferAsync(16384, token);
        return JsonSerializer.Deserialize<T>(await response.Content.ReadAsStringAsync(token), json)
            ?? throw new InvalidDataException("The host returned an empty API response.");
    }

    async Task<HttpResponseMessage> Send(HttpRequestMessage request, CancellationToken token)
    {
        var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, token);
        if (response.IsSuccessStatusCode) return response;
        using (response)
        {
            var detail = await ErrorMessage(response, token);
            if (response.StatusCode == HttpStatusCode.Unauthorized && request.Headers.Authorization?.Parameter is { } rejected) settings.Expire(rejected);
            throw Failure(request, response.StatusCode, detail);
        }
    }

    // The provider explains refusals (not whitelisted, job limit, already processing); show its words.
    static async Task<string> ErrorMessage(HttpResponseMessage response, CancellationToken token)
    {
        try
        {
            await response.Content.LoadIntoBufferAsync(4096, token);
            using var document = JsonDocument.Parse(await response.Content.ReadAsStringAsync(token));
            var root = document.RootElement;
            var message = root.TryGetProperty("error", out var error) && error.TryGetProperty("message", out var text) ? text.GetString()
                : root.TryGetProperty("detail", out var detail) ? detail.GetString() : null;
            return message is { Length: > 0 and <= 300 } ? message : null;
        }
        catch (Exception error) when (error is JsonException or HttpRequestException or InvalidOperationException) { return null; }
    }

    static HttpRequestException Failure(HttpRequestMessage request, HttpStatusCode status, string detail = null)
        => new(status == HttpStatusCode.Unauthorized ? "Your sign-in expired. Sign in to your provider again."
            : detail ?? "The provider returned HTTP " + (int)status + " for " + request.Method + " " + request.RequestUri.AbsolutePath, null, status);

    static HttpRequestMessage Message(Uri endpoint, string key, HttpMethod method, string path)
    {
        var request = new HttpRequestMessage(method, new Uri(endpoint, path));
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
        return request;
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true; lifetime.Cancel();
            if (!active) Release();
        }
    }

    void Release() { http.Dispose(); lifetime.Dispose(); }
}
