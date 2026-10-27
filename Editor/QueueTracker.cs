using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace HumanoidMocapX.EditorTools;

/// <summary>The account's jobs on the provider, refreshed every few seconds while signed in.
/// Videos added beyond the provider's per-account limit wait here and upload one at a time as
/// jobs finish. Raises <see cref="Changed"/> on the editor's main thread.</summary>
sealed class QueueTracker : IDisposable
{
    static readonly TimeSpan PollInterval = TimeSpan.FromSeconds(3);
    readonly HostingSettings settings;
    readonly HostedCapture capture;
    readonly Func<string, Task> openMotion;
    readonly List<string> waiting = new();
    // Local paths of videos sent this session, so opened results can preview their source.
    readonly Dictionary<Guid, string> sources = new();
    readonly CancellationTokenSource lifetime = new();
    bool unreachable;

    internal QueueTracker(HostingSettings settings, HostedCapture capture, Func<string, Task> openMotion)
    {
        this.settings = settings; this.capture = capture; this.openMotion = openMotion;
        settings.Changed += SignInChanged;
        _ = PollAsync();
    }

    internal event Action Changed;
    internal HostedJob[] Jobs { get; private set; } = Array.Empty<HostedJob>();
    internal IReadOnlyList<string> Waiting => waiting;
    internal string Uploading { get; private set; }
    internal HostedCapabilities Limits { get; private set; }
    internal string Problem { get; private set; }
    internal bool Ready => settings.SignedIn && settings.Profile is { CanProcess: true };
    internal int Active => Jobs.Count(job => !Finished(job)) + waiting.Count + (Uploading is null ? 0 : 1);

    internal static bool Finished(HostedJob job) => job.State is "completed" or "failed" or "cancelled";

    void SignInChanged() { Limits = null; Changed?.Invoke(); }

    internal void Add(IEnumerable<string> videos)
    {
        Problem = null;
        waiting.AddRange(videos.Where(File.Exists).Except(waiting));
        Changed?.Invoke();
    }

    internal void Remove(string video) { waiting.Remove(video); Changed?.Invoke(); }

    async Task PollAsync()
    {
        var token = lifetime.Token;
        while (!token.IsCancellationRequested)
        {
            if (Ready)
            {
                try
                {
                    Limits ??= await capture.CapabilitiesAsync(token);
                    var current = await capture.ListAsync(token);
                    await EditorPipeline.SwitchToMainThread();
                    if (token.IsCancellationRequested) return;
                    Jobs = current;
                    if (unreachable) { unreachable = false; Problem = null; }
                    await SubmitWaitingAsync(token);
                }
                catch (Exception failure) when (failure is HttpRequestException or InvalidDataException or ArgumentException
                    or System.Text.Json.JsonException || failure is OperationCanceledException && !token.IsCancellationRequested)
                {
                    await EditorPipeline.SwitchToMainThread();
                    if (token.IsCancellationRequested) return;
                    unreachable = true;
                    Problem = failure is OperationCanceledException ? "The provider did not answer in time." : failure.Message;
                }
            }
            await EditorPipeline.SwitchToMainThread();
            if (token.IsCancellationRequested) return;
            Changed?.Invoke();
            try { await Task.Delay(PollInterval, token); } catch (OperationCanceledException) { return; }
        }
    }

    // One upload at a time, and only while the account is below the provider's job limit.
    async Task SubmitWaitingAsync(CancellationToken token)
    {
        while (waiting.Count > 0 && Uploading is null && Jobs.Count(job => !Finished(job)) < (Limits?.MaximumActiveJobs ?? 1))
        {
            var video = waiting[0]; waiting.RemoveAt(0); Uploading = video; Changed?.Invoke();
            try
            {
                var job = await capture.SubmitAsync(video, 0, null, token);
                await EditorPipeline.SwitchToMainThread();
                sources[job.Id] = video;
                Jobs = await capture.ListAsync(token);
            }
            // One refused video (too long, too large, unreadable) must not hold up the rest.
            catch (Exception failure) when (failure is HttpRequestException or InvalidDataException or IOException or ArgumentException)
            {
                await EditorPipeline.SwitchToMainThread();
                Problem = Path.GetFileName(video) + ": " + failure.Message;
            }
            finally { await EditorPipeline.SwitchToMainThread(); Uploading = null; }
        }
    }

    internal async Task CancelAsync(HostedJob job)
    {
        try { await capture.CancelAsync(job.Id, lifetime.Token); }
        catch (HttpRequestException failure) when (failure.StatusCode != System.Net.HttpStatusCode.NotFound)
        { await EditorPipeline.SwitchToMainThread(); Problem = failure.Message; Changed?.Invoke(); return; }
        await EditorPipeline.SwitchToMainThread();
        Jobs = Jobs.Where(entry => entry.Id != job.Id).ToArray(); sources.Remove(job.Id); Changed?.Invoke();
    }

    internal async Task OpenAsync(HostedJob job)
    {
        try
        {
            var path = await capture.DownloadAsync(job.Id, sources.GetValueOrDefault(job.Id), lifetime.Token);
            await EditorPipeline.SwitchToMainThread();
            await openMotion(path);
        }
        catch (Exception failure) when (failure is HttpRequestException or InvalidDataException or IOException or FormatException
            or System.Text.Json.JsonException)
        { await EditorPipeline.SwitchToMainThread(); Problem = failure.Message; Changed?.Invoke(); }
    }

    public void Dispose()
    {
        settings.Changed -= SignInChanged; lifetime.Cancel();
    }
}
