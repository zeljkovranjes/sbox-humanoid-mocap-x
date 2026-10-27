#nullable enable annotations

using System;
using System.Threading;
using System.Threading.Tasks;

namespace HumanoidMocapHost;

/// <summary>Owns one HTTP listener, queue loop and file-store lifetime. Shutdown drains
/// uploads and GPU captures before disposing their resources.</summary>
internal sealed class HostSession : IAsyncDisposable
{
    readonly object gate = new();
    readonly CancellationTokenSource stopping = new();
    readonly FileJobStorage storage;
    readonly CaptureQueue queue;
    readonly HttpHost http;
    readonly Task worker;
    readonly Action<Exception> reportError;
    Task? shutdown;
    internal bool IsStopping { get { lock (gate) return shutdown is not null; } }

    internal HostSession(string prefix, ApiKeys keys, FileJobStorage storage, CaptureQueue queue, Action<Exception> reportError)
    {
        this.storage = storage; this.queue = queue; this.reportError = reportError;
        try { http = new HttpHost(prefix, keys, queue, reportError); }
        catch { stopping.Dispose(); queue.Dispose(); storage.Dispose(); throw; }
        worker = Task.Run(() => QueueLoop.RunAsync(queue, (id, error) => reportError(new InvalidOperationException($"Capture {id} failed.", error)), stopping.Token));
        _ = ObserveAsync();
    }

    async Task ObserveAsync()
    {
        try { await worker; }
        catch (Exception error) { reportError(error); }
        finally
        {
            try { await StopAsync(); }
            catch (Exception error) { reportError(error); }
        }
    }

    /// <summary>Idempotently stops this instance. The returned task completes only after
    /// cancellation and resource cleanup; callers must await it before starting a replacement.</summary>
    internal Task StopAsync()
    {
        lock (gate) return shutdown ??= StopCoreAsync();
    }

    async Task StopCoreAsync()
    {
        await stopping.CancelAsync();
        try { await http.DisposeAsync(); }
        finally
        {
            try { await worker; }
            finally
            {
                try { queue.Dispose(); }
                finally
                {
                    try { storage.DeleteRun(); }
                    finally { storage.Dispose(); stopping.Dispose(); }
                }
            }
        }
    }

    public ValueTask DisposeAsync() => new(StopAsync());
}
