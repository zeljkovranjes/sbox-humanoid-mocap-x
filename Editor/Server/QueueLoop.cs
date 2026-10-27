using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
namespace HumanoidMocapHost;

/// <summary>Runs queue processing and periodic cleanup without a framework host, so the
/// ASP.NET worker and an editor-embedded service share one loop.</summary>
internal static class QueueLoop
{
    static readonly TimeSpan CleanupInterval = TimeSpan.FromMinutes(1);

    /// <summary>Processes jobs until <paramref name="stopping"/> is cancelled. Processor failures
    /// go to <paramref name="reportFailure"/>; a failure of the loop itself propagates.</summary>
    internal static async Task RunAsync(CaptureQueue queue, Action<Guid, Exception> reportFailure, CancellationToken stopping)
    {
        using var lifetime = CancellationTokenSource.CreateLinkedTokenSource(stopping);
        var processing = queue.RunAsync(reportFailure, lifetime.Token);
        var cleanup = Cleanup(queue, lifetime.Token);
        await Task.WhenAny(processing, cleanup);
        await lifetime.CancelAsync();
        try { await Task.WhenAll(processing, cleanup); }
        catch (OperationCanceledException) when (stopping.IsCancellationRequested) { }
    }

    static async Task Cleanup(CaptureQueue queue, CancellationToken cancellation)
    {
        using var timer = new PeriodicTimer(CleanupInterval);
        while (await timer.WaitForNextTickAsync(cancellation)) queue.Prune();
    }
}
