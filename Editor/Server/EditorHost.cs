using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Editor;
using HumanoidMocapX.EditorTools.Video;
using HumanoidMocapX;
using HumanoidMocapX.EditorTools;
using Sandbox;

namespace HumanoidMocapHost;

/// <summary>Opt-in processing host inside the s&amp;box editor. Operator configuration
/// comes from MOCAP_* environment variables; see docs/editor-host.md.</summary>
public static class EditorHost
{
    // The engine exposes static console/event entry points; session state stays here.
    static readonly SemaphoreSlim Changes = new(1, 1);
    static EditorHostLifetime active;
    static CancellationTokenSource starting;

    /// <summary>Verifies installed Base/Latest weights, then listens on loopback. Requires
    /// Windows Media Foundation and a GPU budget. Startup failures propagate to the console.</summary>
    [ConCmd("mocap_x_host_start", ConVarFlags.Protected, Help = "Start the configured in-editor Humanoid Mocap X processing host")]
    public static async Task StartAsync()
    {
        await Changes.WaitAsync();
        using var cancellation = new CancellationTokenSource();
        starting = cancellation;
        try
        {
            if (active is { IsRunning: true }) { Log.Info("Humanoid Mocap X host is already running."); return; }
            if (active is not null) { await active.StopAsync(); active = null; }
            if (LocalCapture.Availability is { } reason) throw new PlatformNotSupportedException(reason);
            var config = HostConfiguration.Read(Environment.GetEnvironmentVariable,
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HumanoidMocapX"));
            await ModelStore.InstalledAsync(config.Models, FlowModel.Base, message => Log.Info(message), cancellation.Token);
            await ModelStore.InstalledAsync(config.Models, FlowModel.Latest, message => Log.Info(message), cancellation.Token);
            await GameTask.MainThread(); cancellation.Token.ThrowIfCancellationRequested();
            if (!Graphics.IsAvailable || Graphics.VideoMemoryBudget == 0 && Sandbox.Engine.SystemInfo.GpuMemory == 0)
                throw new InvalidOperationException("The host needs an initialized GPU with a reported video-memory budget.");
            active = Start(config);
            Log.Info($"Humanoid Mocap X host listening on http://127.0.0.1:{config.Port}/. Stop with mocap_x_host_stop.");
        }
        finally { starting = null; Changes.Release(); }
    }

    static EditorHostLifetime Start(HostConfiguration config)
    {
        var lanes = config.Lanes == CaptureLanes.Auto ? CaptureLanes.Maximum : config.Lanes;
        var admission = new GpuAdmission(lanes);
        admission.Update(Graphics.VideoMemoryBudget, Graphics.VideoMemoryUsed, Sandbox.Engine.SystemInfo.GpuMemory);
        var storage = new FileJobStorage(config.Data);
        try
        {
            var processor = new CaptureProcessor(storage.CaptureDirectory, config.UploadBytes,
                (selected, workspace, progress) => new LocalCapture(new CaptureOptions
                { Model = selected.Model, CameraInterval = selected.CameraInterval, Lanes = lanes }, config.Models, workspace, progress),
                ReadDurationAsync);
            var queue = new CaptureQueue(storage, processor, TimeProvider.System, config.ActivePerOwner,
                maximumUploadBytes: config.UploadBytes, maximumClipSeconds: config.ClipSeconds,
                concurrentJobs: config.ConcurrentJobs, canStartJob: admission.CanStart);
            var session = new HostSession($"http://127.0.0.1:{config.Port}/", config.Keys, storage, queue, ReportError);
            return new EditorHostLifetime(session, admission);
        }
        catch { storage.Dispose(); throw; }
    }

    static Task<double> ReadDurationAsync(string video, CancellationToken token) => Task.Run(() =>
    {
        token.ThrowIfCancellationRequested();
        using var decoder = IVideoDecoder.Open(video);
        return decoder.Duration;
    }, token);

    /// <summary>Cancels startup, uploads and active captures, then waits for GPU cleanup.
    /// Repeated calls are safe; completed job IDs are invalidated by the next start.</summary>
    [ConCmd("mocap_x_host_stop", ConVarFlags.Protected, Help = "Stop and drain the in-editor Humanoid Mocap X processing host")]
    public static async Task StopAsync()
    {
        try { starting?.Cancel(); } catch (ObjectDisposedException) { } // Startup just finished.
        await Changes.WaitAsync();
        try
        {
            await GameTask.MainThread();
            if (active is not null) { await active.StopAsync(); active = null; }
        }
        finally { Changes.Release(); }
    }

    [EditorEvent.Hotload]
    static void Hotloaded() => _ = StopAfterHotloadAsync();

    static async Task StopAfterHotloadAsync()
    {
        try { await StopAsync(); }
        catch (Exception error) { ReportError(error); }
    }

    static void ReportError(Exception error) => Log.Error($"Humanoid Mocap X host: {error}");
}
