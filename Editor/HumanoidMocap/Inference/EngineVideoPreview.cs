using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX.Core.Inference;
using Sandbox;

namespace HumanoidMocapX.EditorTools.Inference;

/// <summary>Editor-thread adapter for s&amp;box video playback. Pixel callbacks are copied
/// into one synchronized preview buffer; this is not a frame-accurate inference decoder.</summary>
internal sealed class EngineVideoPreview : IDisposable
{
    const double LoadTimeoutSeconds = 30;
    readonly string source;
    readonly BaseFileSystem filesystem;
    readonly CancellationTokenSource stop = new();
    readonly object sync = new();
    readonly PreviewFrameBuffer frame = new();
    readonly PreviewPlayback playback = new();
    VideoPlayer player;
    long deliveredVersion, started;
    bool loading = true, loaded, disposed, stoppedForFailure;
    string failure;
    internal double Duration { get; private set; }
    internal string Error { get { lock (sync) return failure; } }

    internal EngineVideoPreview(string source)
    {
        this.source = source;
        filesystem = global::Editor.FileSystem.ProjectTemporary;
        _ = LoadAsync();
    }

    async Task LoadAsync()
    {
        try
        {
            var path = await VideoFiles.CacheVideoAsync(source, filesystem, stop.Token);
            await EditorPipeline.SwitchToMainThread();
            if (disposed) return;
            player = new VideoPlayer { OnLoaded = Loaded, OnTextureData = ReceivePixels, Repeat = false, Muted = true };
            started = Stopwatch.GetTimestamp();
            player.Play(filesystem, path);
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested) { }
        catch (Exception error)
        {
            await EditorPipeline.SwitchToMainThread();
            Fail(error.Message);
        }
        finally
        {
            await EditorPipeline.SwitchToMainThread();
            loading = false;
            if (disposed) stop.Dispose();
        }
    }

    void Loaded()
    {
        if (disposed) return;
        Duration = player.Duration;
        if (!double.IsFinite(Duration) || Duration <= 0) { Fail("The engine returned no valid video duration."); return; }
        player.Pause(); loaded = true;
    }

    void ReceivePixels(ReadOnlySpan<byte> pixels, Vector2 size)
    {
        lock (sync)
        {
            if (disposed || failure is not null) return;
            try { frame.Copy(pixels, (int)size.x, (int)size.y); }
            catch (Exception error) { Fail(error.Message); }
        }
    }

    /// <summary>Presents the requested preview time and delivers changed pixels on the
    /// editor thread. The callback borrows the buffer only for the duration of the call.</summary>
    internal void Present(double seconds, bool isPlaying, Action<PreviewFrameBuffer, double> receive)
    {
        if (disposed || player is null) return;
        try
        {
            if (Error is not null)
            {
                if (!stoppedForFailure) { player.Stop(); stoppedForFailure = true; }
                return;
            }
            if (!double.IsFinite(seconds) || seconds < 0) throw new ArgumentOutOfRangeException(nameof(seconds));
            if (loaded)
            {
                if (playback.SeekTime(seconds, Duration, player.PlaybackTime, isPlaying) is float target) player.Seek(target);
                if (isPlaying && player.IsPaused) player.Resume();
                else if (!isPlaying && !player.IsPaused) player.Pause();
            }
            player.Present();
            var time = player.PlaybackTime;
            lock (sync)
            {
                if ((!loaded || frame.Version == 0) && Stopwatch.GetElapsedTime(started).TotalSeconds > LoadTimeoutSeconds)
                    Fail("The engine did not decode a preview frame within 30 seconds.");
                if (frame.Version == deliveredVersion) return;
                receive(frame, time); deliveredVersion = frame.Version;
            }
        }
        catch (Exception error) { Fail(error.Message); }
    }

    void Fail(string message) { lock (sync) failure = "Video preview (" + source + "): " + message; }

    public void Dispose()
    {
        lock (sync) { if (disposed) return; disposed = true; }
        stop.Cancel();
        if (player is not null)
        {
            player.OnLoaded = null; player.OnTextureData = null;
            var texture = player.Texture;
            player.Dispose(); texture.Dispose(); player = null;
        }
        if (!loading) stop.Dispose();
    }
}
