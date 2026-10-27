using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocap.Core.Motion;
using HumanoidMocapX;

namespace HumanoidMocapHost;

/// <summary>Adapts the library's capture pipeline to streamed hosted jobs. Each job owns
/// a temporary directory; the caller owns both streams. Capture and cleanup failures propagate.</summary>
internal sealed class CaptureProcessor : IClipProcessor, IClipInspector
{
    readonly string directory;
    readonly long maximumUploadBytes;
    readonly Func<HostedCaptureOptions, string, Action<string, double?>, IBodyCapture> create;
    readonly Func<string, CancellationToken, Task<double>>? duration;

    internal CaptureProcessor(string directory, long maximumUploadBytes,
        Func<HostedCaptureOptions, string, Action<string, double?>, IBodyCapture> create,
        Func<string, CancellationToken, Task<double>>? duration = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        ArgumentNullException.ThrowIfNull(create);
        if (maximumUploadBytes < 1) throw new ArgumentOutOfRangeException(nameof(maximumUploadBytes));
        this.directory = Path.GetFullPath(directory); this.maximumUploadBytes = maximumUploadBytes; this.create = create;
        this.duration = duration;
    }

    // The editor constructs this only after checking the decoder and installed models.
    public bool IsAvailable => true;

    public async Task<HostedCaptureOptions> InspectAsync(HostedCaptureOptions options, string video, CancellationToken cancellation)
    {
        options.Validate(); cancellation.ThrowIfCancellationRequested();
        if (duration is null) return options;
        var seconds = await duration(video, cancellation);
        if (!double.IsFinite(seconds) || seconds <= options.Start)
            throw new InvalidDataException("The uploaded video has no frames in the selected range.");
        return options with { End = Math.Min(options.End ?? seconds, seconds) };
    }

    public async Task ProcessAsync(HostedCaptureOptions options, string extension, Stream video, Stream motion,
        Action<string, double?> progress, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(options); options.Validate();
        ArgumentNullException.ThrowIfNull(video); ArgumentNullException.ThrowIfNull(motion);
        if (!video.CanRead || !motion.CanWrite) throw new ArgumentException("Capture needs a readable video and writable motion stream.");
        if (extension is not (".mp4" or ".mov" or ".m4v" or ".avi" or ".mkv" or ".webm"))
            throw new ArgumentException("Unsupported video extension.", nameof(extension));
        cancellation.ThrowIfCancellationRequested();
        var workspace = Path.Combine(directory, Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(workspace);
        try
        {
            var path = Path.Combine(workspace, "video" + extension);
            await using (var output = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, true))
                await CaptureQueue.CopyBounded(video, output, maximumUploadBytes, cancellation);
            var capture = create(options, workspace, progress);
            if (capture.UnavailableReason is { } reason) throw new InvalidOperationException(reason);
            var result = await capture.CaptureAsync(path, options.Start, options.End, static _ => { }, cancellation);
            await ExportAsync(workspace, result, motion, cancellation);
        }
        finally { Directory.Delete(workspace, true); }
    }

    static async Task ExportAsync(string workspace, string result, Stream destination, CancellationToken cancellation)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(result);
        var relative = Path.GetRelativePath(workspace, Path.GetFullPath(result));
        if (relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) || Path.IsPathRooted(relative))
            throw new InvalidDataException("The capture result is outside its job workspace.");
        await using var input = new FileStream(result, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, true);
        if (input.Length is < 1 or > CaptureQueue.MaximumResultBytes)
            throw new InvalidDataException("The capture result is empty or exceeds 128 MiB.");
        var document = await JsonSerializer.DeserializeAsync<MotionDocument>(input, MotionDocument.JsonOptions, cancellation)
            ?? throw new InvalidDataException("The capture returned no motion document.");
        document.Validate();
        // Local paths belong to the provider; only the reconstructed motion is downloadable.
        document.Name = "capture"; document.SourceVideo = ""; document.OriginalReconstruction = null;
        await JsonSerializer.SerializeAsync(destination, document, MotionDocument.JsonOptions, cancellation);
    }
}
