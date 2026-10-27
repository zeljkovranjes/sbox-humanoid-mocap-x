using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.EditorTools.Inference;

namespace HumanoidMocapX.EditorTools;

/// <summary>Connects installed models and timestamped Windows decoding to in-editor
/// GPU inference. No network, external worker or runtime model download is used.</summary>
internal sealed class LocalCapture(CaptureOptions options, string models, string cache,
    Action<string, double?> reportProgress = null) : IBodyCapture
{
    internal const string LinuxUnavailable = "Local video capture needs Windows Media Foundation. Linux preview, saved-motion export and hosted capture remain available.";
    internal static string Availability => OperatingSystem.IsWindows() ? null : LinuxUnavailable;
    public string UnavailableReason => Availability;
    int active;

    public async Task<string> CaptureAsync(string video, double start, double? end, Action<string> progress, CancellationToken cancellationToken)
    {
        if (UnavailableReason is { } reason) throw new PlatformNotSupportedException(reason);
        var selected = new CaptureOptions { Model = options.Model, CameraInterval = options.CameraInterval, Lanes = options.Lanes };
        selected.Validate(); new HostedCaptureOptions(selected.Model, selected.CameraInterval, start, end).Validate();
        if (Interlocked.CompareExchange(ref active, 1, 0) != 0)
            throw new InvalidOperationException("Wait for the current local capture to finish.");
        try
        {
            void Report(string message) { progress?.Invoke(message); reportProgress?.Invoke(message, null); }
            using var frames = await Task.Run(async () =>
            {
                cancellationToken.ThrowIfCancellationRequested(); Report("Opening video decoder…");
                using var decoder = new WindowsVideoDecoder(video);
                var duration = decoder.Duration;
                if (start > 0) decoder.Seek(start);
                return await VideoFrames.CreateAsync(decoder.Read, duration, start, end, cache, Report, cancellationToken);
            }, cancellationToken);
            var installed = await ModelStore.InstalledAsync(models, selected.Model, Report, cancellationToken);
            var motion = await new FlowVideoProcessor(installed, selected, progress, reportProgress).RunAsync(frames, cancellationToken);
            motion.Video = video; motion.Name = Path.GetFileNameWithoutExtension(video); motion.StartTime = start;
            var result = await Task.Run(() => Save(motion, cancellationToken), cancellationToken);
            reportProgress?.Invoke("Motion is ready", 1);
            return result;
        }
        finally { Volatile.Write(ref active, 0); }
    }

    async Task<string> Save(FlowMotion motion, CancellationToken token)
    {
        Directory.CreateDirectory(cache);
        var path = Path.Combine(cache, Guid.NewGuid().ToString("N") + ".hmotion"); var partial = path + ".partial";
        try
        {
            var document = motion.ToDocument();
            await using (var output = new FileStream(partial, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, true))
                await JsonSerializer.SerializeAsync(output, document, MotionDocument.JsonOptions, token);
            token.ThrowIfCancellationRequested(); File.Move(partial, path);
            return path;
        }
        finally { if (File.Exists(partial)) File.Delete(partial); }
    }
}
