using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.EditorTools.Video;
using HumanoidMocapX.Inference;

namespace HumanoidMocapX.EditorTools;

/// <summary>Connects installed models and timestamped decoding (Windows Media Foundation, or
/// the system FFmpeg elsewhere) to in-editor GPU inference. No network, capture worker or
/// runtime model download is used.</summary>
internal sealed class LocalCapture(CaptureOptions options, string models, string cache,
    Action<string, double?> reportProgress = null) : IBodyCapture
{
    internal const string FfmpegMissing = "Local video capture on Linux needs FFmpeg: install it so ffmpeg and ffprobe are on PATH.";
    internal static string Availability => OperatingSystem.IsWindows() || FfmpegVideoDecoder.IsInstalled ? null : FfmpegMissing;
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
                using var decoder = IVideoDecoder.Open(video);
                var duration = decoder.Duration;
                if (start > 0) decoder.Seek(start);
                return await VideoFrames.CreateAsync(decoder.Read, duration, start, end, cache, Report, cancellationToken);
            }, cancellationToken);
            var installed = await ModelStore.InstalledAsync(models, selected.Model, Report, cancellationToken);
            var (motion, person, detail) = await new FlowVideoProcessor(installed, selected, progress, reportProgress).RunAsync(frames, cancellationToken);
            motion.Video = video; motion.Name = Path.GetFileNameWithoutExtension(video); motion.StartTime += start;
            var note = person.Note(start, frames.Count);
            var result = await Task.Run(() => Save(motion, detail, note, cancellationToken), cancellationToken);
            reportProgress?.Invoke("Motion is ready", 1);
            return result;
        }
        finally { Volatile.Write(ref active, 0); }
    }

    async Task<string> Save(FlowMotion motion, ObservedDetail detail, string note, CancellationToken token)
    {
        Directory.CreateDirectory(cache);
        var path = Path.Combine(cache, Guid.NewGuid().ToString("N") + ".hmotion"); var partial = path + ".partial";
        try
        {
            var document = motion.ToDocument(); detail.Apply(document);
            if (note is not null) document.Diagnostics.Add(note);
            await using (var output = new FileStream(partial, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, true))
                await JsonSerializer.SerializeAsync(output, document, MotionDocument.JsonOptions, token);
            token.ThrowIfCancellationRequested(); File.Move(partial, path);
            return path;
        }
        finally { if (File.Exists(partial)) File.Delete(partial); }
    }
}
