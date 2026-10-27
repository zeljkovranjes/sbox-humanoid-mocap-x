using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace HumanoidMocapX.EditorTools.Video;

/// <summary>Decodes with the system FFmpeg command-line tools (<c>ffmpeg</c>, <c>ffprobe</c>),
/// the Linux counterpart of <see cref="WindowsVideoDecoder"/>. The decoder process starts on the
/// first read and restarts on each seek. Dispose stops it.</summary>
public sealed class FfmpegVideoDecoder : IVideoDecoder
{
    const string Decoder = "ffmpeg", Probe = "ffprobe";
    // Decoding starts this far before a seek target so the frame displayed at it is included.
    const double SeekMargin = 1;
    readonly string path;
    readonly double origin;
    double start;
    Process process;
    Channel<(double Time, int Width, int Height)> frames;
    Queue<string> errors;
    Task stderr;

    /// <summary>Probes <paramref name="path"/>. Throws when it is missing, FFmpeg is not
    /// installed or the file has no video stream.</summary>
    public FfmpegVideoDecoder(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException("Video not found.", path);
        this.path = Path.GetFullPath(path);
        (origin, Duration) = RunProbe(this.path);
    }

    public double Duration { get; }

    /// <summary>Whether <c>ffmpeg</c> and <c>ffprobe</c> are on <c>PATH</c>.</summary>
    public static bool IsInstalled => OnPath(Decoder) && OnPath(Probe);

    public void Seek(double seconds) { Stop(); start = Math.Max(0, seconds); }

    public DecodedVideoFrame Read(CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        if (process is null) Start();
        if (!frames.Reader.WaitToReadAsync(token).AsTask().GetAwaiter().GetResult())
        {
            process.WaitForExit();
            if (process.ExitCode != 0) throw new InvalidDataException("ffmpeg failed: " + string.Join(" | ", errors));
            return null;
        }
        frames.Reader.TryRead(out var frame);
        // Each frame is a new array: the timeline builder keeps the current and next frames alive.
        var rgba = new byte[checked(frame.Width * frame.Height * 4)];
        process.StandardOutput.BaseStream.ReadExactly(rgba);
        return new DecodedVideoFrame(rgba, frame.Width, frame.Height, frame.Time - origin);
    }

    void Start()
    {
        var info = new ProcessStartInfo(Decoder) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (var argument in new[] { "-hide_banner", "-nostdin", "-loglevel", "info" }) info.ArgumentList.Add(argument);
        if (start > SeekMargin)
            foreach (var argument in new[] { "-ss", (start - SeekMargin).ToString("R", CultureInfo.InvariantCulture) }) info.ArgumentList.Add(argument);
        // showinfo reports each frame's timestamp and size just before its pixels are written.
        foreach (var argument in new[] { "-copyts", "-i", path, "-map", "0:v:0", "-vf", "showinfo", "-fps_mode", "passthrough",
            "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1" }) info.ArgumentList.Add(argument);
        frames = Channel.CreateUnbounded<(double, int, int)>(); errors = new();
        process = Process.Start(info) ?? throw new InvalidOperationException("Could not start ffmpeg.");
        var running = process; var output = frames; var log = errors;
        stderr = Task.Run(() => ReadLog(running, output, log));
    }

    static async Task ReadLog(Process process, Channel<(double, int, int)> frames, Queue<string> errors)
    {
        try
        {
            var (numerator, denominator) = (0L, 0L);
            while (await process.StandardError.ReadLineAsync() is { } line)
            {
                if (!line.Contains("Parsed_showinfo", StringComparison.Ordinal))
                {
                    errors.Enqueue(line);
                    if (errors.Count > 8) errors.Dequeue();
                }
                else if (Field(line, "config in time_base:") is { } timeBase) (numerator, denominator) = Rational(timeBase);
                else if (Field(line, " pts:") is { } pts && Field(line, " s:") is { } size)
                {
                    // Exact integer timestamps: pts_time is printed with only six decimals.
                    if (denominator <= 0) throw new InvalidDataException("ffmpeg reported a frame before its time base.");
                    var separator = size.IndexOf('x');
                    frames.Writer.TryWrite((long.Parse(pts, CultureInfo.InvariantCulture) * (double)numerator / denominator,
                        int.Parse(size[..separator], CultureInfo.InvariantCulture), int.Parse(size[(separator + 1)..], CultureInfo.InvariantCulture)));
                }
            }
            frames.Writer.TryComplete();
        }
        catch (Exception error) { frames.Writer.TryComplete(error); }
    }

    static (long Numerator, long Denominator) Rational(string text)
    {
        var parts = text.TrimEnd(',').Split('/');
        if (parts.Length != 2 || !long.TryParse(parts[0], CultureInfo.InvariantCulture, out var numerator)
            || !long.TryParse(parts[1], CultureInfo.InvariantCulture, out var denominator))
            throw new InvalidDataException($"Unreadable time base '{text}'.");
        return (numerator, denominator);
    }

    static string Field(string line, string name)
    {
        var at = line.IndexOf(name, StringComparison.Ordinal);
        if (at < 0) return null;
        var begin = at + name.Length;
        while (begin < line.Length && line[begin] == ' ') begin++;
        var end = line.IndexOf(' ', begin);
        return line[begin..(end < 0 ? line.Length : end)];
    }

    static (double Start, double Duration) RunProbe(string path)
    {
        var info = new ProcessStartInfo(Probe) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (var argument in new[] { "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_pts,time_base:format=duration",
            "-of", "default=noprint_wrappers=1", path }) info.ArgumentList.Add(argument);
        Process probe;
        try { probe = Process.Start(info) ?? throw new InvalidOperationException("Could not start ffprobe."); }
        catch (System.ComponentModel.Win32Exception error)
        { throw new NotSupportedException("Decoding video here needs FFmpeg: install it so ffmpeg and ffprobe are on PATH.", error); }
        using (probe)
        {
            var output = probe.StandardOutput.ReadToEndAsync();
            var error = probe.StandardError.ReadToEnd();
            probe.WaitForExit();
            var values = output.GetAwaiter().GetResult().Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(line => line.Split('=', 2)).Where(pair => pair.Length == 2).ToDictionary(pair => pair[0], pair => pair[1]);
            if (probe.ExitCode != 0 || !values.TryGetValue("duration", out var duration) || !values.TryGetValue("time_base", out var timeBase))
                throw new InvalidDataException($"ffprobe found no video stream with a duration in '{path}': {error.Trim()}");
            var (numerator, denominator) = Rational(timeBase);
            var first = long.TryParse(values.GetValueOrDefault("start_pts"), CultureInfo.InvariantCulture, out var pts) ? pts * (double)numerator / denominator : 0;
            return (first, double.Parse(duration, CultureInfo.InvariantCulture));
        }
    }

    static bool OnPath(string program) =>
        (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Any(directory => File.Exists(Path.Combine(directory, program)));

    void Stop()
    {
        if (process is null) return;
        if (!process.HasExited) process.Kill(entireProcessTree: true);
        process.WaitForExit();
        stderr.Wait();
        process.Dispose(); process = null;
    }

    public void Dispose() => Stop();
}
