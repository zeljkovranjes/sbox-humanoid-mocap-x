using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocap.EditorTools.Inference;
using HumanoidMocapX.Inference;

namespace HumanoidMocapX.EditorTools;

/// <summary>Streams timestamped decoded pixels to a temporary 30 fps timeline.
/// Disposal deletes the cache; reads preserve original resolution and RGBA bytes.</summary>
internal sealed class VideoFrames : ICaptureFrames, IDisposable
{
    internal const long MaximumCacheBytes = 16L * 1024 * 1024 * 1024;
    readonly FileStream stream;
    readonly int frameBytes;
    public int Width { get; }
    public int Height { get; }
    public int Count { get; }

    VideoFrames(FileStream stream, int width, int height, int count)
    {
        this.stream = stream; Width = width; Height = height; Count = count;
        frameBytes = checked(width * height * 4);
    }

    /// <summary>Reads sequential samples, keeping the displayed frame at each target
    /// timestamp. The caller owns the decoder and runs this off the editor thread.</summary>
    internal static async Task<VideoFrames> CreateAsync(Func<CancellationToken, DecodedVideoFrame> read,
        double duration, double start, double? end, string directory, Action<string> progress, CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(read);
        var timeline = new CaptureTimeline(duration, start, end); token.ThrowIfCancellationRequested();
        var current = Read(read, 0, 0, double.NegativeInfinity, token)
            ?? throw new InvalidDataException("Video contains no decoded frames.");
        if (current.Time > start + CaptureTimeline.TimestampTolerance)
            throw new InvalidDataException("The decoder did not return a frame covering the start of the selected range.");
        var bytes = checked(current.Width * current.Height * 4);
        var required = (long)bytes * timeline.Count;
        if (required > MaximumCacheBytes)
            throw new InvalidDataException("Decoded video would exceed the 16 GiB temporary cache limit. Select a shorter range.");
        Directory.CreateDirectory(directory);
        const long reserveBytes = 256L * 1024 * 1024;
        if (OperatingSystem.IsWindows() && new DriveInfo(Path.GetPathRoot(Path.GetFullPath(directory))).AvailableFreeSpace - reserveBytes < required)
            throw new IOException("Not enough free disk space for decoded video. Select a shorter range or free disk space.");
        var path = Path.Combine(directory, Guid.NewGuid().ToString("N") + ".rgba");
        var stream = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 65536,
            FileOptions.Asynchronous | FileOptions.RandomAccess | FileOptions.DeleteOnClose);
        try
        {
            var next = Read(read, current.Width, current.Height, current.Time, token);
            for (var frame = 0; frame < timeline.Count; frame++)
            {
                token.ThrowIfCancellationRequested();
                var time = timeline.Time(frame);
                while (next is not null && next.Time <= time + CaptureTimeline.TimestampTolerance)
                {
                    current = next;
                    next = Read(read, current.Width, current.Height, current.Time, token);
                }
                await stream.WriteAsync(current.Rgba, token).ConfigureAwait(false);
                if (frame % 30 == 0 || frame == timeline.Count - 1)
                    progress?.Invoke($"Decoding video · {frame + 1}/{timeline.Count}");
            }
            await stream.FlushAsync(token).ConfigureAwait(false);
            return new(stream, current.Width, current.Height, timeline.Count);
        }
        catch { await stream.DisposeAsync(); throw; }
    }

    static DecodedVideoFrame Read(Func<CancellationToken, DecodedVideoFrame> read, int width, int height, double previous, CancellationToken token)
    {
        token.ThrowIfCancellationRequested(); var frame = read(token);
        if (frame is null) return null;
        if (frame.Width < 1 || frame.Height < 1 || (long)frame.Width * frame.Height > 33_177_600
            || frame.Rgba is null || frame.Rgba.Length != (long)frame.Width * frame.Height * 4
            || width != 0 && (frame.Width != width || frame.Height != height))
            throw new InvalidDataException("Decoded video has invalid or changing frame dimensions.");
        if (!double.IsFinite(frame.Time) || frame.Time <= previous)
            throw new InvalidDataException("Decoded video timestamps must be finite and strictly increasing.");
        return frame;
    }

    public async ValueTask ReadAsync(int frame, Memory<byte> rgba, CancellationToken cancellation)
    {
        if ((uint)frame >= (uint)Count || rgba.Length != frameBytes)
            throw new ArgumentException("Invalid capture frame or destination size.");
        cancellation.ThrowIfCancellationRequested(); stream.Position = (long)frame * frameBytes;
        await stream.ReadExactlyAsync(rgba, cancellation).ConfigureAwait(false);
    }

    public void Dispose() => stream.Dispose();
}
