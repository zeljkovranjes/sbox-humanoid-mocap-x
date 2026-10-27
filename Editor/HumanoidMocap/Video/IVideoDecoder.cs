using System;
using System.Threading;

namespace HumanoidMocapX.EditorTools.Video;

/// <summary>Timestamped RGBA frames from a video file's first video stream, in presentation
/// order. Times are seconds from the stream start.</summary>
public interface IVideoDecoder : IDisposable
{
    /// <summary>The container duration in seconds.</summary>
    double Duration { get; }

    /// <summary>Continues decoding at or before <paramref name="seconds"/>.</summary>
    void Seek(double seconds);

    /// <summary>The next frame, or null at the end of the stream. Throws when decoding fails.</summary>
    DecodedVideoFrame Read(CancellationToken token);

    /// <summary>Windows Media Foundation on Windows; the system FFmpeg elsewhere. Throws
    /// when the file is missing or no installed decoder can read it.</summary>
    static IVideoDecoder Open(string path) =>
        OperatingSystem.IsWindows() ? new WindowsVideoDecoder(path) : new FfmpegVideoDecoder(path);
}
