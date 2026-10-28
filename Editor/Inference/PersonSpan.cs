using System;
using System.Threading;
using System.Threading.Tasks;

namespace HumanoidMocapX.Inference;

/// <summary>The part of a capture timeline in which the tracked person is visible: its first
/// frame, how many follow, and one XYXY box per visible frame.</summary>
internal sealed record PersonSpan(float[] Boxes, int First, int Count)
{
    const double TimelineFps = 30;

    /// <summary>Seconds from the timeline's start to the first visible frame.</summary>
    internal double StartSeconds => First / TimelineFps;

    /// <summary>The visible frames of <paramref name="frames"/>; the same instance when the
    /// person is visible throughout.</summary>
    internal ICaptureFrames Visible(ICaptureFrames frames) =>
        First == 0 && Count == frames.Count ? frames : new Range(frames, First, Count);

    /// <summary>A diagnostic naming the captured part of the video, whose timeline starts at
    /// <paramref name="start"/> seconds and has <paramref name="frames"/> frames; null when the
    /// person is visible throughout.</summary>
    internal string Note(double start, int frames) => First == 0 && Count == frames ? null
        : FormattableString.Invariant($"The person is in view from {start + StartSeconds:0.00} s to {start + (First + Count) / TimelineFps:0.00} s of the video, so only that part was captured.");

    sealed class Range(ICaptureFrames frames, int first, int count) : ICaptureFrames
    {
        public int Width => frames.Width;
        public int Height => frames.Height;
        public int Count => count;

        public ValueTask ReadAsync(int frame, Memory<byte> rgba, CancellationToken cancellation)
        {
            if ((uint)frame >= (uint)count) throw new ArgumentOutOfRangeException(nameof(frame));
            return frames.ReadAsync(first + frame, rgba, cancellation);
        }
    }
}
