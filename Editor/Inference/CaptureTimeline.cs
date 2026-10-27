using System;

namespace HumanoidMocapX.Inference;

/// <summary>A source-time interval sampled at FlowHMR's fixed 30 fps. The end is exclusive.</summary>
internal readonly struct CaptureTimeline
{
    internal const double TimestampTolerance = 0.0000001;
    internal double Start { get; }
    internal double End { get; }
    internal int Count { get; }

    internal CaptureTimeline(double duration, double start, double? end)
    {
        if (!double.IsFinite(duration) || duration <= 0 || !double.IsFinite(start) || start < 0 || start >= duration
            || end is { } stop && (!double.IsFinite(stop) || stop <= start || stop > duration + TimestampTolerance))
            throw new ArgumentException("Choose a nonempty capture range within the video's duration.");
        Start = start; End = Math.Min(end ?? duration, duration);
        var count = Math.Ceiling((End - Start) * 30 - TimestampTolerance);
        if (count < 1 || count > CaptureOptions.MaximumFrames)
            throw new ArgumentException("Choose a capture range of at most 5,000 frames (166.66 seconds at 30 fps).");
        Count = (int)count;
    }

    internal double Time(int frame)
    {
        if ((uint)frame >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(frame));
        return Start + frame / 30d;
    }
}
