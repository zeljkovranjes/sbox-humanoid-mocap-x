using System;

namespace HumanoidMocapX.Core.Preview;

/// <summary>Synchronizes an engine preview clock to a 30 fps capture timeline.</summary>
public sealed class PreviewPlayback
{
    const double FrameRate = 30, MaximumDrift = .1, EndMargin = .001;
    double lastTarget = double.NaN;
    bool wasPlaying;

    /// <summary>Returns a seek time when the player needs correction, otherwise null.
    /// Normal playback tolerates 100 ms of drift; paused/slow playback seeks only when
    /// its 30 fps frame changes. Rejects invalid times and a nonpositive duration.</summary>
    public float? SeekTime(double requested, double duration, double position, bool isPlaying)
    {
        if (!double.IsFinite(requested) || requested < 0 || !double.IsFinite(duration) || duration <= 0
            || !double.IsFinite(position) || position < 0)
            throw new ArgumentException("Preview playback requires finite non-negative times and a positive duration.");
        var target = Math.Min(requested, Math.Max(0, duration - EndMargin));
        if (!isPlaying) target = Math.Floor(target * FrameRate) / FrameRate;
        var seek = double.IsNaN(lastTarget) || isPlaying != wasPlaying
            || (isPlaying ? Math.Abs(position - target) > MaximumDrift : target != lastTarget);
        lastTarget = target; wasPlaying = isPlaying;
        return seek ? (float)target : null;
    }
}
