using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
namespace HumanoidMocapHost;

/// <summary>Learns how long captures take, as processing seconds per second of requested video,
/// from completed jobs. Starts from a measured default and moves towards recent jobs.</summary>
internal sealed class ProcessingEstimate(double secondsPerVideoSecond = ProcessingEstimate.DefaultSecondsPerVideoSecond)
{
    // 272 s for a 5 s clip on an RX 9070 with four lanes (2026-10-27); the host's own jobs replace it.
    internal const double DefaultSecondsPerVideoSecond = 60;
    // Jobs without an end time use this video length until the processor reports progress.
    internal const double UnknownVideoSeconds = 10;
    const double Weight = 0.3;
    double rate = secondsPerVideoSecond > 0 ? secondsPerVideoSecond
        : throw new ArgumentOutOfRangeException(nameof(secondsPerVideoSecond));

    internal double SecondsPerVideoSecond => rate;

    /// <summary>Expected processing time for a job covering <paramref name="videoSeconds"/>.</summary>
    internal double Expected(double? videoSeconds) => rate * (videoSeconds ?? UnknownVideoSeconds);

    /// <summary>Time left for a running job: extrapolated from reported progress once there is some,
    /// otherwise the expected time minus what has elapsed (never negative).</summary>
    internal double Remaining(double? videoSeconds, double elapsedSeconds, double? progress)
        => progress is > 0.02 and < 1 ? elapsedSeconds * (1 - progress.Value) / progress.Value
            : Math.Max(0, Expected(videoSeconds) - elapsedSeconds);

    /// <summary>Learns from a finished job. Jobs without a known video length are skipped.</summary>
    internal void Completed(double? videoSeconds, double elapsedSeconds)
    {
        if (videoSeconds is not > 0 || !double.IsFinite(elapsedSeconds) || elapsedSeconds <= 0) return;
        rate += Weight * (elapsedSeconds / videoSeconds.Value - rate);
    }
}
