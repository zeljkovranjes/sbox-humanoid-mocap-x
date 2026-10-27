using System;

namespace HumanoidMocapX;

/// <summary>Video-range and model settings sent to a processing host. Times are source seconds.</summary>
public sealed record HostedCaptureOptions(FlowModel Model, int CameraInterval, double Start, double? End)
{
    /// <summary>Rejects unsupported models, sampling intervals and invalid time ranges.</summary>
    public void Validate()
    {
        new CaptureOptions { Model = Model, CameraInterval = CameraInterval }.Validate();
        if (!double.IsFinite(Start) || Start < 0 || End is double end && (!double.IsFinite(end) || end <= Start))
            throw new ArgumentException("The video range needs a finite, nonnegative start and a later end.");
    }
}

/// <summary>Protocol version, processor readiness, the upload limit in bytes, how many unfinished
/// jobs one account may have, how many jobs wait and run across all accounts, how many process at
/// once, and the longest video range in seconds. A provider's site reports the signed-in account's
/// own limits and whether its jobs wait in the priority queue.</summary>
public sealed record HostedCapabilities(int Protocol, bool ProcessingAvailable, long MaximumUploadBytes,
    int MaximumActiveJobs = 1, int QueueLength = 0, int Processing = 0, int ConcurrentJobs = 1,
    double? MaximumClipSeconds = null, bool Priority = false);

/// <summary>Job state returned by the host. QueuePosition is one-based while queued, out of
/// QueueLength waiting jobs; Progress (0 to 1) is set while processing. SecondsUntilStart and
/// SecondsRemaining are the host's estimates. ExpiresAt is when a finished job and its result
/// are deleted. Priority jobs wait ahead of the others. IDs are opaque; result URLs are derived locally.</summary>
#nullable enable annotations
public sealed record HostedJob(Guid Id, string State, string Message, int? QueuePosition = null, string? Name = null,
    DateTimeOffset CreatedAt = default, int QueueLength = 0, double? Progress = null, int? SecondsUntilStart = null,
    int? SecondsRemaining = null, DateTimeOffset? ExpiresAt = null, bool Priority = false);
#nullable restore
