#nullable enable annotations

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX;

namespace HumanoidMocapHost;

/// <summary>Job creation body: the capture settings plus an optional display name, usually the
/// video's file name. <see cref="Policy"/> is honoured only from a signed front end, which sets it
/// from the account's roles; other callers get the host's own limits.</summary>
internal sealed record HostJobRequest(FlowModel Model, int CameraInterval, double Start, double? End, string? Name = null,
    JobPolicy? Policy = null)
{
    const int MaximumName = 120;

    internal HostedCaptureOptions Options => new(Model, CameraInterval, Start, End);

    /// <summary>The trimmed name, or null when absent. Rejects names that are too long or
    /// contain control characters.</summary>
    internal string? DisplayName()
    {
        var name = Name?.Trim();
        if (string.IsNullOrEmpty(name)) return null;
        if (name.Length > MaximumName || name.Any(char.IsControl))
            throw new ArgumentException($"Job names are at most {MaximumName} characters without control characters.");
        return name;
    }
}

/// <summary>One account's queue priority and limits. Null limits use the host's; larger values are
/// capped at the host's.</summary>
internal sealed record JobPolicy(bool Priority = false, double? MaximumClipSeconds = null, long? MaximumUploadBytes = null)
{
    internal void Validate()
    {
        if (MaximumClipSeconds is { } seconds && !(seconds > 0 && double.IsFinite(seconds)) || MaximumUploadBytes is < 1)
            throw new ArgumentException("Job policy limits must be positive.");
    }
}

/// <summary>Host settings an operator can change while it runs; saved with the job storage.</summary>
internal sealed record HostSettings(int ConcurrentJobs);
