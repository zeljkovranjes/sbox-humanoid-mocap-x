using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX;

namespace HumanoidMocapHost;

/// <summary>Host integration point. Implement inside the host; no client-side worker is required.</summary>
public interface IClipProcessor
{
    /// <summary>True only when models and the processing backend are ready.</summary>
    bool IsAvailable { get; }

    /// <summary>Reads a video and writes a full-body .hmotion document in metres.
    /// Report progress as a message plus the overall fraction done (0 to 1), or null when unknown;
    /// the queue turns it into time-left estimates. The caller owns the streams. Respect
    /// cancellation; failures must propagate.</summary>
    Task ProcessAsync(HostedCaptureOptions options, string extension, Stream video, Stream motion,
        Action<string, double?> progress, CancellationToken cancellation);
}

/// <summary>Optionally inspects the completed upload before admission. The file remains
/// owned by storage; returned settings must preserve the model and stay within the requested range.</summary>
internal interface IClipInspector
{
    Task<HostedCaptureOptions> InspectAsync(HostedCaptureOptions options, string video, CancellationToken cancellation);
}

internal sealed class UnconfiguredProcessor : IClipProcessor
{
    public bool IsAvailable => false;
    public Task ProcessAsync(HostedCaptureOptions options, string extension, Stream video, Stream motion,
        Action<string, double?> progress, CancellationToken cancellation)
        => throw new InvalidOperationException("Register the host's IClipProcessor implementation before accepting captures.");
}
