#nullable enable annotations

namespace HumanoidMocapX.Core.Motion;

/// <summary>Reconstructs a video range into a validated, metre-based motion document.</summary>
public interface IBodyCapture
{
    /// <summary>Reason capture cannot start, or null when the backend is ready.</summary>
    string? UnavailableReason { get; }

    /// <summary>Returns the saved motion path. Range times are source-video seconds;
    /// a null end means the remaining video. Failures and cancellation propagate.</summary>
    Task<string> CaptureAsync(string video, double start, double? end,
        Action<string> progress, CancellationToken cancellationToken);
}
