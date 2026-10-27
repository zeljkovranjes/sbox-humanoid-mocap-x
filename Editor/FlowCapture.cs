using System;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.EditorTools;

internal sealed class FlowCapture(CaptureOptions options, HostingSettings hosting, IBodyCapture hosted, IBodyCapture local) : IBodyCapture
{
    /// <summary>Why capture is held back before either backend is asked (setup or model downloads
    /// unfinished), or null. Set by the window; capture is refused with this reason.</summary>
    internal string Blocked { get; set; }

    public string UnavailableReason => Blocked ?? (hosting.Enabled ? hosted.UnavailableReason : local.UnavailableReason);

    public Task<string> CaptureAsync(string video, double start, double? end,
        Action<string> progress, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (Blocked is { } reason) throw new InvalidOperationException(reason);
        options.Validate();
        if (hosting.Enabled) return hosted.CaptureAsync(video, start, end, progress, cancellationToken);
        return local.CaptureAsync(video, start, end, progress, cancellationToken);
    }
}
