using System;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.EditorTools;

internal sealed class FlowCapture(CaptureOptions options, HostingSettings hosting, IBodyCapture hosted,
    SshSettings machine, IBodyCapture remote, IBodyCapture local) : IBodyCapture
{
    /// <summary>Why capture is held back before any backend is asked (setup or model downloads
    /// unfinished), or null. Set by the window; capture is refused with this reason.</summary>
    internal string Blocked { get; set; }

    // The provider, the processing machine over SSH, or this computer.
    IBodyCapture Selected => hosting.Enabled ? hosted : machine.Enabled ? remote : local;

    public string UnavailableReason => Blocked ?? Selected.UnavailableReason;

    public Task<string> CaptureAsync(string video, double start, double? end,
        Action<string> progress, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (Blocked is { } reason) throw new InvalidOperationException(reason);
        options.Validate();
        return Selected.CaptureAsync(video, start, end, progress, cancellationToken);
    }
}
