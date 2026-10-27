using System;
using System.Threading.Tasks;
using Editor;
using HumanoidMocapX.Core.Motion;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

public sealed partial class RetargetWindow
{
    readonly IBodyCapture _bodyCapture;
    readonly Action<Widget> _captureSettings;
    bool HasBodyCapture => _bodyCapture is not null;

    /// <summary>Creates a third-person workspace with an injected body capture backend.
    /// The settings callback adds backend controls to Advanced. Null backends are rejected.</summary>
    public RetargetWindow(Widget parent, string title, IBodyCapture bodyCapture,
        Action<Widget> captureSettings = null) : this(parent, title,
            bodyCapture ?? throw new ArgumentNullException(nameof(bodyCapture)), captureSettings, true) { }

    /// <summary>Loads a finished .hmotion capture into this workspace, as if the backend had
    /// just returned it. Does nothing while a capture is running.</summary>
    public Task OpenMotionAsync(string path) => OpenCaptureFileAsync(path);

    async Task CaptureWithBackendAsync(string video, double start, double? end)
    {
        var path = await _bodyCapture.CaptureAsync(video, start, end,
            ReceiveWorkerProgress, _processing.Token);
        _processing.Token.ThrowIfCancellationRequested();
        await EditorPipeline.SwitchToMainThread();
        if (this.IsValid()) await LoadMotionAsync(path);
    }
}
