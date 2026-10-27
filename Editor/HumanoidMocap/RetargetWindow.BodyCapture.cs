using System;
using System.IO;
using System.Threading.Tasks;
using Editor;
using HumanoidMocapX.Core.Motion;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

public sealed partial class RetargetWindow
{
    readonly IBodyCapture _bodyCapture;
    readonly Action<Widget> _captureSettings;

    /// <summary>Loads a finished .hmotion capture into this workspace, as if the backend had
    /// just returned it. Does nothing while a capture is running.</summary>
    public Task OpenMotionAsync(string path) => OpenCaptureFileAsync(path);

    internal Task OpenCaptureFileAsync(string path)
    {
        if (_processing is not null) return Task.CompletedTask;
        if (!string.Equals(Path.GetExtension(path), ".hmotion", StringComparison.OrdinalIgnoreCase))
        { _captureStatus.Text = "Open a full-body .hmotion capture in this workspace."; return Task.CompletedTask; }
        return LoadMotionAsync(path);
    }

    async Task CaptureWithBackendAsync(string video, double start, double? end)
    {
        var path = await _bodyCapture.CaptureAsync(video, start, end,
            ReceiveWorkerProgress, _processing.Token);
        _processing.Token.ThrowIfCancellationRequested();
        await EditorPipeline.SwitchToMainThread();
        if (this.IsValid()) await LoadMotionAsync(path);
    }
}
