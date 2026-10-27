using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

public sealed partial class RetargetWindow
{
    readonly Queue<(string Path, double Start, double? End)> _queuedVideos = new();
    string _workerMessage;
    void ReceiveWorkerProgress(string message)=>Interlocked.Exchange(ref _workerMessage,message);
    internal bool CaptureIsRunning => _processing is not null;
    internal string CaptureMotionPath => _motionPath;
    internal string CaptureMessage => _captureStatus.Text;
    internal bool CaptureCanExport => _exportMotionButton.Enabled;
    internal bool CapturePreviewReady => _mocapPreview.IsValid() && _mocapPreview.FrameCount > 0 && _mocapPreview.HasModel;
    internal PreviewWidget CapturePreview => _mocapPreview;
    internal void CancelCapture() => _processing?.Cancel();

    public void ImportVideoAndProcess(string path) => ImportVideoAndProcess(path,0,null);

    /// <summary>Process an explicitly selected range; the ordinary upload processes the full video.</summary>
    public void ImportVideoAndProcess(string path, double start, double? end)
    {
        if (_bodyCapture.UnavailableReason is { } reason) { _captureStatus.Text = reason; return; }
        if (_processing is not null)
        {
            if (_queuedVideos.Count >= 8) { _captureStatus.Text = "Upload saved. Eight videos are already queued; open this video after processing finishes."; return; }
            _queuedVideos.Enqueue((path,start,end));
            _captureStatus.Text = $"Video saved · {_queuedVideos.Count} queued.";
            return;
        }
        ResetClipSpeed();
        LoadVideo(path);
        _rangeStart.Text = start.ToString(System.Globalization.CultureInfo.InvariantCulture);
        _rangeEnd.Text = end?.ToString(System.Globalization.CultureInfo.InvariantCulture)??"";
        _ = ProcessImportedVideoAsync();
    }

    async Task ProcessImportedVideoAsync()
    {
        if (_processing is not null) return;
        if (_bodyCapture.UnavailableReason is { } reason) { _captureStatus.Text = reason; return; }
        if (!File.Exists(_sourcePath.Text)) { _captureStatus.Text = "Upload a video first."; return; }
        var video = _sourcePath.Text;
        if (!double.TryParse(_rangeStart.Text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var start)
            || !double.IsFinite(start) || start < 0)
        { _captureStatus.Text = "Start must be a finite, non-negative number of seconds."; return; }
        double? end = null;
        if (!string.IsNullOrWhiteSpace(_rangeEnd.Text))
        {
            if (!double.TryParse(_rangeEnd.Text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var stop)
                || !double.IsFinite(stop) || stop <= start)
            { _captureStatus.Text = "End must be a finite time after Start."; return; }
            end = stop;
        }
        _processing = new CancellationTokenSource();
        SetCaptureBusy(true); _retryCaptureButton.Visible = false;
        _exportMotionButton.Enabled = false; _rawMotion = null; _editedMotion = null; _motionPath = null;
        _editSession=null;_appliedCleanup=null;
        ++_motionLoadRevision;InvalidateMocapPreview();
        _targetHost.Layout.Clear(true);_mocapPreview=null;
        _loader=_targetHost.Layout.Add(new ProcessingIndicator(_targetHost),1);_loader.SetMessage("Preparing video…");_loaderStatus=null;
        ResetPlayback();_motionDetails.Text="";
        try
        {
            await CaptureWithBackendAsync(video, start, end);
        }
        catch (OperationCanceledException)
        {
            await EditorPipeline.SwitchToMainThread();
            if (this.IsValid()) { _captureStatus.Text = "Cancelled. Retry starts a new capture."; _retryCaptureButton.Visible = true; }
            _queuedVideos.Clear();
        }
        catch (Exception e)
        {
            await EditorPipeline.SwitchToMainThread();
            if (this.IsValid()) { _captureStatus.Text = e.Message; _retryCaptureButton.Visible = true; }
        }
        finally
        {
            Interlocked.Exchange(ref _workerMessage,null);
            await EditorPipeline.SwitchToMainThread(); _processing.Dispose(); _processing = null;
            if (this.IsValid())
            {
                SetCaptureBusy(false);
                StartNextQueuedVideo();
            }
        }
    }

    void SetCaptureBusy(bool busy)
    {
        _uploadVideoButton.Enabled = !busy;
        if(_loader.IsValid()){_loader.Busy=busy;_loaderStatus=null;}
        _cancelCaptureButton.Visible = busy; _advancedPanel.Enabled = !busy;
        _previewTargetPicker.Enabled=!busy;
        UpdateExportAvailability();
    }
    void StartNextQueuedVideo()
    {
        if(_queuedVideos.TryDequeue(out var queued))ImportVideoAndProcess(queued.Path,queued.Start,queued.End);
    }
}
