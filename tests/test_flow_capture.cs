using HumanoidMocapX.Core.Motion;
using HumanoidMocapX;
using HumanoidMocapX.EditorTools;

static class FlowCaptureTests
{
    internal static async Task Run()
    {
        var hosting = new HostingSettings();
        var hosted = new RecordingCapture("hosted"); var local = new RecordingCapture("local");
        var capture = new FlowCapture(new CaptureOptions(), hosting, hosted, local) { Blocked = "Finish setup first." };

        // A held-back capture reports its reason and never reaches either backend.
        MotionTests.Equal("Finish setup first.", capture.UnavailableReason);
        MotionTests.Throws<InvalidOperationException>(() => capture.CaptureAsync("clip.mp4", 0, null, null, default));
        MotionTests.Equal(0, hosted.Calls + local.Calls);

        capture.Blocked = null;
        MotionTests.Equal("local", await capture.CaptureAsync("clip.mp4", 0, null, null, default));
        hosting.Enabled = true;
        MotionTests.Equal("hosted", await capture.CaptureAsync("clip.mp4", 0, null, null, default));
        MotionTests.Equal(1, hosted.Calls); MotionTests.Equal(1, local.Calls);
        Console.WriteLine("PASS: capture routing holds back blocked captures and picks the hosted or local backend");
    }

    sealed class RecordingCapture(string result) : IBodyCapture
    {
        internal int Calls;
        public string UnavailableReason => null;
        public Task<string> CaptureAsync(string video, double start, double? end, Action<string> progress, CancellationToken cancellationToken)
        { Calls++; return Task.FromResult(result); }
    }
}
