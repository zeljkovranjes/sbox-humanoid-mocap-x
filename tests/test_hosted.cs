using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX;
using HumanoidMocapX.EditorTools;

static class HostedTests
{
    const string Key = "test-key-00000000000000000000000000000000";
    internal static async Task Run()
    {
        var settings = new HostingSettings { Enabled = true, ServiceUrl = "https://host.example/prefix" };
        var profile = new HostProfile(new HostUser("discord:1", "discord", null, null, null), true, false, true, true, false);
        void SignIn() => settings.SignIn(Key, DateTimeOffset.UtcNow.AddHours(1), profile);
        var changes = 0; settings.Changed += () => changes++;
        MotionTests.Check(new HostedCapture(settings, new CaptureOptions(), Path.GetTempPath()).UnavailableReason!.Contains("Sign in"), "Signed-out state must ask for sign-in");
        SignIn();
        MotionTests.Check(settings.SignedIn && changes == 1 && new HostedCapture(settings, new CaptureOptions(), Path.GetTempPath()).UnavailableReason is null, "Sign-in must enable processing");
        settings.Expire("an-older-token");
        MotionTests.Check(settings.SignedIn && changes == 1, "A 401 for an older token must not end the current sign-in");
        settings.Update(Key, profile with { CanProcess = false });
        MotionTests.Check(new HostedCapture(settings, new CaptureOptions(), Path.GetTempPath()).UnavailableReason!.Contains("whitelisted"), "Profile refresh must apply");
        SignIn();
        MotionTests.Equal("https://host.example/prefix/", settings.Endpoint().AbsoluteUri);
        foreach (var invalid in new[] { "http://host.example", "file:///tmp/file", "https://user:pass@host.example", "https://host.example/?token=secret" })
        { settings.ServiceUrl = invalid; MotionTests.Throws<ArgumentException>(() => settings.Endpoint()); }
        settings.ServiceUrl = "https://host.example/prefix";
        new HostedCaptureOptions(FlowModel.Base, 30, 2, 9).Validate();
        MotionTests.Throws<ArgumentException>(() => new HostedCaptureOptions(FlowModel.Latest, 10, double.NaN, null).Validate());
        MotionTests.Throws<ArgumentException>(() => new HostedCaptureOptions(FlowModel.Latest, 10, 3, 2).Validate());
        var folder = Path.Combine(Path.GetTempPath(), "mocap-hosted-test-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(folder);
        var video = Path.Combine(folder, "input.mp4"); var bytes = new byte[] { 1, 2, 3, 4, 5 };
        await File.WriteAllBytesAsync(video, bytes);
        try
        {
            var motion = MotionTests.Fixture().ToDocument(); motion.SourceVideo = "/host/private/video.mp4";
            motion.OriginalReconstruction = new MotionOrigin { Path = "/host/private/original.hmotion", Sha256 = new string('a', 64) };
            var original = motion.ToJson();
            var handler = new HostHandler(original, bytes);
            using (var capture = new HostedCapture(settings, new CaptureOptions { Model = FlowModel.Base, CameraInterval = 30 }, folder, handler))
            {
                var output = await capture.CaptureAsync(video, 2, 9, null, default);
                var parsed = MotionDocument.Parse(await File.ReadAllBytesAsync(output));
                MotionTests.Equal(video, parsed.SourceVideo); MotionTests.Equal(motion.Frames.Count, parsed.Frames.Count);
                MotionTests.Check(parsed.OriginalReconstruction is null, "Remote sidecar metadata must not select a local file");
                MotionTests.Check(handler.Deleted, "Completed host job must be deleted after download");
                MotionTests.Equal("Base", handler.Options.GetProperty("model").GetString());
                MotionTests.Equal(30, handler.Options.GetProperty("cameraInterval").GetInt32());
                MotionTests.Near(2, handler.Options.GetProperty("start").GetDouble());
                MotionTests.Near(9, handler.Options.GetProperty("end").GetDouble());
            }
            var before = Directory.GetFiles(folder, "*.hmotion").Length;
            handler = new HostHandler("{\"schema\":99}", bytes);
            using (var capture = new HostedCapture(settings, new CaptureOptions(), folder, handler))
                await Throws<FormatException>(() => capture.CaptureAsync(video, 0, null, null, default));
            MotionTests.Equal(before, Directory.GetFiles(folder, "*.hmotion").Length);
            MotionTests.Equal(0, Directory.GetFiles(folder, "*.partial").Length);
            MotionTests.Check(handler.Deleted, "Invalid results must clean up their remote job");
            using var cancelled = new CancellationTokenSource();
            handler = new HostHandler(original, bytes) { OnUpload = cancelled.Cancel };
            using (var capture = new HostedCapture(settings, new CaptureOptions(), folder, handler))
                await Throws<OperationCanceledException>(() => capture.CaptureAsync(video, 0, null, null, cancelled.Token));
            MotionTests.Check(handler.Deleted, "Cancellation must still send remote DELETE");
            handler = new HostHandler(original, bytes) { Available = false };
            using (var capture = new HostedCapture(settings, new CaptureOptions(), folder, handler))
                await Throws<InvalidOperationException>(() => capture.CaptureAsync(video, 0, null, null, default));
            MotionTests.Check(!handler.Created, "An unconfigured host must not receive video");
            handler = new HostHandler(original, bytes) { Rejected = true };
            using (var capture = new HostedCapture(settings, new CaptureOptions(), folder, handler))
                await Throws<HttpRequestException>(() => capture.CaptureAsync(video, 0, null, null, default));
            MotionTests.Check(!handler.Created && !settings.SignedIn, "Rejected credentials must sign out without uploading video");
            SignIn();
            handler = new HostHandler(original, bytes) { Clip = 3 };
            using (var capture = new HostedCapture(settings, new CaptureOptions(), folder, handler))
                await Throws<InvalidDataException>(() => capture.CaptureAsync(video, 0, 5, null, default));
            MotionTests.Check(!handler.Created, "Clips over the account's limit must stop before upload");
            var delays = new List<double>(); var messages = new List<string>();
            handler = new HostHandler(original, bytes) { TransientPolls = 2, QueuedPolls = 1 };
            using (var capture = new HostedCapture(settings, new CaptureOptions(), folder, handler,
                (time, token) => { token.ThrowIfCancellationRequested(); delays.Add(time.TotalSeconds); return Task.CompletedTask; }))
                await capture.CaptureAsync(video, 0, 5, messages.Add, default);
            MotionTests.Check(delays.SequenceEqual(new[] { 2d, 2d, 7d }), "Queue polling ignored Retry-After");
            MotionTests.Check(messages.Any(m => m.Contains("Waiting · 2 in line")), "Queue position was not displayed");
            MotionTests.Equal("Waiting · 2 of 5 in line (priority) · starts in 2 min · about 1 h 5 min left",
                HostedCapture.Describe(new HostedJob(Guid.Empty, "queued", "Waiting", 2, QueueLength: 5, SecondsUntilStart: 100, SecondsRemaining: 3900, Priority: true)));
            MotionTests.Equal(4, handler.PollCount);
            delays.Clear(); handler = new HostHandler(original, bytes) { TransientPolls = 20 };
            using (var capture = new HostedCapture(settings, new CaptureOptions(), folder, handler,
                (time, token) => { delays.Add(time.TotalSeconds); return Task.CompletedTask; }))
                await Throws<HttpRequestException>(() => capture.CaptureAsync(video, 0, 5, null, default));
            MotionTests.Equal(6, delays.Count); MotionTests.Equal(7, handler.PollCount);
            MotionTests.Check(handler.Deleted, "Exhausted status retries must clean up the known job");
            handler = new HostHandler(original, bytes) { QueuedPolls = 1, Position = 0 };
            using (var capture = new HostedCapture(settings, new CaptureOptions(), folder, handler))
                await Throws<InvalidDataException>(() => capture.CaptureAsync(video, 0, 5, null, default));
            MotionTests.Check(handler.Deleted, "Malformed queue position must clean up the known job");
            using var queuedCancellation = new CancellationTokenSource();
            handler = new HostHandler(original, bytes) { QueuedPolls = 1 };
            using (var capture = new HostedCapture(settings, new CaptureOptions(), folder, handler,
                (time, token) => { queuedCancellation.Cancel(); return Task.FromCanceled(token); }))
                await Throws<OperationCanceledException>(() => capture.CaptureAsync(video, 0, 5, null, queuedCancellation.Token));
            MotionTests.Check(handler.Deleted, "Cancellation during queue delay must send remote DELETE");
            var now = DateTimeOffset.UtcNow;
            MotionTests.Near(1, HostedCapture.PollDelay(null, now).TotalSeconds);
            MotionTests.Near(30, HostedCapture.PollDelay(new RetryConditionHeaderValue(TimeSpan.FromMinutes(5)), now).TotalSeconds);
            MotionTests.Near(1, HostedCapture.PollDelay(new RetryConditionHeaderValue(now.AddSeconds(-5)), now).TotalSeconds);
            MotionTests.Near(12, HostedCapture.PollDelay(new RetryConditionHeaderValue(now.AddSeconds(12)), now).TotalSeconds);
            handler = new HostHandler(original, bytes);
            var closing = new HostedCapture(settings, new CaptureOptions(), folder, handler);
            handler.OnUpload = closing.Dispose;
            await Throws<OperationCanceledException>(() => closing.CaptureAsync(video, 0, null, null, default));
            MotionTests.Check(handler.Deleted && handler.Disposed, "Closing capture must cancel remotely before disposing HTTP");
            await Throws<ObjectDisposedException>(() => closing.CaptureAsync(video, 0, null, null, default));
            var remainingVideo = await File.ReadAllBytesAsync(video);
            MotionTests.Check(bytes.SequenceEqual(remainingVideo), "The original video must remain unchanged");
        }
        finally { Directory.Delete(folder, true); }
        Console.WriteLine("PASS: hosted upload/settings, queue positions, bounded Retry-After polling, result validation, cancellation and private metadata replacement");
    }

    sealed class HostHandler(string result, byte[] expectedVideo) : HttpMessageHandler
    {
        readonly Guid id = Guid.NewGuid();
        internal bool Deleted, Created, Rejected, Disposed, Available = true;
        internal int TransientPolls, QueuedPolls, PollCount, Position = 2;
        internal double? Clip;
        internal Action? OnUpload;
        internal JsonElement Options;
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            MotionTests.Equal("host.example", request.RequestUri!.Host);
            MotionTests.Equal(Key, request.Headers.Authorization!.Parameter);
            var path = request.RequestUri.AbsolutePath;
            if (Rejected) return new(HttpStatusCode.Unauthorized);
            if (request.Method == HttpMethod.Delete) { Deleted = true; return new(HttpStatusCode.NoContent); }
            if (path == "/prefix/v1/capabilities") return Json(new HostedCapabilities(1, Available, 1024, MaximumClipSeconds: Clip));
            if (request.Method == HttpMethod.Post)
            {
                Created = true; Options = JsonSerializer.Deserialize<JsonElement>(await request.Content!.ReadAsStringAsync(token));
                return Json(new HostedJob(id, "awaiting-upload", "Ready"));
            }
            if (request.Method == HttpMethod.Put)
            {
                MotionTests.Equal(".mp4", request.Headers.GetValues("X-Video-Extension").Single());
                var uploadedVideo = await request.Content!.ReadAsByteArrayAsync(token);
                MotionTests.Check(expectedVideo.SequenceEqual(uploadedVideo), "Upload bytes changed");
                OnUpload?.Invoke(); return new(HttpStatusCode.Accepted);
            }
            if (path.EndsWith("/result", StringComparison.Ordinal))
                return new(HttpStatusCode.OK) { Content = new StringContent(result, Encoding.UTF8, "application/json") };
            PollCount++;
            if (TransientPolls-- > 0)
            {
                var transient = new HttpResponseMessage(HttpStatusCode.ServiceUnavailable);
                transient.Headers.RetryAfter = new RetryConditionHeaderValue(TimeSpan.FromSeconds(2)); return transient;
            }
            if (QueuedPolls-- > 0)
            {
                var queued = Json(new HostedJob(id, "queued", "Waiting", Position));
                queued.Headers.RetryAfter = new RetryConditionHeaderValue(TimeSpan.FromSeconds(7)); return queued;
            }
            return Json(new HostedJob(id, "completed", "Ready"));
        }

        static HttpResponseMessage Json<T>(T value) => new(HttpStatusCode.OK)
        { Content = new StringContent(JsonSerializer.Serialize(value, new JsonSerializerOptions(JsonSerializerDefaults.Web)), Encoding.UTF8, "application/json") };

        protected override void Dispose(bool disposing) { Disposed = true; base.Dispose(disposing); }
    }

    static async Task Throws<T>(Func<Task> action) where T : Exception
    {
        try { await action(); } catch (T) { return; }
        throw new Exception("Expected " + typeof(T).Name);
    }
}
