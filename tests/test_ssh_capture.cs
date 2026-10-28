using System.Diagnostics;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX;
using HumanoidMocapX.EditorTools;

static class SshCaptureTests
{
    const string Awkward = "Breakin' - Fortnite Battle Royale (Break Dance Emote) - TY_ (720p, h264).mp4";

    // A stand-in ssh: logs its arguments, then runs the remote command here with sh, as sshd runs it
    // with the login shell. FAKE_SSH_FAIL makes it fail the way an unreachable machine does.
    const string FakeSsh = """
        #!/bin/sh
        printf '%s\n' "$@" >> "$FAKE_ROOT/ssh.log"
        [ -n "$FAKE_SSH_FAIL" ] && { echo "ssh: connect to host gpu port 22: Connection refused" >&2; exit 255; }
        for last; do :; done
        exec sh -c "$last"
        """;

    // A stand-in worker: its usage without arguments; otherwise it records its arguments and the
    // uploaded video, then fails, runs until stopped, or reports progress and writes the motion.
    const string FakeWorker = """
        #!/bin/sh
        [ $# -eq 0 ] && { echo "usage:" >&2; echo "  humanoid-mocap-worker capture <video> <output.hmotion>" >&2; exit 2; }
        printf '%s\n' "$@" > "$FAKE_ROOT/arguments.txt"
        cp "$2" "$FAKE_ROOT/uploaded"
        case $FAKE_WORKER in
        fail) echo "GPU: Fake" >&2; echo "error: System.InvalidOperationException: no GPU" >&2; echo "   at Worker.Run()" >&2; exit 1 ;;
        hang) trap 'echo stopped > "$FAKE_ROOT/stopped"; exit 130' TERM
              echo $$ > "$FAKE_ROOT/worker.pid"; echo '{"message":"Tracking person · 1/150","fraction":0}'
              while :; do sleep 0.1; done ;;
        esac
        echo 'a line that is not progress'
        echo '{"message":"Tracking person · 1/2","fraction":0.5}'
        cp "$FAKE_ROOT/motion.json" "$3"
        echo '{"message":"Motion is ready","fraction":1}'
        """;

    internal static async Task Run()
    {
        Parsing();
        // The fake ssh, worker and shells are Linux-only.
        if (!OperatingSystem.IsLinux()) { Console.WriteLine("PASS: SSH argument quoting and progress parsing (process checks run on Linux)"); return; }
        Quoting();
        var root = Path.Combine(Path.GetTempPath(), "mocap-ssh-" + Guid.NewGuid().ToString("N"));
        var bin = Path.Combine(root, "bin"); Directory.CreateDirectory(bin);
        var path = Environment.GetEnvironmentVariable("PATH");
        try
        {
            Script(Path.Combine(bin, "ssh"), FakeSsh); Script(Path.Combine(bin, "ffmpeg"), "#!/bin/sh"); Script(Path.Combine(bin, "ffprobe"), "#!/bin/sh");
            var worker = Path.Combine(root, "worker's dir", "humanoid-mocap-worker");
            Directory.CreateDirectory(Path.GetDirectoryName(worker)!); Script(worker, FakeWorker);
            Environment.SetEnvironmentVariable("FAKE_ROOT", root);
            var settings = new SshSettings { Enabled = true, Address = "me@gpu", Worker = worker };
            var seen = new List<(string Message, double? Fraction)>();
            var cache = Path.Combine(root, "captures");
            var capture = new SshCapture(settings, new CaptureOptions { Model = FlowModel.Base }, cache, (message, fraction) => seen.Add((message, fraction)));

            Environment.SetEnvironmentVariable("PATH", root);
            MotionTests.Equal(SshCapture.ClientMissing, capture.UnavailableReason);
            Environment.SetEnvironmentVariable("PATH", bin + Path.PathSeparator + path);
            MotionTests.Check(capture.UnavailableReason is null, "An address and an ssh client make the machine available");
            MotionTests.Check(new SshCapture(new SshSettings(), new CaptureOptions(), cache).UnavailableReason!.Contains("address"), "An unset address is explained");

            await SshCapture.CheckAsync(settings, default);
            var missing = await Throws<InvalidOperationException>(() => SshCapture.CheckAsync(new SshSettings { Address = "me@gpu", Worker = Path.Combine(root, "absent") }, default));
            MotionTests.Check(missing.Message.Contains("did not start"), "A missing worker fails the check: " + missing.Message);

            var video = Path.Combine(root, Awkward); var bytes = new byte[] { 1, 2, 3, 4, 5 };
            await File.WriteAllBytesAsync(video, bytes);
            var fixture = MotionTests.Fixture().ToDocument();
            await File.WriteAllTextAsync(Path.Combine(root, "motion.json"), fixture.ToJson());
            var messages = new List<string>();
            var output = await capture.CaptureAsync(video, 1.5, 3, messages.Add, default);
            var motion = MotionDocument.Parse(await File.ReadAllBytesAsync(output));
            MotionTests.Equal(cache, Path.GetDirectoryName(output));
            MotionTests.Equal(fixture.Frames.Count, motion.Frames.Count);
            MotionTests.Equal(video, motion.SourceVideo); MotionTests.Equal(Path.GetFileNameWithoutExtension(Awkward), motion.Name);
            var uploaded = await File.ReadAllBytesAsync(Path.Combine(root, "uploaded"));
            MotionTests.Check(bytes.SequenceEqual(uploaded), "The video arrives unchanged");
            var arguments = await File.ReadAllLinesAsync(Path.Combine(root, "arguments.txt"));
            var job = arguments[^3];
            MotionTests.Check(job.StartsWith(SshCapture.JobRoot, StringComparison.Ordinal) && !Directory.Exists(job), "The job directory is removed: " + job);
            MotionTests.Equal($"capture {job}/video.mp4 {job}/capture.hmotion --start 1.5 --model base --camera-interval 10 --cache {job} --end 3",
                string.Join(" ", arguments));
            MotionTests.Check(seen.Contains(("Tracking person · 1/2", 0.5)) && seen.Contains(("Motion is ready", 1))
                && !seen.Any(line => line.Message.Contains("not progress")), "Progress lines are forwarded: " + string.Join(" | ", seen));
            MotionTests.Check(messages.Contains("Uploading video to me@gpu…") && messages.Contains("Tracking person · 1/2"), "Progress reaches the capture callback");
            MotionTests.Check(!(await File.ReadAllTextAsync(Path.Combine(root, "ssh.log"))).Contains("Breakin"), "The user's file name never reaches the machine");

            Environment.SetEnvironmentVariable("FAKE_WORKER", "fail");
            var failure = await Throws<InvalidOperationException>(() => capture.CaptureAsync(video, 0, null, null, default));
            MotionTests.Check(failure.Message.Contains("error: System.InvalidOperationException: no GPU") && !failure.Message.Contains("Worker.Run"),
                "The worker's error, without its stack, fails the capture: " + failure.Message);
            MotionTests.Check(!Directory.Exists((await File.ReadAllLinesAsync(Path.Combine(root, "arguments.txt")))[^1]), "A failed job is removed");

            Environment.SetEnvironmentVariable("FAKE_WORKER", "hang");
            using (var cancellation = new CancellationTokenSource())
            {
                var running = capture.CaptureAsync(video, 0, null, null, cancellation.Token);
                var pid = Path.Combine(root, "worker.pid");
                for (var wait = 0; !File.Exists(pid) && wait < 200; wait++) await Task.Delay(50);
                var clock = Stopwatch.StartNew();
                cancellation.Cancel();
                await Throws<OperationCanceledException>(() => running);
                MotionTests.Check(File.Exists(Path.Combine(root, "stopped")), "Cancelling sends the remote worker TERM");
                MotionTests.Check(!Directory.Exists("/proc/" + (await File.ReadAllTextAsync(pid)).Trim()), "The remote worker has exited");
                MotionTests.Check(clock.Elapsed < TimeSpan.FromSeconds(10), "The worker stops through the session, before any kill fallback");
                MotionTests.Check(!Directory.Exists((await File.ReadAllLinesAsync(Path.Combine(root, "arguments.txt")))[^1]), "A cancelled job is removed");
            }

            Environment.SetEnvironmentVariable("FAKE_SSH_FAIL", "1");
            var unreachable = await Throws<InvalidOperationException>(() => SshCapture.CheckAsync(settings, default));
            MotionTests.Check(unreachable.Message.Contains("Connection refused"), "ssh's own error explains an unreachable machine: " + unreachable.Message);
        }
        finally
        {
            Environment.SetEnvironmentVariable("PATH", path);
            foreach (var name in new[] { "FAKE_ROOT", "FAKE_WORKER", "FAKE_SSH_FAIL" }) Environment.SetEnvironmentVariable(name, null);
            Directory.Delete(root, true);
        }
        Console.WriteLine("PASS: SSH capture quoting, upload, worker arguments, progress, download, failure output, cancellation and cleanup");
    }

    static void Parsing()
    {
        MotionTests.Equal(("Tracking person · 1/2", (double?)0.5), SshCapture.Progress("""{"message":"Tracking person · 1/2","fraction":0.5}"""));
        MotionTests.Equal(("Verifying…", (double?)null), SshCapture.Progress("""{"message":"Verifying…","fraction":null}"""));
        foreach (var other in new[] { "GPU: NVIDIA", "", "[1]", """{"fraction":1}""", """{"message":3}""", "{" })
            MotionTests.Check(SshCapture.Progress(other) is null, "Not a progress line: " + other);
        MotionTests.Equal("GPU: Fake\nerror: X: no GPU\nFile name: 'a'",
            SshCapture.Summary("GPU: Fake\nerror: X: no GPU\nFile name: 'a'\n   at A.B()\n--- End of stack trace from previous location ---\n   at C.D()\n"));
        MotionTests.Equal("capture /j/video.mp4 /j/capture.hmotion --start 1.5 --model base --camera-interval 30 --cache /j",
            string.Join(" ", SshCapture.WorkerArguments("/j", "video.mp4", new HostedCaptureOptions(FlowModel.Base, 30, 1.5, null))));
        MotionTests.Equal("video.mp4", SshCapture.VideoName("/home/a/" + Awkward.Replace(".mp4", ".MP4")));
        MotionTests.Equal("video", SshCapture.VideoName("/home/a/clip.we'rd"));
        MotionTests.Equal("./humanoid-mocap-worker/humanoid-mocap-worker", SshCapture.RemoteWorker(SshSettings.DefaultWorker));
        MotionTests.Equal("/opt/worker", SshCapture.RemoteWorker("/opt/worker"));
        var arguments = SshCapture.Arguments("me@gpu", "true");
        MotionTests.Check(arguments.Contains("BatchMode=yes") && arguments.Contains("-T") && arguments[^2] == "me@gpu" && arguments[^1] == "true",
            "ssh never prompts, allocates no terminal and takes one command");
        foreach (var address in new[] { "", "-oProxyCommand=touch x", "me@gpu host", "me@gpu\n" })
            MotionTests.Check(new SshSettings { Address = address }.Problem is not null, "Rejected address: " + address);
        MotionTests.Check(new SshSettings { Address = "zeljko@192.168.1.215" }.Problem is null, "A user@host address is accepted");
        MotionTests.Check(new SshSettings { Address = "me@gpu", Worker = " " }.Problem is not null, "An empty worker path is rejected");
    }

    // The remote command line must reach the script unchanged whatever the login shell is.
    static void Quoting()
    {
        var values = new[] { Awkward, "a\\b \\\\ 'c' \"d\" $HOME `id` ;|&*? ~ \t", "" };
        var expected = string.Concat(values.Select(value => value + "|"));
        foreach (var shell in new[] { "/bin/sh", "/bin/bash", "/usr/bin/fish" }.Where(File.Exists))
        {
            var info = new ProcessStartInfo(shell) { RedirectStandardOutput = true, UseShellExecute = false };
            info.ArgumentList.Add("-c"); info.ArgumentList.Add(SshCapture.Command("printf '%s|' \"$@\"", values));
            using var process = Process.Start(info)!;
            var output = process.StandardOutput.ReadToEnd(); process.WaitForExit();
            MotionTests.Equal(expected, output);
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("linux")]
    static void Script(string path, string text)
    {
        File.WriteAllText(path, text.Replace("\r\n", "\n") + "\n");
        File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
    }

    static async Task<T> Throws<T>(Func<Task> action) where T : Exception
    {
        try { await action(); } catch (T error) { return error; }
        throw new Exception("Expected " + typeof(T).Name);
    }
}
