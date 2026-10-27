using HumanoidMocapX.EditorTools.Inference;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;

static class VideoFramesTests
{
    internal static async Task Run()
    {
        var directory = Path.Combine(Path.GetTempPath(), "mocap-frames-" + Guid.NewGuid().ToString("N"));
        try
        {
            var timeline = new CaptureTimeline(10, 2, 7);
            MotionTests.Equal(150, timeline.Count); MotionTests.Near(2, timeline.Time(0)); MotionTests.Near(2 + 149 / 30d, timeline.Time(149));
            MotionTests.Throws<ArgumentException>(() => new CaptureTimeline(10, 2, 11));
            MotionTests.Throws<ArgumentException>(() => new CaptureTimeline(10, double.NaN, null));
            MotionTests.Throws<ArgumentException>(() => new CaptureTimeline(200, 0, null));
            MotionTests.Throws<ArgumentException>(() => new CaptureTimeline(10, 2, 2));
            MotionTests.Throws<ArgumentOutOfRangeException>(() => timeline.Time(150));
            var samples = new[] { Frame(0, 10), Frame(.04, 20), Frame(.09, 30), Frame(.14, 40) };
            using (var frames = await Cache(samples, 0, .2, directory))
            {
                MotionTests.Equal(6, frames.Count); MotionTests.Equal(1, frames.Width); MotionTests.Equal(1, frames.Height);
                byte[] expected = [10, 10, 20, 30, 30, 40]; var rgba = new byte[4];
                // Out-of-order reads must use the cache rather than decoder seeks.
                for (var i = expected.Length - 1; i >= 0; i--)
                {
                    await frames.ReadAsync(i, rgba, default);
                    MotionTests.Check(rgba.SequenceEqual(new byte[] { expected[i], 17, 29, 255 }), "Decoded RGBA or timestamp selection changed");
                }
                await Reject<ArgumentException>(() => frames.ReadAsync(6, rgba, default).AsTask());
                await Reject<ArgumentException>(() => frames.ReadAsync(0, new byte[3], default).AsTask());
                using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
                await Reject<OperationCanceledException>(() => frames.ReadAsync(0, rgba, cancelled.Token).AsTask());
            }
            Empty(directory);
            using (var trimmed = await Cache(samples, .05, .15, directory))
            {
                var rgba = new byte[4]; MotionTests.Equal(3, trimmed.Count);
                await trimmed.ReadAsync(0, rgba, default); MotionTests.Equal((byte)20, rgba[0]);
                await trimmed.ReadAsync(2, rgba, default); MotionTests.Equal((byte)30, rgba[0]);
            }
            var sixty = Enumerable.Range(0, 12).Select(i => Frame(i / 60d, (byte)i)).ToArray();
            using (var frames = await Cache(sixty, 0, .2, directory))
            {
                var rgba = new byte[4];
                for (var i = 0; i < 6; i++) { await frames.ReadAsync(i, rgba, default); MotionTests.Equal((byte)(i * 2), rgba[0]); }
            }
            await Reject<InvalidDataException>(() => Cache([], 0, .2, directory));
            await Reject<InvalidDataException>(() => Cache([Frame(.01, 0)], 0, .2, directory));
            await Reject<InvalidDataException>(() => Cache([Frame(0, 0), Frame(0, 1)], 0, .2, directory));
            await Reject<InvalidDataException>(() => Cache([Frame(0, 0), Frame(double.NaN, 1)], 0, .2, directory));
            await Reject<InvalidDataException>(() => Cache([Frame(0, 0), new(new byte[8], 2, 1, .1)], 0, .2, directory));
            await Reject<InvalidDataException>(() => Cache([new(new byte[3], 1, 1, 0)], 0, .2, directory));
            using (var cancellation = new CancellationTokenSource())
                await Reject<OperationCanceledException>(() => Cache(samples, 0, .2, directory, _ => cancellation.Cancel(), cancellation.Token));
            await Reject<IOException>(() => VideoFrames.CreateAsync(_ => throw new IOException("decode fixture failed"), .2, 0, .2, directory, null, default));
            Empty(directory);
            Console.WriteLine("PASS: timestamped 30 fps cache, variable/high frame rates, trimmed ranges, exact RGBA, random reads, malformed frames and cancellation cleanup");
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
    }

    static DecodedVideoFrame Frame(double time, byte value) => new([value, 17, 29, 255], 1, 1, time);
    static Task<VideoFrames> Cache(DecodedVideoFrame[] samples, double start, double end, string directory, Action<string> progress = null, CancellationToken token = default)
    {
        var index = 0;
        return VideoFrames.CreateAsync(_ => index == samples.Length ? null : samples[index++], .2, start, end, directory, progress, token);
    }
    static void Empty(string directory) => MotionTests.Equal(0, Directory.GetFiles(directory).Length);
    internal static async Task Reject<T>(Func<Task> action) where T : Exception
    {
        try { await action(); }
        catch (T) { return; }
        throw new Exception("Expected " + typeof(T).Name);
    }
}
