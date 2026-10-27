using System.Security.Cryptography;
using HumanoidMocapX.Core.Preview;
using HumanoidMocapX.EditorTools.Video;

static class PreviewTests
{
    internal static async Task Run()
    {
        Pixels(); Playback(); await Cache();
        Console.WriteLine("PASS: bounded preview pixels, allocation-free playback decisions, streamed video cache and cancellation");
    }

    static void Pixels()
    {
        var frame = new PreviewFrameBuffer();
        byte[] source = { 255, 10, 20, 255, 30, 240, 50, 128, 60, 70, 230, 0, 80, 90, 100, 255 };
        frame.Copy(source, 2, 2);
        MotionTests.Equal(2, frame.Width); MotionTests.Equal(2, frame.Height); MotionTests.Equal(1L, frame.Version);
        MotionTests.Check(frame.Pixels.SequenceEqual(source), "Preview changed RGBA channel order or alpha");
        source[0] = 0; MotionTests.Equal((byte)255, frame.Pixels[0]);
        var large = new byte[1920 * 1080 * 4];
        for (var y = 0; y < 1080; y++)
            for (var x = 0; x < 1920; x++)
            {
                var offset = (y * 1920 + x) * 4;
                large[offset] = (byte)x; large[offset + 1] = (byte)y; large[offset + 2] = (byte)(x + y); large[offset + 3] = 255;
            }
        frame.Copy(large, 1920, 1080); MotionTests.Equal(960, frame.Width); MotionTests.Equal(540, frame.Height);
        for (var y = 0; y < 540; y++)
            for (var x = 0; x < 960; x++)
                MotionTests.Check(frame.Pixels.Slice((y * 960 + x) * 4, 4).SequenceEqual(large.AsSpan((y * 2 * 1920 + x * 2) * 4, 4)), "Preview sampled the wrong pixel");
        var portrait = new byte[1080 * 1920 * 4]; frame.Copy(portrait, 1080, 1920);
        MotionTests.Equal(303, frame.Width); MotionTests.Equal(540, frame.Height);
        for (var i = 0; i < 3; i++) frame.Copy(portrait, 1080, 1920);
        var allocated = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 8; i++) frame.Copy(portrait, 1080, 1920);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - allocated);
        var version = frame.Version;
        MotionTests.Throws<ArgumentException>(() => frame.Copy(new byte[4], 0, 1));
        MotionTests.Throws<ArgumentException>(() => frame.Copy(new byte[4], 2, 1));
        MotionTests.Throws<ArgumentException>(() => frame.Copy(new byte[4], int.MaxValue, int.MaxValue));
        MotionTests.Equal(version, frame.Version);
    }

    static void Playback()
    {
        var playback = new PreviewPlayback();
        MotionTests.Near(0, playback.SeekTime(0, 8, 0, true)!.Value);
        MotionTests.Check(playback.SeekTime(.033, 8, .025, true) is null, "Normal playback unnecessarily seeks");
        MotionTests.Near(4, playback.SeekTime(4, 8, .05, true)!.Value);
        MotionTests.Near(0, playback.SeekTime(0, 8, 7.99, true)!.Value);
        MotionTests.Near(1, playback.SeekTime(1, 8, 1.05, false)!.Value);
        MotionTests.Check(playback.SeekTime(1.01, 8, 1, false) is null, "Paused sub-frame movement unnecessarily seeks");
        MotionTests.Near(31d / 30, playback.SeekTime(1.04, 8, 1, false)!.Value);
        MotionTests.Near(7.999, playback.SeekTime(20, 8, 1, true)!.Value, 1e-6);
        MotionTests.Near(0, playback.SeekTime(0, .0001, 0, false)!.Value);
        MotionTests.Throws<ArgumentException>(() => playback.SeekTime(double.NaN, 8, 0, true));
        MotionTests.Throws<ArgumentException>(() => playback.SeekTime(0, 0, 0, true));
        MotionTests.Throws<ArgumentException>(() => playback.SeekTime(0, 8, -1, true));
        playback.SeekTime(1, 8, 1, true);
        var allocated = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 1000; i++) playback.SeekTime(1, 8, 1, true);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - allocated);
    }

    static async Task Cache()
    {
        var folder = Path.Combine(Path.GetTempPath(), "mocap-preview-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(folder);
        try
        {
            var source = Path.Combine(folder, "source.MP4"); var cache = Path.Combine(folder, "cache");
            var data = Enumerable.Range(0, 200_003).Select(i => (byte)(i * 37)).ToArray(); await File.WriteAllBytesAsync(source, data);
            var expected = Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant() + ".mp4";
            var filename = await VideoPreviewCache.CopyAsync(source, cache, default);
            MotionTests.Equal(expected, filename);
            MotionTests.Check((await File.ReadAllBytesAsync(Path.Combine(cache, filename))).SequenceEqual(data), "Preview cache changed the source bytes");
            MotionTests.Equal(filename, await VideoPreviewCache.CopyAsync(source, cache, default));
            await File.WriteAllBytesAsync(Path.Combine(cache, filename), new byte[] { 1, 2, 3 });
            await VideoPreviewCache.CopyAsync(source, cache, default);
            MotionTests.Check((await File.ReadAllBytesAsync(Path.Combine(cache, filename))).SequenceEqual(data), "Preview cache reused corrupted footage");
            MotionTests.Check((await File.ReadAllBytesAsync(source)).SequenceEqual(data), "Preview staging modified the source");
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            await Throws<OperationCanceledException>(() => VideoPreviewCache.CopyAsync(source, cache, cancelled.Token));
            var empty = Path.Combine(folder, "empty.mp4"); await File.WriteAllBytesAsync(empty, Array.Empty<byte>());
            await Throws<InvalidDataException>(() => VideoPreviewCache.CopyAsync(empty, cache, default));
            await Throws<FileNotFoundException>(() => VideoPreviewCache.CopyAsync(Path.Combine(folder, "missing.mp4"), cache, default));
            MotionTests.Equal(1, Directory.GetFiles(cache).Length);
            MotionTests.Equal(0, Directory.GetFiles(cache, "*.partial").Length);
        }
        finally { Directory.Delete(folder, true); }
    }

    static async Task Throws<T>(Func<Task> action) where T : Exception
    {
        try { await action(); } catch (T) { return; }
        throw new Exception("Expected " + typeof(T).Name);
    }
}
