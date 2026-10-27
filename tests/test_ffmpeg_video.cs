using System.Diagnostics;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.EditorTools.Video;

static class FfmpegVideoTests
{
    internal static async Task Run()
    {
        // Windows decodes through Media Foundation; FFmpeg is the decoder elsewhere.
        if (OperatingSystem.IsWindows() || !FfmpegVideoDecoder.IsInstalled) return;
        var directory = Path.Combine(Path.GetTempPath(), "mocap-ffmpeg-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            // Two seconds of FFmpeg's test pattern: 60 frames at 30 fps, 64x48, H.264 4:2:0.
            var clip = Path.Combine(directory, "clip.mp4");
            var encode = Process.Start(new ProcessStartInfo("ffmpeg", ["-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30",
                "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip]) { RedirectStandardError = true })!;
            await encode.WaitForExitAsync();
            MotionTests.Check(encode.ExitCode == 0, "ffmpeg could not encode the test clip: " + await encode.StandardError.ReadToEndAsync());

            using (var decoder = new FfmpegVideoDecoder(clip))
            {
                MotionTests.Near(2, decoder.Duration, .05);
                var times = new List<double>();
                while (decoder.Read(default) is { } frame)
                {
                    MotionTests.Equal(64, frame.Width); MotionTests.Equal(48, frame.Height);
                    MotionTests.Equal(64 * 48 * 4, frame.Rgba.Length);
                    times.Add(frame.Time);
                }
                MotionTests.Equal(60, times.Count); MotionTests.Near(0, times[0], 1e-6);
                for (var i = 1; i < times.Count; i++) MotionTests.Near(i / 30d, times[i], 1e-6);
                // A seek decodes again from at or before the target.
                decoder.Seek(1.5);
                var resumed = decoder.Read(default);
                MotionTests.Check(resumed is not null && resumed.Time <= 1.5 + 1e-6, "Seek skipped past its target");
            }

            using (var decoder = IVideoDecoder.Open(clip))
            {
                decoder.Seek(.5);
                using var frames = await VideoFrames.CreateAsync(decoder.Read, decoder.Duration, .5, 1.5, directory, null, default);
                MotionTests.Equal(30, frames.Count); MotionTests.Equal(64, frames.Width); MotionTests.Equal(48, frames.Height);
            }
            MotionTests.Throws<FileNotFoundException>(() => new FfmpegVideoDecoder(Path.Combine(directory, "missing.mp4")));
            var text = Path.Combine(directory, "notes.mp4"); await File.WriteAllTextAsync(text, "not a video");
            MotionTests.Throws<InvalidDataException>(() => new FfmpegVideoDecoder(text));
        }
        finally { Directory.Delete(directory, true); }
        Console.WriteLine("PASS: FFmpeg decoding of H.264 frames, timestamps, seeks, the 30 fps timeline, missing and invalid files");
    }
}
