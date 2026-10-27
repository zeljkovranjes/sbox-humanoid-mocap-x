using System.Diagnostics;
using System.Text.Json;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;
using Sandbox;
using static Sandbox.Internal.GlobalGameNamespace;

namespace HumanoidMocapX.Checks;

/// <summary>Validates body/left/right feature extraction on actual RGBA video frames.</summary>
public static class SamFeatureChecks
{
    public sealed record Frame(string RgbaPath, int Width, int Height);

    /// <summary>Loads released models once, checks each frame twice and writes full
    /// feature/landmark evidence. This is not video decoding, temporal tracking or export.</summary>
    public static async Task<object> Run(string samPath, string mhrPath, string detectorPath, string framesPath, string resultPath, CancellationToken cancellation)
    {
        var frames = JsonSerializer.Deserialize<Frame[]>(await File.ReadAllTextAsync(framesPath, cancellation));
        if (frames is null || frames.Length == 0) throw new ArgumentException("At least one feature-validation frame is required.");
        var pixels = new byte[frames.Length][]; var boxes = new float[frames.Length][];
        var network = new YoloNetwork(); var detectorWeights = await Task.Run(() => YoloCheckpoint.Load(detectorPath, network, cancellation), cancellation);
        await GameTask.MainThread(); var detector = new YoloGpuDetector(network, detectorWeights);
        try
        {
            for (var i = 0; i < frames.Length; i++)
            {
                pixels[i] = await File.ReadAllBytesAsync(frames[i].RgbaPath, cancellation);
                await GameTask.MainThread(); var found = new YoloBox[YoloNetwork.Predictions];
                var count = await detector.DetectAsync(pixels[i], frames[i].Width, frames[i].Height, found, cancellation);
                if (count == 0) throw new Exception("No person found in " + frames[i].RgbaPath);
                boxes[i] = [found[0].Left, found[0].Top, found[0].Right, found[0].Bottom];
            }
        }
        finally { await GameTask.MainThread(); detector.Dispose(); }
        var timer = Stopwatch.StartNew();
        var extractor = await SamFeatures.LoadAsync(samPath, mhrPath, stage => Log.Info(stage), cancellation);
        var loadMilliseconds = timer.Elapsed.TotalMilliseconds; var results = new List<object>(); var timings = new List<double[]>();
        try
        {
            for (var i = 0; i < frames.Length; i++)
            {
                var frame = frames[i]; var focal = (float)Math.Sqrt((double)frame.Width * frame.Width + (double)frame.Height * frame.Height);
                var intrinsics = new float[] { focal, 0, frame.Width / 2f, 0, focal, frame.Height / 2f, 0, 0, 1 };
                var features = new float[SamGpuFeatures.Elements];
                await GameTask.MainThread();
                using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
                await Reject<OperationCanceledException>(() => extractor.RunAsync(pixels[i], frame.Width, frame.Height, boxes[i], intrinsics, features, cancelled.Token));
                intrinsics[0] = float.NaN;
                await Reject<ArgumentException>(() => extractor.RunAsync(pixels[i], frame.Width, frame.Height, boxes[i], intrinsics, features, cancellation));
                intrinsics[0] = focal; timer.Restart();
                var pending = extractor.RunAsync(pixels[i], frame.Width, frame.Height, boxes[i], intrinsics, features, cancellation);
                try { await Reject<InvalidOperationException>(() => extractor.RunAsync(pixels[i], frame.Width, frame.Height, boxes[i], intrinsics, features, cancellation)); }
                finally { await pending; }
                var firstTime = timer.Elapsed.TotalMilliseconds;
                var previous = features.ToArray(); var points = extractor.ImageKeypoints.ToArray(); var hands = extractor.HandBoxes.ToArray();
                await GameTask.MainThread(); timer.Restart();
                await extractor.RunAsync(pixels[i], frame.Width, frame.Height, boxes[i], intrinsics, features, cancellation);
                var repeatedTime = timer.Elapsed.TotalMilliseconds;
                if (!previous.SequenceEqual(features) || !points.AsSpan().SequenceEqual(extractor.ImageKeypoints) || !hands.AsSpan().SequenceEqual(extractor.HandBoxes))
                    throw new Exception("SAM body/left/right feature extraction did not repeat exactly for " + frame.RgbaPath);
                SamDecoder.Finite(features);
                if (features.AsSpan(1024, 1024).SequenceEqual(features.AsSpan(2048, 1024))) throw new Exception("Both hands unexpectedly produced identical tokens.");
                timings.Add([firstTime, repeatedTime]);
                results.Add(new
                {
                    frame.RgbaPath,
                    frame.Width,
                    frame.Height,
                    PersonBox = boxes[i],
                    Intrinsics = intrinsics,
                    HandBoxes = hands,
                    ImageKeypoints = points,
                    Features = features,
                    Milliseconds = timings[^1]
                });
            }
        }
        finally { await GameTask.MainThread(); extractor.Dispose(); }
        await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(new { LoadMilliseconds = loadMilliseconds, Frames = results },
            new JsonSerializerOptions { WriteIndented = true }), cancellation);
        return new
        {
            Passed = true,
            Frames = frames.Length,
            FeaturesPerFrame = SamGpuFeatures.Elements,
            ExactRepeats = true,
            LoadMilliseconds = loadMilliseconds,
            FrameMilliseconds = timings,
            ResultPath = resultPath
        };
    }

    static async Task Reject<T>(Func<Task> action) where T : Exception
    {
        try { await action(); } catch (T) { return; }
        throw new Exception("SAM feature adapter failed to reject input with " + typeof(T).Name);
    }
}
