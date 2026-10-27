using System.Diagnostics;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;
using Sandbox;

namespace HumanoidMocapX.Checks;

/// <summary>Live-editor checks using installed checkpoints and supplied image fixtures.</summary>
public static class ReleasedGpuChecks
{
    /// <summary>Loads a verified YOLOX-L checkpoint and detects people twice in an
    /// RGBA8 fixture. Throws on invalid input, missing people or unstable results.
    /// Input must depict at least one visible person; this is not a video-capture test.</summary>
    public static async Task<object> Yolo(string checkpoint, string rgbaFile, int width, int height,
        CancellationToken cancellation)
    {
        var network = new YoloNetwork();
        var filters = await Task.Run(() => YoloCheckpoint.Load(checkpoint, network, cancellation), cancellation);
        var image = await File.ReadAllBytesAsync(rgbaFile, cancellation);
        await GameTask.MainThread();
        var detector = new YoloGpuDetector(network, filters);
        try
        {
            var first = new YoloBox[YoloNetwork.Predictions]; var second = new YoloBox[first.Length];
            var timer = Stopwatch.StartNew();
            var count = await detector.DetectAsync(image, width, height, first, cancellation);
            var firstMilliseconds = timer.Elapsed.TotalMilliseconds;
            if (count == 0) throw new Exception("Released YOLOX-L weights detected no people in the supplied fixture.");
            await GameTask.MainThread(); timer.Restart();
            var repeated = await detector.DetectAsync(image, width, height, second, cancellation);
            var repeatMilliseconds = timer.Elapsed.TotalMilliseconds;
            if (repeated != count) throw new Exception("Repeated detection changed the person count.");
            for (var i = 0; i < count; i++)
                if (first[i] != second[i]) throw new Exception("Repeated detection changed person " + i + ".");
            return new { People = first.AsSpan(0, count).ToArray(), FirstMilliseconds = firstMilliseconds, RepeatMilliseconds = repeatMilliseconds };
        }
        finally { await GameTask.MainThread(); detector.Dispose(); }
    }
}
