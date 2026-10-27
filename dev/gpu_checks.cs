using HumanoidMocapX.Inference;
using Sandbox;

namespace HumanoidMocapX.Checks;

/// <summary>Live-editor numerical comparison of the complete GPU sampler and the CPU reference.</summary>
public static class FlowGpuChecks
{
    /// <summary>Runs synthetic 64-channel-head models through all eighteen blocks and twenty
    /// sampling steps. Throws on disagreement, malformed-input acceptance or cancellation failure.</summary>
    public static async Task<object> Run(CancellationToken cancellation)
    {
        double largestError = 0;
        var captures = 0;
        foreach (var (frames, heads) in new[] { (1, 1), (3, 2), (67, 1) })
        {
            cancellation.ThrowIfCancellationRequested();
            var width = heads * FlowGpuWorkspace.HeadWidth;
            var weights = new FlowWeights(DenoiserTests.Fixture(width, heads));
            var body = DenoiserTests.Wave(frames * FlowDenoiser.FeatureChannels, .037f);
            var camera = DenoiserTests.Wave(frames * FlowDenoiser.CameraChannels, .17f);
            var initial = DenoiserTests.Wave(frames * FlowSampler.Channels, .019f);
            var cpu = new FlowDenoiser(weights, frames, width, heads);
            await GameTask.MainThread();
            var gpu = new FlowGpuSampler(weights, frames, width, heads);
            try
            {
                using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
                await MustCancel(() => gpu.SampleAsync(body, camera, initial, new float[initial.Length], cancelled.Token));
                try
                {
                    await gpu.SampleAsync(body, camera, initial, new float[1], cancellation);
                    throw new Exception("GPU sampler accepted an invalid output shape.");
                }
                catch (ArgumentException) { }
                for (var pass = 0; pass < 2; pass++)
                {
                    var expected = await Task.Run(() =>
                    {
                        var values = (float[])initial.Clone();
                        cpu.SetVideo(body, camera, cancellation);
                        FlowSampler.Sample(values, new float[values.Length], cpu.Predict, cancellation);
                        return values;
                    }, cancellation);
                    await GameTask.MainThread();
                    var actual = new float[expected.Length];
                    await gpu.SampleAsync(body, camera, initial, actual, cancellation);
                    for (var i = 0; i < actual.Length; i++)
                    {
                        var error = Math.Abs((double)expected[i] - actual[i]);
                        largestError = Math.Max(largestError, error);
                        if (!float.IsFinite(actual[i]) || error > 1e-3 * Math.Max(1, Math.Abs(expected[i])))
                            throw new Exception($"GPU sampler [{frames} frames, {heads} heads], capture {pass}, value {i}: expected {expected[i]:R}, got {actual[i]:R}.");
                    }
                    captures++;
                    for (var i = 0; i < body.Length; i++) body[i] *= -.4f;
                    for (var i = 0; i < camera.Length; i++) camera[i] *= .7f;
                }
            }
            finally { await GameTask.MainThread(); gpu.Dispose(); }
        }
        return new { Captures = captures, LargestAbsoluteError = largestError, Steps = FlowSampler.Steps };
    }

    static async Task MustCancel(Func<Task> run)
    {
        try { await run(); } catch (OperationCanceledException) { return; }
        throw new Exception("GPU sampler ignored cancellation.");
    }
}
