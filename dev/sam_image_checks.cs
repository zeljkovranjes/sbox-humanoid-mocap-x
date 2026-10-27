using HumanoidMocapX.Inference;
using Sandbox;

namespace HumanoidMocapX.Checks;

/// <summary>Validates the reusable SAM image adapter against CPU inference.</summary>
public static class SamImageChecks
{
    /// <summary>Checks both conditioners, changed pixels, cancellation, invalid
    /// pixels and recovery with synthetic weights; requires the live editor.</summary>
    public static async Task<object> Run(CancellationToken cancellation)
    {
        var shape = DinoShape.Sam(32, 48, 64, 1); var backbone = new FlowWeights(OmegaFixture.Sam(shape.Channels, shape.Heads));
        var body = new FlowWeights(SamFixture.ImageConditioning(shape.Channels, false));
        var hand = new FlowWeights(SamFixture.ImageConditioning(shape.Channels, true));
        var pixels = Enumerable.Range(0, shape.ImageElements).Select(i => .5f + .4f * MathF.Sin(i * .013f)).ToArray();
        var rayEncoder = new SamRays(shape); var rays = new float[rayEncoder.Elements];
        rayEncoder.Prepare(new float[] { 700, 0, 320, 0, 530, 240, 0, 0, 1 }, new float[] { .5f, 0, 120, 0, .7f, -13 }, rays, cancellation);
        var cpu = new DinoImageEncoder(backbone, shape); var bodyCpu = new SamImageConditioning(body, shape);
        var handCpu = new SamImageConditioning(hand, shape, true);
        var features = new float[shape.Patches * shape.Channels]; var expected = new float[features.Length]; var actual = new float[features.Length];
        await GameTask.MainThread(); var gpu = new SamGpuImageEncoder(backbone, body, hand, shape); double maximum = 0;
        try
        {
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            await Reject<OperationCanceledException>(() => gpu.RunAsync(pixels, rays, actual, false, cancelled.Token));
            pixels[0] = float.NaN;
            await Reject<ArgumentException>(() => gpu.RunAsync(pixels, rays, actual, false, cancellation)); pixels[0] = .5f;
            for (var pass = 0; pass < 4; pass++)
            {
                var isHand = pass % 2 != 0; var conditioning = isHand ? handCpu : bodyCpu;
                if (pass == 2) Array.Reverse(pixels);
                await Task.Run(() => { cpu.Run(pixels, features, cancellation); conditioning.Run(features, rays, expected, cancellation); }, cancellation);
                await GameTask.MainThread(); await gpu.RunAsync(pixels, rays, actual, isHand, cancellation); Compare();
                conditioning.CopyPositions(expected); gpu.CopyPositions(actual, isHand); Compare();
            }
        }
        finally { await GameTask.MainThread(); gpu.Dispose(); }
        return new { Passed = true, LargestAbsoluteError = maximum };

        void Compare()
        {
            for (var i = 0; i < expected.Length; i++)
            {
                var error = Math.Abs((double)expected[i] - actual[i]); maximum = Math.Max(maximum, error);
                if (!float.IsFinite(actual[i]) || error > 2e-4 * Math.Max(1, Math.Abs(expected[i])))
                    throw new Exception($"SAM image adapter value {i}: expected {expected[i]:R}, got {actual[i]:R}.");
            }
        }
    }

    static async Task Reject<T>(Func<Task> action) where T : Exception
    {
        try { await action(); }
        catch (T) { return; }
        throw new Exception("SAM image adapter failed to reject input with " + typeof(T).Name);
    }
}
