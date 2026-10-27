using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Creates standard normal initial latents without an external tensor runtime.</summary>
internal static class FlowNoise
{
    /// <summary>Fills complete 491-channel frames using a seeded normal distribution.
    /// The seed is repeatable in this runtime, but does not reproduce PyTorch's RNG.</summary>
    internal static void Fill(Span<float> output, int seed, CancellationToken cancellation = default)
    {
        if (output.Length == 0 || output.Length % FlowSampler.Channels != 0 || output.Length / FlowSampler.Channels > CaptureOptions.MaximumFrames)
            throw new ArgumentException("FlowHMR noise requires 1–5000 complete frames.");
        cancellation.ThrowIfCancellationRequested(); var random = new Random(seed);
        for (var i = 0; i < output.Length; i += 2)
        {
            if ((i & 4095) == 0) cancellation.ThrowIfCancellationRequested();
            double x, y, radius;
            do { x = random.NextDouble() * 2 - 1; y = random.NextDouble() * 2 - 1; radius = x * x + y * y; } while (radius >= 1 || radius == 0);
            var scale = Math.Sqrt(-2 * Math.Log(radius) / radius);
            output[i] = (float)(x * scale); if (i + 1 < output.Length) output[i + 1] = (float)(y * scale);
        }
    }
}
