// C# adaptation of FlowHMR's x1 Euler sampling. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;

namespace HumanoidMocapX;

/// <summary>Predicts the clean 491-channel motion latent at a normalized time in [0,1).
/// Writes every output element; failures and cancellation propagate.</summary>
public delegate void FlowPrediction(ReadOnlySpan<float> state, float time, Span<float> output, CancellationToken cancellation);

/// <summary>The released Base and Latest models use twenty Euler steps and x1 prediction.</summary>
public static class FlowSampler
{
    public const int Channels = 491;
    public const int Steps = 20;

    /// <summary>Replaces caller-supplied initial noise with sampled motion using a reusable
    /// prediction buffer. Rejects invalid shapes and non-finite predictions. A failed or
    /// cancelled run may leave state partially advanced; it must not be exported.</summary>
    public static void Sample(float[] state, float[] prediction, FlowPrediction predict,
        CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(predict);
        if (state.Length == 0 || state.Length % Channels != 0 || state.Length / Channels > FlowMotion.MaximumFrames
            || prediction.Length != state.Length || ReferenceEquals(state, prediction))
            throw new ArgumentException("Sampling needs separate, equally sized [frames,491] state and prediction buffers.");
        CheckFinite(state);
        const float minimumRemaining = 1f / Steps;
        for (var step = 0; step < Steps; step++)
        {
            cancellation.ThrowIfCancellationRequested();
            var time = step / (float)Steps;
            var delta = (step + 1) / (float)Steps - time;
            predict(state, time, prediction, cancellation);
            cancellation.ThrowIfCancellationRequested();
            CheckFinite(prediction);
            var remaining = Math.Max(1f - time, minimumRemaining);
            for (var i = 0; i < state.Length; i++)
                state[i] += (prediction[i] - state[i]) / remaining * delta;
            CheckFinite(state);
        }
    }

    static void CheckFinite(ReadOnlySpan<float> values)
    {
        foreach (var value in values)
            if (!float.IsFinite(value)) throw new ArithmeticException("FlowHMR sampling produced or received a non-finite latent.");
    }
}
