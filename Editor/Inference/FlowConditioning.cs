// C# adaptation of FlowHMR's timestep embedding. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

internal static class FlowConditioning
{
    internal static void Silu(Span<float> values)
    {
        for (var i = 0; i < values.Length; i++) values[i] /= 1 + MathF.Exp(-values[i]);
    }

    internal static void Gelu(Span<float> values)
    {
        const float sqrtTwoOverPi = .7978845608028654f, cubicCoefficient = .044715f;
        for (var i = 0; i < values.Length; i++)
        {
            var x = values[i];
            values[i] = .5f * x * (1 + MathF.Tanh(sqrtTwoOverPi * (x + cubicCoefficient * x * x * x)));
        }
    }

    internal static void Modulate(Span<float> values, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        if (shift.Length == 0 || shift.Length != scale.Length || values.Length % shift.Length != 0)
            throw new ArgumentException("Modulation needs complete rows and equal shift/scale vectors.");
        for (var i = 0; i < values.Length; i++) values[i] = values[i] * (1 + scale[i % scale.Length]) + shift[i % shift.Length];
    }

    internal static void Time(float time, float factor, Span<float> output)
    {
        if (!float.IsFinite(time) || !float.IsFinite(factor) || output.Length < 2)
            throw new ArgumentException("Time embedding needs a finite time/factor and at least two channels.");
        var half = output.Length / 2;
        for (var i = 0; i < half; i++)
        {
            var angle = time * factor * MathF.Exp(-MathF.Log(10000f) * i / half);
            output[i] = MathF.Cos(angle); output[half + i] = MathF.Sin(angle);
        }
        if (output.Length % 2 != 0) output[^1] = 0;
    }

    internal static void AddGated(Span<float> values, ReadOnlySpan<float> delta, ReadOnlySpan<float> gate)
    {
        if (gate.Length == 0 || values.Length % gate.Length != 0 || delta.Length != values.Length)
            throw new ArgumentException("Gated residual needs equal [tokens,width] tensors and a [width] gate.");
        for (var i = 0; i < values.Length; i++) values[i] += delta[i] * gate[i % gate.Length];
    }
}

/// <summary>Two linear layers separated by SiLU, with reusable intermediate storage.</summary>
internal sealed class FlowConditionMlp
{
    readonly FlowLinear first, second;
    readonly float[] hidden;

    internal FlowConditionMlp(FlowWeights weights, string name, int inputs, int width)
    {
        first = new(weights, name + ".0", inputs, width);
        second = new(weights, name + ".2", width, width);
        hidden = new float[width];
    }

    internal void Run(ReadOnlySpan<float> input, Span<float> output, CancellationToken cancellation)
    {
        first.Run(input, 1, hidden, cancellation);
        FlowConditioning.Silu(hidden);
        second.Run(hidden, 1, output, cancellation);
    }
}
