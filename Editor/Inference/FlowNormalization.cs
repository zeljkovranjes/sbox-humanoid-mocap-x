// C# adaptation of FlowHMR's RMS normalization. See THIRD_PARTY_NOTICES.md.
using System;
using HumanoidMocapX.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

internal static class FlowNormalization
{
    internal static void Rms(Span<float> values, ReadOnlySpan<float> scale)
    {
        const float epsilon = 1e-6f;
        if (scale.Length < 1 || values.Length % scale.Length != 0)
            throw new ArgumentException("RMS normalization needs complete rows and one scale per channel.");
        for (var offset = 0; offset < values.Length; offset += scale.Length)
        {
            var row = values.Slice(offset, scale.Length);
            var squares = 0f;
            foreach (var value in row) squares += value * value;
            var inverse = 1f / MathF.Sqrt(squares / row.Length + epsilon);
            for (var channel = 0; channel < row.Length; channel++) row[channel] *= inverse * scale[channel];
        }
    }
}

/// <summary>Non-affine normalization using Humanoid Mocap's existing layer norm.</summary>
internal sealed class FlowLayerNorm
{
    readonly int width;
    readonly float epsilon;
    readonly Tensor scale, bias;

    internal FlowLayerNorm(int width, float epsilon)
    {
        if (width < 1 || !float.IsFinite(epsilon) || epsilon <= 0) throw new ArgumentException("Invalid layer normalization dimensions or epsilon.");
        this.width = width; this.epsilon = epsilon;
        var ones = new float[width]; Array.Fill(ones, 1f);
        scale = new("unit scale", ones, new[] { width }); bias = new("zero bias", new float[width], new[] { width });
    }

    internal void Run(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length % width != 0 || output.Length != input.Length) throw new ArgumentException("Layer normalization needs equal tensors with complete rows.");
        input.CopyTo(output);
        NeuralOps.LayerNorm(output, output.Length / width, width, scale, bias, epsilon);
    }
}
