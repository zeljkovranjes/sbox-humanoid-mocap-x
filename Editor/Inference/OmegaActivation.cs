using System;

namespace HumanoidMocapX.Inference;

internal static class OmegaActivation
{
    /// <summary>Error-function GELU with an erfc polynomial, preserving negative tails.
    /// Uses the same coefficients as the compute shader; input must already be finite.</summary>
    internal static void Gelu(Span<float> values)
    {
        for (var index = 0; index < values.Length; index++)
        {
            var value = values[index]; var z = MathF.Abs(value) * .7071067811865475f;
            var t = 1 / (1 + .3275911f * z);
            // Abramowitz-Stegun 7.1.26; maximum erfc approximation error is 1.5e-7.
            var erfc = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - .284496736f) * t + .254829592f) * t * MathF.Exp(-z * z);
            values[index] = value * (value >= 0 ? 1 - .5f * erfc : .5f * erfc);
        }
    }
}
