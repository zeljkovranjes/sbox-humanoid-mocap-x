// C# adaptation of VGGT-Omega's max-normalized axial RoPE. See LICENSE-VGGT and THIRD_PARTY_NOTICES.md.
// SAM's per-axis configuration also retains the DINOv3 terms in LICENSE-DINOV3.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
using System;

namespace HumanoidMocapX.Inference;

internal static class OmegaPosition
{
    /// <summary>Builds interleaved cos/sin pairs for split-half heads, with identity
    /// factors for prefix tokens. SAM normalizes each axis separately; Omega uses
    /// the longer side. Periods come from the checkpoint's rope buffer.</summary>
    internal static float[] Factors(int height, int width, int prefix, ReadOnlySpan<float> periods, bool separateAxes = false)
    {
        if (height < 1 || width < 1 || prefix < 0 || periods.Length < 1
            || ((long)height * width + prefix) > int.MaxValue / (4L * periods.Length))
            throw new ArgumentException("Invalid Omega position dimensions.");
        foreach (var period in periods)
            if (!float.IsFinite(period) || period <= 0) throw new ArgumentException("Omega rotary periods must be finite and positive.");
        var headWidth = periods.Length * 4;
        var result = new float[(height * width + prefix) * headWidth];
        for (var token = 0; token < prefix; token++)
            for (var pair = 0; pair < headWidth / 2; pair++) result[token * headWidth + pair * 2] = 1;
        var extent = Math.Max(height, width);
        for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
                for (var axis = 0; axis < 2; axis++)
                    for (var frequency = 0; frequency < periods.Length; frequency++)
                    {
                        var denominator = separateAxes ? (axis == 0 ? height : width) : extent;
                        var coordinate = 2 * ((axis == 0 ? y : x) + .5f) / denominator - 1;
                        var angle = 2 * MathF.PI * coordinate / periods[frequency];
                        var at = ((y * width + x + prefix) * headWidth) + (axis * periods.Length + frequency) * 2;
                        result[at] = MathF.Cos(angle); result[at + 1] = MathF.Sin(angle);
                    }
        return result;
    }

    internal static void Apply(Span<float> values, ReadOnlySpan<float> factors, int width, int headWidth)
    {
        if (headWidth < 4 || headWidth % 4 != 0 || width < headWidth || width % headWidth != 0
            || values.Length % width != 0 || factors.Length != (long)values.Length / width * headWidth || values.Overlaps(factors))
            throw new ArgumentException("Omega position tensors have inconsistent dimensions or overlap.");
        for (var row = 0; row < values.Length / width; row++)
            for (var head = 0; head < width / headWidth; head++)
                for (var pair = 0; pair < headWidth / 2; pair++)
                {
                    var index = row * width + head * headWidth + pair;
                    var other = index + headWidth / 2; var factor = row * headWidth + pair * 2;
                    var real = values[index]; var imaginary = values[other];
                    values[index] = real * factors[factor] - imaginary * factors[factor + 1];
                    values[other] = real * factors[factor + 1] + imaginary * factors[factor];
                }
    }
}
