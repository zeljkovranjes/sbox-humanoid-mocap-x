// C# adaptation of SAM 3D Body's camera encoding. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Encodes crop-camera rays at DINO patch resolution without allocating a full image of rays.</summary>
internal sealed class SamRays
{
    internal const int Bands = 16, Channels = 99;
    readonly float[] horizontal, vertical;
    internal int Elements => horizontal.Length * vertical.Length * Channels;

    internal SamRays(DinoShape shape)
    {
        if (!shape.IsSam) throw new ArgumentException("Camera conditioning requires a SAM image shape.", nameof(shape));
        horizontal = Coordinates(shape.Width); vertical = Coordinates(shape.Height);
    }

    /// <summary>Consumes row-major full-image intrinsics[3,3] and full-to-crop affine[2,3].
    /// Writes [patches,99]. Rotation/skew, invalid cameras and overlapping buffers fail.
    /// Coordinates refer to the already flipped image when preparing the left hand.</summary>
    internal void Prepare(ReadOnlySpan<float> intrinsics, ReadOnlySpan<float> affine, Span<float> output,
        CancellationToken cancellation = default)
    {
        if (intrinsics.Length != 9 || affine.Length != 6 || output.Length != Elements
            || intrinsics.Overlaps(output) || affine.Overlaps(output)) throw new ArgumentException("SAM ray tensors have invalid dimensions or overlap.");
        SamDecoder.Finite(intrinsics); SamDecoder.Finite(affine); cancellation.ThrowIfCancellationRequested();
        if (intrinsics[0] <= 0 || intrinsics[4] <= 0 || affine[0] <= 0 || affine[4] <= 0
            || intrinsics[1] != 0 || intrinsics[3] != 0 || intrinsics[6] != 0 || intrinsics[7] != 0 || intrinsics[8] != 1
            || affine[1] != 0 || affine[3] != 0)
            throw new ArgumentException("SAM rays need positive focal/crop scales, no rotation/skew and a canonical camera last row.");
        for (var y = 0; y < vertical.Length; y++)
        {
            cancellation.ThrowIfCancellationRequested();
            for (var x = 0; x < horizontal.Length; x++)
            {
                var row = output.Slice((y * horizontal.Length + x) * Channels, Channels);
                row[0] = (horizontal[x] / affine[0] - affine[2] / affine[0] - intrinsics[2]) / intrinsics[0];
                row[1] = (vertical[y] / affine[4] - affine[5] / affine[4] - intrinsics[5]) / intrinsics[4]; row[2] = 1;
                for (var axis = 0; axis < 3; axis++)
                    for (var band = 0; band < Bands; band++)
                    {
                        var angle = MathF.PI * (row[axis] * (1 + 31f * band / (Bands - 1)));
                        row[3 + axis * Bands + band] = MathF.Sin(angle);
                        row[3 + 3 * Bands + axis * Bands + band] = MathF.Cos(angle);
                    }
            }
        }
        SamDecoder.Finite(output);
    }

    static float[] Coordinates(int pixels)
    {
        const int scale = DinoShape.PatchSize;
        var coordinates = new float[pixels / scale];
        // Rays are affine in pixel position; filtering coordinates preserves edge antialiasing without a 512x512 ray buffer.
        for (var patch = 0; patch < coordinates.Length; patch++)
        {
            var center = (patch + .5f) * scale; double sum = 0, total = 0;
            var start = Math.Max(0, (int)(center - scale + .5f)); var end = Math.Min(pixels, (int)(center + scale + .5f));
            for (var pixel = start; pixel < end; pixel++)
            {
                var weight = Math.Max(0, 1 - Math.Abs((pixel + .5 - center) / scale));
                sum += weight * pixel; total += weight;
            }
            coordinates[patch] = (float)(sum / total);
        }
        return coordinates;
    }
}
