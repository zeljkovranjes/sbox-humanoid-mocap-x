// C# adaptation of SAM 3D Body's MHR parameter conversion. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;

namespace HumanoidMocapX.Inference;

/// <summary>Converts SAM's continuous body/hand predictions to MHR parameter order.</summary>
internal static class SamCompactPose
{
    internal const int BodyContinuous = 260, BodyParameters = 133, HandContinuous = 54, HandParameters = 27;
    static readonly int[] Triples = [0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 39, 44, 45, 46, 53, 54, 55,
        64, 65, 66, 85, 69, 73, 86, 70, 79, 87, 71, 82, 88, 72, 76, 91, 92, 93, 112, 96, 100,
        113, 97, 106, 114, 98, 109, 115, 99, 103, 130, 131, 132];
    static readonly int[] Singles = [1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52,
        56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90,
        94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123];
    static readonly int[] HandDofs = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1];

    /// <summary>Writes 133 parameters, including the hand and jaw slots that the
    /// head masks later. Rejects malformed, non-finite or overlapping buffers.</summary>
    internal static void Body(ReadOnlySpan<float> continuous, Span<float> output)
    {
        Validate(continuous, output, BodyContinuous, BodyParameters);
        Span<float> angles = stackalloc float[3];
        var triples = Triples; var singles = Singles;
        for (var i = 0; i < triples.Length; i += 3)
        {
            SamRotation.CompactXYZ(continuous.Slice(i * 2, 6), angles);
            for (var axis = 0; axis < 3; axis++) output[triples[i + axis]] = angles[axis];
        }
        var start = triples.Length * 2;
        for (var i = 0; i < singles.Length; i++) output[singles[i]] = MathF.Atan2(continuous[start + i * 2], continuous[start + i * 2 + 1]);
        continuous.Slice(BodyContinuous - 6).CopyTo(output.Slice(124, 6));
    }

    /// <summary>Writes 27 hand parameters in joint/axis order. Two-axis joints use
    /// two sine/cosine pairs; three-axis joints use two rotation-matrix columns.</summary>
    internal static void Hand(ReadOnlySpan<float> continuous, Span<float> output)
    {
        Validate(continuous, output, HandContinuous, HandParameters);
        var at = 0;
        foreach (var count in HandDofs)
        {
            if (count == 3) SamRotation.CompactXYZ(continuous.Slice(at * 2, 6), output.Slice(at, 3));
            else for (var axis = 0; axis < count; axis++) output[at + axis] = MathF.Atan2(continuous[(at + axis) * 2], continuous[(at + axis) * 2 + 1]);
            at += count;
        }
    }

    static void Validate(ReadOnlySpan<float> input, Span<float> output, int inputs, int outputs)
    {
        if (input.Length != inputs || output.Length != outputs || input.Overlaps(output))
            throw new ArgumentException("SAM compact pose dimensions are invalid or overlap.");
        foreach (var value in input) if (!float.IsFinite(value)) throw new ArgumentException("SAM compact pose must be finite.");
    }
}
