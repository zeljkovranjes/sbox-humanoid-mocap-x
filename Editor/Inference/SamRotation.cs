// C# adaptation of SAM 3D Body's rotation conventions. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Global Euler conversion adapts RoMa/SciPy. See LICENSE-ROMA and LICENSE-SCIPY.
using System;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

internal static class SamRotation
{
    const float NormalizeEpsilon = 1e-12f, CompactSingularity = 1e-6f, GlobalSingularity = 1e-7f;

    // SAM's compact angles use cross-product orthogonalization, unlike its global head.
    internal static void CompactXYZ(ReadOnlySpan<float> columns, Span<float> angles)
    {
        var x = Normalize(new(columns[0], columns[1], columns[2]));
        var z = Normalize(Vector3.Cross(x, new(columns[3], columns[4], columns[5])));
        var y = Vector3.Cross(z, x);
        var horizontal = MathF.Sqrt(x.X * x.X + x.Y * x.Y);
        angles[0] = horizontal < CompactSingularity ? MathF.Atan2(-z.Y, y.Y) : MathF.Atan2(y.Z, z.Z);
        angles[1] = MathF.Atan2(-x.Z, horizontal);
        angles[2] = horizontal < CompactSingularity ? 0 : MathF.Atan2(x.Y, x.X);
    }

    internal static void GlobalZYX(ReadOnlySpan<float> columns, Span<float> angles)
    {
        var x = Normalize(new(columns[0], columns[1], columns[2]));
        var raw = new Vector3(columns[3], columns[4], columns[5]);
        var y = Normalize(raw - Vector3.Dot(x, raw) * x); var z = Vector3.Cross(x, y);
        Span<float> matrix = stackalloc float[9] { x.X, y.X, z.X, x.Y, y.Y, z.Y, x.Z, y.Z, z.Z };
        Euler(matrix, angles, true);
    }

    internal static void MatrixXYZ(ReadOnlySpan<float> angles, Span<float> matrix)
    {
        var sx = MathF.Sin(angles[0]); var cx = MathF.Cos(angles[0]);
        var sy = MathF.Sin(angles[1]); var cy = MathF.Cos(angles[1]);
        var sz = MathF.Sin(angles[2]); var cz = MathF.Cos(angles[2]);
        matrix[0] = cy * cz; matrix[1] = sx * sy * cz - cx * sz; matrix[2] = cx * sy * cz + sx * sz;
        matrix[3] = cy * sz; matrix[4] = sx * sy * sz + cx * cz; matrix[5] = cx * sy * sz - sx * cz;
        matrix[6] = -sy; matrix[7] = sx * cy; matrix[8] = cx * cy;
    }

    internal static void Euler(ReadOnlySpan<float> matrix, Span<float> angles, bool intrinsicZYX = false)
    {
        var rotation = Quaternion(matrix);
        var a = rotation.W - rotation.Y; var b = rotation.X + rotation.Z;
        var c = rotation.Y + rotation.W; var d = rotation.Z - rotation.X;
        var middle = 2 * MathF.Atan2(float.Hypot(c, d), float.Hypot(a, b));
        var halfSum = MathF.Atan2(b, a); var halfDifference = MathF.Atan2(d, c);
        angles[0] = intrinsicZYX ? halfSum + halfDifference : halfSum - halfDifference;
        angles[2] = intrinsicZYX ? halfSum - halfDifference : halfSum + halfDifference;
        if (MathF.Abs(middle) <= GlobalSingularity || MathF.Abs(middle - MathF.PI) <= GlobalSingularity)
        {
            angles[2] = 0;
            angles[0] = MathF.Abs(middle) <= GlobalSingularity ? 2 * halfSum : 2 * (intrinsicZYX ? 1 : -1) * halfDifference;
        }
        angles[1] = middle - MathF.PI / 2;
        for (var i = 0; i < 3; i++)
        {
            if (angles[i] < -MathF.PI) angles[i] += 2 * MathF.PI;
            if (angles[i] > MathF.PI) angles[i] -= 2 * MathF.PI;
        }
    }

    static System.Numerics.Quaternion Quaternion(ReadOnlySpan<float> m)
    {
        Span<float> diagonal = stackalloc float[4] { 1 + m[0] - m[4] - m[8], 1 + m[4] - m[8] - m[0],
            1 + m[8] - m[0] - m[4], 1 + m[0] + m[4] + m[8] };
        var largest = 0;
        for (var i = 1; i < 4; i++) if (diagonal[i] > diagonal[largest]) largest = i;
        var value = largest switch
        {
            0 => new System.Numerics.Quaternion(diagonal[0], m[1] + m[3], m[2] + m[6], m[7] - m[5]),
            1 => new(m[1] + m[3], diagonal[1], m[5] + m[7], m[2] - m[6]),
            2 => new(m[2] + m[6], m[5] + m[7], diagonal[2], m[3] - m[1]),
            _ => new(m[7] - m[5], m[2] - m[6], m[3] - m[1], diagonal[3])
        };
        return System.Numerics.Quaternion.Normalize(value);
    }

    static Vector3 Normalize(Vector3 value) => value / MathF.Max(NormalizeEpsilon, value.Length());
}
