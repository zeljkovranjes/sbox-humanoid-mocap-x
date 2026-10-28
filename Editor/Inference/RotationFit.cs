using System;
using System.Numerics;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Best-fit rotation between two sets of paired directions, such as SAM's camera-space
/// body directions and the same directions in a capture's world.</summary>
internal static class RotationFit
{
    /// <summary>The rotation taking directions <paramref name="seen"/> to directions
    /// <paramref name="posed"/>, refitted twice without pairs more than <paramref name="residual"/>
    /// radians off. Zero-length pairs are skipped; with no usable pair it returns identity.</summary>
    internal static Quaternion Fit(ReadOnlySpan<Vector3> seen, ReadOnlySpan<Vector3> posed, float residual)
    {
        var rotation = Quaternion.Identity; var everyPair = true;
        Span<double> s = stackalloc double[9];
        for (var pass = 0; pass < 3; pass++)
        {
            s.Clear();
            for (var i = 0; i < seen.Length; i++)
            {
                if (!(seen[i].LengthSquared() > 1e-12f) || !(posed[i].LengthSquared() > 1e-12f)) continue;
                var a = Vector3.Normalize(seen[i]); var b = Vector3.Normalize(posed[i]);
                if (!everyPair && Vector3.Dot(Vector3.Transform(a, rotation), b) < MathF.Cos(residual)) continue;
                s[0] += a.X * b.X; s[1] += a.X * b.Y; s[2] += a.X * b.Z;
                s[3] += a.Y * b.X; s[4] += a.Y * b.Y; s[5] += a.Y * b.Z;
                s[6] += a.Z * b.X; s[7] += a.Z * b.Y; s[8] += a.Z * b.Z;
            }
            rotation = Horn(s); everyPair = false;
        }
        return rotation;
    }

    /// <summary>Horn's closed-form absolute orientation: the unit quaternion maximising the
    /// alignment is the dominant eigenvector of a symmetric 4×4 built from the cross-covariance
    /// <paramref name="s"/> (row-major Σ a bᵀ). Found by power iteration, shifted so every
    /// eigenvalue is positive.</summary>
    static Quaternion Horn(ReadOnlySpan<double> s)
    {
        double xx = s[0], xy = s[1], xz = s[2], yx = s[3], yy = s[4], yz = s[5], zx = s[6], zy = s[7], zz = s[8];
        Span<double> n = stackalloc double[]
        {
            xx + yy + zz, yz - zy, zx - xz, xy - yx,
            yz - zy, xx - yy - zz, xy + yx, zx + xz,
            zx - xz, xy + yx, -xx + yy - zz, yz + zy,
            xy - yx, zx + xz, yz + zy, -xx - yy + zz
        };
        double shift = 0;
        for (var i = 0; i < 16; i++) shift += Math.Abs(n[i]);
        for (var i = 0; i < 4; i++) n[i * 5] += shift;
        Span<double> q = stackalloc double[] { 1, .1, .1, .1 }; Span<double> next = stackalloc double[4];
        for (var iteration = 0; iteration < 500; iteration++)
        {
            double length = 0;
            for (var r = 0; r < 4; r++) { next[r] = n[r * 4] * q[0] + n[r * 4 + 1] * q[1] + n[r * 4 + 2] * q[2] + n[r * 4 + 3] * q[3]; length += next[r] * next[r]; }
            length = Math.Sqrt(length);
            for (var r = 0; r < 4; r++) q[r] = next[r] / length;
        }
        // Horn orders the quaternion (w, x, y, z).
        return Quaternion.Normalize(new Quaternion((float)q[1], (float)q[2], (float)q[3], (float)q[0]));
    }
}
