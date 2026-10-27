using System;
using System.Numerics;

namespace HumanoidMocapX.Inference;

// Momentum composes float local states in double precision before returning floats.
internal readonly record struct MhrState(double X, double Y, double Z,
    double Qx, double Qy, double Qz, double Qw, double Scale)
{
    internal const int Elements = 8;

    internal static MhrState Local(ReadOnlySpan<float> pose, ReadOnlySpan<float> offset, ReadOnlySpan<float> prerotation)
    {
        var sx = MathF.Sin(pose[3] * .5f); var cx = MathF.Cos(pose[3] * .5f);
        var sy = MathF.Sin(pose[4] * .5f); var cy = MathF.Cos(pose[4] * .5f);
        var sz = MathF.Sin(pose[5] * .5f); var cz = MathF.Cos(pose[5] * .5f);
        var euler = new Quaternion(sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz);
        var rotation = new Quaternion(prerotation[0], prerotation[1], prerotation[2], prerotation[3]) * euler;
        return new(pose[0] + offset[0], pose[1] + offset[1], pose[2] + offset[2],
            rotation.X, rotation.Y, rotation.Z, rotation.W, MathF.Exp(pose[6] * 0.69314718246459961f));
    }

    internal MhrState Then(MhrState child)
    {
        var a = Normalized(); var b = child.Normalized();
        var vx = a.Qy * b.Z - a.Qz * b.Y; var vy = a.Qz * b.X - a.Qx * b.Z; var vz = a.Qx * b.Y - a.Qy * b.X;
        return new(a.X + a.Scale * (b.X + 2 * (a.Qw * vx + a.Qy * vz - a.Qz * vy)),
            a.Y + a.Scale * (b.Y + 2 * (a.Qw * vy + a.Qz * vx - a.Qx * vz)),
            a.Z + a.Scale * (b.Z + 2 * (a.Qw * vz + a.Qx * vy - a.Qy * vx)),
            a.Qw * b.Qx + a.Qx * b.Qw + a.Qy * b.Qz - a.Qz * b.Qy,
            a.Qw * b.Qy - a.Qx * b.Qz + a.Qy * b.Qw + a.Qz * b.Qx,
            a.Qw * b.Qz + a.Qx * b.Qy - a.Qy * b.Qx + a.Qz * b.Qw,
            a.Qw * b.Qw - a.Qx * b.Qx - a.Qy * b.Qy - a.Qz * b.Qz, a.Scale * b.Scale);
    }

    MhrState Normalized()
    {
        var norm = Math.Max(1e-12, Math.Sqrt(Qx * Qx + Qy * Qy + Qz * Qz + Qw * Qw));
        return this with { Qx = Qx / norm, Qy = Qy / norm, Qz = Qz / norm, Qw = Qw / norm };
    }

    internal void Write(Span<float> output)
    {
        output[0] = (float)X; output[1] = (float)Y; output[2] = (float)Z;
        output[3] = (float)Qx; output[4] = (float)Qy; output[5] = (float)Qz; output[6] = (float)Qw; output[7] = (float)Scale;
    }
}
