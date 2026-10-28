using System;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>World-space pose of one motion frame, for corrections that compare a capture with
/// what the video shows.</summary>
internal static class MotionPose
{
    /// <summary>Fills each bone's world rotation and joint position; spans are bone-sized.</summary>
    internal static void World(MotionDocument document, MotionFrame frame, Span<Quaternion> rotations, Span<Vector3> positions)
    {
        for (var i = 0; i < document.Bones.Count; i++)
        {
            var parent = document.Bones[i].Parent; var local = MotionDocument.Q(frame.Rotations[i]); var offset = MotionDocument.V(frame.Positions[i]);
            rotations[i] = parent < 0 ? local : rotations[parent] * local;
            positions[i] = parent < 0 ? offset : positions[parent] + Vector3.Transform(offset, rotations[parent]);
        }
    }

    /// <summary>The bone with <paramref name="role"/>, or -1.</summary>
    internal static int Bone(MotionDocument document, BoneRole role) => document.Bones.FindIndex(b => b.Role == role);

    /// <summary>Signed angle in radians from <paramref name="from"/> to <paramref name="to"/>
    /// about <paramref name="axis"/>, after projecting both onto the plane normal to it, and how
    /// much of <paramref name="to"/> survives that projection (0 to 1).</summary>
    internal static float Angle(Vector3 from, Vector3 to, Vector3 axis, out float support)
    {
        axis = Vector3.Normalize(axis);
        var length = to.Length();
        from -= axis * Vector3.Dot(from, axis); to -= axis * Vector3.Dot(to, axis);
        support = length > 0 ? to.Length() / length : 0;
        return MathF.Atan2(Vector3.Dot(Vector3.Cross(from, to), axis), Vector3.Dot(from, to));
    }

    /// <summary>Weighted Gaussian smoothing of <paramref name="values"/> (frames × width) with
    /// per-frame weights. Trust is each frame's local mean weight; frames without any weighted
    /// neighbour keep zero values and zero trust.</summary>
    internal static void Smooth(ReadOnlySpan<float> values, ReadOnlySpan<double> weights, int width, double sigma, Span<float> smoothed, Span<double> trust)
    {
        var frames = weights.Length; var radius = (int)Math.Ceiling(3 * sigma);
        smoothed.Clear();
        for (var f = 0; f < frames; f++)
        {
            double total = 0, kernel = 0;
            for (var j = Math.Max(0, f - radius); j <= Math.Min(frames - 1, f + radius); j++)
            {
                var g = Math.Exp(-0.5 * (j - f) * (j - f) / (sigma * sigma)); kernel += g;
                var w = g * weights[j]; if (w <= 0) continue; total += w;
                for (var s = 0; s < width; s++) smoothed[f * width + s] += (float)(values[j * width + s] * w);
            }
            trust[f] = total / kernel;
            if (total > 0) for (var s = 0; s < width; s++) smoothed[f * width + s] /= (float)total;
        }
    }
}
