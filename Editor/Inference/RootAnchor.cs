using System;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Holds the root where the video puts it. FlowHMR integrates its own root velocity,
/// so a dancer who stays on one spot can wander off it (0.5 m over 9 s on the breakin clip).
/// SAM places the pelvis in camera space in every frame independently, which does not drift.
/// With a static camera, the capture's horizontal root path is pulled to SAM's, smoothed hard
/// in depth because SAM's depth is far noisier than its image position. Height stays FlowHMR's,
/// and so does the start position.</summary>
internal static class RootAnchor
{
    // A camera that turns further than this over the clip is not static.
    const double MaxCameraTurn = 2 * Math.PI / 180;
    // Body directions that disagree by more than this after the camera fit are left out of it.
    const float FitResidual = 30 * MathF.PI / 180;
    // Smoothing in frames: across the image and in depth.
    const double LateralSigma = 10, DepthSigma = 60;
    const int LeftShoulder = 5, RightShoulder = 6, LeftHip = 9, RightHip = 10;

    /// <summary>Moves each root of <paramref name="document"/> horizontally to follow SAM's
    /// pelvis in <paramref name="observations"/>. <paramref name="cameraRotations"/> holds one
    /// row-major 3×3 camera rotation per frame. Returns the largest root shift in metres, or
    /// null when the camera turns or the document lacks a root, shoulders or hips.</summary>
    internal static double? Apply(MotionDocument document, ReadOnlySpan<float> observations, ReadOnlySpan<float> cameraRotations)
    {
        var frames = document.Frames.Count;
        if (frames == 0 || observations.Length != frames * SamObservations.Elements || cameraRotations.Length != frames * 9)
            throw new ArgumentException("Root anchoring needs one SAM observation and camera rotation per motion frame.");
        if (!IsStatic(cameraRotations)) return null;
        int armL = MotionPose.Bone(document, BoneRole.UpperArmL), armR = MotionPose.Bone(document, BoneRole.UpperArmR),
            legL = MotionPose.Bone(document, BoneRole.UpperLegL), legR = MotionPose.Bone(document, BoneRole.UpperLegR);
        if (Math.Min(Math.Min(armL, armR), Math.Min(legL, legR)) < 0 || document.Bones[0].Parent >= 0) return null;

        var world = new Quaternion[document.Bones.Count]; var position = new Vector3[world.Length];
        var seen = new Vector3[frames * 3]; var posed = new Vector3[frames * 3]; var pelvis = new Vector3[frames]; var observed = new Vector3[frames];
        for (var f = 0; f < frames; f++)
        {
            var k = observations.Slice(f * SamObservations.Elements, SamObservations.LeftHand);
            MotionPose.World(document, document.Frames[f], world, position);
            pelvis[f] = (position[legL] + position[legR]) / 2;
            // SAM's translation is the body origin; its keypoints are relative to it.
            observed[f] = Point(k, SamObservations.Translation / 3) + (Point(k, LeftHip) + Point(k, RightHip)) / 2;
            seen[f * 3] = Point(k, RightShoulder) - Point(k, LeftShoulder);
            seen[f * 3 + 1] = Point(k, RightHip) - Point(k, LeftHip);
            seen[f * 3 + 2] = Point(k, LeftShoulder) + Point(k, RightShoulder) - Point(k, LeftHip) - Point(k, RightHip);
            posed[f * 3] = position[armR] - position[armL];
            posed[f * 3 + 1] = position[legR] - position[legL];
            posed[f * 3 + 2] = position[armL] + position[armR] - position[legL] - position[legR];
        }
        var toWorld = Fit(seen, posed);

        // Where the capture's pelvis sits relative to SAM's, in camera axes; constant without drift.
        var offsets = new float[frames * 2]; var weights = new double[frames];
        for (var f = 0; f < frames; f++)
        {
            var offset = observed[f] - Vector3.Transform(pelvis[f], Quaternion.Conjugate(toWorld));
            offsets[f * 2] = offset.X; offsets[f * 2 + 1] = offset.Z; weights[f] = 1;
        }
        var lateral = new float[offsets.Length]; var depth = new float[offsets.Length]; var trust = new double[frames];
        MotionPose.Smooth(offsets, weights, 2, LateralSigma, lateral, trust);
        MotionPose.Smooth(offsets, weights, 2, DepthSigma, depth, trust);
        double largest = 0;
        for (var f = 0; f < frames; f++)
        {
            var shift = Vector3.Transform(new Vector3(lateral[f * 2] - lateral[0], 0, depth[f * 2 + 1] - depth[1]), toWorld) with { Y = 0 };
            var root = document.Frames[f].Positions[0];
            root[0] += shift.X; root[2] += shift.Z;
            largest = Math.Max(largest, shift.Length());
        }
        return largest;
    }

    static bool IsStatic(ReadOnlySpan<float> rotations)
    {
        for (var f = 1; f < rotations.Length / 9; f++)
        {
            double trace = 0;
            for (var i = 0; i < 9; i++) trace += rotations[i] * rotations[f * 9 + i];
            if (Math.Acos(Math.Clamp((trace - 1) / 2, -1, 1)) > MaxCameraTurn) return false;
        }
        return true;
    }

    /// <summary>The rotation taking camera-axis directions <paramref name="seen"/> to world
    /// directions <paramref name="posed"/>, refitted twice without pairs it cannot explain.</summary>
    static Quaternion Fit(Vector3[] seen, Vector3[] posed)
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
                if (!everyPair && Vector3.Dot(Vector3.Transform(a, rotation), b) < MathF.Cos(FitResidual)) continue;
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

    static Vector3 Point(ReadOnlySpan<float> points, int i) => new(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
}
