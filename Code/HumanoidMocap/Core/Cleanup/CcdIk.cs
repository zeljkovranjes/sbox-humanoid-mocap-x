// Copyright (c) 2007-2016, DigitalRune GmbH. All rights reserved.
// Adapted from DigitalRune CcdIKSolver; see DigitalRune.LICENSE and THIRD_PARTY_NOTICES.md.
using System;
using System.Numerics;
using HumanoidMocapX.Core.Maths;
using SkeletonModel = HumanoidMocapX.Core.Skeleton.Skeleton;

namespace HumanoidMocapX.Core.Cleanup;
using Vector3 = System.Numerics.Vector3;

/// <summary>
/// DigitalRune cyclic-coordinate descent over a fixed chain of movable joints.
/// Uses reusable scratch storage; create one instance per independently solved chain.
/// </summary>
public sealed class CcdIk
{
    readonly SkeletonModel skeleton;
    readonly int[] joints;
    readonly int tip;
    readonly XForm[] world;

    /// <summary>
    /// Copies root-to-tip joint indices, excluding the tip itself. Skipped helper bones
    /// keep their local transforms. Throws for an empty, reversed or disconnected chain.
    /// </summary>
    public CcdIk(SkeletonModel skeleton, int[] joints, int tip)
    {
        ArgumentNullException.ThrowIfNull(skeleton);
        ArgumentNullException.ThrowIfNull(joints);
        if (tip < 0 || tip >= skeleton.Count || joints.Length == 0)
            throw new ArgumentException("CCD needs a valid tip and at least one movable ancestor.");
        var child = tip;
        for (var i = joints.Length - 1; i >= 0; i--)
        {
            var ancestor = skeleton[child].ParentIndex;
            while (ancestor >= 0 && ancestor != joints[i]) ancestor = skeleton[ancestor].ParentIndex;
            if (joints[i] < 0 || ancestor < 0)
                throw new ArgumentException("CCD joints must be distinct ancestors ordered from root to tip.", nameof(joints));
            child = joints[i];
        }
        this.skeleton = skeleton;
        this.joints = new int[joints.Length];
        Array.Copy(joints, this.joints, joints.Length);
        this.tip = tip;
        world = new XForm[skeleton.Count];
    }

    /// <summary>
    /// Rotates the chain toward a world-space target and returns the remaining distance.
    /// The optional constraint callback runs after each joint change. Tip offset is in
    /// tip-local space. Preserves all translations and all rotations outside the chain.
    /// Throws for malformed poses, non-finite inputs or invalid iteration/gain settings.
    /// Unreachable targets return a nonzero residual; this method is not thread-safe.
    /// </summary>
    public float Solve(XForm[] pose, Vector3 target, int iterations = 32, float tolerance = .001f,
        float gain = .7f, Action<XForm[], int>? constrain = null, Vector3 tipOffset = default)
    {
        Validate(pose, target, tipOffset, iterations, tolerance, gain);
        FkUtil.ToWorld(pose, skeleton, world);
        var toleranceSquared = tolerance * tolerance;
        for (var iteration = 0; iteration < iterations; iteration++)
        {
            for (var j = joints.Length - 1; j >= 0; j--)
            {
                var position = world[tip].Pos + Vector3.Transform(tipOffset, world[tip].Rot);
                if (Vector3.DistanceSquared(position, target) <= toleranceSquared)
                    return Vector3.Distance(position, target);
                var bone = joints[j];
                var inverse = Quaternion.Conjugate(world[bone].Rot);
                var tipLocal = Vector3.Transform(position - world[bone].Pos, inverse);
                var targetLocal = Vector3.Transform(target - world[bone].Pos, inverse);
                var axis = Vector3.Cross(tipLocal, targetLocal);
                var axisLength = axis.Length();
                // A rounded dot of one must not discard a still-measurable tip error.
                var rotation = axisLength > 1e-12f
                    ? Quaternion.CreateFromAxisAngle(axis / axisLength, MathF.Atan2(axisLength, Vector3.Dot(tipLocal, targetLocal)))
                    : MathQ.FromTo(tipLocal, targetLocal);
                if (gain < 1) rotation = Quaternion.Slerp(Quaternion.Identity, rotation, gain);
                pose[bone].Rot = MathQ.Normalize(pose[bone].Rot * rotation);
                constrain?.Invoke(pose, bone);
                FkUtil.ToWorld(pose, skeleton, world);
            }
        }
        return Vector3.Distance(world[tip].Pos + Vector3.Transform(tipOffset, world[tip].Rot), target);
    }

    void Validate(XForm[] pose, Vector3 target, Vector3 offset, int iterations, float tolerance, float gain)
    {
        ArgumentNullException.ThrowIfNull(pose);
        if (pose.Length != skeleton.Count) throw new ArgumentException("CCD pose length must match its skeleton.", nameof(pose));
        if (!Finite(target) || !Finite(offset)) throw new ArgumentException("CCD target and tip offset must be finite.");
        if (iterations < 1 || iterations > 256) throw new ArgumentOutOfRangeException(nameof(iterations), "Use 1 to 256 CCD iterations.");
        if (!float.IsFinite(tolerance) || tolerance <= 0) throw new ArgumentOutOfRangeException(nameof(tolerance));
        if (!float.IsFinite(gain) || gain <= 0 || gain > 1) throw new ArgumentOutOfRangeException(nameof(gain));
        for (var i = 0; i < pose.Length; i++)
        {
            var lengthSquared = pose[i].Rot.LengthSquared();
            if (!Finite(pose[i].Pos) || !float.IsFinite(lengthSquared) || lengthSquared < 1e-12f)
                throw new ArgumentException($"CCD pose bone {i} has a non-finite position or invalid rotation.", nameof(pose));
        }
    }

    static bool Finite(Vector3 value) => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}
