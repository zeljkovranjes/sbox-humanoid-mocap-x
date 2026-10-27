using System;
using System.Collections.Generic;
using System.Numerics;
using HumanoidMocapX.Core.Cleanup;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Skeleton;
using HumanoidMocapX.Core.Target;

namespace HumanoidMocapX.Core.Motion;
using Vector3 = System.Numerics.Vector3;

/// <summary>
/// Keeps forearms and hand-bone segments outside a bone-derived torso proxy, using
/// DigitalRune CCD with temporal easing. Preserves bone lengths, wrist world attitude
/// and finger articulation. Proxy clearance does not guarantee skinned-mesh clearance.
/// </summary>
public static class CaptureArmClearance
{
    public const float HalfWidth = .9f, HalfDepth = .7f, ArmRadius = .25f, SmoothSeconds = .1f;

    /// <summary>
    /// Corrects target frames; returns the number of moved hand samples and largest wrist
    /// displacement in target units. Source arguments retain compatibility with retargeting
    /// callers; contact is measured on the target body. Throws for malformed target frames or FPS.
    /// Use the overload with residual outputs to detect contacts the fixed chain cannot resolve.
    /// </summary>
    public static (int Samples, float Largest) Apply(List<XForm[]> frames, IReadOnlyList<XForm[]> sourceFrames,
        Skeleton.Skeleton source, MappingResult mapping, TargetRig target, float fps)
        => Apply(frames, sourceFrames, source, mapping, target, fps, out _, out _);

    /// <summary>
    /// Also reports the number of unresolved hand samples and largest remaining proxy
    /// projection distance. Unreachable contacts keep the best measured pose, without stretching.
    /// </summary>
    public static (int Samples, float Largest) Apply(List<XForm[]> frames, IReadOnlyList<XForm[]> sourceFrames,
        Skeleton.Skeleton source, MappingResult mapping, TargetRig target, float fps,
        out int unresolved, out float maximumPenetration)
    {
        ArgumentNullException.ThrowIfNull(frames); ArgumentNullException.ThrowIfNull(sourceFrames);
        ArgumentNullException.ThrowIfNull(source); ArgumentNullException.ThrowIfNull(mapping);
        ArgumentNullException.ThrowIfNull(target);
        if (!float.IsFinite(fps) || fps <= 0 || fps > 1000) throw new ArgumentOutOfRangeException(nameof(fps));
        ValidateFrames(frames, target.Skeleton.Count);
        unresolved = 0; maximumPenetration = 0;
        if (frames.Count == 0 || CreateArm(target, true) is not { } left || CreateArm(target, false) is not { } right
            || target.BoneForRole(BoneRole.Hips) is not int hips
            || (target.BoneForRole(BoneRole.Neck) ?? target.BoneForRole(BoneRole.Head)) is not int neck) return (0, 0);
        var rig = target.Skeleton;
        var world = new XForm[rig.Count];
        var arms = new[] { left, right };
        var offsets = new[] { new Vector3[frames.Count], new Vector3[frames.Count] };
        for (var frameIndex = 0; frameIndex < frames.Count; frameIndex++)
        {
            var pose = frames[frameIndex];
            FkUtil.ToWorld(pose, rig, world);
            if (!ArmContactSurface.TryCreate(world, hips, neck, left.Upper, right.Upper, out var surface)) continue;
            for (var side = 0; side < arms.Length; side++)
            {
                var arm = arms[side];
                var upper = pose[arm.Upper].Rot; var lower = pose[arm.Lower].Rot; var hand = pose[arm.Hand].Rot;
                var solved = arm.Solve(pose, surface, Vector3.Zero, surface.Across * (side == 0 ? 1 : -1));
                offsets[side][frameIndex] = surface.ToLocalOffset(solved.Move);
                pose[arm.Upper].Rot = upper; pose[arm.Lower].Rot = lower; pose[arm.Hand].Rot = hand;
            }
        }
        var weights = Weights(fps);
        var samples = 0; var largest = 0f;
        for (var frameIndex = 0; frameIndex < frames.Count; frameIndex++)
        {
            var pose = frames[frameIndex];
            FkUtil.ToWorld(pose, rig, world);
            if (!ArmContactSurface.TryCreate(world, hips, neck, left.Upper, right.Upper, out var surface)) continue;
            for (var side = 0; side < arms.Length; side++)
            {
                var offset = surface.ToWorldOffset(Smooth(offsets[side], frameIndex, weights));
                var solved = arms[side].Solve(pose, surface, offset, surface.Across * (side == 0 ? 1 : -1));
                var distance = solved.Move.Length();
                if (distance > 1e-5f) { samples++; largest = Math.Max(largest, distance); }
                if (solved.Residual > 0) { unresolved++; maximumPenetration = Math.Max(maximumPenetration, solved.Residual); }
            }
        }
        return (samples, largest);
    }

    static ArmContactSolver? CreateArm(TargetRig target, bool left)
        => target.BoneForRole(left ? BoneRole.UpperArmL : BoneRole.UpperArmR) is int upper
        && target.BoneForRole(left ? BoneRole.LowerArmL : BoneRole.LowerArmR) is int lower
        && target.BoneForRole(left ? BoneRole.HandL : BoneRole.HandR) is int hand
            ? new(target, upper, lower, hand) : null;

    static void ValidateFrames(List<XForm[]> frames, int bones)
    {
        for (var frame = 0; frame < frames.Count; frame++)
        {
            if (frames[frame] is not { } pose || pose.Length != bones)
                throw new ArgumentException($"Body-contact frame {frame} does not match the target skeleton.", nameof(frames));
            for (var bone = 0; bone < bones; bone++)
            {
                var rotationLength = pose[bone].Rot.LengthSquared();
                if (!float.IsFinite(pose[bone].Pos.LengthSquared()) || !float.IsFinite(rotationLength) || rotationLength < 1e-12f)
                    throw new ArgumentException($"Body-contact frame {frame}, bone {bone} has an invalid transform.", nameof(frames));
            }
        }
    }

    static float[] Weights(float fps)
    {
        var sigma = Math.Max(1, SmoothSeconds * fps);
        var reach = (int)MathF.Ceiling(2 * sigma);
        var weights = new float[reach + 1];
        for (var i = 0; i <= reach; i++) weights[i] = MathF.Exp(-i * i / (2 * sigma * sigma));
        return weights;
    }

    static Vector3 Smooth(Vector3[] offsets, int frame, float[] weights)
    {
        var sum = Vector3.Zero; var total = 0f;
        var reach = weights.Length - 1;
        for (var neighbor = Math.Max(0, frame - reach); neighbor <= Math.Min(offsets.Length - 1, frame + reach); neighbor++)
        {
            var weight = weights[Math.Abs(neighbor - frame)];
            sum += offsets[neighbor] * weight; total += weight;
        }
        return sum / total;
    }
}
