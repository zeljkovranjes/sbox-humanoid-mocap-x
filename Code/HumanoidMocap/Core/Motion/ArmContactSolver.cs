using System;
using System.Collections.Generic;
using System.Numerics;
using HumanoidMocapX.Core.Cleanup;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Target;
using HumanoidMocapX.Core.Solve;
using SkeletonModel = HumanoidMocapX.Core.Skeleton.Skeleton;

namespace HumanoidMocapX.Core.Motion;
using Vector3 = System.Numerics.Vector3;

/// <summary>Projects arm contacts with DigitalRune CCD while preserving captured wrist and finger rotations.</summary>
internal sealed class ArmContactSolver
{
    const int ContactIterations = 24;
    readonly SkeletonModel skeleton;
    readonly int upper, lower, hand;
    readonly int[] handBones;
    readonly XForm[] world;
    readonly CcdIk wristSolver, elbowSolver;
    readonly float tolerance;
    internal int Upper => upper;
    internal int Lower => lower;
    internal int Hand => hand;

    internal ArmContactSolver(TargetRig target, int upper, int lower, int hand)
    {
        var skeleton = target.Skeleton;
        this.skeleton = skeleton; this.upper = upper; this.lower = lower; this.hand = hand;
        world = new XForm[skeleton.Count];
        wristSolver = new CcdIk(skeleton, new[] { upper, lower }, hand);
        elbowSolver = new CcdIk(skeleton, new[] { upper }, lower);
        var bones = new List<int>();
        for (var bone = hand + 1; bone < skeleton.Count; bone++)
        {
            if (target.RoleOf(bone) is not { } role || !FingerSolver.IsFingerRole(role)) continue;
            var parent = skeleton[bone].ParentIndex;
            while (parent >= 0 && parent != hand) parent = skeleton[parent].ParentIndex;
            if (parent == hand) bones.Add(bone);
        }
        handBones = bones.ToArray();
        var armLength = Vector3.Distance(skeleton.RestWorld[upper].Pos, skeleton.RestWorld[lower].Pos)
            + Vector3.Distance(skeleton.RestWorld[lower].Pos, skeleton.RestWorld[hand].Pos);
        tolerance = Math.Max(1e-5f, armLength * 1e-5f);
    }

    internal (Vector3 Move, float Residual) Solve(XForm[] pose, ArmContactSurface surface, Vector3 offset, Vector3 fallback)
    {
        FkUtil.ToWorld(pose, skeleton, world);
        var originalPosition = world[hand].Pos;
        var orientation = world[hand].Rot;
        fallback = surface.Outward(originalPosition, fallback);
        if (offset.LengthSquared() > tolerance * tolerance)
            MoveWrist(pose, originalPosition + offset, orientation, fallback);
        var bestUpper = pose[upper].Rot; var bestLower = pose[lower].Rot; var bestHand = pose[hand].Rot;
        var bestResidual = float.PositiveInfinity;
        for (var iteration = 0; iteration <= ContactIterations; iteration++)
        {
            var push = Penetration(surface, fallback, out var fraction, out var isForearm);
            var residual = push.Length();
            if (residual < bestResidual)
            {
                bestResidual = residual;
                bestUpper = pose[upper].Rot; bestLower = pose[lower].Rot; bestHand = pose[hand].Rot;
            }
            if (residual <= tolerance || iteration == ContactIterations) break;
            if (isForearm && fraction < .1f)
            {
                elbowSolver.Solve(pose, world[lower].Pos + push, tolerance: tolerance);
                RestoreWrist(pose, orientation);
            }
            else
            {
                var scale = isForearm ? Math.Max(.25f, fraction) : 1f;
                MoveWrist(pose, world[hand].Pos + push / scale, orientation, fallback);
            }
        }
        pose[upper].Rot = bestUpper; pose[lower].Rot = bestLower; pose[hand].Rot = bestHand;
        FkUtil.ToWorld(pose, skeleton, world);
        return (world[hand].Pos - originalPosition, bestResidual <= tolerance ? 0 : bestResidual);
    }

    Vector3 Penetration(ArmContactSurface surface, Vector3 fallback, out float fraction, out bool isForearm)
    {
        var push = surface.ProjectSegment(world[lower].Pos, world[hand].Pos, fallback, out fraction);
        isForearm = true;
        var largest = push.LengthSquared();
        for (var i = 0; i < handBones.Length; i++)
        {
            var bone = handBones[i];
            var candidate = surface.ProjectSegment(world[skeleton[bone].ParentIndex].Pos, world[bone].Pos, fallback, out _);
            var size = candidate.LengthSquared();
            if (size <= largest) continue;
            push = candidate; largest = size; fraction = 1; isForearm = false;
        }
        return push;
    }

    void MoveWrist(XForm[] pose, Vector3 goal, Quaternion orientation, Vector3 bendFallback)
    {
        var shoulder = world[upper].Pos; var elbow = world[lower].Pos; var wrist = world[hand].Pos;
        var first = elbow - shoulder; var second = wrist - elbow;
        // CCD cannot choose a bend direction for an exactly straight chain aimed inward.
        if (first.LengthSquared() > 1e-12f && second.LengthSquared() > 1e-12f
            && Vector3.Cross(first, second).LengthSquared() < 1e-7f * first.LengthSquared() * second.LengthSquared())
        {
            var seed = TwoBoneIk.Solve(shoulder, elbow, wrist, goal, soften: 0, stableBendAxis: bendFallback);
            EffectorIk.ApplyWorldDeltas(pose, skeleton, upper, lower, hand, seed.UpperWorldDelta, seed.LowerWorldDelta, world);
        }
        wristSolver.Solve(pose, goal, iterations: 64, tolerance: tolerance);
        RestoreWrist(pose, orientation);
    }

    void RestoreWrist(XForm[] pose, Quaternion orientation)
    {
        FkUtil.ToWorld(pose, skeleton, world);
        var parent = skeleton[hand].ParentIndex;
        pose[hand].Rot = MathQ.Normalize(Quaternion.Conjugate(world[parent].Rot) * orientation);
        FkUtil.ToWorld(pose, skeleton, world);
    }
}
