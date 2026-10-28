using System;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Turns each hand to the palm orientation SAM sees in its hand close-up. FlowHMR's
/// palms face 55–89° away from SAM's on the breakin clip (25–33° on a real person) while the
/// forearms agree within about 20°, so fingers posed on them land wrong in the image. The camera
/// is related to the capture in each frame through the torso's up and shoulder line plus that
/// arm's upper arm and forearm, so the wrist keeps following the forearm even where the body is
/// turned differently; the torso keeps the fit determined when the arm lies along the shoulders.</summary>
internal static class HandTransfer
{
    const double Sigma = 2;
    // Frames whose arm directions cannot be matched this well say nothing reliable about the wrist.
    const float FitResidual = 30 * MathF.PI / 180;
    const float ForearmFull = 15 * MathF.PI / 180, ForearmNone = 35 * MathF.PI / 180;
    const int LeftShoulder = 5, RightShoulder = 6, LeftElbow = 7, RightElbow = 8, LeftHip = 9, RightHip = 10;
    const int LeftWrist = SamObservations.LeftHandStart + FingerTransfer.Wrist, RightWrist = SamObservations.RightHandStart + FingerTransfer.Wrist;

    /// <summary>Rewrites the hand rotations of <paramref name="document"/>, whose frames match the
    /// per-frame <paramref name="observations"/> one to one. Returns the share of hand-frames
    /// that used observations, from 0 to 1; 0 when the arm or finger-base bones are missing.</summary>
    internal static double Apply(MotionDocument document, ReadOnlySpan<float> observations)
    {
        var frames = document.Frames.Count;
        if (frames == 0 || observations.Length != frames * SamObservations.Elements)
            throw new ArgumentException("Hand transfer needs one SAM observation per motion frame.");
        var world = new Quaternion[document.Bones.Count]; var position = new Vector3[world.Length];
        double used = 0; var hands = 0;
        foreach (var left in new[] { true, false })
        {
            if (Bones(document, left) is not { } bones) continue;
            hands++;
            var targets = new float[frames * 4]; var weights = new double[frames];
            for (var f = 0; f < frames; f++)
            {
                MotionPose.World(document, document.Frames[f], world, position);
                var target = Target(observations.Slice(f * SamObservations.Elements, SamObservations.Elements), left, bones, world, position, out var weight);
                // Same hemisphere as the current rotation, so averaging neighbours cannot cancel.
                var current = MotionDocument.Q(document.Frames[f].Rotations[bones.Hand]);
                if (Quaternion.Dot(target, current) < 0) target = Quaternion.Negate(target);
                targets[f * 4] = target.X; targets[f * 4 + 1] = target.Y; targets[f * 4 + 2] = target.Z; targets[f * 4 + 3] = target.W;
                weights[f] = weight;
            }
            var smoothed = new float[targets.Length]; var trust = new double[frames];
            MotionPose.Smooth(targets, weights, 4, Sigma, smoothed, trust);
            for (var f = 0; f < frames; f++)
            {
                if (trust[f] <= 0) continue;
                var target = new Quaternion(smoothed[f * 4], smoothed[f * 4 + 1], smoothed[f * 4 + 2], smoothed[f * 4 + 3]);
                if (!(target.LengthSquared() > 1e-8f)) continue;
                var original = MotionDocument.Q(document.Frames[f].Rotations[bones.Hand]);
                var blended = Quaternion.Normalize(Quaternion.Slerp(original, Quaternion.Normalize(target), (float)trust[f]));
                document.Frames[f].Rotations[bones.Hand] = new[] { blended.X, blended.Y, blended.Z, blended.W };
                used += trust[f];
            }
        }
        return hands == 0 ? 0 : used / (hands * frames);
    }

    readonly record struct HandBones(int Shoulder, int OtherShoulder, int Elbow, int Hand, int Index, int Middle, int Pinky, int HipL, int HipR);

    static HandBones? Bones(MotionDocument document, bool left)
    {
        var bones = new HandBones(
            MotionPose.Bone(document, left ? BoneRole.UpperArmL : BoneRole.UpperArmR), MotionPose.Bone(document, left ? BoneRole.UpperArmR : BoneRole.UpperArmL),
            MotionPose.Bone(document, left ? BoneRole.LowerArmL : BoneRole.LowerArmR), MotionPose.Bone(document, left ? BoneRole.HandL : BoneRole.HandR),
            MotionPose.Bone(document, left ? BoneRole.IndexProxL : BoneRole.IndexProxR), MotionPose.Bone(document, left ? BoneRole.MiddleProxL : BoneRole.MiddleProxR),
            MotionPose.Bone(document, left ? BoneRole.PinkyProxL : BoneRole.PinkyProxR),
            MotionPose.Bone(document, BoneRole.UpperLegL), MotionPose.Bone(document, BoneRole.UpperLegR));
        ReadOnlySpan<int> all = stackalloc int[] { bones.Shoulder, bones.OtherShoulder, bones.Elbow, bones.Hand, bones.Index, bones.Middle, bones.Pinky, bones.HipL, bones.HipR };
        foreach (var bone in all) if (bone < 0) return null;
        return document.Bones[bones.Hand].Parent == bones.Elbow ? bones : null;
    }

    /// <summary>The hand's local rotation that puts its palm where SAM sees it, and how far to trust it.</summary>
    static Quaternion Target(ReadOnlySpan<float> observation, bool left, HandBones bones, Quaternion[] world, Vector3[] position, out double weight)
    {
        weight = 0;
        var current = MotionPose.Local(world, bones.Hand, bones.Elbow);
        var hand = observation.Slice(left ? SamObservations.LeftHand : SamObservations.RightHand, SamObservations.HandPoints * 3);
        var body = observation.Slice(SamObservations.BodyKeypoints, SamKeypoints.Count * 3);
        if (!FingerTransfer.Palm(SamObservations.Point(hand, FingerTransfer.Wrist), SamObservations.Point(hand, FingerTransfer.IndexBase),
                SamObservations.Point(hand, FingerTransfer.MiddleBase), SamObservations.Point(hand, FingerTransfer.PinkyBase), out var seenPalm)
            || !FingerTransfer.Palm(position[bones.Hand], position[bones.Index], position[bones.Middle], position[bones.Pinky], out var posedPalm))
            return current;
        var shoulder = SamObservations.Point(body, left ? LeftShoulder : RightShoulder); var other = SamObservations.Point(body, left ? RightShoulder : LeftShoulder);
        var elbow = SamObservations.Point(body, left ? LeftElbow : RightElbow); var wrist = SamObservations.Point(body, left ? LeftWrist : RightWrist);
        var hips = SamObservations.Point(body, LeftHip) + SamObservations.Point(body, RightHip);
        ReadOnlySpan<Vector3> seen = stackalloc Vector3[] { shoulder + other - hips, other - shoulder, elbow - shoulder, wrist - elbow };
        ReadOnlySpan<Vector3> posed = stackalloc Vector3[]
        {
            position[bones.Shoulder] + position[bones.OtherShoulder] - position[bones.HipL] - position[bones.HipR], position[bones.OtherShoulder] - position[bones.Shoulder],
            position[bones.Elbow] - position[bones.Shoulder], position[bones.Hand] - position[bones.Elbow]
        };
        var toWorld = RotationFit.Fit(seen, posed, FitResidual);
        var forearm = Vector3.Normalize(Vector3.Transform(seen[3], toWorld)); var posedForearm = Vector3.Normalize(posed[3]);
        var miss = MathF.Acos(Math.Clamp(Vector3.Dot(forearm, posedForearm), -1f, 1f));
        // The palm in the hand's own frame is fixed by the skeleton; only its world turn changes.
        var palmInHand = Quaternion.Inverse(world[bones.Hand]) * Quaternion.CreateFromRotationMatrix(posedPalm);
        var handWorld = toWorld * Quaternion.CreateFromRotationMatrix(seenPalm) * Quaternion.Inverse(palmInHand);
        weight = FingerTransfer.Agreement(observation, left) * Math.Clamp((ForearmNone - miss) / (ForearmNone - ForearmFull), 0, 1);
        return Quaternion.Normalize(Quaternion.Inverse(world[bones.Elbow]) * handWorld);
    }
}
