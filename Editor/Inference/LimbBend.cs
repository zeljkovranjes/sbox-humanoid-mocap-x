using System;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Bends each elbow and knee to the angle SAM sees between the limb's two bones. Through the breakin clip's
/// one-handed spin FlowHMR bent the straight support arm 10–22° more than SAM's arm keypoints, and the dancer read as
/// folding onto the hand; in the closing freeze it left the folded leg nearly straight. The lower bone turns about the
/// joint's hinge; the hand or foot keeps its attitude.</summary>
internal static class LimbBend
{
    // Larger disagreements are keypoint failures (a limb SAM swapped or lost), not posture; they fade out over Fade.
    const float MaxCorrection = 45 * MathF.PI / 180, Fade = 10 * MathF.PI / 180;
    // A captured limb this close to straight has no hinge to turn about; its correction fades in beyond Straight + Fade.
    const float Straight = 4 * MathF.PI / 180;
    // SAM's upper bone, carried into the capture by the trunk, must point within these of the capture's to be the same limb.
    const float Agree = 30 * MathF.PI / 180, Disagree = 60 * MathF.PI / 180;
    const double Sigma = 2;
    const int LeftShoulder = 5, RightShoulder = 6, LeftElbow = 7, RightElbow = 8, LeftHip = 9, RightHip = 10, LeftKnee = 11, RightKnee = 12,
        LeftAnkle = 13, RightAnkle = 14;
    const int LeftWrist = SamObservations.LeftHandStart + FingerTransfer.Wrist, RightWrist = SamObservations.RightHandStart + FingerTransfer.Wrist;

    /// <summary>Turns the forearms and shins of each frame of <paramref name="document"/> toward the elbow and knee angles in
    /// the matching <paramref name="observations"/>. Returns the largest applied correction in degrees; 0 when the document
    /// lacks the arms or legs.</summary>
    internal static double Apply(MotionDocument document, ReadOnlySpan<float> observations)
    {
        var frames = document.Frames.Count;
        if (frames == 0 || observations.Length != frames * SamObservations.Elements)
            throw new ArgumentException("Limb bend needs one SAM observation per motion frame.");
        int armL = MotionPose.Bone(document, BoneRole.UpperArmL), armR = MotionPose.Bone(document, BoneRole.UpperArmR),
            legL = MotionPose.Bone(document, BoneRole.UpperLegL), legR = MotionPose.Bone(document, BoneRole.UpperLegR);
        if (Math.Min(Math.Min(armL, armR), Math.Min(legL, legR)) < 0) return 0;
        double largest = 0;
        foreach (var (upper, lower, hand, shoulder, elbow, wrist) in new[]
        {
            (armL, MotionPose.Bone(document, BoneRole.LowerArmL), MotionPose.Bone(document, BoneRole.HandL), LeftShoulder, LeftElbow, LeftWrist),
            (armR, MotionPose.Bone(document, BoneRole.LowerArmR), MotionPose.Bone(document, BoneRole.HandR), RightShoulder, RightElbow, RightWrist),
            (legL, MotionPose.Bone(document, BoneRole.LowerLegL), MotionPose.Bone(document, BoneRole.FootL), LeftHip, LeftKnee, LeftAnkle),
            (legR, MotionPose.Bone(document, BoneRole.LowerLegR), MotionPose.Bone(document, BoneRole.FootR), RightHip, RightKnee, RightAnkle)
        })
        {
            if (lower < 0 || hand < 0 || document.Bones[lower].Parent != upper || document.Bones[hand].Parent != lower) continue;
            largest = Math.Max(largest, Bend(document, observations, upper, lower, hand, shoulder, elbow, wrist, armL, armR, legL, legR));
        }
        return largest;
    }

    static double Bend(MotionDocument document, ReadOnlySpan<float> observations, int upper, int lower, int hand,
        int shoulder, int elbow, int wrist, int armL, int armR, int legL, int legR)
    {
        var frames = document.Frames.Count;
        var world = new Quaternion[document.Bones.Count]; var position = new Vector3[world.Length];
        var corrections = new float[frames]; var weights = new double[frames];
        Span<Vector3> seen = stackalloc Vector3[3]; Span<Vector3> posed = stackalloc Vector3[3];
        for (var f = 0; f < frames; f++)
        {
            var k = observations.Slice(f * SamObservations.Elements + SamObservations.BodyKeypoints, SamKeypoints.Count * 3);
            MotionPose.World(document, document.Frames[f], world, position);
            var observed = Between(SamObservations.Point(k, shoulder) - SamObservations.Point(k, elbow), SamObservations.Point(k, wrist) - SamObservations.Point(k, elbow));
            var captured = Between(position[upper] - position[lower], position[hand] - position[lower]);
            // The same limb: SAM's upper bone, turned into the capture by the trunk, points where the capture's does.
            seen[0] = SamObservations.Point(k, RightShoulder) - SamObservations.Point(k, LeftShoulder);
            seen[1] = SamObservations.Point(k, RightHip) - SamObservations.Point(k, LeftHip);
            seen[2] = SamObservations.Point(k, LeftShoulder) + SamObservations.Point(k, RightShoulder) - SamObservations.Point(k, LeftHip) - SamObservations.Point(k, RightHip);
            posed[0] = position[armR] - position[armL]; posed[1] = position[legR] - position[legL];
            posed[2] = position[armL] + position[armR] - position[legL] - position[legR];
            var trunk = RotationFit.Fit(seen, posed, MathF.PI / 6);
            var apart = Between(Vector3.Transform(SamObservations.Point(k, elbow) - SamObservations.Point(k, shoulder), trunk), position[lower] - position[upper]);
            var correction = observed - captured;
            corrections[f] = correction;
            weights[f] = Ramp(MaxCorrection + Fade - MathF.Abs(correction), 0, Fade) * Ramp(MathF.PI - captured, Straight, Straight + Fade)
                * Ramp(Disagree - apart, 0, Disagree - Agree);
        }
        var smoothed = new float[frames]; var trust = new double[frames];
        MotionPose.Smooth(corrections, weights, 1, Sigma, smoothed, trust);
        double largest = 0;
        for (var f = 0; f < frames; f++)
        {
            var angle = smoothed[f] * (float)trust[f];
            if (angle == 0) continue;
            MotionPose.World(document, document.Frames[f], world, position);
            var hinge = Vector3.Cross(position[upper] - position[lower], position[hand] - position[lower]);
            if (!(hinge.LengthSquared() > 1e-12f)) continue;
            // Turning the lower bone about this hinge opens the joint for a positive angle.
            var lowerWorld = Quaternion.Normalize(Quaternion.CreateFromAxisAngle(Vector3.Normalize(hinge), angle) * world[lower]);
            Set(document.Frames[f], lower, Quaternion.Inverse(world[upper]) * lowerWorld);
            Set(document.Frames[f], hand, Quaternion.Inverse(lowerWorld) * world[hand]);
            largest = Math.Max(largest, Math.Abs(angle) * 180 / Math.PI);
        }
        return largest;
    }

    static float Between(Vector3 a, Vector3 b)
        => a.LengthSquared() < 1e-12f || b.LengthSquared() < 1e-12f ? 0 : MathF.Acos(Math.Clamp(Vector3.Dot(Vector3.Normalize(a), Vector3.Normalize(b)), -1, 1));

    static void Set(MotionFrame frame, int bone, Quaternion rotation)
    {
        rotation = Quaternion.Normalize(rotation);
        frame.Rotations[bone] = new[] { rotation.X, rotation.Y, rotation.Z, rotation.W };
    }

    static double Ramp(double value, double zero, double one) => Math.Clamp((value - zero) / (one - zero), 0, 1);
}
