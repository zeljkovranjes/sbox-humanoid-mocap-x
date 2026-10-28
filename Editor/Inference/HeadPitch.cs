using System;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Nods the head to the pitch SAM sees in the face. FlowHMR tilts the head further
/// down than SAM's nose and eye keypoints on both validation clips, which reads as a hunch.
/// Pitch is measured against the trunk in the body's sagittal plane, so it holds for any body
/// orientation; yaw and roll stay FlowHMR's. The neck and head share the correction.</summary>
internal static class HeadPitch
{
    // In the SMPL template (zero pose, level head) the head's forward axis is 42.33° from the
    // line from the eyes' midpoint to the nose, using SMPL's COCO face vertices 2800, 6260 and 332.
    // Eyes and nose sit at the same landmarks in SMPL, MHR and MediaPipe; ears do not, and an
    // ear-based line raised the five-second clip's head past what MediaPipe's face mesh shows.
    const float ForwardFromFace = 42.33f * MathF.PI / 180;
    // Larger disagreements are keypoint failures, not posture; they fade out over Fade.
    const float MaxCorrection = 40 * MathF.PI / 180, Fade = 10 * MathF.PI / 180;
    // The face line and head axis must keep this much length off the lateral axis to be measured.
    const float MinSupport = .5f, FullSupport = .8f;
    const double Sigma = 3;
    const int Nose = 0, LeftEye = 1, RightEye = 2, LeftShoulder = 5, RightShoulder = 6, LeftHip = 9, RightHip = 10;

    /// <summary>Rotates the neck and head of each frame of <paramref name="document"/> toward
    /// the head pitch in the matching <paramref name="observations"/>. Returns the largest
    /// applied correction in degrees; 0 when the document has no neck, head, shoulders or hips.</summary>
    internal static double Apply(MotionDocument document, ReadOnlySpan<float> observations)
    {
        var frames = document.Frames.Count;
        if (frames == 0 || observations.Length != frames * SamObservations.Elements)
            throw new ArgumentException("Head pitch needs one SAM observation per motion frame.");
        int neck = MotionPose.Bone(document, BoneRole.Neck), head = MotionPose.Bone(document, BoneRole.Head),
            armL = MotionPose.Bone(document, BoneRole.UpperArmL), armR = MotionPose.Bone(document, BoneRole.UpperArmR),
            legL = MotionPose.Bone(document, BoneRole.UpperLegL), legR = MotionPose.Bone(document, BoneRole.UpperLegR);
        if (Math.Min(Math.Min(neck, head), Math.Min(Math.Min(armL, armR), Math.Min(legL, legR))) < 0 || document.Bones[head].Parent != neck) return 0;
        var world = new Quaternion[document.Bones.Count]; var position = new Vector3[world.Length];
        var corrections = new float[frames]; var weights = new double[frames]; var lateral = new Vector3[frames];
        for (var f = 0; f < frames; f++)
        {
            var k = observations.Slice(f * SamObservations.Elements, SamKeypoints.Count * 3);
            var observed = MotionPose.Angle(Mid(k, LeftShoulder, RightShoulder) - Mid(k, LeftHip, RightHip),
                SamObservations.Point(k, Nose) - Mid(k, LeftEye, RightEye), SamObservations.Point(k, RightShoulder) - SamObservations.Point(k, LeftShoulder), out var faceSupport);
            MotionPose.World(document, document.Frames[f], world, position);
            lateral[f] = position[armR] - position[armL];
            var captured = MotionPose.Angle((position[armL] + position[armR] - position[legL] - position[legR]) / 2,
                Vector3.Transform(Vector3.UnitZ, world[head]), lateral[f], out var headSupport);
            var correction = Wrap(observed + ForwardFromFace - captured);
            corrections[f] = correction;
            weights[f] = Ramp(Math.Min(faceSupport, headSupport), MinSupport, FullSupport) * Ramp(MaxCorrection + Fade - MathF.Abs(correction), 0, Fade);
        }
        var smoothed = new float[frames]; var trust = new double[frames];
        MotionPose.Smooth(corrections, weights, 1, Sigma, smoothed, trust);
        double largest = 0;
        for (var f = 0; f < frames; f++)
        {
            var angle = smoothed[f] * (float)trust[f];
            if (angle == 0 || !(lateral[f].LengthSquared() > 1e-12f)) continue;
            MotionPose.World(document, document.Frames[f], world, position);
            var axis = Vector3.Normalize(lateral[f]);
            var neckParent = document.Bones[neck].Parent;
            var neckWorld = Quaternion.CreateFromAxisAngle(axis, angle / 2) * world[neck];
            var headWorld = Quaternion.CreateFromAxisAngle(axis, angle) * world[head];
            Set(document.Frames[f], neck, neckParent < 0 ? neckWorld : Quaternion.Inverse(world[neckParent]) * neckWorld);
            Set(document.Frames[f], head, Quaternion.Inverse(neckWorld) * headWorld);
            largest = Math.Max(largest, Math.Abs(angle) * 180 / Math.PI);
        }
        return largest;
    }

    static void Set(MotionFrame frame, int bone, Quaternion rotation)
    {
        rotation = Quaternion.Normalize(rotation);
        frame.Rotations[bone] = new[] { rotation.X, rotation.Y, rotation.Z, rotation.W };
    }

    static Vector3 Mid(ReadOnlySpan<float> points, int a, int b) => (SamObservations.Point(points, a) + SamObservations.Point(points, b)) / 2;
    static float Wrap(float angle) => MathF.IEEERemainder(angle, 2 * MathF.PI);
    static double Ramp(double value, double zero, double one) => Math.Clamp((value - zero) / (one - zero), 0, 1);
}
