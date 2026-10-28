using System;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Records the frames in which SAM sees a hand on the floor under a raised body (handstands, freezes,
/// headstands) as stationary-joint tracks on the capture's hand bones, for retargeting to plant. In the breakin
/// clip's closing headstand SAM puts both wrists 5–9 cm above the dancer's lowest point while the capture held the
/// right hand 22–26 cm up. Heights are measured along the capture's up in SAM's camera space, fitted per frame from
/// the shoulder, hip and trunk directions, so a moving camera does not tilt them.</summary>
internal static class HandContact
{
    // A wrist this close above the lowest body keypoint is on the floor; both ankles this far above it raise the body.
    const float OnFloorMetres = .12f, FeetUpMetres = .4f;
    // Shorter runs are flicker, not a plant.
    const int MinimumFrames = 5;
    const string Source = "SAM 3D Body: wrist on the floor under a raised body";
    const int LeftShoulder = 5, RightShoulder = 6, LeftHip = 9, RightHip = 10, LeftAnkle = 13, RightAnkle = 14;
    const int LeftWrist = SamObservations.LeftHandStart + FingerTransfer.Wrist, RightWrist = SamObservations.RightHandStart + FingerTransfer.Wrist;

    /// <summary>Writes a track for each hand of <paramref name="document"/>, replacing any earlier one, from the matching
    /// per-frame <paramref name="observations"/>. Returns the number of hand-frames on the floor; 0 when the document
    /// lacks hands, shoulders or hips.</summary>
    internal static int Apply(MotionDocument document, ReadOnlySpan<float> observations)
    {
        var frames = document.Frames.Count;
        if (frames == 0 || observations.Length != frames * SamObservations.Elements)
            throw new ArgumentException("Hand contact needs one SAM observation per motion frame.");
        int handL = MotionPose.Bone(document, BoneRole.HandL), handR = MotionPose.Bone(document, BoneRole.HandR),
            armL = MotionPose.Bone(document, BoneRole.UpperArmL), armR = MotionPose.Bone(document, BoneRole.UpperArmR),
            legL = MotionPose.Bone(document, BoneRole.UpperLegL), legR = MotionPose.Bone(document, BoneRole.UpperLegR);
        if (Math.Min(Math.Min(handL, handR), Math.Min(Math.Min(armL, armR), Math.Min(legL, legR))) < 0) return 0;
        var world = new Quaternion[document.Bones.Count]; var position = new Vector3[world.Length];
        var left = new bool[frames]; var right = new bool[frames];
        Span<Vector3> seen = stackalloc Vector3[3]; Span<Vector3> posed = stackalloc Vector3[3];
        for (var f = 0; f < frames; f++)
        {
            var k = observations.Slice(f * SamObservations.Elements + SamObservations.BodyKeypoints, SamKeypoints.Count * 3);
            MotionPose.World(document, document.Frames[f], world, position);
            seen[0] = SamObservations.Point(k, RightShoulder) - SamObservations.Point(k, LeftShoulder);
            seen[1] = SamObservations.Point(k, RightHip) - SamObservations.Point(k, LeftHip);
            seen[2] = SamObservations.Point(k, LeftShoulder) + SamObservations.Point(k, RightShoulder) - SamObservations.Point(k, LeftHip) - SamObservations.Point(k, RightHip);
            posed[0] = position[armR] - position[armL]; posed[1] = position[legR] - position[legL];
            posed[2] = position[armL] + position[armR] - position[legL] - position[legR];
            // The capture's up (+Y) in SAM's camera axes.
            var up = Vector3.Transform(Vector3.UnitY, Quaternion.Conjugate(RotationFit.Fit(seen, posed, MathF.PI / 6)));
            var lowest = float.MaxValue;
            for (var i = 0; i < SamKeypoints.Count; i++) lowest = MathF.Min(lowest, Vector3.Dot(SamObservations.Point(k, i), up));
            if (MathF.Min(Height(k, LeftAnkle, up, lowest), Height(k, RightAnkle, up, lowest)) < FeetUpMetres) continue;
            left[f] = Height(k, LeftWrist, up, lowest) < OnFloorMetres; right[f] = Height(k, RightWrist, up, lowest) < OnFloorMetres;
        }
        return Write(document, handL, left) + Write(document, handR, right);
    }

    static float Height(ReadOnlySpan<float> keypoints, int index, Vector3 up, float lowest) => Vector3.Dot(SamObservations.Point(keypoints, index), up) - lowest;

    static int Write(MotionDocument document, int hand, bool[] onFloor)
    {
        var probability = new float[onFloor.Length]; var count = 0;
        for (var f = 0; f < onFloor.Length;)
        {
            if (!onFloor[f]) { f++; continue; }
            var e = f; while (e + 1 < onFloor.Length && onFloor[e + 1]) e++;
            if (e - f + 1 >= MinimumFrames) for (var i = f; i <= e; i++) { probability[i] = 1; count++; }
            f = e + 1;
        }
        var name = document.Bones[hand].Name;
        document.StationaryJoints.RemoveAll(t => t.Bone == name);
        if (count > 0) document.StationaryJoints.Add(new StationaryJointTrack { Bone = name, Source = Source, Probability = probability });
        return count;
    }
}
