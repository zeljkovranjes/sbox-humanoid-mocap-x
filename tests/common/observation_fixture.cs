using System.Numerics;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Inference;

/// <summary>SAM observations that see exactly a motion document: body, face and hand keypoints
/// from its joints, through a camera whose axes are world axes turned upside down (x right,
/// y down, z forward) at a fixed position. Hand passes and the body pass agree in the image.</summary>
static class ObservationFixture
{
    internal static readonly Quaternion Camera = Quaternion.CreateFromAxisAngle(Vector3.UnitX, MathF.PI);
    static readonly Vector3 CameraPosition = new(0, 1, 5);
    // The template's eye-to-nose line runs 42.33° below the head's forward axis.
    static readonly Vector3 Face = new(0, -MathF.Sin(42.33f * MathF.PI / 180) * .05f, MathF.Cos(42.33f * MathF.PI / 180) * .05f);
    static readonly string[] Fingers = { "Thumb", "Index", "Middle", "Ring", "Pinky" };

    internal static float[] Observe(MotionDocument motion)
    {
        var observations = new float[motion.Frames.Count * SamObservations.Elements];
        for (var f = 0; f < motion.Frames.Count; f++)
        {
            var observation = observations.AsSpan(f * SamObservations.Elements, SamObservations.Elements);
            var position = Positions(motion, motion.Frames[f], out var rotation, out var beyond);
            Vector3 At(string role) => position[Bone(motion, role)];
            var head = Bone(motion, "Head");
            var eyes = position[head] + Vector3.Transform(new Vector3(0, .095f, .062f), rotation[head]);
            var root = Camera3(At("UpperLegL") + At("UpperLegR")) / 2 - new Vector3(0, -.9f, 0);
            var body = observation.Slice(SamObservations.BodyKeypoints, SamKeypoints.Count * 3);
            (int Index, Vector3 World)[] keypoints =
            {
                (0, eyes + Vector3.Transform(Face, rotation[head])),
                (1, eyes + Vector3.Transform(new Vector3(.033f, 0, 0), rotation[head])),
                (2, eyes - Vector3.Transform(new Vector3(.033f, 0, 0), rotation[head])),
                (5, At("UpperArmL")), (6, At("UpperArmR")), (7, At("LowerArmL")), (8, At("LowerArmR")), (9, At("UpperLegL")), (10, At("UpperLegR")),
                (SamObservations.RightHandStart + 20, At("HandR")), (SamObservations.LeftHandStart + 20, At("HandL"))
            };
            foreach (var (index, world) in keypoints) Write(body, index, Camera3(world) - root);
            Write(observation, SamObservations.Translation / 3, root);
            foreach (var left in new[] { true, false })
            {
                var hand = observation.Slice(left ? SamObservations.LeftHand : SamObservations.RightHand, SamObservations.HandPoints * 3);
                var side = left ? "L" : "R";
                Write(hand, 20, Camera3(At("Hand" + side)));
                for (var g = 0; g < 5; g++)
                {
                    Write(hand, g * 4 + 3, Camera3(At(Fingers[g] + "Prox" + side)));
                    Write(hand, g * 4 + 2, Camera3(At(Fingers[g] + "Mid" + side)));
                    var dist = Bone(motion, Fingers[g] + "Dist" + side);
                    Write(hand, g * 4 + 1, Camera3(position[dist])); Write(hand, g * 4, Camera3(beyond[dist]));
                }
                var image = observation.Slice(SamObservations.ImageKeypoints);
                var bodyPass = image.Slice((left ? SamObservations.LeftHandStart : SamObservations.RightHandStart) * 2, SamObservations.HandPoints * 2);
                var handPass = image.Slice((left ? 1 : 2) * SamKeypoints.Count * 2 + SamObservations.RightHandStart * 2, SamObservations.HandPoints * 2);
                for (var i = 0; i < SamObservations.HandPoints; i++)
                {
                    bodyPass[i * 2] = handPass[i * 2] = 400 + hand[i * 3] * 1000;
                    bodyPass[i * 2 + 1] = handPass[i * 2 + 1] = 300 + hand[i * 3 + 1] * 1000;
                }
            }
        }
        return observations;
    }

    /// <summary>One identity camera rotation per frame of <paramref name="motion"/>.</summary>
    internal static float[] StaticCamera(MotionDocument motion)
    {
        var rotations = new float[motion.Frames.Count * 9];
        for (var f = 0; f < motion.Frames.Count; f++) { rotations[f * 9] = 1; rotations[f * 9 + 4] = 1; rotations[f * 9 + 8] = 1; }
        return rotations;
    }

    internal static int Bone(MotionDocument motion, string role) => motion.Bones.FindIndex(b => b.Role == Enum.Parse<BoneRole>(role));

    /// <summary>World joint positions and rotations, and where each bone's offset would put a further joint.</summary>
    internal static Vector3[] Positions(MotionDocument motion, MotionFrame frame, out Quaternion[] rotation, out Vector3[] beyond)
    {
        rotation = new Quaternion[motion.Bones.Count]; var position = new Vector3[rotation.Length]; beyond = new Vector3[rotation.Length];
        for (var i = 0; i < rotation.Length; i++)
        {
            var parent = motion.Bones[i].Parent; var offset = MotionDocument.V(frame.Positions[i]);
            rotation[i] = parent < 0 ? MotionDocument.Q(frame.Rotations[i]) : rotation[parent] * MotionDocument.Q(frame.Rotations[i]);
            position[i] = parent < 0 ? offset : position[parent] + Vector3.Transform(offset, rotation[parent]);
            beyond[i] = position[i] + Vector3.Transform(offset, rotation[i]);
        }
        return position;
    }

    static Vector3 Camera3(Vector3 world) => Vector3.Transform(world - CameraPosition, Camera);

    static void Write(Span<float> points, int index, Vector3 value)
    {
        points[index * 3] = value.X; points[index * 3 + 1] = value.Y; points[index * 3 + 2] = value.Z;
    }
}
