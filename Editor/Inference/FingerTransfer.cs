using System;
using System.Numerics;
using System.Runtime.InteropServices;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Takes finger articulation from SAM's per-frame hand passes. FlowHMR returns nearly the
/// same hand pose in every frame; SAM sees each hand in a close crop. Each finger bone is swung so its
/// direction relative to the palm matches the observed one, keeping the bone's twist. A frame is
/// trusted less when its hand pass and body pass disagree about where the hand is in the image;
/// directions are smoothed over time with those weights, and stretches without trusted
/// observations keep FlowHMR's fingers.</summary>
internal static class FingerTransfer
{
    // Gaussian width of the temporal smoothing in frames, about 50 ms at 30 fps.
    const double Sigma = 1.5;
    // Hand-pass and body-pass landmarks this many hand sizes apart in the image get no weight.
    const double Disagreement = 1.5;
    const int Segments = 15;
    static readonly BoneRole[] Left =
    {
        BoneRole.ThumbProxL, BoneRole.ThumbMidL, BoneRole.ThumbDistL, BoneRole.IndexProxL, BoneRole.IndexMidL, BoneRole.IndexDistL,
        BoneRole.MiddleProxL, BoneRole.MiddleMidL, BoneRole.MiddleDistL, BoneRole.RingProxL, BoneRole.RingMidL, BoneRole.RingDistL,
        BoneRole.PinkyProxL, BoneRole.PinkyMidL, BoneRole.PinkyDistL
    };
    static readonly BoneRole[] Right =
    {
        BoneRole.ThumbProxR, BoneRole.ThumbMidR, BoneRole.ThumbDistR, BoneRole.IndexProxR, BoneRole.IndexMidR, BoneRole.IndexDistR,
        BoneRole.MiddleProxR, BoneRole.MiddleMidR, BoneRole.MiddleDistR, BoneRole.RingProxR, BoneRole.RingMidR, BoneRole.RingDistR,
        BoneRole.PinkyProxR, BoneRole.PinkyMidR, BoneRole.PinkyDistR
    };
    // MHR70 hand order: per finger from thumb to pinky, tip, two middle joints, then the base; wrist last.
    const int Wrist = 20, IndexBase = 7, MiddleBase = 11, PinkyBase = 19;

    /// <summary>Rewrites finger rotations of <paramref name="document"/>, whose frames match the
    /// per-frame <paramref name="observations"/> one to one. Returns the share of hand-frames that
    /// used observations, from 0 to 1. Skips a hand whose finger bones are missing.</summary>
    internal static double Apply(MotionDocument document, ReadOnlySpan<float> observations)
    {
        var frames = document.Frames.Count;
        if (frames == 0 || observations.Length != frames * SamObservations.Elements)
            throw new ArgumentException("Finger transfer needs one SAM observation per motion frame.");
        double used = 0; var hands = 0;
        var world = new Quaternion[document.Bones.Count]; var position = new Vector3[world.Length];
        foreach (var left in new[] { true, false })
        {
            if (Bones(document, left ? BoneRole.HandL : BoneRole.HandR, left ? Left : Right) is not { } bones) continue;
            hands++;
            var directions = new Vector3[frames * Segments]; var weights = new double[frames];
            for (var f = 0; f < frames; f++)
                weights[f] = Observe(observations.Slice(f * SamObservations.Elements, SamObservations.Elements), left, directions.AsSpan(f * Segments, Segments));
            var smoothed = new Vector3[directions.Length]; var trust = new double[frames];
            MotionPose.Smooth(MemoryMarshal.Cast<Vector3, float>(directions), weights, Segments * 3, Sigma, MemoryMarshal.Cast<Vector3, float>(smoothed.AsSpan()), trust);
            for (var f = 0; f < frames; f++)
            {
                if (trust[f] <= 0) continue;
                Pose(document, document.Frames[f], bones, smoothed.AsSpan(f * Segments, Segments), (float)trust[f], world, position);
                used += trust[f];
            }
        }
        return hands == 0 ? 0 : used / (hands * frames);
    }

    /// <summary>The hand bone followed by the finger bones in <paramref name="roles"/> order, or null.</summary>
    static int[] Bones(MotionDocument document, BoneRole hand, BoneRole[] roles)
    {
        var bones = new int[roles.Length + 1];
        bones[0] = MotionPose.Bone(document, hand);
        for (var i = 0; i < roles.Length; i++) bones[i + 1] = MotionPose.Bone(document, roles[i]);
        return Array.IndexOf(bones, -1) < 0 ? bones : null;
    }

    /// <summary>Palm-frame unit directions of the 15 observed finger segments, and how much to trust them.</summary>
    static double Observe(ReadOnlySpan<float> observation, bool left, Span<Vector3> directions)
    {
        var hand = observation.Slice(left ? SamObservations.LeftHand : SamObservations.RightHand, SamObservations.HandPoints * 3);
        if (!Palm(Point(hand, Wrist), Point(hand, IndexBase), Point(hand, MiddleBase), Point(hand, PinkyBase), out var palm)) return 0;
        Matrix4x4.Invert(palm, out var toPalm);
        for (var finger = 0; finger < 5; finger++)
            for (var segment = 0; segment < 3; segment++)
            {
                var from = Point(hand, finger * 4 + 3 - segment); var to = Point(hand, finger * 4 + 2 - segment);
                var direction = Vector3.TransformNormal(to - from, toPalm);
                if (!(direction.LengthSquared() > 1e-12f) || !float.IsFinite(direction.X + direction.Y + direction.Z)) return 0;
                directions[finger * 3 + segment] = Vector3.Normalize(direction);
            }
        return Math.Clamp(Disagreement - ImageDistance(observation, left), 0, 1);
    }

    /// <summary>Mean image distance between the hand pass's and the body pass's hand landmarks,
    /// in hand sizes (wrist to middle-finger base, as the body pass sees it).</summary>
    static double ImageDistance(ReadOnlySpan<float> observation, bool left)
    {
        const int Pass = SamKeypoints.Count * 2;
        var image = observation.Slice(SamObservations.ImageKeypoints);
        var handPass = image.Slice((left ? 1 : 2) * Pass + SamObservations.RightHandStart * 2, SamObservations.HandPoints * 2);
        var bodyPass = image.Slice((left ? SamObservations.LeftHandStart : SamObservations.RightHandStart) * 2, SamObservations.HandPoints * 2);
        var size = Math.Sqrt(Square(bodyPass[MiddleBase * 2] - bodyPass[Wrist * 2]) + Square(bodyPass[MiddleBase * 2 + 1] - bodyPass[Wrist * 2 + 1]));
        if (!(size > 1e-3)) return double.PositiveInfinity;
        double total = 0;
        for (var i = 0; i < SamObservations.HandPoints; i++)
            total += Math.Sqrt(Square(handPass[i * 2] - bodyPass[i * 2]) + Square(handPass[i * 2 + 1] - bodyPass[i * 2 + 1]));
        return total / SamObservations.HandPoints / size;
    }

    static Vector3 Point(ReadOnlySpan<float> points, int i) => new(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);

    static double Square(double value) => value * value;

    /// <summary>Palm axes as matrix rows: along the middle finger, across the palm, and its normal.</summary>
    static bool Palm(Vector3 wrist, Vector3 index, Vector3 middle, Vector3 pinky, out Matrix4x4 palm)
    {
        palm = Matrix4x4.Identity;
        var along = middle - wrist; var across = index - pinky;
        var normal = Vector3.Cross(along, across);
        if (!(along.LengthSquared() > 1e-12f) || !(normal.LengthSquared() > 1e-12f)) return false;
        along = Vector3.Normalize(along); normal = Vector3.Normalize(normal); across = Vector3.Cross(normal, along);
        palm = new Matrix4x4(along.X, along.Y, along.Z, 0, across.X, across.Y, across.Z, 0, normal.X, normal.Y, normal.Z, 0, 0, 0, 0, 1);
        return true;
    }

    /// <summary>Swings each finger joint, base to tip, toward the observed palm-frame directions,
    /// blending with the original rotation by <paramref name="trust"/>.</summary>
    static void Pose(MotionDocument document, MotionFrame frame, int[] bones, ReadOnlySpan<Vector3> directions, float trust,
        Quaternion[] world, Vector3[] position)
    {
        MotionPose.World(document, frame, world, position);
        // Proximal joints sit at the finger bases, matching the observed palm landmarks.
        if (!Palm(position[bones[0]], position[bones[4]], position[bones[7]], position[bones[13]], out var palm)) return;
        for (var finger = 0; finger < 5; finger++)
        {
            var parentWorld = world[bones[0]];
            for (var segment = 0; segment < 3; segment++)
            {
                var bone = bones[1 + finger * 3 + segment];
                // A distal joint has no child; its bone continues the middle segment's rest direction.
                var child = segment < 2 ? bones[2 + finger * 3 + segment] : bone;
                var along = MotionDocument.V(frame.Positions[child]);
                if (!(along.LengthSquared() > 1e-12f)) break;
                var original = MotionDocument.Q(frame.Rotations[bone]);
                var target = Vector3.TransformNormal(directions[finger * 3 + segment], palm);
                // Opposite observations averaged out; nothing to aim at.
                if (!(target.LengthSquared() > 1e-8f)) break;
                var desired = Vector3.Normalize(Vector3.Transform(target, Quaternion.Inverse(parentWorld)));
                var current = Vector3.Normalize(Vector3.Transform(along, original));
                var posed = Quaternion.Normalize(MathQ.FromTo(current, desired) * original);
                var blended = Quaternion.Normalize(Quaternion.Slerp(original, posed, trust));
                frame.Rotations[bone] = new[] { blended.X, blended.Y, blended.Z, blended.W };
                parentWorld *= blended;
            }
        }
    }
}
