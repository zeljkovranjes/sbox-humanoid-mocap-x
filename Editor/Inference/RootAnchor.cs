using System;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Holds the root where the video puts it. FlowHMR integrates its own root velocity,
/// so a dancer who stays on one spot can wander off it (0.5 m over 9 s on the breakin clip).
/// SAM places the pelvis in camera space in every frame independently, which does not drift.
/// With a camera that neither turns nor travels, the capture's horizontal root path is pulled to
/// SAM's, smoothed hard in depth because SAM's depth is far noisier than its image position.
/// Height stays FlowHMR's, and so does the start position.</summary>
internal static class RootAnchor
{
    // A camera that turns further than this over the clip, or whose centre moves further than
    // this share of the scene's distance, is not static. Measured travel: tripods 0.001–0.002,
    // a camera panning from one spot 0.007, a handheld camera walking a deep park path 0.023
    // (about a metre there). Leaning low only costs the anchor on a still camera; leaning high
    // would pull the root after a moving one.
    const double MaxCameraTurn = 2 * Math.PI / 180;
    internal const float MaxCameraTravel = .015f;
    // Body directions that disagree by more than this after the camera fit are left out of it.
    const float FitResidual = 30 * MathF.PI / 180;
    // Smoothing in frames: across the image and in depth.
    const double LateralSigma = 10, DepthSigma = 60;
    const int LeftShoulder = 5, RightShoulder = 6, LeftHip = 9, RightHip = 10;

    /// <summary>Moves each root of <paramref name="document"/> horizontally to follow SAM's
    /// pelvis in <paramref name="observations"/>. <paramref name="cameraRotations"/> holds one
    /// row-major 3×3 camera rotation per frame; <paramref name="cameraTravel"/> is
    /// <see cref="FlowCamera.Travel"/>, NaN when unknown. Returns the largest root shift in
    /// metres, or null when the camera turns, travels or may have, or the document lacks a root,
    /// shoulders or hips.</summary>
    internal static double? Apply(MotionDocument document, ReadOnlySpan<float> observations, ReadOnlySpan<float> cameraRotations, float cameraTravel)
    {
        var frames = document.Frames.Count;
        if (frames == 0 || observations.Length != frames * SamObservations.Elements || cameraRotations.Length != frames * 9)
            throw new ArgumentException("Root anchoring needs one SAM observation and camera rotation per motion frame.");
        if (!(cameraTravel <= MaxCameraTravel) || !IsStatic(cameraRotations)) return null;
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
            observed[f] = SamObservations.Point(k, SamObservations.Translation / 3) + (SamObservations.Point(k, LeftHip) + SamObservations.Point(k, RightHip)) / 2;
            seen[f * 3] = SamObservations.Point(k, RightShoulder) - SamObservations.Point(k, LeftShoulder);
            seen[f * 3 + 1] = SamObservations.Point(k, RightHip) - SamObservations.Point(k, LeftHip);
            seen[f * 3 + 2] = SamObservations.Point(k, LeftShoulder) + SamObservations.Point(k, RightShoulder) - SamObservations.Point(k, LeftHip) - SamObservations.Point(k, RightHip);
            posed[f * 3] = position[armR] - position[armL];
            posed[f * 3 + 1] = position[legR] - position[legL];
            posed[f * 3 + 2] = position[armL] + position[armR] - position[legL] - position[legR];
        }
        var toWorld = RotationFit.Fit(seen, posed, FitResidual);

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

}
