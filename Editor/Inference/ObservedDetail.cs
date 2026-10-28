using System;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

/// <summary>What the video shows beyond FlowHMR's motion: per-frame <see cref="SamObservations"/>,
/// the camera rotations and how far the camera travels (<see cref="FlowCamera.Travel"/>; NaN
/// when unknown). Applying it to a capture takes wrist turn, fingers and head pitch
/// from SAM 3D Body and, with a static camera, holds the root where the video puts it.</summary>
internal sealed record ObservedDetail(float[] Observations, float[] CameraRotations, float CameraTravel)
{
    /// <summary>Corrects <paramref name="document"/>, whose frames match the observed frames one
    /// to one, in place, and notes each correction in its diagnostics. Throws
    /// <see cref="ArgumentException"/> when the frame counts differ.</summary>
    internal void Apply(MotionDocument document)
    {
        var hands = HandTransfer.Apply(document, Observations);
        var fingers = FingerTransfer.Apply(document, Observations);
        var head = HeadPitch.Apply(document, Observations);
        var root = RootAnchor.Apply(document, Observations, CameraRotations, CameraTravel);
        if (HandContact.Apply(document, Observations) is var planted && planted > 0)
            document.Diagnostics.Add(FormattableString.Invariant($"Hands on the floor under a raised body, as SAM 3D Body sees them: {planted} hand frames marked for planting."));
        if (CaptureSpin.Apply(document) is var spun && spun > 0)
            document.Diagnostics.Add(FormattableString.Invariant($"Fast spins turn evenly: the body's heading was smoothed in {spun} frames."));
        if (CaptureDespike.Apply(document) is var despiked && despiked > 0)
            document.Diagnostics.Add(FormattableString.Invariant($"Snaps removed: {despiked} joint samples that left the smooth path through their neighbours were pulled back onto it."));
        document.Diagnostics.Add(FormattableString.Invariant($"Wrists and fingers follow SAM 3D Body's hand close-ups in {hands * 100:0}% and {fingers * 100:0}% of hand frames; head pitch follows its face keypoints, by up to {head:0}°."));
        document.Diagnostics.Add(root is { } shift
            ? FormattableString.Invariant($"The camera is static, so the root follows the person's position in the video, up to {shift:0.00} m from FlowHMR's path.")
            : "The camera moves or its motion is unknown, so the root follows FlowHMR's path.");
    }
}
