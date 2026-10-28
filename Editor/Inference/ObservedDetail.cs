using System;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.Inference;

/// <summary>What the video shows beyond FlowHMR's motion: per-frame <see cref="SamObservations"/>
/// and the camera rotations. Applying it to a capture takes fingers and head pitch from SAM 3D
/// Body and, with a static camera, holds the root where the video puts it.</summary>
internal sealed record ObservedDetail(float[] Observations, float[] CameraRotations)
{
    /// <summary>Corrects <paramref name="document"/>, whose frames match the observed frames one
    /// to one, in place, and notes each correction in its diagnostics. Throws
    /// <see cref="ArgumentException"/> when the frame counts differ.</summary>
    internal void Apply(MotionDocument document)
    {
        var fingers = FingerTransfer.Apply(document, Observations);
        var head = HeadPitch.Apply(document, Observations);
        var root = RootAnchor.Apply(document, Observations, CameraRotations);
        document.Diagnostics.Add(FormattableString.Invariant($"Fingers follow SAM 3D Body's hand close-ups in {fingers * 100:0}% of hand frames; head pitch follows its face keypoints, by up to {head:0}°."));
        document.Diagnostics.Add(root is { } shift
            ? FormattableString.Invariant($"The camera is static, so the root follows the person's position in the video, up to {shift:0.00} m from FlowHMR's path.")
            : "The camera moves, so the root follows FlowHMR's path.");
    }
}
