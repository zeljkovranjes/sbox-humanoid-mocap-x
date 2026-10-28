namespace HumanoidMocapX.Inference;

/// <summary>Per-frame SAM 3D Body results kept beside FlowHMR's features. Keypoints follow
/// MHR70: 0–4 face, 5–20 body, 21–41 right hand and 42–62 left hand (tip, two middle joints and
/// base of each finger from thumb to pinky, then the wrist), 69 neck. 3D keypoints are in camera
/// axes, in metres relative to the body root.</summary>
internal static class SamObservations
{
    internal const int HandPoints = 21, RightHandStart = 21, LeftHandStart = 42;

    /// <summary>The body pass's 70 3D keypoints.</summary>
    internal const int BodyKeypoints = 0;
    /// <summary>The body root's camera-space position from the body pass.</summary>
    internal const int Translation = BodyKeypoints + SamKeypoints.Count * 3;
    /// <summary>The left hand pass's 21 hand keypoints, un-mirrored into camera axes.</summary>
    internal const int LeftHand = Translation + 3;
    /// <summary>The right hand pass's 21 hand keypoints.</summary>
    internal const int RightHand = LeftHand + HandPoints * 3;
    /// <summary>Original-image XY keypoints of the body, left hand and right hand passes.</summary>
    internal const int ImageKeypoints = RightHand + HandPoints * 3;
    internal const int Elements = ImageKeypoints + 3 * SamKeypoints.Count * 2;
}
