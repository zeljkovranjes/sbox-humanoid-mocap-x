using System.Numerics;
using HumanoidMocapX;
using HumanoidMocapX.Core.Motion;

static class CaptureDespikeTests
{
    internal static void Run()
    {
        // A steady fast spin, 40° a frame, is not a snap.
        var spin = MotionTests.Fixture();
        for (var f = 0; f < spin.Poses.Length; f++) spin.Poses[f][1] = f * 40 * MathF.PI / 180;
        var spinning = spin.ToDocument();
        MotionTests.Equal(0, CaptureDespike.Apply(spinning));

        // One frame where a knee snaps 20° out of line and the body jumps 6 cm.
        var snap = MotionTests.Fixture();
        snap.Poses[30][3 * 4] = 20 * MathF.PI / 180; snap.Translations[30][0] = .06f;
        var snapped = snap.ToDocument();
        var bone = ObservationFixture.Bone(snapped, "LowerLegL");
        MotionTests.Check(CaptureDespike.Apply(snapped) > 0, "The snap was not corrected");
        var q = MotionDocument.Q(snapped.Frames[30].Rotations[bone]);
        var off = 2 * Math.Acos(Math.Min(1, Math.Abs(q.W))) * 180 / Math.PI;
        MotionTests.Check(off < 7, $"The knee still snaps {off:F1} degrees out of line");
        var root = MotionDocument.V(snapped.Frames[30].Positions[0]) - MotionDocument.V(snapped.Frames[29].Positions[0]);
        MotionTests.Check(root.Length() < .02f, $"The body still jumps {root.Length() * 100:F1} cm");
        Console.WriteLine("PASS: capture despike pulls a one-frame snap back onto its neighbours' path and leaves a steady spin alone");
    }
}
