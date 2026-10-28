using System.Numerics;
using HumanoidMocapX;
using HumanoidMocapX.Core.Motion;

static class CaptureSpinTests
{
    internal static void Run()
    {
        // 60 frames (2 s) turning 40° a frame on average, unevenly: 20° and 60° in turn.
        var spinning = Turning(f => f * 40 + (f % 2 == 0 ? 0 : 20)).ToDocument();
        var before = Headings(spinning);
        MotionTests.Check(CaptureSpin.Apply(spinning) > 40, "A sustained fast spin must be smoothed");
        var after = Headings(spinning);
        var steps = after.Zip(after.Skip(1), (a, b) => b - a).Skip(10).Take(40).ToArray();
        MotionTests.Check(steps.Max() - steps.Min() < 6, $"Spin rate still uneven: {steps.Min():F1}..{steps.Max():F1} degrees per frame");
        MotionTests.Near(before[30], after[30], 12);

        var slow = Turning(f => f * 5).ToDocument(); var original = Turning(f => f * 5).ToDocument();
        MotionTests.Equal(0, CaptureSpin.Apply(slow));
        MotionTests.Near(1, Quaternion.Dot(MotionDocument.Q(original.Frames[30].Rotations[0]), MotionDocument.Q(slow.Frames[30].Rotations[0])), 1e-7);
        Console.WriteLine("PASS: capture spin evens out fast uneven turns, keeps their heading and leaves slow turns alone");
    }

    /// <summary>The motion fixture turned about the vertical by <paramref name="degrees"/> per frame index.</summary>
    static FlowMotion Turning(Func<int, float> degrees)
    {
        var fixture = MotionTests.Fixture();
        for (var f = 0; f < fixture.Poses.Length; f++) fixture.Poses[f][1] = degrees(f) * MathF.PI / 180;
        return fixture;
    }

    /// <summary>Unwrapped heading of the hip line in degrees, positive about +Y.</summary>
    static double[] Headings(MotionDocument motion)
    {
        var left = ObservationFixture.Bone(motion, "UpperLegL"); var right = ObservationFixture.Bone(motion, "UpperLegR");
        var headings = new double[motion.Frames.Count];
        for (var f = 0; f < headings.Length; f++)
        {
            var position = ObservationFixture.Positions(motion, motion.Frames[f], out _, out _);
            var line = position[right] - position[left];
            headings[f] = Math.Atan2(-line.Z, line.X) * 180 / Math.PI;
            if (f > 0) headings[f] = headings[f - 1] + Math.IEEERemainder(headings[f] - headings[f - 1], 360);
        }
        return headings;
    }
}
