using System.Numerics;
using HumanoidMocapX;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Inference;

static class HeadPitchTests
{
    internal static void Run()
    {
        var nodding = Nodding(.35f).ToDocument(); var observations = ObservationFixture.Observe(nodding);
        var same = Nodding(.35f).ToDocument();
        MotionTests.Near(0, HeadPitch.Apply(same, observations), .01);

        var level = MotionTests.Fixture().ToDocument();
        var applied = HeadPitch.Apply(level, observations);
        MotionTests.Near(.35 * 180 / Math.PI, applied, .1);
        var head = ObservationFixture.Bone(level, "Head"); var neck = ObservationFixture.Bone(level, "Neck");
        for (var f = 0; f < level.Frames.Count; f++)
        {
            ObservationFixture.Positions(level, level.Frames[f], out var rotation, out _);
            ObservationFixture.Positions(nodding, nodding.Frames[f], out var expected, out _);
            MotionTests.Near(1, Math.Abs(Quaternion.Dot(expected[head], rotation[head])), 1e-5);
            // The neck carries half of the nod.
            MotionTests.Near(.35 / 2, 2 * Math.Acos(Math.Min(1, Math.Abs(MotionDocument.Q(level.Frames[f].Rotations[neck]).W))), 1e-3);
        }

        var wild = Nodding(1.2f).ToDocument(); var untouched = MotionTests.Fixture().ToDocument();
        MotionTests.Near(0, HeadPitch.Apply(untouched, ObservationFixture.Observe(wild)));
        MotionTests.Throws<ArgumentException>(() => HeadPitch.Apply(untouched, observations.AsSpan(1)));
        Console.WriteLine("PASS: head pitch reaches the observed nod across neck and head, keeps matching heads and ignores implausible faces");
    }

    /// <summary>The motion fixture with the head nodded forward by <paramref name="angle"/> radians.</summary>
    static FlowMotion Nodding(float angle)
    {
        var fixture = MotionTests.Fixture();
        // SMPL's head is joint 15; positive rotation about +x pitches the face down.
        foreach (var pose in fixture.Poses) pose[15 * 3] = angle;
        return fixture;
    }
}
