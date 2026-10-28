using System.Numerics;
using HumanoidMocapX;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Inference;

static class ElbowBendTests
{
    internal static void Run()
    {
        // The video shows the left elbow bent 30°; the capture bends it 60°.
        var observations = ObservationFixture.Observe(Bent(.5f).ToDocument());
        var same = Bent(.5f).ToDocument();
        MotionTests.Near(0, ElbowBend.Apply(same, observations), .5);

        var bent = Bent(1.05f).ToDocument(); var before = Bent(1.05f).ToDocument();
        var applied = ElbowBend.Apply(bent, observations);
        MotionTests.Check(applied > 25, $"Elbow opened only {applied:F1} degrees");
        int upper = ObservationFixture.Bone(bent, "UpperArmL"), lower = ObservationFixture.Bone(bent, "LowerArmL"), hand = ObservationFixture.Bone(bent, "HandL");
        for (var f = 0; f < bent.Frames.Count; f++)
        {
            var position = ObservationFixture.Positions(bent, bent.Frames[f], out var rotation, out _);
            var angle = Angle(position[upper] - position[lower], position[hand] - position[lower]);
            MotionTests.Near(180 - .5 * 180 / Math.PI, angle, 2);
            ObservationFixture.Positions(before, before.Frames[f], out var original, out _);
            MotionTests.Near(1, Math.Abs(Quaternion.Dot(original[hand], rotation[hand])), 1e-5);
        }

        // A disagreement this large is a keypoint failure, not posture.
        var wild = Bent(1.8f).ToDocument();
        MotionTests.Near(0, ElbowBend.Apply(wild, observations), .5);
        MotionTests.Throws<ArgumentException>(() => ElbowBend.Apply(same, observations.AsSpan(1)));
        Console.WriteLine("PASS: elbow bend opens an elbow to the angle SAM sees, keeps the hand's attitude, and leaves matching and implausible elbows alone");
    }

    /// <summary>The motion fixture with the left elbow bent by <paramref name="angle"/> radians.</summary>
    static FlowMotion Bent(float angle)
    {
        var fixture = MotionTests.Fixture();
        // SMPL's left elbow is joint 18; the T-pose arm lies along +x, so a turn about +y bends the forearm forward.
        foreach (var pose in fixture.Poses) pose[18 * 3 + 1] = angle;
        return fixture;
    }

    static double Angle(Vector3 a, Vector3 b) => Math.Acos(Math.Clamp(Vector3.Dot(Vector3.Normalize(a), Vector3.Normalize(b)), -1, 1)) * 180 / Math.PI;
}
