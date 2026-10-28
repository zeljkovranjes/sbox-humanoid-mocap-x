using System.Numerics;
using HumanoidMocapX;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Inference;

static class LimbBendTests
{
    internal static void Run()
    {
        // The video shows the left elbow bent 30°; the capture bends it 60°.
        var observations = ObservationFixture.Observe(Bent(.5f).ToDocument());
        var same = Bent(.5f).ToDocument();
        MotionTests.Near(0, LimbBend.Apply(same, observations), .5);

        var bent = Bent(1.05f).ToDocument(); var before = Bent(1.05f).ToDocument();
        var applied = LimbBend.Apply(bent, observations);
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
        MotionTests.Near(0, LimbBend.Apply(wild, observations), .5);
        MotionTests.Throws<ArgumentException>(() => LimbBend.Apply(same, observations.AsSpan(1)));
        // A knee bent 60° in the video and 20° in the capture folds to it, the foot keeping its attitude.
        var seen = Knee(1.05f).ToDocument(); var knee = ObservationFixture.Observe(seen); var straight = Knee(.35f).ToDocument(); var kept = Knee(.35f).ToDocument();
        MotionTests.Check(LimbBend.Apply(straight, knee) > 25, "The knee was not folded");
        int thigh = ObservationFixture.Bone(straight, "UpperLegL"), shin = ObservationFixture.Bone(straight, "LowerLegL"), foot = ObservationFixture.Bone(straight, "FootL");
        for (var f = 0; f < straight.Frames.Count; f++)
        {
            var position = ObservationFixture.Positions(straight, straight.Frames[f], out var rotation, out _);
            // The fixture's rest leg is not quite straight, so the angle to reach is the observed one.
            var observed = ObservationFixture.Positions(seen, seen.Frames[f], out _, out _);
            MotionTests.Near(Angle(observed[thigh] - observed[shin], observed[foot] - observed[shin]), Angle(position[thigh] - position[shin], position[foot] - position[shin]), 2);
            ObservationFixture.Positions(kept, kept.Frames[f], out var original, out _);
            MotionTests.Near(1, Math.Abs(Quaternion.Dot(original[foot], rotation[foot])), 1e-5);
        }
        Console.WriteLine("PASS: limb bend opens an elbow and folds a knee to the angles SAM sees, keeps hand and foot attitude, and leaves matching and implausible limbs alone");
    }

    /// <summary>The motion fixture with the left elbow bent by <paramref name="angle"/> radians.</summary>
    static FlowMotion Bent(float angle)
    {
        var fixture = MotionTests.Fixture();
        // SMPL's left elbow is joint 18; the T-pose arm lies along +x, so a turn about +y bends the forearm forward.
        foreach (var pose in fixture.Poses) pose[18 * 3 + 1] = angle;
        return fixture;
    }

    /// <summary>The motion fixture with the left knee bent by <paramref name="angle"/> radians.</summary>
    static FlowMotion Knee(float angle)
    {
        var fixture = MotionTests.Fixture();
        // SMPL's left knee is joint 4; a turn about +x swings the shin back.
        foreach (var pose in fixture.Poses) pose[4 * 3] = angle;
        return fixture;
    }

    static double Angle(Vector3 a, Vector3 b) => Math.Acos(Math.Clamp(Vector3.Dot(Vector3.Normalize(a), Vector3.Normalize(b)), -1, 1)) * 180 / Math.PI;
}
