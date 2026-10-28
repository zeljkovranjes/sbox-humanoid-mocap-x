using System.Numerics;
using HumanoidMocapX;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Inference;

static class HandTransferTests
{
    internal static void Run()
    {
        var turned = Arms(.9f).ToDocument(); var observations = ObservationFixture.Observe(turned);
        var same = Arms(.9f).ToDocument();
        MotionTests.Near(1, HandTransfer.Apply(same, observations));
        foreach (var side in new[] { "HandL", "HandR" })
        {
            var hand = ObservationFixture.Bone(same, side);
            MotionTests.Near(1, Math.Abs(Quaternion.Dot(MotionDocument.Q(turned.Frames[0].Rotations[hand]), MotionDocument.Q(same.Frames[0].Rotations[hand]))), 1e-5);
        }

        var level = Arms(0).ToDocument();
        MotionTests.Near(1, HandTransfer.Apply(level, observations));
        ObservationFixture.Positions(turned, turned.Frames[^1], out var expected, out _);
        ObservationFixture.Positions(level, level.Frames[^1], out var actual, out _);
        foreach (var side in new[] { "HandL", "HandR" })
        {
            var hand = ObservationFixture.Bone(level, side);
            MotionTests.Near(1, Math.Abs(Quaternion.Dot(expected[hand], actual[hand])), 1e-5);
        }

        var distrusted = ObservationFixture.Observe(turned);
        for (var f = 0; f < turned.Frames.Count; f++)
            for (var i = SamObservations.ImageKeypoints + SamKeypoints.Count * 2; i < SamObservations.Elements; i++)
                distrusted[f * SamObservations.Elements + i] += 500;
        var kept = Arms(0).ToDocument(); var original = Arms(0).ToDocument();
        MotionTests.Near(0, HandTransfer.Apply(kept, distrusted));
        for (var b = 0; b < kept.Bones.Count; b++)
            MotionTests.Near(1, Quaternion.Dot(MotionDocument.Q(original.Frames[0].Rotations[b]), MotionDocument.Q(kept.Frames[0].Rotations[b])), 1e-6);
        MotionTests.Throws<ArgumentException>(() => HandTransfer.Apply(kept, observations.AsSpan(1)));
        Console.WriteLine("PASS: hand transfer keeps matching wrists, turns wrists to the observed palm, and ignores hand passes that disagree with the body");
    }

    /// <summary>The motion fixture with bent elbows and both wrists turned by <paramref name="turn"/> radians.</summary>
    static FlowMotion Arms(float turn)
    {
        var fixture = MotionTests.Fixture();
        foreach (var pose in fixture.Poses)
        {
            // SMPL joints 18 and 19 are the elbows, 20 and 21 the wrists.
            pose[18 * 3 + 1] = .8f; pose[19 * 3 + 1] = -.8f;
            pose[20 * 3] = turn; pose[20 * 3 + 2] = turn / 2;
            pose[21 * 3] = -turn; pose[21 * 3 + 1] = turn / 3;
        }
        return fixture;
    }
}
