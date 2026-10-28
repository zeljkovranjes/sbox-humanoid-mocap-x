using System.Numerics;
using HumanoidMocapX;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Inference;

static class FingerTransferTests
{
    internal static void Run()
    {
        var curled = Curled().ToDocument(); var observations = ObservationFixture.Observe(curled);
        var unchanged = Curled().ToDocument();
        MotionTests.Near(1, FingerTransfer.Apply(unchanged, observations));
        for (var f = 0; f < curled.Frames.Count; f++)
            for (var b = 0; b < curled.Bones.Count; b++)
                MotionTests.Near(1, Math.Abs(Quaternion.Dot(MotionDocument.Q(curled.Frames[f].Rotations[b]), MotionDocument.Q(unchanged.Frames[f].Rotations[b]))), 1e-5);

        var flat = MotionTests.Fixture().ToDocument();
        MotionTests.Near(1, FingerTransfer.Apply(flat, observations));
        var actual = ObservationFixture.Observe(flat);
        foreach (var start in new[] { SamObservations.LeftHand, SamObservations.RightHand })
            for (var i = 0; i < SamObservations.HandPoints * 3; i++) MotionTests.Near(observations[start + i], actual[start + i], 1e-4);

        var distrusted = ObservationFixture.Observe(curled);
        for (var f = 0; f < curled.Frames.Count; f++)
            for (var i = SamObservations.ImageKeypoints + SamKeypoints.Count * 2; i < SamObservations.Elements; i++)
                distrusted[f * SamObservations.Elements + i] += 500;
        var kept = MotionTests.Fixture().ToDocument(); var original = MotionTests.Fixture().ToDocument();
        MotionTests.Near(0, FingerTransfer.Apply(kept, distrusted));
        for (var b = 0; b < kept.Bones.Count; b++)
            MotionTests.Near(1, Quaternion.Dot(MotionDocument.Q(original.Frames[0].Rotations[b]), MotionDocument.Q(kept.Frames[0].Rotations[b])), 1e-6);
        MotionTests.Throws<ArgumentException>(() => FingerTransfer.Apply(kept, observations.AsSpan(1)));
        Console.WriteLine("PASS: finger transfer keeps matching fingers, reaches observed finger directions, and ignores hand passes that disagree with the body");
    }

    /// <summary>The motion fixture with every finger joint bent by its own angle.</summary>
    static FlowMotion Curled()
    {
        var fixture = MotionTests.Fixture();
        foreach (var pose in fixture.Poses)
            for (var joint = 22; joint < 52; joint++)
            {
                pose[joint * 3 + 1] = .2f - (joint - 22) * .01f;
                pose[joint * 3 + 2] = .3f + (joint - 22) % 3 * .25f;
            }
        return fixture;
    }
}
