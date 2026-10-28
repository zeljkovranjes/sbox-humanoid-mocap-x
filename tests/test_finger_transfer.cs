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

        var stiff = Stiff().ToDocument(); var coupled = MotionTests.Fixture().ToDocument();
        FingerTransfer.Apply(coupled, ObservationFixture.Observe(stiff));
        foreach (var side in new[] { "L", "R" })
            foreach (var finger in new[] { "Index", "Middle", "Ring", "Pinky" })
            {
                var (observed, straight) = Bends(stiff, finger, side);
                var (middle, tip) = Bends(coupled, finger, side);
                MotionTests.Check(observed > .4 && straight < .01, $"Stiff fixture bends {observed} and {straight}");
                MotionTests.Near(observed, middle, .01);
                MotionTests.Near(observed * 2 / 3, tip, .01);
            }
        Console.WriteLine("PASS: finger transfer keeps matching fingers, reaches observed finger directions, bends straight tips with their middle joints, and ignores hand passes that disagree with the body");
    }

    /// <summary>Middle and tip joint bends in radians of one finger in the first frame.</summary>
    static (double Middle, double Tip) Bends(MotionDocument motion, string finger, string side)
    {
        var position = ObservationFixture.Positions(motion, motion.Frames[0], out _, out var beyond);
        int prox = ObservationFixture.Bone(motion, finger + "Prox" + side), mid = ObservationFixture.Bone(motion, finger + "Mid" + side), dist = ObservationFixture.Bone(motion, finger + "Dist" + side);
        Vector3 a = Vector3.Normalize(position[mid] - position[prox]), b = Vector3.Normalize(position[dist] - position[mid]), c = Vector3.Normalize(beyond[dist] - position[dist]);
        return (Math.Acos(Math.Clamp(Vector3.Dot(a, b), -1, 1)), Math.Acos(Math.Clamp(Vector3.Dot(b, c), -1, 1)));
    }

    /// <summary>The motion fixture with bent middle finger joints and straight tips, as SAM reports relaxed hands.</summary>
    static FlowMotion Stiff()
    {
        var fixture = MotionTests.Fixture();
        foreach (var pose in fixture.Poses)
            for (var joint = 22; joint < 52; joint++)
                pose[joint * 3 + 2] = (joint - 22) % 3 == 1 ? .6f : 0;
        return fixture;
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
