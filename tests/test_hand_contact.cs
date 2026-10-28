using System.Numerics;
using HumanoidMocapX;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Inference;

static class HandContactTests
{
    internal static void Run()
    {
        var handstand = Handstand().ToDocument();
        var marked = HandContact.Apply(handstand, ObservationFixture.Observe(handstand));
        MotionTests.Equal(2 * handstand.Frames.Count, marked);
        foreach (var hand in new[] { "HandL", "HandR" })
        {
            var track = handstand.StationaryJoints.Single(t => t.Bone == handstand.Bones[ObservationFixture.Bone(handstand, hand)].Name);
            MotionTests.Check(track.Probability.All(p => p == 1), hand + " must be marked on the floor throughout");
        }
        // Marking again replaces the tracks instead of adding a second set.
        HandContact.Apply(handstand, ObservationFixture.Observe(handstand));
        MotionTests.Equal(2, handstand.StationaryJoints.Count(t => t.Bone.Contains("Hand", StringComparison.OrdinalIgnoreCase) || t.Source.StartsWith("SAM")));

        var standing = MotionTests.Fixture().ToDocument();
        MotionTests.Equal(0, HandContact.Apply(standing, ObservationFixture.Observe(standing)));
        MotionTests.Throws<ArgumentException>(() => HandContact.Apply(standing, new float[3]));
        Console.WriteLine("PASS: hand contact marks hands SAM sees on the floor under a raised body, and not a standing body's");
    }

    /// <summary>The motion fixture upside down with both arms straight overhead, now below it: a two-handed handstand.</summary>
    static FlowMotion Handstand()
    {
        var fixture = MotionTests.Fixture();
        foreach (var pose in fixture.Poses)
        {
            pose[0] = MathF.PI;                          // pelvis turned over about X
            pose[16 * 3 + 2] = MathF.PI / 2;             // left arm (along +X) raised about Z
            pose[17 * 3 + 2] = -MathF.PI / 2;            // right arm (along -X) raised about Z
        }
        return fixture;
    }
}
