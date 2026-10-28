using HumanoidMocapX;
using HumanoidMocapX.Inference;

static class RootAnchorTests
{
    const float Drift = .5f;

    internal static void Run()
    {
        var still = Long(MotionTests.Fixture()).ToDocument(); var observations = ObservationFixture.Observe(still);
        var drifting = Drifting().ToDocument(); var rotations = ObservationFixture.StaticCamera(drifting);
        var shift = RootAnchor.Apply(drifting, observations, rotations);
        MotionTests.Check(shift is > .3 and < .55, $"Expected the anchor to undo most of the drift, shifted {shift}");
        var last = drifting.Frames[^1].Positions[0]; var start = drifting.Frames[0].Positions[0];
        MotionTests.Check(Math.Abs(last[0] - still.Frames[^1].Positions[0][0]) < .03, $"Root still drifts: {last[0]}");
        MotionTests.Near(still.Frames[^1].Positions[0][1], last[1], 1e-6);
        MotionTests.Near(0, start[0] - still.Frames[0].Positions[0][0], .02);

        var turning = ObservationFixture.StaticCamera(still);
        for (var f = 0; f < still.Frames.Count; f++)
        {
            var angle = f * .002f;
            turning[f * 9] = MathF.Cos(angle); turning[f * 9 + 2] = MathF.Sin(angle); turning[f * 9 + 6] = -MathF.Sin(angle); turning[f * 9 + 8] = MathF.Cos(angle);
        }
        var untouched = Drifting().ToDocument();
        MotionTests.Check(RootAnchor.Apply(untouched, observations, turning) is null, "A turning camera must not anchor the root");
        MotionTests.Near(Drift, untouched.Frames[^1].Positions[0][0], 1e-6);
        MotionTests.Throws<ArgumentException>(() => RootAnchor.Apply(untouched, observations, rotations.AsSpan(9)));
        Console.WriteLine("PASS: root anchor removes drift the camera does not see, keeps height and start, and leaves turning cameras alone");
    }

    /// <summary>The motion fixture over 10 s, wandering sideways by <see cref="Drift"/> metres.</summary>
    static FlowMotion Drifting()
    {
        var fixture = Long(MotionTests.Fixture());
        for (var f = 0; f < fixture.Translations.Length; f++) fixture.Translations[f][0] = Drift * f / (fixture.Translations.Length - 1);
        return fixture;
    }

    static FlowMotion Long(FlowMotion fixture)
    {
        const int Frames = 300;
        fixture.Poses = Enumerable.Range(0, Frames).Select(_ => new float[156]).ToArray();
        fixture.Translations = Enumerable.Range(0, Frames).Select(_ => new float[3]).ToArray();
        fixture.Contacts = Enumerable.Range(0, Frames).Select(_ => new[] { 1f, 1f, 1f, 1f }).ToArray();
        return fixture;
    }
}
