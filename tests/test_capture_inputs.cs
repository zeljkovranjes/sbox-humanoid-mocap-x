using HumanoidMocapX.Inference;

static class CaptureInputTests
{
    internal static void Run()
    {
        var person = new YoloBox(10, 20, 110, 220, .8f); var other = new YoloBox(250, 10, 290, 100, .95f);
        var track = new PersonTrack(5); track.Add([other, person]); track.Add([person with { Left = 12, Right = 112 }, other]);
        track.Add([]); track.Add([person with { Left = 16, Right = 116 }]); track.Add([person with { Left = 18, Right = 118 }]);
        var span = track.Complete(); var boxes = span.Boxes;
        MotionTests.Equal(0, span.First); MotionTests.Equal(5, span.Count); MotionTests.Check(span.Note(2, 5) is null, "A fully visible person needs no note");
        for (var i = 0; i < 5; i++) { MotionTests.Equal(10f + i * 2, boxes[i * 4]); MotionTests.Equal(110f + i * 2, boxes[i * 4 + 2]); }
        MotionTests.Throws<InvalidOperationException>(() => track.Add([person]));
        var nobody = new PersonTrack(2); nobody.Add([]); nobody.Add([]); MotionTests.Throws<InvalidOperationException>(() => nobody.Complete());
        var unfinished = new PersonTrack(2); unfinished.Add([person]); MotionTests.Throws<InvalidOperationException>(() => unfinished.Complete());
        // The capture starts where the person first appears and ends at their last detection.
        var late = new PersonTrack(6); late.Add([]); late.Add([]); late.Add([person]); late.Add([]); late.Add([person with { Left = 14, Right = 114 }]); late.Add([]);
        var visible = late.Complete();
        MotionTests.Equal(2, visible.First); MotionTests.Equal(3, visible.Count); MotionTests.Equal(12, visible.Boxes.Length);
        MotionTests.Equal(12f, visible.Boxes[4]); MotionTests.Equal(14f, visible.Boxes[8]); MotionTests.Near(2 / 30d, visible.StartSeconds);
        MotionTests.Equal("The person is in view from 1.07 s to 1.17 s of the video, so only that part was captured.", visible.Note(1, 6));
        var uncertain = new PersonTrack(2); uncertain.Add([person]);
        MotionTests.Throws<InvalidOperationException>(() => uncertain.Add([person, person with { Left = 11, Right = 111 }]));
        // A person lost for more than half a second ends the capture; a later return is not joined on.
        var lost = new PersonTrack(18); lost.Add([person]); for (var i = 0; i < 15; i++) lost.Add([]);
        lost.Add([other]); lost.Add([person]);
        var kept = lost.Complete(); MotionTests.Equal(0, kept.First); MotionTests.Equal(1, kept.Count);
        MotionTests.Throws<ArgumentException>(() => new PersonTrack(1).Add([person with { Left = float.NaN }]));
        var reuse = new PersonTrack(20); var input = new[] { person }; for (var i = 0; i < 4; i++) reuse.Add(input);
        var before = GC.GetAllocatedBytesForCurrentThread(); reuse.Add(input); MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        var noise = new float[491 * 501]; var repeated = new float[noise.Length]; FlowNoise.Fill(noise, 0); FlowNoise.Fill(repeated, 0);
        MotionTests.Check(noise.SequenceEqual(repeated), "Initial noise must repeat for the same seed");
        var mean = noise.Average(v => (double)v); var variance = noise.Average(v => (double)v * v) - mean * mean;
        MotionTests.Near(0, mean, .01); MotionTests.Near(1, variance, .02);
        MotionTests.Check(noise.Min() < -3 && noise.Max() > 3 && noise.All(float.IsFinite), "Normal noise must cover both tails and remain finite");
        FlowNoise.Fill(repeated, 1); MotionTests.Check(!noise.SequenceEqual(repeated), "Different seeds must vary initial noise");
        using var stop = new CancellationTokenSource(); stop.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => FlowNoise.Fill(noise, 0, stop.Token));
        MotionTests.Throws<ArgumentException>(() => FlowNoise.Fill(new float[490], 0));
        Console.WriteLine("PASS: person identity continuity, short-gap interpolation, ambiguous/lost-person rejection, allocation-free tracking and seeded standard-normal latents");
    }
}
