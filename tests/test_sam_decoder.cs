using HumanoidMocapX.Inference;
using HumanoidMocapX.EditorTools;

static class SamDecoderTests
{
    internal static void Run()
    {
        foreach (var hand in new[] { false, true }) Decoder(hand);
        Checkpoint();
        MotionTests.Throws<ArgumentException>(() => new SamDecoderShape(0, 1024));
        MotionTests.Throws<ArgumentException>(() => new SamDecoderShape(145, 1025));
        MotionTests.Throws<ArgumentException>(() => new FlowAttention(2, 0, 8, 2));
        Console.WriteLine("PASS: SAM body/hand decoder, asymmetric cross-attention, positional normalization, five intermediate updates and allocation-free reuse");
    }

    static void Checkpoint()
    {
        var path = Path.Combine(Path.GetTempPath(), "sam-decoder-" + Guid.NewGuid().ToString("N") + ".pt");
        try
        {
            var shape = new SamDecoderShape(5, 7, 8, 12, 2, 4, 16); var tensors = SamFixture.Decoder(shape, false);
            foreach (var pair in SamFixture.Decoder(shape, true)) tensors.Add(pair.Key, pair.Value);
            CheckpointFixture.Write(path, tensors, "state_dict", "");
            foreach (var hand in new[] { false, true })
            {
                var weights = VisionCheckpoint.LoadSamDecoder(path, hand); _ = new SamDecoder(weights, shape, hand);
                MotionTests.Throws<InvalidDataException>(() => new SamDecoder(weights, shape, !hand));
            }
        }
        finally { File.Delete(path); }
    }

    static void Decoder(bool hand)
    {
        var shape = new SamDecoderShape(5, 7, 8, 12, 2, 4, 16); var tensors = SamFixture.Decoder(shape, hand);
        var decoder = new SamDecoder(new(tensors), shape, hand); var name = hand ? "decoder_hand" : "decoder";
        var tokens = DenoiserTests.Wave(shape.TokenElements, .017f); var image = DenoiserTests.Wave(shape.ImageElements, .023f);
        var tokenPe = DenoiserTests.Wave(tokens.Length, .007f); var imagePe = DenoiserTests.Wave(image.Length, .011f);
        var actual = new float[tokens.Length]; var expected = tokens.Select(v => (double)v).ToArray(); var expectedPe = tokenPe.Select(v => (double)v).ToArray();
        var updates = 0;
        SamPoseFeedback feedback = (layer, normalized, current, positions, cancellation) =>
        { cancellation.ThrowIfCancellationRequested(); updates++; SamFixture.Feedback(shape.Width, layer, normalized, current, positions); };
        decoder.Run(tokens, image, tokenPe, imagePe, actual, feedback);
        MotionTests.Equal(5, updates);
        for (var layer = 0; layer < 6; layer++)
        {
            expected = SamReference.Block(tensors, name + ".layers." + layer, expected, image.Select(v => (double)v).ToArray(),
                expectedPe, imagePe.Select(v => (double)v).ToArray(), shape.Heads, layer == 0);
            var normalized = OmegaReference.Normalize(tensors, expected, name + ".norm_final", 1e-6);
            if (layer == 5) { expected = normalized; break; }
            for (var c = 0; c < shape.Width; c++)
            {
                expected[expected.Length - shape.Width + c] += normalized[c] * (.03 * (layer + 1));
                expectedPe[expectedPe.Length - shape.Width + c] = normalized[c] * .04 + layer * .01;
            }
        }
        for (var i = 0; i < actual.Length; i++) MotionTests.Near(expected[i], actual[i], 2e-4);
        for (var warm = 0; warm < 3; warm++) decoder.Run(tokens, image, tokenPe, imagePe, actual, feedback);
        var allocated = GC.GetAllocatedBytesForCurrentThread(); decoder.Run(tokens, image, tokenPe, imagePe, actual, feedback);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - allocated);
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => decoder.Run(tokens, image, tokenPe, imagePe, actual, feedback, cancelled.Token));
        MotionTests.Throws<ArgumentNullException>(() => decoder.Run(tokens, image, tokenPe, imagePe, actual, null));
        MotionTests.Throws<ArgumentException>(() => decoder.Run(tokens, image, tokenPe, imagePe, tokens, feedback));
        SamPoseFeedback invalid = (_, _, current, _, _) => current[0] = float.NaN;
        MotionTests.Throws<ArgumentException>(() => decoder.Run(tokens, image, tokenPe, imagePe, actual, invalid));
    }
}
