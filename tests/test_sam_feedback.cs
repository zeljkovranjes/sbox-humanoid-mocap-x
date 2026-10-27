using HumanoidMocapX.EditorTools.MotionBricks;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;

static class SamFeedbackTests
{
    internal static void Run()
    {
        foreach (var hand in new[] { false, true }) Feedback(hand);
        Console.WriteLine("PASS: SAM projected keypoint feedback, bilinear image sampling, zero padding, visibility masks, projection bias and pelvis-relative 3D embeddings");
    }

    static void Feedback(bool hand)
    {
        var shape = new SamDecoderShape(145, 6, 8, 12, 2, 4, 16); var tensors = SamPoseFixture.Feedback(shape, hand);
        var feedback = new SamKeypointFeedback(new(tensors), shape, 3, 2, hand);
        var image = DenoiserTests.Wave(shape.ImageElements, .17f); var points = DenoiserTests.Wave(210, .07f);
        var cropped = DenoiserTests.Wave(140, .11f).Select(v => v * .45f).ToArray(); var depth = Enumerable.Repeat(1f, 70).ToArray();
        cropped[0] = cropped[1] = -.5f; cropped[2] = cropped[3] = .5f;
        cropped[4] = -.501f; cropped[6] = .501f; depth[4] = -1; depth[5] = 0; depth[6] = 1e-5f;
        var tokens = DenoiserTests.Wave(shape.TokenElements, .03f); var positions = DenoiserTests.Wave(shape.TokenElements, .05f);
        var expectedTokens = tokens.ToArray(); var expectedPositions = positions.ToArray(); var suffix = hand ? "_hand" : "";
        for (var point = 0; point < 70; point++)
        {
            var x = cropped[point * 2]; var y = cropped[point * 2 + 1];
            var valid = x >= -.5 && x <= .5 && y >= -.5 && y <= .5 && depth[point] >= 1e-5f;
            var encoded = Mlp(tensors, "keypoint_posemb_linear" + suffix, new double[] { x, y });
            for (var c = 0; c < shape.Width; c++) expectedPositions[(5 + point) * shape.Width + c] = valid ? (float)encoded[c] : 0;
            var sample = new double[shape.ContextWidth];
            if (valid)
                for (var row = 0; row < 2; row++)
                    for (var col = 0; col < 3; col++)
                    {
                        var weight = Math.Max(0, 1 - Math.Abs((x + .5) * 3 - .5 - col)) * Math.Max(0, 1 - Math.Abs((y + .5) * 2 - .5 - row));
                        for (var c = 0; c < sample.Length; c++) sample[c] += image[(row * 3 + col) * sample.Length + c] * weight;
                    }
            var projected = RefinerTests.Linear(tensors, sample, "keypoint_feat_linear" + suffix);
            for (var c = 0; c < shape.Width; c++) expectedTokens[(5 + point) * shape.Width + c] += (float)projected[c];
            var centered = new double[3];
            for (var c = 0; c < 3; c++) centered[c] = points[point * 3 + c] - ((double)points[27 + c] + points[30 + c]) / 2;
            encoded = Mlp(tensors, "keypoint3d_posemb_linear" + suffix, centered);
            for (var c = 0; c < shape.Width; c++) expectedPositions[(75 + point) * shape.Width + c] = (float)encoded[c];
        }
        feedback.Run(image, cropped, depth, points, tokens, positions);
        SamCompactPoseTests.Compare(expectedTokens, tokens, 2e-7); SamCompactPoseTests.Compare(expectedPositions, positions, 2e-7);
        for (var i = 0; i < 4; i++) feedback.Run(image, cropped, depth, points, tokens, positions);
        var before = GC.GetAllocatedBytesForCurrentThread(); feedback.Run(image, cropped, depth, points, tokens, positions);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        MotionTests.Throws<ArgumentException>(() => feedback.Run(image, cropped, depth, points, tokens, tokens));
        MotionTests.Throws<ArgumentException>(() => feedback.Run(image.AsSpan(1), cropped, depth, points, tokens, positions));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => feedback.Run(image, cropped, depth, points, tokens, positions, cancelled.Token));
        cropped[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => feedback.Run(image, cropped, depth, points, tokens, positions));
        Checkpoint(tensors, shape, hand);
    }

    static double[] Mlp(Dictionary<string, Tensor> tensors, string name, double[] input)
        => RefinerTests.Linear(tensors, RefinerTests.Linear(tensors, input, name + ".layers.0.0").Select(v => Math.Max(0, v)).ToArray(), name + ".layers.1");

    static void Checkpoint(Dictionary<string, Tensor> tensors, SamDecoderShape shape, bool hand)
    {
        var path = Path.Combine(Path.GetTempPath(), "sam-feedback-" + Guid.NewGuid().ToString("N") + ".pt");
        try
        {
            CheckpointFixture.Write(path, tensors, "state_dict", "");
            _ = new SamKeypointFeedback(VisionCheckpoint.LoadSamFeedback(path, hand), shape, 3, 2, hand);
            MotionTests.Throws<InvalidDataException>(() => VisionCheckpoint.LoadSamFeedback(path, !hand));
        }
        finally { File.Delete(path); }
    }
}
