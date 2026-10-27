using HumanoidMocapX.Inference;
using HumanoidMocapX.EditorTools;

static class SamInitialTests
{
    internal static void Run()
    {
        foreach (var hand in new[] { false, true }) Initial(hand);
        Console.WriteLine("PASS: SAM learned initial pose/camera tokens, CLIFF crop conditioning, invalid point prompt, hand/keypoint embeddings and allocation-free reuse");
    }

    static void Initial(bool hand)
    {
        const int width = 8, contextWidth = 12;
        var tensors = new Dictionary<string, Tensor>(); var suffix = hand ? "_hand" : "";
        OmegaFixture.Add(tensors, "init_pose" + suffix + ".weight", 519, 1);
        OmegaFixture.Add(tensors, "init_camera" + suffix + ".weight", 3, 1);
        OmegaFixture.Linear(tensors, "init_to_token_mhr" + suffix, 525, width);
        OmegaFixture.Linear(tensors, "prev_to_token_mhr" + suffix, 522, width);
        OmegaFixture.Linear(tensors, "prompt_to_token", contextWidth, width);
        OmegaFixture.Add(tensors, "prompt_encoder.invalid_point_embed.weight", contextWidth, 1);
        OmegaFixture.Add(tensors, "hand_box_embedding.weight", width, 2);
        OmegaFixture.Add(tensors, "keypoint_embedding" + suffix + ".weight", width, 70);
        OmegaFixture.Add(tensors, "keypoint3d_embedding" + suffix + ".weight", width, 70);
        var initial = new SamInitialTokens(new(tensors), hand, width, contextWidth);
        var tokens = new float[145 * width]; var positions = new float[tokens.Length];
        var projection = new SamProjection(new float[] { 700, 0, 320, 0, 530, 240, 0, 0, 1 },
            new float[] { .1f, 0, -8, 0, .1f, -8 }, 410, 285, 480, DinoShape.Sam(32, 48), hand);
        var estimate = tensors["init_pose" + suffix + ".weight"].Data.Concat(tensors["init_camera" + suffix + ".weight"].Data).Select(v => (double)v).ToArray();
        var input = new[] { 90.0 / 700, 45.0 / 700, 480.0 / 700 }.Concat(estimate).ToArray();
        var expected = RefinerTests.Linear(tensors, input, "init_to_token_mhr" + suffix);
        var previous = RefinerTests.Linear(tensors, estimate, "prev_to_token_mhr" + suffix);
        var prompt = RefinerTests.Linear(tensors, tensors["prompt_encoder.invalid_point_embed.weight"].Data.Select(v => (double)v).ToArray(), "prompt_to_token");
        Array.Fill(positions, float.NaN); initial.Run(projection, tokens, positions);
        for (var i = 0; i < width; i++)
        {
            MotionTests.Near(expected[i], tokens[i], 2e-6); MotionTests.Near(previous[i], tokens[width + i], 2e-6);
            MotionTests.Near(prompt[i], tokens[width * 2 + i], 2e-7);
            MotionTests.Equal(tokens[width + i], positions[width + i]); MotionTests.Equal(tokens[width * 2 + i], positions[width * 2 + i]);
        }
        MotionTests.Check(positions.AsSpan(0, width).SequenceEqual(new float[width]) && positions.AsSpan(width * 3).ToArray().All(v => v == 0), "Only previous/prompt positions are initialized");
        MotionTests.Check(tokens.AsSpan(width * 3, width * 2).SequenceEqual(tensors["hand_box_embedding.weight"].Data), "Hand box tokens");
        MotionTests.Check(tokens.AsSpan(width * 5, width * 70).SequenceEqual(tensors["keypoint_embedding" + suffix + ".weight"].Data), "2D keypoint tokens");
        MotionTests.Check(tokens.AsSpan(width * 75).SequenceEqual(tensors["keypoint3d_embedding" + suffix + ".weight"].Data), "3D keypoint tokens");
        for (var i = 0; i < 4; i++) initial.Run(projection, tokens, positions);
        var before = GC.GetAllocatedBytesForCurrentThread(); initial.Run(projection, tokens, positions);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        MotionTests.Throws<ArgumentException>(() => initial.Run(projection, tokens, tokens));
        MotionTests.Throws<ArgumentException>(() => initial.Run(default, tokens, positions));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => initial.Run(projection, tokens, positions, cancelled.Token));
        var path = Path.Combine(Path.GetTempPath(), "sam-initial-" + Guid.NewGuid().ToString("N") + ".pt");
        try
        {
            CheckpointFixture.Write(path, tensors, "state_dict", "");
            _ = new SamInitialTokens(VisionCheckpoint.LoadSamInitialTokens(path, hand), hand, width, contextWidth);
            MotionTests.Throws<InvalidDataException>(() => new SamInitialTokens(VisionCheckpoint.LoadSamInitialTokens(path, !hand), !hand, width, contextWidth));
        }
        finally { File.Delete(path); }
    }
}
