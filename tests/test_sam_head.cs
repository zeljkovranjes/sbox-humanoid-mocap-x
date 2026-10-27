using HumanoidMocap.EditorTools.MotionBricks;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;

static class SamHeadTests
{
    internal static void Run()
    {
        foreach (var hand in new[] { false, true })
        {
            var tensors = SamPoseFixture.Head(hand); var weights = new FlowWeights(tensors);
            foreach (var camera in new[] { false, true }) Prediction(tensors, weights, hand, camera);
            Mapping(tensors, weights, hand);
        }
        Checkpoint();
        Console.WriteLine("PASS: SAM body/hand pose and camera heads, fixed initialization, MHR scales, dedicated fingers, wrist transform, non-hand mask and allocation-free reuse");
    }

    static void Prediction(Dictionary<string, Tensor> tensors, FlowWeights weights, bool hand, bool camera)
    {
        var head = new SamPredictionHead(weights, hand, camera, 8); var input = DenoiserTests.Wave(8, .17f);
        var output = new float[head.Parameters]; var initial = new float[head.Parameters]; head.CopyInitial(initial);
        var part = camera ? "camera" : "pose"; var suffix = hand ? "_hand" : "";
        var name = "head_" + part + suffix + ".proj.layers.";
        var hidden = Linear(tensors, name + "0.0", input.Select(v => (double)v).ToArray());
        var expected = Linear(tensors, name + "1", hidden.Select(v => Math.Max(0, v)).ToArray());
        head.Run(input, output);
        for (var i = 0; i < output.Length; i++) MotionTests.Near(expected[i] + initial[i], output[i], 2e-7);
        var previous = output.ToArray();
        for (var i = 0; i < 4; i++) head.Run(input, output);
        MotionTests.Check(previous.SequenceEqual(output), "Head must not accumulate previous predictions");
        var before = GC.GetAllocatedBytesForCurrentThread(); head.Run(input, output);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        MotionTests.Throws<ArgumentException>(() => head.Run(output.AsSpan(0, Math.Min(8, output.Length)), output));
        MotionTests.Throws<ArgumentException>(() => head.Run(input.AsSpan(1), output));
        MotionTests.Throws<ArgumentException>(() => head.CopyInitial(output.AsSpan(1)));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => head.Run(input, output, cancelled.Token));
        input[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => head.Run(input, output));
    }

    static double[] Linear(Dictionary<string, Tensor> tensors, string name, double[] input)
    {
        var weight = tensors[name + ".weight"]; var output = tensors[name + ".bias"].Data.Select(v => (double)v).ToArray();
        for (var r = 0; r < output.Length; r++) for (var c = 0; c < input.Length; c++) output[r] += weight.Data[r * input.Length + c] * input[c];
        return output;
    }

    static void Mapping(Dictionary<string, Tensor> tensors, FlowWeights weights, bool hand)
    {
        var mapping = new SamMhrParameters(weights, SamPoseFixture.Left, SamPoseFixture.Right, SamPoseFixture.Nonhand, hand);
        var prediction = SamPoseFixture.Prediction(); var model = new float[204];
        var expected = new double[204]; var name = hand ? "head_pose_hand." : "head_pose.";
        expected[3] = .47; expected[4] = -.31; expected[5] = .23;
        var body = DenoiserTests.Wave(133, .13f);
        for (var i = 0; i < 130; i++) expected[i + 6] = i is >= 62 and < 116 ? 0 : body[i];
        for (var i = 0; i < 68; i++)
        {
            expected[136 + i] = tensors[name + "scale_mean"].Data[i];
            for (var k = 0; k < 28; k++) expected[136 + i] += (double)prediction[311 + k] * tensors[name + "scale_comps"].Data[k * 68 + i];
        }
        for (var side = 0; side < 2; side++)
        {
            var continuous = new float[54]; var angles = new float[27];
            for (var i = 0; i < 54; i++)
            {
                double value = tensors[name + "hand_pose_mean"].Data[i];
                for (var k = 0; k < 54; k++) value += (double)prediction[339 + side * 54 + k] * tensors[name + "hand_pose_comps"].Data[k * 54 + i];
                continuous[i] = (float)value;
            }
            SamCompactPose.Hand(continuous, angles);
            var indices = side == 0 ? SamPoseFixture.Left : SamPoseFixture.Right;
            for (var i = 0; i < 27; i++) expected[indices[i]] = angles[i];
        }
        mapping.Run(prediction, model);
        if (hand)
        {
            var world = SamPoseFixture.Multiply(SamPoseFixture.Matrix(.47, -.31, .23), SamPoseFixture.Matrix(.2, -.3, .4));
            var actual = SamPoseFixture.Matrix(model[3], model[4], model[5]);
            for (var r = 0; r < 3; r++)
            {
                double position = tensors[name + "root_coords"].Data[r];
                for (var c = 0; c < 3; c++)
                {
                    MotionTests.Near(world[r, c], actual[r, c], 5e-7);
                    position += world[r, c] * (tensors[name + "right_wrist_coords"].Data[c] - tensors[name + "root_coords"].Data[c]);
                }
                expected[r] = -position * 10; expected[r + 3] = model[r + 3];
            }
            foreach (var index in SamPoseFixture.Nonhand) expected[index] = 0;
        }
        for (var i = 0; i < model.Length; i++) MotionTests.Near(expected[i], model[i], 2e-6);
        for (var i = 0; i < 4; i++) mapping.Run(prediction, model);
        var before = GC.GetAllocatedBytesForCurrentThread(); mapping.Run(prediction, model);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        MotionTests.Throws<ArgumentException>(() => mapping.Run(prediction, prediction.AsSpan(0, 204)));
        MotionTests.Throws<ArgumentException>(() => mapping.Run(prediction.AsSpan(1), model));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => mapping.Run(prediction, model, cancelled.Token));
        prediction[311] = float.NaN; MotionTests.Throws<ArgumentException>(() => mapping.Run(prediction, model));
        MotionTests.Throws<InvalidDataException>(() => new SamMhrParameters(weights, SamPoseFixture.Left, SamPoseFixture.Left, SamPoseFixture.Nonhand, hand));
        var invalid = SamPoseFixture.Nonhand.ToArray(); invalid[0] = 204;
        MotionTests.Throws<InvalidDataException>(() => new SamMhrParameters(weights, SamPoseFixture.Left, SamPoseFixture.Right, invalid, hand));
    }

    static void Checkpoint()
    {
        var path = Path.Combine(Path.GetTempPath(), "sam-head-" + Guid.NewGuid().ToString("N") + ".pt");
        try
        {
            var tensors = SamPoseFixture.Head(false);
            foreach (var pair in SamPoseFixture.Head(true)) tensors.Add(pair.Key, pair.Value);
            CheckpointFixture.Write(path, tensors, "state_dict", "");
            foreach (var hand in new[] { false, true })
            {
                var weights = VisionCheckpoint.LoadSamHead(path, hand);
                _ = new SamPredictionHead(weights, hand, false, 8); _ = new SamPredictionHead(weights, hand, true, 8);
                MotionTests.Throws<InvalidDataException>(() => new SamPredictionHead(weights, !hand, false, 8));
                MotionTests.Throws<InvalidDataException>(() => SamCheckpoint.LoadParameters(path, weights, hand));
            }
        }
        finally { File.Delete(path); }
    }

    internal static void Released(string samPath, string mhrPath)
    {
        var body = MhrCheckpoint.LoadBody(mhrPath); var input = DenoiserTests.Wave(1024, .017f);
        foreach (var hand in new[] { false, true })
        {
            var head = SamCheckpoint.LoadPoseHead(samPath, body, hand); head.Run(input);
            var before = head.Keypoints.ToArray(); var vertices = head.Vertices.ToArray(); var model = head.Model.ToArray();
            MotionTests.Check(before.All(float.IsFinite) && vertices.All(float.IsFinite), "Released SAM pose head output must be finite");
            head.Run(input);
            MotionTests.Check(before.AsSpan().SequenceEqual(head.Keypoints) && vertices.AsSpan().SequenceEqual(head.Vertices), "Released SAM head must repeat exactly");
            if (hand) for (var i = 0; i < 70; i++) if (i < 21 || i >= 42) MotionTests.Check(head.Keypoints.Slice(i * 3, 3).SequenceEqual(new float[3]), "Hand head keypoint mask");
            var varied = input.Select(v => v * -.7f).ToArray(); head.Run(varied);
            MotionTests.Check(!model.AsSpan().SequenceEqual(head.Model) && !before.AsSpan().SequenceEqual(head.Keypoints), "Pose head must respond to its token");
            Console.WriteLine($"PASS: released SAM {(hand ? "hand" : "body")} head + complete MHR, {head.Vertices.Length / 3} vertices, {head.Keypoints.Length / 3} keypoints, finite and exactly repeatable");
            Decoder(samPath, head, hand);
        }
    }

    static void Decoder(string path, SamPoseHead pose, bool hand)
    {
        var shape = new SamDecoderShape(145, 6); var imageShape = DinoShape.Sam(32, 48);
        var camera = new SamPredictionHead(VisionCheckpoint.LoadSamHead(path, hand), hand, true);
        var update = new SamKeypointFeedback(VisionCheckpoint.LoadSamFeedback(path, hand), shape, 3, 2, hand);
        var feedback = new SamDecoderFeedback(shape, pose, camera, update);
        var decoder = new SamDecoder(VisionCheckpoint.LoadSamDecoder(path, hand), shape, hand);
        var image = DenoiserTests.Wave(shape.ImageElements, .017f); var tokens = new float[shape.TokenElements];
        var positions = new float[tokens.Length]; var imagePositions = new float[image.Length]; var output = new float[tokens.Length];
        var projection = new SamProjection(new float[] { 700, 0, 320, 0, 700, 240, 0, 0, 1 }, new float[] { .1f, 0, -8, 0, .1f, -8 }, 320, 240, 480, imageShape, hand);
        new SamInitialTokens(VisionCheckpoint.LoadSamInitialTokens(path, hand), hand).Run(projection, tokens, positions);
        MotionTests.Throws<ArgumentException>(() => feedback.Finish(output));
        feedback.Prepare(projection, image);
        var calls = 0;
        SamPoseFeedback callback = (layer, normalized, current, tokenPositions, cancellation) =>
        { calls++; feedback.Update(layer, normalized, current, tokenPositions, cancellation); };
        decoder.Run(tokens, image, positions, imagePositions, output, callback); feedback.Finish(output);
        MotionTests.Equal(5, calls);
        var expected = output.ToArray(); var points = pose.Keypoints.ToArray();
        MotionTests.Check(points.All(float.IsFinite), "Released decoder pose feedback must stay finite");
        decoder.Run(tokens, image, positions, imagePositions, output, callback); feedback.Finish(output);
        MotionTests.Equal(10, calls);
        MotionTests.Check(expected.SequenceEqual(output) && points.AsSpan().SequenceEqual(pose.Keypoints), "Released decoder with real pose feedback must repeat exactly");
        Console.WriteLine($"PASS: released SAM {(hand ? "hand" : "body")} six-layer decoder, learned initial tokens, five real MHR/keypoint feedback updates, final pose and exact repeat with synthetic image tokens");
    }
}
