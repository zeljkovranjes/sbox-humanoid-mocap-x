using HumanoidMocapX.Inference;
using HumanoidMocapX.EditorTools;
using HumanoidMocap.EditorTools.Inference;
using HumanoidMocap.EditorTools.MotionBricks;

static class YoloTests
{
    internal static void Run()
    {
        Architecture(); Normalization(); Checkpoints(); Images(); Boxes();
        Console.WriteLine("PASS: YOLOX-L graph, live tensor storage, fused batch normalization, BGR images and class-agnostic person suppression");
    }

    static void Checkpoints()
    {
        var path = Path.Combine(Path.GetTempPath(), "yolo-" + Guid.NewGuid().ToString("N") + ".pth");
        try
        {
            var nodes = new YoloNetwork().Nodes;
            var layer = nodes.First(n => n.Weight == "backbone.backbone.stem.conv");
            var raw = nodes.First(n => n.Weight == "head.reg_preds.0");
            var tensors = new Dictionary<string, Tensor>();
            Add(layer.Weight + ".conv.weight", Enumerable.Range(0, 64 * 12 * 3 * 3).Select(v => v * .001f).ToArray(), 3, 3, 12, 64);
            foreach (var suffix in new[] { "weight", "bias", "running_mean", "running_var" })
                Add(layer.Weight + ".bn." + suffix, Enumerable.Repeat(suffix == "running_var" ? 4f : 1f, 64).ToArray(), 64);
            Add(raw.Weight + ".weight", Enumerable.Range(0, 256 * 4).Select(v => v * .01f).ToArray(), 1, 1, 256, 4);
            Add(raw.Weight + ".bias", new float[] { 1, 2, 3, 4 }, 4);
            CheckpointFixture.Write(path, tensors, "model", "");
            using (var checkpoint = new TorchCheckpoint(path, "model"))
            {
                var loaded = YoloCheckpoint.LoadFilter(checkpoint, layer, 12, default);
                MotionTests.Near(.251 / Math.Sqrt(4.001), loaded.Weight[251], 1e-6);
                MotionTests.Near(1 - 1 / Math.Sqrt(4.001), loaded.Bias[3], 1e-6);
                loaded = YoloCheckpoint.LoadFilter(checkpoint, raw, 256, default);
                MotionTests.Check(loaded.Weight.AsSpan().SequenceEqual(tensors[raw.Weight + ".weight"].Data), "YOLOX changed OIHW weights");
                MotionTests.Check(loaded.Bias.AsSpan().SequenceEqual(new float[] { 1, 2, 3, 4 }), "YOLOX lost prediction bias");
                MotionTests.Throws<InvalidDataException>(() => YoloCheckpoint.LoadFilter(checkpoint, layer, 3, default));
                using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
                MotionTests.Throws<OperationCanceledException>(() => YoloCheckpoint.LoadFilter(checkpoint, layer, 12, cancelled.Token));
            }
            MotionTests.Throws<InvalidDataException>(() => YoloCheckpoint.Load(path, new YoloNetwork()));
            File.Delete(path); tensors[layer.Weight + ".bn.running_var"].Data[0] = -1;
            CheckpointFixture.Write(path, tensors, "model", "");
            using var invalid = new TorchCheckpoint(path, "model");
            MotionTests.Throws<InvalidDataException>(() => YoloCheckpoint.LoadFilter(invalid, layer, 12, default));

            void Add(string name, float[] values, params int[] shape) => tensors.Add(name, new Tensor(name, values, shape));
        }
        finally { File.Delete(path); }
    }

    static void Architecture()
    {
        var network = new YoloNetwork(); var nodes = network.Nodes;
        MotionTests.Equal(640 * 640 * 3, nodes[0].Elements);
        var expectedSizes = new[] { 80, 40, 20 };
        for (var index = 0; index < 3; index++)
        {
            var head = nodes[network.Heads[index]];
            MotionTests.Equal(expectedSizes[index], head.Size); MotionTests.Equal(85, head.Channels);
        }
        var names = new HashSet<string>(); long parameters = 0;
        var content = Enumerable.Repeat(-1, network.Capacities.Length).ToArray();
        for (var index = 0; index < nodes.Length; index++)
        {
            var node = nodes[index];
            if (node.First >= 0) MotionTests.Equal(node.First, content[network.Slots[node.First]]);
            if (node.Second >= 0) MotionTests.Equal(node.Second, content[network.Slots[node.Second]]);
            MotionTests.Check(network.Capacities[network.Slots[index]] >= node.Elements, "YOLOX storage is undersized");
            if (node.First >= 0) MotionTests.Check(network.Slots[index] != network.Slots[node.First], "YOLOX output aliases its input");
            if (node.Second >= 0) MotionTests.Check(network.Slots[index] != network.Slots[node.Second], "YOLOX output aliases its second input");
            content[network.Slots[index]] = index;
            if (node.Operation != YoloOperation.Convolution) continue;
            MotionTests.Check(names.Add(node.Weight), "YOLOX reused a learned convolution");
            parameters += (long)node.Channels * nodes[node.First].Channels * node.Kernel * node.Kernel
                + node.Channels * (node.Normalize ? 2 : 1);
        }
        foreach (var head in network.Heads) MotionTests.Equal(head, content[network.Slots[head]]);
        foreach (var prefix in new[] { "head.cls_preds.", "head.obj_preds.", "head.reg_preds." })
            for (var index = 0; index < 3; index++) MotionTests.Check(names.Contains(prefix + index), "Missing YOLOX output head");
        foreach (var stage in new[] { 3, 4 })
            MotionTests.Check(names.Contains($"backbone.backbone.dark{stage}.1.m.8.conv2"), "YOLOX-L must have nine bottlenecks in each middle stage");
        MotionTests.Check(network.Capacities.Sum(v => (long)v) < nodes.Sum(v => (long)v.Elements) / 4, "YOLOX retains dead image tensors");
        Console.WriteLine($"YOLOX-L: {parameters:N0} trainable parameters, {nodes.Length} operations, {network.Capacities.Sum(v => (long)v) * 4:N0} scratch bytes");
    }

    static void Normalization()
    {
        float[] weight = { .2f, -.3f, 2, 4, -1, .4f }, scale = { .4f, -1f }, bias = { -.7f, .2f }, mean = { 1, -2 }, variance = { .3f, 0 };
        var original = (float[])weight.Clone(); var fusedBias = YoloWeights.Fold(weight, scale, bias, mean, variance);
        double[] input = { -.3, .8, 2.1 };
        for (var channel = 0; channel < 2; channel++)
        {
            double originalValue = 0, fusedValue = fusedBias[channel];
            for (var index = 0; index < 3; index++)
            { originalValue += original[channel * 3 + index] * input[index]; fusedValue += weight[channel * 3 + index] * input[index]; }
            var expected = (originalValue - mean[channel]) * scale[channel] / Math.Sqrt(variance[channel] + .001) + bias[channel];
            MotionTests.Near(expected, fusedValue, 1e-5);
        }
        MotionTests.Throws<InvalidDataException>(() => YoloWeights.Fold(new float[3], scale, bias, mean, variance));
        variance[1] = -1; MotionTests.Throws<InvalidDataException>(() => YoloWeights.Fold(original, scale, bias, mean, variance));
        variance[1] = 1; scale[0] = float.NaN;
        MotionTests.Throws<InvalidDataException>(() => YoloWeights.Fold(original, scale, bias, mean, variance));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => YoloWeights.Fold(original, scale, bias, mean, variance, cancelled.Token));
    }

    static void Images()
    {
        var source = new byte[640 * 320 * 4]; var output = new float[640 * 640 * 3];
        for (var pixel = 0; pixel < source.Length / 4; pixel++)
        { source[pixel * 4] = (byte)(pixel % 251); source[pixel * 4 + 1] = 100; source[pixel * 4 + 2] = 200; }
        YoloImage.Prepare(source, 640, 320, output);
        for (var pixel = 0; pixel < 640 * 320; pixel++)
        { MotionTests.Equal(200f, output[pixel * 3]); MotionTests.Equal(100f, output[pixel * 3 + 1]); MotionTests.Equal((float)(pixel % 251), output[pixel * 3 + 2]); }
        MotionTests.Check(output.AsSpan(640 * 320 * 3).IndexOfAnyExcept(114f) < 0, "YOLOX letterbox padding moved or changed");
        byte[] small = { 0, 0, 0, 255, 255, 255, 255, 0 };
        YoloImage.Prepare(small, 2, 1, output);
        MotionTests.Equal(0f, output[0]); MotionTests.Equal(127f, output[319 * 3]);
        MotionTests.Equal(128f, output[320 * 3]); MotionTests.Equal(255f, output[639 * 3]);
        for (var warm = 0; warm < 3; warm++) YoloImage.Prepare(source, 640, 320, output);
        var allocated = GC.GetAllocatedBytesForCurrentThread(); YoloImage.Prepare(source, 640, 320, output);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - allocated);
        MotionTests.Throws<ArgumentException>(() => YoloImage.Prepare(source, 0, 320, output));
        MotionTests.Throws<ArgumentException>(() => YoloImage.Prepare(source, 320, 320, output));
        MotionTests.Throws<ArgumentException>(() => YoloImage.Prepare(source, 640, 320, new float[1]));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => YoloImage.Prepare(source, 640, 320, output, cancelled.Token));
    }

    static void Boxes()
    {
        var decoder = new YoloDetection(); var logits = new float[8400 * 85]; var boxes = new YoloBox[8400];
        Array.Fill(logits, -20);
        MotionTests.Equal(0, decoder.Decode(logits, 1280, 720, boxes));
        Set(10 * 80 + 10, 0, 5); Set(10 * 80 + 11, 0, 4);
        MotionTests.Equal(1, decoder.Decode(logits, 1280, 720, boxes));
        MotionTests.Near(96, boxes[0].Left, 1e-4); MotionTests.Near(224, boxes[0].Right, 1e-4);
        MotionTests.Near(96, boxes[0].Top, 1e-4); MotionTests.Near(224, boxes[0].Bottom, 1e-4);
        Set(10 * 80 + 11, 1, 6);
        MotionTests.Equal(0, decoder.Decode(logits, 1280, 720, boxes));
        Array.Fill(logits, -20); Set(6400 + 3 * 40 + 4, 0, 5);
        MotionTests.Equal(1, decoder.Decode(logits, 640, 640, boxes));
        MotionTests.Near(0, boxes[0].Left, 1e-4); MotionTests.Near(-16, boxes[0].Top, 1e-4);
        Array.Fill(logits, -20); Set(8000 + 10 * 20 + 11, 0, 5);
        MotionTests.Equal(1, decoder.Decode(logits, 640, 640, boxes));
        MotionTests.Near(224, boxes[0].Left, 1e-4); MotionTests.Near(192, boxes[0].Top, 1e-4);
        for (var warm = 0; warm < 3; warm++) decoder.Decode(logits, 640, 640, boxes);
        var allocated = GC.GetAllocatedBytesForCurrentThread(); decoder.Decode(logits, 640, 640, boxes);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - allocated);
        logits[0] = float.NaN; MotionTests.Throws<ArithmeticException>(() => decoder.Decode(logits, 640, 640, boxes));
        logits[0] = 0;
        MotionTests.Throws<ArgumentException>(() => decoder.Decode(logits, -1, 640, boxes));
        MotionTests.Throws<ArgumentException>(() => decoder.Decode(logits, 640, 640, new YoloBox[1]));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => decoder.Decode(logits, 640, 640, boxes, cancelled.Token));

        void Set(int anchor, int category, float score)
        {
            var row = logits.AsSpan(anchor * 85, 85); row.Fill(-20);
            row[0] = 0; row[1] = 0; row[2] = MathF.Log(8); row[3] = MathF.Log(8);
            row[4] = score; row[5 + category] = score;
        }
    }
}
