using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;

static class SamBackboneTests
{
    internal static void Run()
    {
        Dimensions(); Encoder(); Checkpoint();
        Console.WriteLine("PASS: 32-layer SAM DINOv3 encoder, SwiGLU gates, per-axis positions, selective checkpoint loading and allocation-free reuse");
    }

    static void Dimensions()
    {
        var shape = DinoShape.Sam();
        MotionTests.Equal(512, shape.Width); MotionTests.Equal(512, shape.Height);
        MotionTests.Equal(1280, shape.Channels); MotionTests.Equal(20, shape.Heads); MotionTests.Equal(32, shape.Depth);
        MotionTests.Equal(1024, shape.Patches); MotionTests.Equal(1029, shape.Tokens);
        MotionTests.Throws<ArgumentException>(() => DinoShape.Sam(511, 512));
        MotionTests.Throws<ArgumentException>(() => DinoShape.Sam(channels: 1281));
        MotionTests.Throws<ArgumentException>(() => DinoShape.Sam(heads: 3));
        var positions = OmegaPosition.Factors(2, 3, 5, new float[] { 1 }, true);
        // Row zero's Y coordinate is -1/2; column zero's X coordinate is -2/3.
        MotionTests.Near(-1, positions[5 * 4], 1e-6);
        MotionTests.Near(Math.Cos(-4 * Math.PI / 3), positions[5 * 4 + 2], 1e-6);
    }

    static void Encoder()
    {
        var shape = DinoShape.Sam(32, 48, 8, 2); var tensors = OmegaFixture.Sam(shape.Channels, shape.Heads);
        var encoder = new DinoImageEncoder(new(tensors), shape); var output = new float[shape.Patches * shape.Channels];
        var image = OmegaVisionTests.Image(shape.ImageElements, 0);
        for (var pass = 0; pass < 2; pass++)
        {
            if (pass > 0) Array.Reverse(image);
            encoder.Run(image, output);
            var expected = OmegaVisionReference.Image(tensors, image, shape.Height, shape.Width, shape.Channels, shape.Heads, true);
            for (var i = 0; i < output.Length; i++) MotionTests.Near(expected[i], output[i], 2e-4);
        }
        for (var warm = 0; warm < 3; warm++) encoder.Run(image, output);
        var before = GC.GetAllocatedBytesForCurrentThread(); encoder.Run(image, output);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => encoder.Run(image, output, cancelled.Token));
        MotionTests.Throws<ArgumentException>(() => encoder.Run(image, image.AsSpan(0, output.Length)));
        image[0] = 1.1f; MotionTests.Throws<ArgumentException>(() => encoder.Run(image, output));
        image[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => encoder.Run(image, output));
        var block = new OmegaBlock(new(tensors), "backbone.encoder.blocks.0", 8, 2, false, true);
        var values = new float[shape.Tokens * shape.Channels];
        MotionTests.Throws<ArgumentException>(() => block.Run(values, new(shape.Tokens, 8), new(shape.Tokens, 8, 2, false), default));
    }

    static void Checkpoint()
    {
        var path = Path.Combine(Path.GetTempPath(), "sam-backbone-" + Guid.NewGuid().ToString("N") + ".pt");
        try
        {
            var tensors = OmegaFixture.Sam(8, 2);
            tensors.Add("decoder.unused", new("decoder.unused", new[] { float.NaN }, new[] { 1, 1, 1, 1, 1 }));
            foreach (var component in new string[] { null, "state_dict" })
            {
                CheckpointFixture.Write(path, tensors, component, ""); var loaded = VisionCheckpoint.LoadSamBackbone(path);
                foreach (var pair in tensors.Where(pair => pair.Key.StartsWith("backbone.encoder.")))
                    MotionTests.Check(loaded.Get(pair.Key, pair.Value.Shape).Data.AsSpan().SequenceEqual(pair.Value.Data), "SAM backbone checkpoint changed tensor layout");
                _ = new DinoImageEncoder(loaded, DinoShape.Sam(16, 32, 8, 2));
                using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
                MotionTests.Throws<OperationCanceledException>(() => VisionCheckpoint.LoadSamBackbone(path, cancelled.Token));
                File.Delete(path);
            }
            tensors["backbone.encoder.blocks.31.mlp.w3.bias"].Data[0] = float.NaN;
            CheckpointFixture.Write(path, tensors, "state_dict", "");
            MotionTests.Throws<InvalidDataException>(() => VisionCheckpoint.LoadSamBackbone(path));
        }
        finally { File.Delete(path); }
    }
}
