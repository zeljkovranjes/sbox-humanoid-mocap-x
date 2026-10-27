using HumanoidMocap.EditorTools.MotionBricks;
using HumanoidMocapX;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;

static class OmegaVisionTests
{
    internal static void Run()
    {
        Dimensions(); Images(); ImageEdges(); Network(); Checkpoint();
        Console.WriteLine("PASS: complete Omega image encoder and alternating aggregator, first-frame tokens, camera output, streamed-model dimensions and image preparation");
    }

    static void Dimensions()
    {
        var shape = new OmegaShape(25, 288, 512);
        MotionTests.Equal(576, shape.Patches); MotionTests.Equal(581, shape.ImageTokens); MotionTests.Equal(593, shape.FrameTokens);
        MotionTests.Equal(25 * 593, shape.Tokens);
        MotionTests.Throws<ArgumentException>(() => new OmegaShape(0, 288, 512));
        MotionTests.Throws<ArgumentException>(() => new OmegaShape(1, 287, 512));
        MotionTests.Throws<ArgumentException>(() => new OmegaShape(1, 288, 513));
        MotionTests.Throws<ArgumentException>(() => new OmegaShape(252, 512, 512));
        MotionTests.Throws<ArgumentException>(() => new OmegaShape(1, 288, 512, 1000, 16));
        MotionTests.Equal(5, Enumerable.Range(0, 24).Count(OmegaShape.RegistersOnly));
    }

    static void Images()
    {
        foreach (var (width, height, targetWidth, targetHeight) in new[] { (1920, 1080, 512, 288), (100, 800, 256, 512), (800, 100, 512, 256), (400, 775, 272, 512) })
        {
            var image = new OmegaImage(width, height); MotionTests.Equal(targetWidth, image.Width); MotionTests.Equal(targetHeight, image.Height);
            var rgba = new byte[width * height * 4]; var output = new float[image.Width * image.Height * 3];
            for (var index = 0; index < rgba.Length; index += 4) { rgba[index] = 32; rgba[index + 1] = 64; rgba[index + 2] = 128; rgba[index + 3] = 255; }
            image.Prepare(rgba, output);
            for (var index = 0; index < output.Length; index++) MotionTests.Equal((index % 3 == 0 ? 32 : index % 3 == 1 ? 64 : 128) / 255f, output[index]);
            Array.Clear(rgba); image.Prepare(rgba, output);
            MotionTests.Check(output.AsSpan().IndexOfAnyExcept(1f) < 0, "Omega did not composite transparency onto white");
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            MotionTests.Throws<OperationCanceledException>(() => image.Prepare(rgba, output, cancelled.Token));
            MotionTests.Throws<ArgumentException>(() => image.Prepare(new byte[4], output));
        }
        var square = new OmegaImage(512, 512); var pixels = new byte[512 * 512 * 4]; var rgb = new float[512 * 512 * 3];
        for (var pixel = 0; pixel < 512 * 512; pixel++)
        { pixels[pixel * 4] = (byte)(pixel % 251); pixels[pixel * 4 + 1] = 99; pixels[pixel * 4 + 2] = 101; pixels[pixel * 4 + 3] = 255; }
        square.Prepare(pixels, rgb);
        for (var i = 0; i < rgb.Length; i++) MotionTests.Equal(pixels[i / 3 * 4 + i % 3] / 255f, rgb[i]);
        for (var warm = 0; warm < 3; warm++) square.Prepare(pixels, rgb);
        var allocated = GC.GetAllocatedBytesForCurrentThread(); square.Prepare(pixels, rgb);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - allocated);
    }

    static void Network()
    {
        var shape = new OmegaShape(2, 16, 32, 8, 2); var tensors = OmegaFixture.Full(shape.Channels, shape.Heads); var weights = new FlowWeights(tensors);
        var encoder = new DinoImageEncoder(weights, shape); var aggregator = new OmegaAggregator(weights, shape);
        var head = new OmegaCameraHead(weights, shape.Frames, shape.Channels * 2, shape.Heads);
        var images = Enumerable.Range(0, shape.Frames).Select(frame => Image(shape.ImageElements, frame)).ToArray();
        var patches = new float[shape.Frames * shape.Patches * shape.Channels]; var expectedPatches = new double[shape.Frames][];
        for (var frame = 0; frame < shape.Frames; frame++)
        {
            encoder.Run(images[frame], patches.AsSpan(frame * shape.Patches * shape.Channels, shape.Patches * shape.Channels));
            expectedPatches[frame] = OmegaVisionReference.Image(tensors, images[frame], shape.Height, shape.Width, shape.Channels, shape.Heads);
            for (var i = 0; i < expectedPatches[frame].Length; i++) MotionTests.Near(expectedPatches[frame][i], patches[frame * expectedPatches[frame].Length + i], 2e-4);
        }
        var combined = new float[shape.CameraFeatures]; aggregator.Run(patches, combined);
        var expected = OmegaVisionReference.Aggregate(tensors, expectedPatches, shape.Height, shape.Width, shape.Channels, shape.Heads);
        for (var i = 0; i < expected.Length; i++) MotionTests.Near(expected[i], combined[i], 2e-4);
        var poses = new float[shape.Frames * 9]; head.Run(combined, poses);
        var expectedPoses = OmegaReference.Camera(tensors, expected.Select(v => (float)v).ToArray(), shape.Frames, shape.Channels * 2, shape.Heads);
        for (var i = 0; i < poses.Length; i++) MotionTests.Near(expectedPoses[i], poses[i], 2e-4);
        var rotations = new float[shape.Frames * 9]; var intrinsics = new float[rotations.Length];
        FlowCamera.Decode(poses, shape.Frames, 1, 1920, 1080, rotations, intrinsics);
        MotionTests.Near(1, rotations[0], 1e-5); MotionTests.Near(1, rotations[4], 1e-5); MotionTests.Near(1, rotations[8], 1e-5);
        for (var warm = 0; warm < 3; warm++) aggregator.Run(patches, combined);
        var allocated = GC.GetAllocatedBytesForCurrentThread(); aggregator.Run(patches, combined);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - allocated);
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => aggregator.Run(patches, combined, cancelled.Token));
        MotionTests.Throws<OperationCanceledException>(() => encoder.Run(images[0], new float[shape.Patches * shape.Channels], cancelled.Token));
        MotionTests.Throws<ArgumentException>(() => aggregator.Run(patches, new float[1]));
        images[0][0] = float.NaN;
        MotionTests.Throws<ArgumentException>(() => encoder.Run(images[0], new float[shape.Patches * shape.Channels]));
    }

    static void ImageEdges()
    {
        var image = new OmegaImage(2, 1); var output = new float[image.Width * image.Height * 3];
        image.Prepare(new byte[] { 0, 0, 0, 255, 255, 255, 255, 255 }, output);
        // Two samples reduce the normalized cubic to two weights; linear resizing gives 64 and 191 here.
        foreach (var (x, value) in new[] { (0, 0), (192, 53), (255, 127), (256, 128), (319, 202), (511, 255) })
            MotionTests.Equal(value / 255f, output[x * 3]);

        var cropped = new OmegaImage(1025, 256); var rgba = new byte[1025 * 256 * 4];
        var result = new float[cropped.Width * cropped.Height * 3];
        for (var y = 0; y < 256; y++)
            for (var x = 0; x < 1025; x++)
            {
                var at = (y * 1025 + x) * 4;
                rgba[at] = x >= 256 && x < 768 ? (byte)((x - 256) % 251) : (byte)255;
                rgba[at + 1] = 10; rgba[at + 2] = 200; rgba[at + 3] = 255;
            }
        cropped.Prepare(rgba, result);
        for (var x = 0; x < 512; x++) MotionTests.Equal((x % 251) / 255f, result[x * 3]);
        image = new OmegaImage(1, 1); output = new float[512 * 512 * 3];
        image.Prepare(new byte[] { 0, 127, 255, 128 }, output);
        MotionTests.Equal(127 / 255f, output[0]); MotionTests.Equal(191 / 255f, output[1]); MotionTests.Equal(1f, output[2]);
    }

    static void Checkpoint()
    {
        var path = Path.Combine(Path.GetTempPath(), "omega-full-" + Guid.NewGuid().ToString("N") + ".pt");
        try
        {
            var tensors = OmegaFixture.Full(8, 2);
            tensors.Add("dense_head.unused", new("dense_head.unused", new[] { float.NaN }, new[] { 1, 1, 1, 1, 1 }));
            CheckpointFixture.Write(path, tensors, null, ""); var loaded = VisionCheckpoint.LoadOmega(path);
            foreach (var pair in tensors.Where(pair => !pair.Key.StartsWith("dense_head.")))
                MotionTests.Check(loaded.Get(pair.Key, pair.Value.Shape).Data.AsSpan().SequenceEqual(pair.Value.Data), "Omega model checkpoint changed tensor shape or values");
            _ = new DinoImageEncoder(loaded, new OmegaShape(2, 16, 32, 8, 2)); _ = new OmegaAggregator(loaded, new(2, 16, 32, 8, 2));
            // Part-by-part loading (OmegaGpuModel.LoadAsync's prefixes) covers every tensor the full
            // load does, each part holds only its own tensors (others are absent), and an empty part
            // fails loudly.
            var parts = new[] { new[] { "aggregator.patch_embed.", "aggregator.camera_token", "aggregator.register_token", "aggregator.rope_embed." },
                new[] { "aggregator.frame_blocks." }, new[] { "aggregator.inter_frame_blocks." }, new[] { "camera_head." } };
            var covered = 0;
            foreach (var prefixes in parts)
            {
                var part = VisionCheckpoint.LoadOmegaPart(path, default, prefixes);
                foreach (var pair in tensors.Where(pair => !pair.Key.StartsWith("dense_head.")))
                {
                    if (!prefixes.Any(prefix => pair.Key.StartsWith(prefix)))
                    {
                        MotionTests.Throws<InvalidDataException>(() => part.Get(pair.Key, pair.Value.Shape));
                        continue;
                    }
                    MotionTests.Check(part.Get(pair.Key, pair.Value.Shape).Data.AsSpan().SequenceEqual(pair.Value.Data), "Omega part changed tensor values");
                    covered++;
                }
            }
            MotionTests.Equal(tensors.Count(pair => !pair.Key.StartsWith("dense_head.")), covered);
            MotionTests.Throws<InvalidDataException>(() => VisionCheckpoint.LoadOmegaPart(path, default, "no_such_part."));
        }
        finally { File.Delete(path); }
    }

    internal static float[] Image(int elements, int frame) => Enumerable.Range(0, elements).Select(i => .5f + .4f * MathF.Sin(i * .013f + frame)).ToArray();
}
