using HumanoidMocapX.Inference;
using HumanoidMocapX.EditorTools;

static class SamConditioningTests
{
    internal static void Run()
    {
        foreach (var hand in new[] { false, true }) Conditioning(hand);
        Rays(); Checkpoint();
        Console.WriteLine("PASS: SAM crop-camera rays, edge antialiasing, body/hand image conditioning, positional context and allocation-free reuse");
    }

    static void Conditioning(bool hand)
    {
        var shape = DinoShape.Sam(32, 48, 8, 2); var tensors = SamFixture.ImageConditioning(shape.Channels, hand);
        var model = new SamImageConditioning(new(tensors), shape, hand); var rays = new SamRays(shape);
        var input = DenoiserTests.Wave(shape.Patches * shape.Channels, .07f); var camera = new float[rays.Elements];
        var intrinsics = new float[] { 700, 0, 320, 0, 530, 240, 0, 0, 1 }; var affine = new float[] { .5f, 0, 120, 0, .7f, -13 };
        rays.Prepare(intrinsics, affine, camera); var output = new float[input.Length]; var positions = new float[input.Length];
        var expectedRays = RayReference(shape, intrinsics, affine);
        for (var i = 0; i < camera.Length; i++) MotionTests.Near(expectedRays[i], camera[i], 2e-5);
        for (var pass = 0; pass < 2; pass++)
        {
            if (pass > 0) Array.Reverse(input);
            model.Run(input, camera, output); var expected = ContextReference(tensors, input, expectedRays, shape, hand);
            for (var i = 0; i < output.Length; i++) MotionTests.Near(expected[i], output[i], 3e-4);
        }
        model.CopyPositions(positions);
        var matrix = tensors[(hand ? "hand_pe_layer" : "prompt_encoder.pe_layer") + ".positional_encoding_gaussian_matrix"].Data;
        for (var patch = 0; patch < shape.Patches; patch++)
            for (var i = 0; i < shape.Channels / 2; i++)
            {
                var x = 2 * ((patch % 3 + .5) / 3) - 1; var y = 2 * ((patch / 3 + .5) / 2) - 1;
                var angle = 2 * Math.PI * (x * matrix[i] + y * matrix[shape.Channels / 2 + i]);
                MotionTests.Near(Math.Sin(angle), positions[patch * shape.Channels + i], 2e-6);
                MotionTests.Near(Math.Cos(angle), positions[patch * shape.Channels + shape.Channels / 2 + i], 2e-6);
            }
        for (var warm = 0; warm < 4; warm++) { rays.Prepare(intrinsics, affine, camera); model.Run(input, camera, output); model.CopyPositions(positions); }
        var before = GC.GetAllocatedBytesForCurrentThread();
        rays.Prepare(intrinsics, affine, camera); model.Run(input, camera, output); model.CopyPositions(positions);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        MotionTests.Throws<ArgumentException>(() => model.Run(input, camera, input));
        MotionTests.Throws<ArgumentException>(() => model.Run(input, camera, camera.AsSpan(0, input.Length)));
        MotionTests.Throws<ArgumentException>(() => model.CopyPositions(new float[1]));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => model.Run(input, camera, output, cancelled.Token));
        input[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => model.Run(input, camera, output));
    }

    static void Rays()
    {
        var shape = DinoShape.Sam(16, 16, 8, 2); var rays = new SamRays(shape); var output = new float[rays.Elements];
        var camera = new float[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; var affine = new float[] { 1, 0, 0, 0, 1, 0 };
        rays.Prepare(camera, affine, output); MotionTests.Near(7.5, output[0], 1e-6); MotionTests.Near(7.5, output[1], 1e-6);
        foreach (var index in new[] { 0, 4, 8 })
        {
            var invalid = camera.ToArray(); invalid[index] = 0;
            MotionTests.Throws<ArgumentException>(() => rays.Prepare(invalid, affine, output));
        }
        foreach (var index in new[] { 1, 3, 6, 7 })
        {
            var invalid = camera.ToArray(); invalid[index] = .1f;
            MotionTests.Throws<ArgumentException>(() => rays.Prepare(invalid, affine, output));
        }
        affine[1] = .1f; MotionTests.Throws<ArgumentException>(() => rays.Prepare(camera, affine, output)); affine[1] = 0;
        affine[0] = 0; MotionTests.Throws<ArgumentException>(() => rays.Prepare(camera, affine, output)); affine[0] = 1;
        camera[2] = float.NaN; MotionTests.Throws<ArgumentException>(() => rays.Prepare(camera, affine, output)); camera[2] = 0;
        MotionTests.Throws<ArgumentException>(() => rays.Prepare(output.AsSpan(0, 9), affine, output));
        MotionTests.Throws<ArgumentException>(() => new SamRays(default));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => rays.Prepare(camera, affine, output, cancelled.Token));
    }

    static double[] RayReference(DinoShape shape, float[] camera, float[] affine)
    {
        var output = new double[shape.Patches * 99]; var columns = shape.Width / 16;
        for (var patch = 0; patch < shape.Patches; patch++)
        {
            double rx = 0, ry = 0, total = 0;
            for (var y = 0; y < shape.Height; y++)
                for (var x = 0; x < shape.Width; x++)
                {
                    var w = Math.Max(0, 1 - Math.Abs((x + .5) / 16 - (patch % columns + .5)))
                        * Math.Max(0, 1 - Math.Abs((y + .5) / 16 - (patch / columns + .5)));
                    rx += w * ((x - affine[2]) / (double)affine[0] - camera[2]) / camera[0];
                    ry += w * ((y - affine[5]) / (double)affine[4] - camera[5]) / camera[4]; total += w;
                }
            var row = output.AsSpan(patch * 99, 99); row[0] = rx / total; row[1] = ry / total; row[2] = 1;
            for (var axis = 0; axis < 3; axis++)
                for (var band = 0; band < 16; band++)
                {
                    var angle = Math.PI * row[axis] * (1 + 31.0 * band / 15);
                    row[3 + axis * 16 + band] = Math.Sin(angle); row[51 + axis * 16 + band] = Math.Cos(angle);
                }
        }
        return output;
    }

    static double[] ContextReference(Dictionary<string, Tensor> tensors, float[] image, double[] rays, DinoShape shape, bool hand)
    {
        var name = hand ? "ray_cond_emb_hand" : "ray_cond_emb"; var width = shape.Channels; var result = new double[image.Length];
        var mask = tensors["prompt_encoder.no_mask_embed.weight"].Data; var weights = tensors[name + ".conv.weight"].Data;
        var scale = tensors[name + ".norm.weight"].Data; var bias = tensors[name + ".norm.bias"].Data;
        for (var patch = 0; patch < shape.Patches; patch++)
        {
            var row = result.AsSpan(patch * width, width);
            for (var channel = 0; channel < width; channel++)
            {
                for (var i = 0; i < width; i++) row[channel] += ((double)image[patch * width + i] + mask[i]) * weights[channel * (width + 99) + i];
                for (var i = 0; i < 99; i++) row[channel] += rays[patch * 99 + i] * weights[channel * (width + 99) + width + i];
            }
            var values = row.ToArray(); var mean = values.Average(); var variance = values.Average(x => (x - mean) * (x - mean));
            for (var i = 0; i < width; i++) row[i] = (row[i] - mean) / Math.Sqrt(variance + 1e-6) * scale[i] + bias[i];
        }
        return result;
    }

    static void Checkpoint()
    {
        var path = Path.Combine(Path.GetTempPath(), "sam-context-" + Guid.NewGuid().ToString("N") + ".pt");
        try
        {
            var shape = DinoShape.Sam(16, 16, 8, 2); var tensors = SamFixture.ImageConditioning(8, false);
            foreach (var pair in SamFixture.ImageConditioning(8, true)) tensors[pair.Key] = pair.Value;
            tensors.Add("backbone.unused", new("backbone.unused", new[] { float.NaN }, new[] { 1 }));
            foreach (var root in new string[] { null, "state_dict" })
            {
                CheckpointFixture.Write(path, tensors, root, "");
                foreach (var hand in new[] { false, true })
                {
                    var weights = VisionCheckpoint.LoadSamImageConditioning(path, hand); _ = new SamImageConditioning(weights, shape, hand);
                    MotionTests.Throws<InvalidDataException>(() => new SamImageConditioning(weights, shape, !hand));
                }
                File.Delete(path);
            }
        }
        finally { File.Delete(path); }
    }
}
