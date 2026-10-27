using HumanoidMocap.EditorTools.MotionBricks;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;

static class OmegaTests
{
    internal static void Run()
    {
        Positions(); Activation(); Blocks(); Camera(); Checkpoint();
        Console.WriteLine("PASS: Omega axial positions, erf GELU, masked bias, Q/K normalization, transformer and four-block camera head against double-precision equations");
    }

    static void Positions()
    {
        var table = OmegaPosition.Factors(2, 3, 2, new float[] { 1, 10 });
        MotionTests.Equal(8 * 8, table.Length);
        for (var index = 0; index < 16; index++) MotionTests.Equal(index % 2 == 0 ? 1f : 0f, table[index]);
        for (var y = 0; y < 2; y++)
            for (var x = 0; x < 3; x++)
                for (var axis = 0; axis < 2; axis++)
                    for (var f = 0; f < 2; f++)
                    {
                        var angle = 2 * Math.PI * (2 * ((axis == 0 ? y : x) + .5) / 3 - 1) / (f == 0 ? 1 : 10);
                        var at = ((2 + y * 3 + x) * 8) + axis * 4 + f * 2;
                        MotionTests.Near(Math.Cos(angle), table[at], 1e-6); MotionTests.Near(Math.Sin(angle), table[at + 1], 1e-6);
                    }
        var values = Enumerable.Range(0, 8 * 16).Select(i => i * .1f).ToArray(); var original = (float[])values.Clone();
        OmegaPosition.Apply(values, table, 16, 8);
        MotionTests.Check(values.AsSpan(0, 32).SequenceEqual(original.AsSpan(0, 32)), "Omega rotated its prefix tokens");
        for (var offset = 0; offset < values.Length; offset += 8)
            MotionTests.Near(original.AsSpan(offset, 8).ToArray().Sum(v => (double)v * v), values.AsSpan(offset, 8).ToArray().Sum(v => (double)v * v), 1e-4);
        MotionTests.Throws<ArgumentException>(() => OmegaPosition.Factors(1, 1, 0, new float[] { 0 }));
        MotionTests.Throws<ArgumentException>(() => OmegaPosition.Factors(int.MaxValue, int.MaxValue, 1, new float[] { 1 }));
        MotionTests.Throws<ArgumentException>(() => OmegaPosition.Apply(values, values, 16, 8));
    }

    static void Activation()
    {
        var values = Enumerable.Range(0, 1001).Select(i => (i - 500) * .02f).ToArray(); var original = (float[])values.Clone();
        OmegaActivation.Gelu(values);
        for (var i = 0; i < values.Length; i++) MotionTests.Near(OmegaReference.Gelu(original[i]), values[i], 2e-6);
    }

    static void Blocks()
    {
        const int width = 16, heads = 2, rows = 8;
        foreach (var qk in new[] { false, true })
            foreach (var maskAll in new[] { false, true })
            {
                var tensors = OmegaFixture.Create(width, heads); OmegaFixture.Block(tensors, "block", width, heads, qk);
                var mask = tensors["block.attn.qkv.bias_mask"].Data;
                if (maskAll) Array.Clear(mask);
                var block = new OmegaBlock(new(tensors), "block", width, heads, qk);
                var input = DenoiserTests.Wave(rows * width, .17f); var values = (float[])input.Clone();
                var factors = OmegaPosition.Factors(2, 3, 2, new float[] { 1, 10 });
                var expected = OmegaReference.Block(tensors, "block", input.Select(v => (double)v).ToArray(), width, heads, qk, factors);
                var workspace = new OmegaWorkspace(rows, width); var attention = new FlowAttention(rows, width, heads, false);
                block.Run(values, workspace, attention, factors);
                for (var i = 0; i < values.Length; i++) MotionTests.Near(expected[i], values[i], 1e-5);
                mask[width] = 2;
                MotionTests.Throws<InvalidDataException>(() => new OmegaBlock(new(tensors), "block", width, heads, qk));
            }
    }

    static void Camera()
    {
        const int frames = 2, width = 16, heads = 2;
        var tensors = OmegaFixture.Create(width, heads); var model = new OmegaCameraHead(new(tensors), frames, width, heads);
        var input = DenoiserTests.Wave(frames * 17 * width, .083f); var output = new float[frames * 9];
        var expected = OmegaReference.Camera(tensors, input, frames, width, heads);
        model.Run(input, output);
        for (var i = 0; i < output.Length; i++) MotionTests.Near(expected[i], output[i], 1e-5);
        var previous = output.ToArray(); input[width + 3] += 7; model.Run(input, output);
        MotionTests.Check(output.Where((value, i) => Math.Abs(value - previous[i]) > 1e-6).Any(), "Omega ignored register tokens");
        for (var warm = 0; warm < 3; warm++) model.Run(input, output);
        var allocated = GC.GetAllocatedBytesForCurrentThread(); model.Run(input, output);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - allocated);
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => model.Run(input, output, cancelled.Token));
        MotionTests.Throws<ArgumentException>(() => model.Run(input, new float[1]));
        input[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => model.Run(input, output));
        float[] raw = { 1, -2, 3, -4, 5, -6, 7, -.5f, .4f }; OmegaCameraHead.Activate(raw);
        MotionTests.Equal(-4f, raw[3]); MotionTests.Near(.01, raw[7]); MotionTests.Near(.41, raw[8]);
    }

    static void Checkpoint()
    {
        var path = Path.Combine(Path.GetTempPath(), "omega-" + Guid.NewGuid().ToString("N") + ".pt");
        try
        {
            var tensors = OmegaFixture.Create(16, 2); CheckpointFixture.Write(path, tensors, null, "");
            var loaded = VisionCheckpoint.LoadOmegaCameraHead(path);
            foreach (var pair in tensors) MotionTests.Check(loaded.Get(pair.Key, pair.Value.Shape).Data.AsSpan().SequenceEqual(pair.Value.Data), "Omega camera checkpoint changed tensor layout");
            _ = new OmegaCameraHead(loaded, 1, 16, 2);
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            MotionTests.Throws<OperationCanceledException>(() => VisionCheckpoint.LoadOmegaCameraHead(path, cancelled.Token));
            File.Delete(path); tensors["camera_head.token_norm.weight"].Data[0] = float.PositiveInfinity;
            CheckpointFixture.Write(path, tensors, "state_dict", "");
            MotionTests.Throws<InvalidDataException>(() => VisionCheckpoint.LoadOmegaCameraHead(path));
        }
        finally { File.Delete(path); }
    }

}
