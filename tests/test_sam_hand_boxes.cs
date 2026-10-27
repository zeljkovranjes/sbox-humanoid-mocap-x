using HumanoidMocapX.Inference;

static class SamHandBoxTests
{
    internal static void Run()
    {
        const int width = 8;
        var tensors = new Dictionary<string, Tensor>();
        OmegaFixture.Linear(tensors, "bbox_embed.layers.0", width, width);
        OmegaFixture.Linear(tensors, "bbox_embed.layers.1", width, width);
        OmegaFixture.Linear(tensors, "bbox_embed.layers.2", width, 4);
        var model = new SamHandBoxes(new(tensors), width); var tokens = DenoiserTests.Wave(width * 2, 3);
        var shape = DinoShape.Sam(32, 32, 8, 2); var body = new SamCrop(-80, 12, 120, 330, shape);
        var actual = new float[8]; var affine = new float[6]; body.CopyAffine(affine); model.Run(tokens, body, actual);
        for (var hand = 0; hand < 2; hand++)
        {
            var values = tokens.Skip(hand * width).Take(width).Select(v => (double)v).ToArray();
            for (var layer = 0; layer < 3; layer++)
            {
                var w = tensors["bbox_embed.layers." + layer + ".weight"]; var b = tensors["bbox_embed.layers." + layer + ".bias"];
                var output = new double[b.Data.Length];
                for (var row = 0; row < output.Length; row++)
                {
                    output[row] = b.Data[row];
                    for (var col = 0; col < values.Length; col++) output[row] += w.Data[row * values.Length + col] * values[col];
                    if (layer < 2) output[row] = Math.Max(0, output[row]);
                }
                values = output;
            }
            for (var i = 0; i < 4; i++) values[i] = 32 / (1 + Math.Exp(-values[i]));
            var centerX = (values[0] - affine[2]) / affine[0]; var centerY = (values[1] - affine[5]) / affine[0];
            var size = Math.Max(values[2], values[3]) / affine[0];
            var expected = new[] { centerX - size / 2, centerY - size / 2, centerX + size / 2, centerY + size / 2 };
            for (var i = 0; i < 4; i++) MotionTests.Near(expected[i], actual[hand * 4 + i], 3e-5);
        }
        MotionTests.Check(!actual.AsSpan(0, 4).SequenceEqual(actual.AsSpan(4)), "Different hand tokens must produce different boxes");
        for (var i = 0; i < 4; i++) model.Run(tokens, body, actual);
        var before = GC.GetAllocatedBytesForCurrentThread(); model.Run(tokens, body, actual);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        var box = new float[] { -5, 7, 19, 31 };
        var right = SamHandBoxes.Crop(box, shape, 100, false); var left = SamHandBoxes.Crop(box, shape, 100, true);
        MotionTests.Equal(7f, right.CenterX); MotionTests.Equal(92f, left.CenterX);
        MotionTests.Equal(right.CenterY, left.CenterY); MotionTests.Equal(right.Size, left.Size);
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => model.Run(tokens, body, actual, cancelled.Token));
        MotionTests.Throws<ArgumentException>(() => model.Run(tokens.AsSpan(1), body, actual));
        MotionTests.Throws<ArgumentException>(() => model.Run(tokens, body, tokens.AsSpan(0, 8)));
        MotionTests.Throws<ArgumentException>(() => model.Run(tokens, default, actual));
        MotionTests.Throws<ArgumentException>(() => SamHandBoxes.Crop(box.AsSpan(1), shape, 100, true));
        MotionTests.Throws<ArgumentException>(() => SamHandBoxes.Crop(box, shape, 0, true));
        tokens[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => model.Run(tokens, body, actual));
        Console.WriteLine("PASS: SAM hand-box MLP against double precision, square crop coordinates, left-hand reflection, validation and allocation-free reuse");
    }
}
