using HumanoidMocapX.Inference;

static class SamCropTests
{
    internal static void Run()
    {
        var shape = DinoShape.Sam(16, 16, 8, 2); var affine = new float[6]; var pixels = new byte[32 * 32 * 4];
        for (var y = 0; y < 32; y++)
            for (var x = 0; x < 32; x++)
            { var at = (y * 32 + x) * 4; pixels[at] = (byte)(x * 8); pixels[at + 1] = (byte)(y * 8); pixels[at + 2] = 160; pixels[at + 3] = 17; }
        var output = new float[shape.ImageElements];
        foreach (var hand in new[] { false, true })
        {
            var crop = new SamCrop(3, 2, 15, 18, shape, hand); crop.CopyAffine(affine);
            var size = hand ? 14.4 : 20;
            MotionTests.Near(9, crop.CenterX, 0); MotionTests.Near(10, crop.CenterY, 0); MotionTests.Near(size, crop.Size, 2e-6);
            MotionTests.Near(8, affine[0] * crop.CenterX + affine[2], 1e-6); MotionTests.Near(8, affine[4] * crop.CenterY + affine[5], 1e-6);
            var projected = new float[] { 0, 0, -.25f, .25f }; crop.ToImage(projected);
            MotionTests.Near(crop.CenterX, projected[0], 1e-6); MotionTests.Near(crop.CenterY, projected[1], 1e-6);
            MotionTests.Near(crop.CenterX - crop.Size * .25, projected[2], 1e-6); MotionTests.Near(crop.CenterY + crop.Size * .25, projected[3], 1e-6);
            crop.Prepare(pixels, 32, 32, output);
            for (var y = 0; y < 16; y++)
                for (var x = 0; x < 16; x++)
                {
                    var sx = crop.CenterX + (x / 16.0 - .5) * crop.Size; var sy = crop.CenterY + (y / 16.0 - .5) * crop.Size;
                    // This fixture avoids fractional-bin boundaries, except the separately asserted tie case.
                    var qx = Math.Floor(sx * 32 + .5) / 32; var qy = Math.Floor(sy * 32 + .5) / 32;
                    for (var c = 0; c < 3; c++)
                    {
                        double value = 0;
                        for (var row = (int)Math.Floor(qy); row <= (int)Math.Floor(qy) + 1; row++)
                            for (var col = (int)Math.Floor(qx); col <= (int)Math.Floor(qx) + 1; col++)
                                if (row >= 0 && row < 32 && col >= 0 && col < 32)
                                    value += pixels[(row * 32 + col) * 4 + c] * (1 - Math.Abs(qx - col)) * (1 - Math.Abs(qy - row));
                        MotionTests.Equal((float)Math.Round(value, MidpointRounding.AwayFromZero) / 255f, output[(y * 16 + x) * 3 + c]);
                    }
                }
            for (var i = 0; i < 4; i++) crop.Prepare(pixels, 32, 32, output);
            var before = GC.GetAllocatedBytesForCurrentThread(); crop.Prepare(pixels, 32, 32, output);
            MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        }
        var wide = new SamCrop(0, 0, 24, 12, shape); MotionTests.Equal(40f, wide.Size);
        var halfBin = new SamCrop(7.015625f, 2, 13.015625f, 18, shape);
        pixels[0] = 0; pixels[4] = 32; halfBin.Prepare(pixels, 32, 32, output); MotionTests.Equal(1 / 255f, output[0]);
        var beforeAlpha = output.ToArray(); for (var i = 3; i < pixels.Length; i += 4) pixels[i] = 255;
        halfBin.Prepare(pixels, 32, 32, output); MotionTests.Check(beforeAlpha.SequenceEqual(output), "SAM must ignore alpha");
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => halfBin.Prepare(pixels, 32, 32, output, cancelled.Token));
        MotionTests.Throws<ArgumentException>(() => halfBin.Prepare(pixels, 31, 32, output));
        MotionTests.Throws<ArgumentException>(() => halfBin.Prepare(pixels, 32, 32, output.AsSpan(1)));
        MotionTests.Throws<ArgumentException>(() => default(SamCrop).Prepare(pixels, 32, 32, output));
        MotionTests.Throws<ArgumentException>(() => default(SamCrop).ToImage(new float[2]));
        MotionTests.Throws<ArgumentException>(() => halfBin.ToImage(new float[3]));
        MotionTests.Throws<ArgumentException>(() => halfBin.ToImage(new[] { float.NaN, 0f }));
        MotionTests.Throws<ArgumentException>(() => new SamCrop(0, 0, 0, 16, shape));
        MotionTests.Throws<ArgumentException>(() => new SamCrop(float.NaN, 0, 16, 16, shape));
        MotionTests.Throws<ArgumentException>(() => new SamCrop(0, 0, 16, 16, DinoShape.Sam(16, 32, 8, 2)));
        Mirrored();
        Console.WriteLine("PASS: SAM body/hand padding, aspect expansion, affine centres, RGB crops, black borders, fixed-point rounding, alpha handling and allocation-free reuse");
    }

    static void Mirrored()
    {
        const int width = 27, height = 19;
        var shape = DinoShape.Sam(16, 16, 8, 2); var pixels = new byte[width * height * 4]; new Random(31).NextBytes(pixels);
        var flipped = new byte[pixels.Length]; var expected = new float[shape.ImageElements]; var actual = new float[expected.Length];
        for (var y = 0; y < height; y++) for (var x = 0; x < width; x++)
            pixels.AsSpan((y * width + x) * 4, 4).CopyTo(flipped.AsSpan((y * width + width - x - 1) * 4));
        foreach (var box in new[] { new float[] { -5.27f, -9.01f, 19.83f, 15.12f }, new float[] { 15, 8, 34, 27 } })
        {
            var crop = new SamCrop(box[0], box[1], box[2], box[3], shape, true);
            crop.Prepare(flipped, width, height, expected); crop.Prepare(pixels, width, height, actual, flipHorizontal: true);
            MotionTests.Check(expected.SequenceEqual(actual), "Mirrored sampling must equal a flipped image, including borders and fractional pixels");
        }
    }
}
