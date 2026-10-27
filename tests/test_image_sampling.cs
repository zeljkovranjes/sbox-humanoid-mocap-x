using HumanoidMocapX.EditorTools.Video;

static class ImageSamplingTests
{
    internal static void Run()
    {
        var pixels = Enumerable.Range(0, 5 * 4 * 4).Select(i => (byte)(i * 37 % 256)).ToArray();
        foreach (var replicate in new[] { false, true })
        {
            var image = new LinearRgbImage(pixels, 5, 4);
            for (var y = -40; y < 160; y += 3)
                for (var x = -40; x < 192; x += 5)
                {
                    var actual = image.Sample(x, y, replicate);
                    for (var c = 0; c < 3; c++)
                    {
                        double value = 0; var sx = x / 32.0; var sy = y / 32.0;
                        for (var row = (int)Math.Floor(sy); row <= (int)Math.Floor(sy) + 1; row++)
                            for (var col = (int)Math.Floor(sx); col <= (int)Math.Floor(sx) + 1; col++)
                            {
                                var rx = replicate ? Math.Clamp(col, 0, 4) : col; var ry = replicate ? Math.Clamp(row, 0, 3) : row;
                                if (rx < 0 || ry < 0 || rx >= 5 || ry >= 4) continue;
                                value += pixels[(ry * 5 + rx) * 4 + c] * (1 - Math.Abs(sx - col)) * (1 - Math.Abs(sy - row));
                            }
                        var expected = (float)Math.Round(value, MidpointRounding.AwayFromZero) / 255f;
                        MotionTests.Equal(expected, c == 0 ? actual.X : c == 1 ? actual.Y : actual.Z);
                    }
                }
            var before = GC.GetAllocatedBytesForCurrentThread(); _ = image.Sample(39, 74, replicate);
            MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        }
        MotionTests.Throws<ArgumentException>(() => new LinearRgbImage(pixels, 4, 5 + 1));
        MotionTests.Throws<ArgumentException>(() => new LinearRgbImage(pixels, 0, 4));
        MotionTests.Throws<InvalidOperationException>(() => default(LinearRgbImage).Sample(0, 0));
        Console.WriteLine("PASS: shared byte-rounded RGB sampling, original hand border behaviour, pixel order, negative coordinates and allocation-free reuse");
    }
}
