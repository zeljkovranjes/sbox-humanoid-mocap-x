using HumanoidMocapX.Inference;

static class SamProjectionTests
{
    internal static void Run()
    {
        var intrinsics = new float[] { 700, 0, 320, 0, 530, 240, 0, 0, 1 };
        var affine = new float[] { .8f, 0, -18, 0, .6f, 23 }; var shape = DinoShape.Sam(32, 48, 8, 2);
        var points = DenoiserTests.Wave(210, .21f); var cropped = new float[140]; var depth = new float[70]; var translation = new float[3];
        foreach (var hand in new[] { false, true })
            foreach (var scale in new[] { -1.3f, 1.3f })
            {
                var projection = new SamProjection(intrinsics, affine, 410, 285, 350, shape, hand);
                var camera = new float[] { scale, .13f, -.17f };
                projection.Run(points, camera, cropped, depth, translation);
                var denominator = 350 * -(double)scale * (hand ? 10 : 1) + 1e-8;
                var t = new[] { camera[1] + 2 * (410.0 - 320) / denominator, -camera[2] + 2 * (285.0 - 240) / denominator, 2 * 700 / denominator };
                for (var i = 0; i < 3; i++) MotionTests.Near(t[i], translation[i], Math.Max(1, Math.Abs(t[i])) * 2e-7);
                for (var i = 0; i < 70; i++)
                {
                    var z = points[i * 3 + 2] + t[2]; MotionTests.Near(z, depth[i], Math.Max(1, Math.Abs(z)) * 2e-7);
                    for (var axis = 0; axis < 2; axis++)
                    {
                        var image = intrinsics[axis * 4] * ((points[i * 3 + axis] + t[axis]) / z) + intrinsics[axis * 3 + 2];
                        var expected = (affine[axis * 4] * image + affine[axis * 3 + 2]) / (axis == 0 ? shape.Width : shape.Height) - .5;
                        MotionTests.Near(expected, cropped[i * 2 + axis], Math.Max(3e-5, Math.Abs(expected) * 2e-5));
                    }
                }
                for (var i = 0; i < 4; i++) projection.Run(points, camera, cropped, depth, translation);
                var before = GC.GetAllocatedBytesForCurrentThread(); projection.Run(points, camera, cropped, depth, translation);
                MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
                MotionTests.Throws<ArgumentException>(() => projection.Run(points, camera, cropped, depth, camera));
                MotionTests.Throws<ArgumentException>(() => projection.Run(points.AsSpan(1), camera, cropped, depth, translation));
            }
        var valid = new SamProjection(intrinsics, affine, 320, 240, 350, shape);
        var parameters = new float[] { -1, 0, 0 };
        valid.Run(points, parameters, cropped, depth, translation); points[2] = -translation[2];
        MotionTests.Throws<ArgumentException>(() => valid.Run(points, parameters, cropped, depth, translation));
        MotionTests.Throws<ArgumentException>(() => default(SamProjection).Run(points, parameters, cropped, depth, translation));
        MotionTests.Throws<ArgumentException>(() => new SamProjection(intrinsics, affine, 320, 240, 0, shape));
        intrinsics[1] = .1f; MotionTests.Throws<ArgumentException>(() => new SamProjection(intrinsics, affine, 320, 240, 350, shape));
        Console.WriteLine("PASS: SAM perspective translation, hand camera scale, full-image/crop coordinates, signed depths and allocation-free projection");
    }
}
