using System.Numerics;
using HumanoidMocapX;

static class CameraTests
{
    internal static void Run()
    {
        Sampling(); Conversion(); ShortestTurn(); InvalidInputs(); Allocations();
        Console.WriteLine("PASS: VGGT camera sampling, original-pixel intrinsics, shortest-arc interpolation, relative rotations and allocation-free decoding");
    }

    static void Sampling()
    {
        foreach (var frames in new[] { 1, 2, 10, 11, 30, 31, 150, 300, 5000 })
            foreach (var interval in new[] { 1, 10, 30 })
            {
                var indices = Enumerable.Range(0, frames).Where(frame => frame % interval == 0 || frame == frames - 1).ToArray();
                MotionTests.Equal(indices.Length, FlowCamera.SampleCount(frames, interval));
                for (var i = 0; i < indices.Length; i++) MotionTests.Equal(indices[i], FlowCamera.SampleFrame(i, frames, interval));
                MotionTests.Throws<ArgumentOutOfRangeException>(() => FlowCamera.SampleFrame(-1, frames, interval));
                MotionTests.Throws<ArgumentOutOfRangeException>(() => FlowCamera.SampleFrame(indices.Length, frames, interval));
            }
    }

    static void Conversion()
    {
        const int frames = 24, width = 1920, height = 1080;
        var baseline = Quaternion.CreateFromAxisAngle(Vector3.UnitX, .7f);
        foreach (var interval in new[] { 1, 10, 30 })
        {
            var count = FlowCamera.SampleCount(frames, interval);
            var poses = new float[count * 9];
            for (var sample = 0; sample < count; sample++)
            {
                var frame = FlowCamera.SampleFrame(sample, frames, interval);
                var camera = Quaternion.CreateFromAxisAngle(Vector3.UnitY, frame * .03f) * baseline;
                Pose(poses.AsSpan(sample * 9, 9), camera, width, height, 800 + frame * 7, 450 + frame * 3);
            }
            var rotations = new float[frames * 9]; var intrinsics = new float[frames * 9];
            FlowCamera.Decode(poses, frames, interval, width, height, rotations, intrinsics);
            for (var frame = 0; frame < frames; frame++)
            {
                var angle = frame * .03;
                var c = Math.Cos(angle); var s = Math.Sin(angle);
                double[] expected = [c, 0, s, 0, 1, 0, -s, 0, c];
                double[] expectedK = [800 + frame * 7, 0, width * .5, 0, 450 + frame * 3, height * .5, 0, 0, 1];
                for (var i = 0; i < 9; i++)
                {
                    MotionTests.Near(expected[i], rotations[frame * 9 + i], 3e-6);
                    MotionTests.Near(expectedK[i], intrinsics[frame * 9 + i], .0002);
                }
            }
            var original = (float[])rotations.Clone();
            var worldChange = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, .4f);
            for (var sample = 0; sample < count; sample++)
            {
                var offset = sample * 9;
                var quaternion = new Quaternion(poses[offset + 3], poses[offset + 4], poses[offset + 5], poses[offset + 6]) * worldChange;
                for (var axis = 0; axis < 4; axis++) poses[offset + 3 + axis] = quaternion[axis];
            }
            FlowCamera.Decode(poses, frames, interval, width, height, rotations, intrinsics);
            for (var i = 0; i < rotations.Length; i++) MotionTests.Near(original[i], rotations[i], 3e-6);
        }
    }

    static void ShortestTurn()
    {
        var poses = new float[18];
        Pose(poses.AsSpan(0, 9), Quaternion.CreateFromAxisAngle(Vector3.UnitZ, 170 * MathF.PI / 180));
        Pose(poses.AsSpan(9, 9), Quaternion.CreateFromAxisAngle(Vector3.UnitZ, -170 * MathF.PI / 180));
        var rotations = new float[45]; var intrinsics = new float[45];
        foreach (var scale in new[] { 1f, -1f, 1e-35f, 1e35f })
        {
            var scaled = (float[])poses.Clone();
            for (var i = 3; i < 7; i++) scaled[i + 9] *= scale;
            FlowCamera.Decode(scaled, 5, 10, 640, 480, rotations, intrinsics);
            for (var frame = 0; frame < 5; frame++)
            {
                var angle = frame * 5 * Math.PI / 180;
                MotionTests.Near(Math.Cos(angle), rotations[frame * 9], 3e-6);
                MotionTests.Near(-Math.Sin(angle), rotations[frame * 9 + 1], 3e-6);
                MotionTests.Near(Math.Sin(angle), rotations[frame * 9 + 3], 3e-6);
            }
        }
        FlowCamera.Decode(poses.AsSpan(0, 9), 1, 30, 640, 480, rotations.AsSpan(0, 9), intrinsics.AsSpan(0, 9));
        double[] identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        for (var i = 0; i < 9; i++) MotionTests.Near(identity[i], rotations[i], 1e-6);
    }

    static void InvalidInputs()
    {
        var poses = new float[9]; Pose(poses, Quaternion.Identity);
        var rotations = new float[9]; var intrinsics = new float[9];
        foreach (var frames in new[] { 0, 5001, int.MaxValue })
            MotionTests.Throws<ArgumentOutOfRangeException>(() => FlowCamera.SampleCount(frames, 1));
        MotionTests.Throws<ArgumentOutOfRangeException>(() => FlowCamera.SampleCount(5, 2));
        MotionTests.Throws<ArgumentOutOfRangeException>(() => FlowCamera.Decode(poses, 1, 1, 0, 480, rotations, intrinsics));
        MotionTests.Throws<ArgumentException>(() => FlowCamera.Decode(new float[8], 1, 1, 640, 480, rotations, intrinsics));
        MotionTests.Throws<ArgumentException>(() => FlowCamera.Decode(poses, 1, 1, 640, 480, new float[10], intrinsics));
        MotionTests.Throws<ArgumentException>(() => FlowCamera.Decode(poses, 1, 1, 640, 480, rotations, new float[8]));
        MotionTests.Throws<ArgumentException>(() => FlowCamera.Decode(poses, 1, 1, 640, 480, rotations, rotations));
        MotionTests.Throws<ArgumentException>(() => FlowCamera.Decode(poses, 1, 1, 640, 480, poses, intrinsics));
        var overlapping = new float[10]; poses.CopyTo(overlapping, 0);
        MotionTests.Throws<ArgumentException>(() => FlowCamera.Decode(overlapping.AsSpan(0, 9), 1, 1, 640, 480, rotations, overlapping.AsSpan(1, 9)));
        for (var channel = 0; channel < 9; channel++)
        {
            var invalid = (float[])poses.Clone(); invalid[channel] = float.NaN;
            MotionTests.Throws<ArgumentException>(() => FlowCamera.Decode(invalid, 1, 1, 640, 480, rotations, intrinsics));
        }
        foreach (var fov in new[] { 0f, -.1f, MathF.PI, 4f, float.Epsilon })
        {
            var invalid = (float[])poses.Clone(); invalid[7] = fov;
            MotionTests.Throws<ArgumentException>(() => FlowCamera.Decode(invalid, 1, 1, 640, 480, rotations, intrinsics));
        }
        var zero = (float[])poses.Clone(); zero[6] = 0;
        MotionTests.Throws<ArgumentException>(() => FlowCamera.Decode(zero, 1, 1, 640, 480, rotations, intrinsics));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => FlowCamera.Decode(poses, 1, 1, 640, 480, rotations, intrinsics, cancelled.Token));
    }

    static void Allocations()
    {
        var poses = new float[18]; Pose(poses.AsSpan(0, 9), Quaternion.Identity); Pose(poses.AsSpan(9, 9), Quaternion.Identity);
        var rotations = new float[99]; var intrinsics = new float[99];
        for (var i = 0; i < 20; i++) FlowCamera.Decode(poses, 11, 10, 640, 480, rotations, intrinsics);
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 100; i++) FlowCamera.Decode(poses, 11, 10, 640, 480, rotations, intrinsics);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    static void Pose(Span<float> pose, Quaternion rotation, int width = 640, int height = 480, float fx = 320, float fy = 240)
    {
        pose[0] = 4; pose[1] = -5; pose[2] = 6;
        pose[3] = rotation.X; pose[4] = rotation.Y; pose[5] = rotation.Z; pose[6] = rotation.W;
        pose[7] = (float)(2 * Math.Atan(height * .5 / fy)); pose[8] = (float)(2 * Math.Atan(width * .5 / fx));
    }
}
