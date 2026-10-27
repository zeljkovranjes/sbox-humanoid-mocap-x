using System.Diagnostics;
using System.Numerics;
using HumanoidMocapX.Inference;
using HumanoidMocapX.EditorTools;

static class MhrSkeletonTests
{
    internal static void Run()
    {
        var parents = new[] { -1, 0, 1, 1, 3, -1, 5 };
        var tensors = Fixture(parents.Length, 9);
        var model = new MhrSkeleton(new(tensors), parents, 9);
        var input = DenoiserTests.Wave(9, .3f); var pose = new float[parents.Length * 7]; var states = new float[parents.Length * 8];
        for (var pass = 0; pass < 3; pass++)
        {
            if (pass == 1) Array.Reverse(input);
            if (pass == 2) Array.Clear(input);
            Check(model, tensors, parents, input, pose, states);
        }
        for (var warm = 0; warm < 4; warm++) model.Run(input, pose, states);
        var before = GC.GetAllocatedBytesForCurrentThread(); model.Run(input, pose, states);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => model.Run(input, pose, states, cancelled.Token));
        MotionTests.Throws<ArgumentException>(() => model.Run(input, pose, states.AsSpan(1)));
        MotionTests.Throws<ArgumentException>(() => model.Run(input.AsSpan(1), pose, states));
        MotionTests.Throws<ArgumentException>(() => model.Run(input, pose.AsSpan(1), states));
        MotionTests.Throws<ArgumentException>(() => model.Run(states.AsSpan(0, input.Length), pose, states));
        MotionTests.Throws<ArgumentException>(() => model.Run(pose.AsSpan(0, input.Length), pose, states));
        MotionTests.Throws<ArgumentException>(() => model.Run(input, states.AsSpan(0, pose.Length), states));
        var mapping = tensors[MhrSkeleton.Prefix + "parameter_transform.parameter_transform"].Data;
        MotionTests.Throws<ArgumentException>(() => model.Run(input, mapping.AsSpan(0, pose.Length), states));
        MotionTests.Throws<ArgumentException>(() => model.Run(input, pose, mapping.AsSpan(0, states.Length)));
        input[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => model.Run(input, pose, states)); input[0] = 0;
        foreach (var invalid in new[] { -2, 1, 3, int.MaxValue })
        {
            var bad = parents.ToArray(); bad[1] = invalid;
            MotionTests.Throws<InvalidDataException>(() => new MhrSkeleton(new(tensors), bad, 9));
        }
        MotionTests.Throws<InvalidDataException>(() => new MhrSkeleton(new(tensors), parents, 10));
        MotionTests.Throws<ArgumentException>(() => new MhrSkeleton(new(tensors), Array.Empty<int>(), 9));
        tensors[MhrSkeleton.Prefix + "skeleton.joint_prerotations"].Data[3] = 0;
        MotionTests.Throws<InvalidDataException>(() => new MhrSkeleton(new(tensors), parents, 9));
        Scales();
        Console.WriteLine("PASS: MHR parameter mapping, XYZ rotation order, prerotations, branched global transforms, scale, validation and allocation-free reuse");
    }

    static void Scales()
    {
        var tensors = Fixture(1, 7); var mapping = tensors[MhrSkeleton.Prefix + "parameter_transform.parameter_transform"].Data;
        Array.Clear(mapping); for (var i = 0; i < 7; i++) mapping[i * 7 + i] = 1;
        var model = new MhrSkeleton(new(tensors), new[] { -1 }, 7);
        var input = new float[7]; var pose = new float[7]; var states = new float[8];
        input[6] = 2; model.Run(input, pose, states); MotionTests.Near(4, states[7], 1e-6);
        input[6] = -2; model.Run(input, pose, states); MotionTests.Near(.25, states[7], 1e-6);
        input[6] = 200; MotionTests.Throws<ArithmeticException>(() => model.Run(input, pose, states));
        input[6] = -200; MotionTests.Throws<ArithmeticException>(() => model.Run(input, pose, states));
    }

    internal static void Released(string path)
    {
        var watch = Stopwatch.StartNew(); var model = MhrCheckpoint.LoadSkeleton(path);
        using var source = TorchCheckpoint.OpenScriptData(path, "__torch__.MHRDemo");
        var tensors = new Dictionary<string, Tensor>();
        foreach (var suffix in new[] { "parameter_transform.parameter_transform", "skeleton.joint_translation_offsets", "skeleton.joint_prerotations" })
        {
            var name = MhrSkeleton.Prefix + suffix;
            tensors.Add(name, new(name, source.ReadFloat(name), source.Tensors[name].Shape.Reverse().ToArray()));
        }
        var parents = source.ReadInt64(MhrSkeleton.Prefix + "skeleton.joint_parents").Select(v => checked((int)v)).ToArray();
        MotionTests.Equal(127, model.Joints); MotionTests.Equal(249, model.Parameters);
        var input = new float[model.Parameters]; var pose = new float[model.Joints * 7]; var states = new float[model.Joints * 8];
        Check(model, tensors, parents, input, pose, states);
        var inverseBind = source.ReadFloat(MhrSkeleton.Prefix + "linear_blend_skinning.inverse_bind_pose");
        for (var joint = 0; joint < model.Joints; joint++)
        {
            var product = Multiply(StateMatrix(states.AsSpan(joint * 8, 8)), StateMatrix(inverseBind.AsSpan(joint * 8, 8)));
            for (var row = 0; row < 4; row++)
                for (var col = 0; col < 4; col++) MotionTests.Near(row == col ? 1 : 0, product[row, col], 5e-4);
        }
        for (var i = 0; i < input.Length; i++) input[i] = .015f * MathF.Sin(i * .17f);
        Check(model, tensors, parents, input, pose, states);
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => MhrCheckpoint.LoadSkeleton(path, cancelled.Token));
        Console.WriteLine($"PASS: released MHR {model.Joints} joints/{model.Parameters} parameters, independent matrix reference and inverse-bind identity ({watch.ElapsedMilliseconds} ms including reads)");
    }

    static void Check(MhrSkeleton model, Dictionary<string, Tensor> tensors, int[] parents, float[] input, float[] pose, float[] states)
    {
        model.Run(input, pose, states);
        var mapping = tensors[MhrSkeleton.Prefix + "parameter_transform.parameter_transform"].Data;
        var offsets = tensors[MhrSkeleton.Prefix + "skeleton.joint_translation_offsets"].Data;
        var pre = tensors[MhrSkeleton.Prefix + "skeleton.joint_prerotations"].Data;
        var expected = new double[pose.Length];
        for (var row = 0; row < pose.Length; row++)
        {
            for (var col = 0; col < input.Length; col++) expected[row] += (double)mapping[row * input.Length + col] * input[col];
            MotionTests.Near(expected[row], pose[row], 1e-5);
        }
        var matrices = new double[parents.Length][,];
        for (var joint = 0; joint < parents.Length; joint++)
        {
            var p = expected.AsSpan(joint * 7, 7);
            var rotation = Multiply(QuaternionMatrix(pre.AsSpan(joint * 4, 4)), Multiply(Axis(2, p[5]), Multiply(Axis(1, p[4]), Axis(0, p[3]))));
            var scale = Math.Pow(2, p[6]);
            for (var row = 0; row < 3; row++)
            {
                for (var col = 0; col < 3; col++) rotation[row, col] *= scale;
                rotation[row, 3] = p[row] + offsets[joint * 3 + row];
            }
            matrices[joint] = parents[joint] < 0 ? rotation : Multiply(matrices[parents[joint]], rotation);
            var actual = StateMatrix(states.AsSpan(joint * 8, 8));
            // Float local rotations accumulate before the final float cast; scale
            // the error bound by the transform, including near-zero matrix entries.
            double magnitude = 1;
            foreach (var value in matrices[joint]) magnitude = Math.Max(magnitude, Math.Abs(value));
            for (var row = 0; row < 4; row++)
                for (var col = 0; col < 4; col++)
                {
                    var value = matrices[joint][row, col];
                    MotionTests.Near(value, actual[row, col], Math.Max(8e-5, magnitude * 3e-6));
                }
        }
    }

    static Dictionary<string, Tensor> Fixture(int joints, int parameters)
    {
        var tensors = new Dictionary<string, Tensor>();
        Add("parameter_transform.parameter_transform", DenoiserTests.Wave(joints * 7 * parameters, .11f), parameters, joints * 7);
        Add("skeleton.joint_translation_offsets", DenoiserTests.Wave(joints * 3, .9f), 3, joints);
        var rotations = new float[joints * 4];
        for (var joint = 0; joint < joints; joint++)
        {
            var q = Quaternion.CreateFromAxisAngle(Vector3.Normalize(new Vector3(1, 2, -3)), joint * .2f);
            rotations[joint * 4] = q.X; rotations[joint * 4 + 1] = q.Y; rotations[joint * 4 + 2] = q.Z; rotations[joint * 4 + 3] = q.W;
        }
        Add("skeleton.joint_prerotations", rotations, 4, joints);
        return tensors;
        void Add(string suffix, float[] data, params int[] shape)
        { var name = MhrSkeleton.Prefix + suffix; tensors.Add(name, new(name, data, shape)); }
    }

    internal static double[,] StateMatrix(ReadOnlySpan<float> state)
    {
        var m = QuaternionMatrix(state.Slice(3, 4));
        for (var row = 0; row < 3; row++)
        {
            for (var col = 0; col < 3; col++) m[row, col] *= state[7];
            m[row, 3] = state[row];
        }
        return m;
    }

    static double[,] QuaternionMatrix(ReadOnlySpan<float> quaternion)
    {
        double x = quaternion[0], y = quaternion[1], z = quaternion[2], w = quaternion[3];
        var norm = x * x + y * y + z * z + w * w; var k = 2 / norm;
        return new double[,] {
            { 1-k*(y*y+z*z), k*(x*y-z*w), k*(x*z+y*w), 0 },
            { k*(x*y+z*w), 1-k*(x*x+z*z), k*(y*z-x*w), 0 },
            { k*(x*z-y*w), k*(y*z+x*w), 1-k*(x*x+y*y), 0 }, { 0,0,0,1 } };
    }

    static double[,] Axis(int axis, double angle)
    {
        var m = new double[4, 4]; for (var i = 0; i < 4; i++) m[i, i] = 1;
        var a = (axis + 1) % 3; var b = (axis + 2) % 3;
        m[a, a] = m[b, b] = Math.Cos(angle); m[a, b] = -Math.Sin(angle); m[b, a] = Math.Sin(angle);
        return m;
    }

    internal static double[,] Multiply(double[,] a, double[,] b)
    {
        var output = new double[4, 4];
        for (var row = 0; row < 4; row++)
            for (var col = 0; col < 4; col++)
                for (var k = 0; k < 4; k++) output[row, col] += a[row, k] * b[k, col];
        return output;
    }
}
