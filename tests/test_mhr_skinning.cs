using System.Numerics;
using HumanoidMocap.EditorTools.Inference;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;

static class MhrSkinningTests
{
    internal static void Run()
    {
        var bind = States(.2f); var states = States(.7f);
        var joints = new[] { 0, 1, 1, 0, 0 }; var vertices = new[] { 0, 0, 2, 2, 2 }; var weights = new[] { .25f, .75f, .2f, 1.1f, .2f };
        var rest = new float[] { 1, -2, 3, 4, 5, 6, -2, 1, 3 }; var output = new float[rest.Length];
        var model = new MhrSkinning(bind, joints, vertices, weights, 3);
        for (var pass = 0; pass < 2; pass++)
        {
            if (pass > 0) Array.Reverse(rest);
            model.Run(states, rest, output);
            var expected = Reference(bind, states, joints, vertices, weights, rest);
            for (var i = 0; i < output.Length; i++) MotionTests.Near(expected[i], output[i], 3e-6);
            MotionTests.Check(output.AsSpan(3, 3).SequenceEqual(new float[3]), "Unweighted vertex must remain zero");
        }
        for (var warm = 0; warm < 4; warm++) model.Run(states, rest, output);
        var before = GC.GetAllocatedBytesForCurrentThread(); model.Run(states, rest, output);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        MotionTests.Throws<ArgumentException>(() => model.Run(states, rest, rest));
        MotionTests.Throws<ArgumentException>(() => model.Run(states, rest, states.AsSpan(0, rest.Length)));
        MotionTests.Throws<ArgumentException>(() => model.Run(states.AsSpan(1), rest, output));
        MotionTests.Throws<ArgumentException>(() => model.Run(states, rest.AsSpan(1), output));
        MotionTests.Throws<ArgumentException>(() => model.Run(states, rest, output.AsSpan(1)));
        rest[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => model.Run(states, rest, output)); rest[0] = 1;
        foreach (var invalid in new[] { 0f, -1f, float.NaN, float.PositiveInfinity })
        {
            var bad = states.ToArray(); bad[7] = invalid;
            MotionTests.Throws<ArgumentException>(() => model.Run(bad, rest, output));
        }
        var badRotation = states.ToArray(); Array.Clear(badRotation, 3, 4);
        MotionTests.Throws<ArgumentException>(() => model.Run(badRotation, rest, output));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => model.Run(states, rest, output, cancelled.Token));
        foreach (var invalid in new[] { -1, 2, int.MaxValue })
        {
            var bad = joints.ToArray(); bad[0] = invalid;
            MotionTests.Throws<InvalidDataException>(() => new MhrSkinning(bind, bad, vertices, weights, 3));
        }
        foreach (var invalid in new[] { -1, 3, int.MaxValue })
        {
            var bad = vertices.ToArray(); bad[0] = invalid;
            MotionTests.Throws<InvalidDataException>(() => new MhrSkinning(bind, joints, bad, weights, 3));
        }
        foreach (var invalid in new[] { -.1f, float.NaN, float.PositiveInfinity })
        {
            var bad = weights.ToArray(); bad[0] = invalid;
            MotionTests.Throws<InvalidDataException>(() => new MhrSkinning(bind, joints, vertices, bad, 3));
        }
        MotionTests.Throws<InvalidDataException>(() => new MhrSkinning(bind.AsSpan(1), joints, vertices, weights, 3));
        MotionTests.Throws<InvalidDataException>(() => new MhrSkinning(bind, joints, vertices, weights.AsSpan(1), 3));
        MotionTests.Throws<ArgumentException>(() => new MhrSkinning(badRotation, joints, vertices, weights, 3));
        Console.WriteLine("PASS: MHR weighted mesh skinning, inverse bind composition, repeated influences, caller buffers, validation and allocation-free reuse");
    }

    internal static void Released(string path)
    {
        var skeleton = MhrCheckpoint.LoadSkeleton(path); var skinning = MhrCheckpoint.LoadSkinning(path);
        using var source = TorchCheckpoint.OpenScriptData(path, "__torch__.MHRDemo");
        const string prefix = MhrSkeleton.Prefix + "linear_blend_skinning.";
        var bind = source.ReadFloat(prefix + "inverse_bind_pose");
        var joints = source.ReadInt64(prefix + "skin_indices_flattened").Select(v => checked((int)v)).ToArray();
        var vertices = source.ReadInt64(prefix + "vert_indices_flattened").Select(v => checked((int)v)).ToArray();
        var weights = source.ReadFloat(prefix + "skin_weights_flattened");
        var rest = source.ReadFloat(MhrSkeleton.Prefix + "mesh.rest_vertices"); var output = new float[rest.Length];
        var input = new float[skeleton.Parameters]; var pose = new float[skeleton.Joints * 7]; var states = new float[skeleton.Joints * 8];
        double maximum = 0;
        for (var pass = 0; pass < 2; pass++)
        {
            if (pass > 0) for (var i = 0; i < input.Length; i++) input[i] = .02f * MathF.Sin(i * .37f);
            skeleton.Run(input, pose, states); skinning.Run(states, rest, output);
            var expected = Reference(bind, states, joints, vertices, weights, rest);
            for (var i = 0; i < output.Length; i++)
            {
                MotionTests.Near(expected[i], output[i], 1e-4); maximum = Math.Max(maximum, Math.Abs(expected[i] - output[i]));
                if (pass == 0) MotionTests.Near(rest[i], output[i], 1e-4);
            }
        }
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => MhrCheckpoint.LoadSkinning(path, cancelled.Token));
        Console.WriteLine($"PASS: released MHR {skinning.Vertices} vertices/{weights.Length} influences, rest mesh and posed matrix reference; max absolute difference {maximum:G6} model units");
    }

    static float[] States(float angle)
    {
        var output = new float[16];
        for (var joint = 0; joint < 2; joint++)
        {
            var q = Quaternion.CreateFromAxisAngle(Vector3.Normalize(new Vector3(1, -2, 3)), angle + joint * .31f);
            var i = joint * 8; output[i] = joint + .3f; output[i + 1] = angle; output[i + 2] = -.8f;
            output[i + 3] = q.X; output[i + 4] = q.Y; output[i + 5] = q.Z; output[i + 6] = q.W; output[i + 7] = 1 + angle;
        }
        return output;
    }

    static double[] Reference(float[] bind, float[] states, int[] joints, int[] vertices, float[] weights, float[] rest)
    {
        var matrices = new double[states.Length / 8][,]; var output = new double[rest.Length];
        for (var joint = 0; joint < matrices.Length; joint++)
            matrices[joint] = MhrSkeletonTests.Multiply(MhrSkeletonTests.StateMatrix(states.AsSpan(joint * 8, 8)), MhrSkeletonTests.StateMatrix(bind.AsSpan(joint * 8, 8)));
        for (var influence = 0; influence < joints.Length; influence++)
        {
            var m = matrices[joints[influence]]; var vertex = vertices[influence] * 3;
            for (var axis = 0; axis < 3; axis++)
                output[vertex + axis] += weights[influence] * (m[axis, 0] * rest[vertex] + m[axis, 1] * rest[vertex + 1] + m[axis, 2] * rest[vertex + 2] + m[axis, 3]);
        }
        return output;
    }
}
