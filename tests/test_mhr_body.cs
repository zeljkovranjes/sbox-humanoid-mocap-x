using HumanoidMocap.EditorTools.Inference;
using HumanoidMocap.EditorTools.MotionBricks;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;

static class MhrBodyTests
{
    internal static void Run()
    {
        var tensors = MhrFixture.Tensors(); var weights = new FlowWeights(tensors);
        var blend = new MhrBlendShapes(weights, MhrFixture.Vertices, MhrFixture.Identities, MhrFixture.Expressions);
        var correctives = new MhrCorrectives(weights, MhrFixture.Indices, MhrFixture.Joints, MhrFixture.Hidden, MhrFixture.Vertices);
        var identity = new float[] { .2f, -.4f, .3f }; var expression = new float[] { -.2f, .7f };
        var pose = DenoiserTests.Wave(MhrFixture.Joints * 7, .19f); var output = new float[MhrFixture.Vertices * 3];
        for (var pass = 0; pass < 3; pass++)
        {
            if (pass == 1) { Array.Reverse(identity); Array.Reverse(expression); Array.Reverse(pose); }
            if (pass == 2) { Array.Clear(identity); Array.Clear(expression); Array.Clear(pose); }
            blend.Run(identity, expression, output);
            Compare(BlendReference(tensors, identity, expression, MhrFixture.Vertices), output, 2e-7);
            correctives.Run(pose, output);
            Compare(CorrectiveReference(tensors, MhrFixture.Indices, pose, MhrFixture.Hidden, output.Length), output, 2e-7);
            if (pass == 2) MotionTests.Check(output.All(v => v == 0), "Neutral pose must have zero corrections");
        }
        for (var warm = 0; warm < 4; warm++) { blend.Run(identity, expression, output); correctives.Run(pose, output); }
        var before = GC.GetAllocatedBytesForCurrentThread(); blend.Run(identity, expression, output); correctives.Run(pose, output);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        RestrictedProjection();
        SharedProjection();
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => blend.Run(identity, expression, output, cancelled.Token));
        MotionTests.Throws<OperationCanceledException>(() => correctives.Run(pose, output, cancelled.Token));
        MotionTests.Throws<ArgumentException>(() => blend.Run(output.AsSpan(0, identity.Length), expression, output));
        MotionTests.Throws<ArgumentException>(() => blend.Run(identity, output.AsSpan(0, expression.Length), output));
        MotionTests.Throws<ArgumentException>(() => blend.Run(identity.AsSpan(1), expression, output));
        MotionTests.Throws<ArgumentException>(() => blend.Run(identity, expression.AsSpan(1), output));
        MotionTests.Throws<ArgumentException>(() => blend.Run(identity, expression, output.AsSpan(1)));
        MotionTests.Throws<ArgumentException>(() => blend.Run(identity, expression, tensors[MhrSkeleton.Prefix + "blend_shape.base_shape"].Data));
        identity[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => blend.Run(identity, expression, output)); identity[0] = 0;
        expression[0] = float.PositiveInfinity; MotionTests.Throws<ArgumentException>(() => blend.Run(identity, expression, output)); expression[0] = 0;
        MotionTests.Throws<ArgumentException>(() => correctives.Run(pose, pose.AsSpan(0, output.Length)));
        MotionTests.Throws<ArgumentException>(() => correctives.Run(pose.AsSpan(1), output));
        MotionTests.Throws<ArgumentException>(() => correctives.Run(pose, output.AsSpan(1)));
        MotionTests.Throws<ArgumentException>(() => correctives.Run(pose, correctives.Projection.Data.AsSpan(0, output.Length)));
        pose[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => correctives.Run(pose, output)); pose[0] = 0;
        foreach (var bad in new[] { -1L, long.MaxValue })
            foreach (var index in new[] { 0, MhrFixture.Indices.Length / 2 })
            {
                var indices = MhrFixture.Indices.ToArray(); indices[index] = bad;
                MotionTests.Throws<InvalidDataException>(() => new MhrCorrectives(weights, indices, MhrFixture.Joints, MhrFixture.Hidden, MhrFixture.Vertices));
            }
        Integration(tensors);
        Console.WriteLine("PASS: MHR identity/expression blending, sparse pose features, duplicate COO entries, ReLU, corrective projection and complete allocation-free body evaluation");
    }

    static void Integration(Dictionary<string, Tensor> tensors)
    {
        var body = MhrFixture.Body(tensors); var identity = new float[MhrFixture.Identities]; var expression = new float[MhrFixture.Expressions];
        var model = DenoiserTests.Wave(MhrFixture.ModelParameters, .11f); var vertices = new float[MhrFixture.Vertices * 3]; var states = new float[MhrFixture.Joints * 8];
        body.Run(identity, model, expression, vertices, states); var expectedStates = states.ToArray(); var unshaped = vertices.ToArray();
        identity[0] = .4f; expression[0] = -.3f; body.Run(identity, model, expression, vertices, states);
        MotionTests.Check(states.SequenceEqual(expectedStates), "Identity/expression incorrectly modified skeleton parameters");
        MotionTests.Check(vertices.Zip(unshaped).Any(v => Math.Abs(v.First - v.Second) > 1e-4), "Identity/expression did not deform the mesh");
        var joint = new float[MhrFixture.Joints * 7]; var expanded = new float[MhrFixture.ModelParameters + MhrFixture.Identities]; model.CopyTo(expanded, 0);
        var mapping = tensors[MhrSkeleton.Prefix + "parameter_transform.parameter_transform"].Data;
        for (var i = 0; i < joint.Length; i++)
            for (var j = 0; j < expanded.Length; j++) joint[i] += mapping[i * expanded.Length + j] * expanded[j];
        var blend = BlendReference(tensors, identity, expression, MhrFixture.Vertices);
        var correction = CorrectiveReference(tensors, MhrFixture.Indices, joint, MhrFixture.Hidden, vertices.Length);
        var unposed = blend.Zip(correction).Select(p => (float)(p.First + p.Second)).ToArray(); var expected = new float[vertices.Length];
        MhrFixture.Skinning().Run(expectedStates, unposed, expected); Compare(expected.Select(v => (double)v).ToArray(), vertices, 1e-6);
        for (var warm = 0; warm < 4; warm++) body.Run(identity, model, expression, vertices, states);
        var before = GC.GetAllocatedBytesForCurrentThread(); body.Run(identity, model, expression, vertices, states);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        MotionTests.Throws<ArgumentException>(() => body.Run(identity, model, expression, vertices, vertices));
        MotionTests.Throws<ArgumentException>(() => body.Run(identity, model, expression, states.AsSpan(0, vertices.Length), states));
        MotionTests.Throws<ArgumentException>(() => body.Run(identity, model.AsSpan(1), expression, vertices, states));
        MotionTests.Throws<ArgumentException>(() => body.Run(identity.AsSpan(1), model, expression, vertices, states));
        MotionTests.Throws<ArgumentException>(() => body.Run(identity, model, expression.AsSpan(1), vertices, states));
        MotionTests.Throws<ArgumentException>(() => body.Run(vertices.AsSpan(0, identity.Length), model, expression, vertices, states));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => body.Run(identity, model, expression, vertices, states, cancelled.Token));
    }

    // Restricted vertices must match a full run exactly; all other corrections are zero.
    static void RestrictedProjection()
    {
        var tensors = MhrFixture.Tensors(); var weights = new FlowWeights(tensors);
        var full = new MhrCorrectives(weights, MhrFixture.Indices, MhrFixture.Joints, MhrFixture.Hidden, MhrFixture.Vertices);
        var restricted = new MhrCorrectives(weights, MhrFixture.Indices, MhrFixture.Joints, MhrFixture.Hidden, MhrFixture.Vertices);
        int[] kept = { 1, 3 }; restricted.Restrict(kept);
        var pose = DenoiserTests.Wave(MhrFixture.Joints * 7, .23f);
        var expected = new float[MhrFixture.Vertices * 3]; var actual = new float[expected.Length];
        Array.Fill(actual, 9f); full.Run(pose, expected); restricted.Run(pose, actual);
        for (var row = 0; row < expected.Length; row++)
        {
            var bits = BitConverter.SingleToInt32Bits(kept.Contains(row / 3) ? expected[row] : 0f);
            MotionTests.Equal(bits, BitConverter.SingleToInt32Bits(actual[row]));
        }
        MotionTests.Check(kept.Any(v => expected[v * 3] != 0), "Restricted fixture needs nonzero corrections");
        MotionTests.Throws<ArgumentOutOfRangeException>(() => restricted.Restrict(new[] { MhrFixture.Vertices }));
    }

    // Capture lanes share one projection. Like lanes 2..n, the shared instance is cloned from an
    // already restricted source: it holds the same tensor object, stays unrestricted, and runs
    // concurrently with the source without either one's scratch buffers leaking into the other.
    static void SharedProjection()
    {
        var weights = new FlowWeights(MhrFixture.Tensors());
        MhrCorrectives Build() => new(weights, MhrFixture.Indices, MhrFixture.Joints, MhrFixture.Hidden, MhrFixture.Vertices);
        int[] kept = { 1, 3 };
        var source = Build(); source.Restrict(kept);
        var shared = new MhrCorrectives(source);
        MotionTests.Check(ReferenceEquals(source.Projection, shared.Projection), "Lanes must share one corrective projection");
        var restrictedReference = Build(); restrictedReference.Restrict(kept);
        var poseA = DenoiserTests.Wave(MhrFixture.Joints * 7, .23f); var poseB = DenoiserTests.Wave(MhrFixture.Joints * 7, .41f);
        var expectedA = new float[MhrFixture.Vertices * 3]; var expectedB = new float[expectedA.Length];
        restrictedReference.Run(poseA, expectedA); Build().Run(poseB, expectedB);
        MotionTests.Check(expectedB.Count(value => value != 0) > kept.Length * 3, "Shared projection must stay unrestricted");
        void Repeat(MhrCorrectives correctives, float[] pose, float[] expected)
        {
            var actual = new float[expected.Length];
            for (var i = 0; i < 50; i++)
            {
                correctives.Run(pose, actual);
                for (var row = 0; row < expected.Length; row++)
                    MotionTests.Equal(BitConverter.SingleToInt32Bits(expected[row]), BitConverter.SingleToInt32Bits(actual[row]));
            }
        }
        Task.WaitAll(Task.Run(() => Repeat(source, poseA, expectedA)), Task.Run(() => Repeat(shared, poseB, expectedB)));
    }

    internal static void Released(string path)
    {
        using var checkpoint = TorchCheckpoint.OpenScriptData(path, "__torch__.MHRDemo");
        var skeleton = MhrCheckpoint.LoadSkeleton(path); var skinning = MhrCheckpoint.LoadSkinning(path);
        var blends = MhrCheckpoint.LoadBlendShapes(path); var correctives = MhrCheckpoint.LoadCorrectives(path);
        var body = new MhrBody(skeleton, skinning, blends, correctives);
        var identity = new float[body.Identities]; var expression = new float[body.Expressions]; var model = new float[body.ModelParameters];
        var vertices = new float[body.Vertices * 3]; var states = new float[body.Joints * 8];
        body.Run(identity, model, expression, vertices, states);
        var rest = checkpoint.ReadFloat(MhrSkeleton.Prefix + "blend_shape.base_shape");
        for (var i = 0; i < vertices.Length; i++) MotionTests.Near(rest[i], vertices[i], 1e-4);
        var neutral = vertices.ToArray();
        for (var i = 0; i < model.Length; i++) model[i] = .02f * MathF.Sin(i * .37f);
        for (var i = 0; i < identity.Length; i++) identity[i] = .1f * MathF.Sin(i * .13f);
        for (var i = 0; i < expression.Length; i++) expression[i] = .02f * MathF.Sin(i * .27f);
        body.Run(identity, model, expression, vertices, states);
        MotionTests.Check(vertices.All(float.IsFinite), "Real MHR body contains non-finite vertices");
        MotionTests.Check(vertices.Zip(neutral).Any(v => Math.Abs(v.First - v.Second) > .1), "Posed body did not move");
        var first = vertices.ToArray(); body.Run(identity, model, expression, vertices, states);
        MotionTests.Check(first.SequenceEqual(vertices), "Repeated MHR body inference changed outputs");
        var full = new float[skeleton.Parameters]; model.CopyTo(full, 0); var joint = new float[body.Joints * 7]; skeleton.Run(full, joint, states);
        var hidden = new float[correctives.Hidden]; var offsets = new float[vertices.Length]; correctives.Prepare(joint, hidden); correctives.Run(joint, offsets);
        double maximum = 0;
        for (var row = 0; row < offsets.Length; row += 173)
        {
            double expected = 0;
            for (var col = 0; col < hidden.Length; col++) expected += (double)correctives.Projection.Data[row * hidden.Length + col] * hidden[col];
            maximum = Math.Max(maximum, Math.Abs(expected - offsets[row])); MotionTests.Near(expected, offsets[row], 1e-5);
        }
        Console.WriteLine($"PASS: released complete MHR body, neutral base mesh, shaped/posed expression inputs and deterministic repeat; sampled dense-projection max error {maximum:G6}");
    }

    static double[] BlendReference(Dictionary<string, Tensor> tensors, float[] identity, float[] expression, int vertices)
    {
        var basis = tensors[MhrSkeleton.Prefix + "blend_shape.base_shape"].Data;
        var identities = tensors[MhrSkeleton.Prefix + "blend_shape.shape_vectors"].Data;
        var expressions = tensors["face_expressions_model.shape_vectors"].Data; var output = new double[vertices * 3];
        for (var coordinate = 0; coordinate < output.Length; coordinate++)
        {
            output[coordinate] = basis[coordinate];
            for (var i = 0; i < identity.Length; i++) output[coordinate] += (double)identity[i] * identities[i * output.Length + coordinate];
            for (var i = 0; i < expression.Length; i++) output[coordinate] += (double)expression[i] * expressions[i * output.Length + coordinate];
        }
        return output;
    }

    static double[] CorrectiveReference(Dictionary<string, Tensor> tensors, long[] indices, float[] pose, int width, int coordinates)
    {
        var features = new double[(pose.Length / 7 - 2) * 6]; var hidden = new double[width]; var output = new double[coordinates];
        for (var joint = 2; joint < pose.Length / 7; joint++)
        {
            var matrix = MhrSkeletonTests.Multiply(Axis(2, pose[joint * 7 + 5]), MhrSkeletonTests.Multiply(Axis(1, pose[joint * 7 + 4]), Axis(0, pose[joint * 7 + 3])));
            for (var col = 0; col < 2; col++)
                for (var row = 0; row < 3; row++) features[(joint - 2) * 6 + col * 3 + row] = matrix[row, col] - (row == col ? 1 : 0);
        }
        var sparse = tensors[MhrCorrectives.Prefix + "0.sparse_weight"].Data; var projection = tensors[MhrCorrectives.Prefix + "2.weight"].Data;
        for (var i = 0; i < sparse.Length; i++) hidden[indices[i]] += (double)sparse[i] * features[indices[sparse.Length + i]];
        for (var row = 0; row < coordinates; row++)
            for (var col = 0; col < width; col++) output[row] += projection[row * width + col] * Math.Max(0, hidden[col]);
        return output;
    }

    static double[,] Axis(int axis, double angle)
    {
        var m = new double[4, 4]; for (var i = 0; i < 4; i++) m[i, i] = 1;
        var a = (axis + 1) % 3; var b = (axis + 2) % 3;
        m[a, a] = m[b, b] = Math.Cos(angle); m[a, b] = -Math.Sin(angle); m[b, a] = Math.Sin(angle); return m;
    }

    static void Compare(double[] expected, float[] actual, double tolerance)
    { for (var i = 0; i < expected.Length; i++) MotionTests.Near(expected[i], actual[i], tolerance); }
}
