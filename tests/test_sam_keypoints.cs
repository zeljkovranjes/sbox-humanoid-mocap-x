using HumanoidMocapX.EditorTools.MotionBricks;
using HumanoidMocapX.Inference;

static class SamKeypointTests
{
    internal static void Run()
    {
        const int points = SamKeypoints.Vertices + SamKeypoints.Joints;
        foreach (var hand in new[] { false, true })
        {
            var name = (hand ? "head_pose_hand." : "head_pose.") + "keypoint_mapping";
            var matrix = new float[points * 308];
            for (var i = 0; i < 308; i++)
            {
                matrix[i * points + i * 37 % SamKeypoints.Vertices] = .7f;
                matrix[i * points + SamKeypoints.Vertices + i % SamKeypoints.Joints] = -.2f;
            }
            var mapper = new SamKeypoints(new(new Dictionary<string, Tensor> { [name] = new(name, matrix, new[] { points, 308 }) }), hand);
            var vertices = DenoiserTests.Wave(SamKeypoints.Vertices * 3, .17f);
            var states = DenoiserTests.Wave(SamKeypoints.Joints * 8, .23f); var output = new float[70 * 3];
            mapper.Run(vertices, states, output);
            for (var i = 0; i < 70; i++)
                for (var axis = 0; axis < 3; axis++)
                {
                    var expected = hand && (i < 21 || i >= 42) ? 0 : (.7 * vertices[(i * 37 % SamKeypoints.Vertices) * 3 + axis]
                        - .2 * states[(i % SamKeypoints.Joints) * 8 + axis]) / 100 * (axis == 0 ? 1 : -1);
                    MotionTests.Near(expected, output[i * 3 + axis], 2e-9);
                }
            // Unmapped vertices must not change any keypoint bit, which lets MHR skip their correctives.
            var mapped = mapper.MappedVertices();
            MotionTests.Check(mapped.SequenceEqual(Enumerable.Range(0, 70).Select(i => i * 37 % SamKeypoints.Vertices).Distinct().Order()),
                "Keypoint mapping reported the wrong vertices");
            var perturbed = vertices.ToArray(); var again = new float[output.Length];
            for (var vertex = 0; vertex < SamKeypoints.Vertices; vertex++)
                if (Array.BinarySearch(mapped, vertex) < 0) for (var axis = 0; axis < 3; axis++) perturbed[vertex * 3 + axis] = -3.5f - axis;
            mapper.Run(perturbed, states, again);
            MotionTests.Check(output.Select(BitConverter.SingleToInt32Bits).SequenceEqual(again.Select(BitConverter.SingleToInt32Bits)),
                "Unmapped vertices changed keypoints");
            for (var i = 0; i < 4; i++) mapper.Run(vertices, states, output);
            var before = GC.GetAllocatedBytesForCurrentThread(); mapper.Run(vertices, states, output);
            MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
            MotionTests.Throws<ArgumentException>(() => mapper.Run(vertices.AsSpan(1), states, output));
            MotionTests.Throws<ArgumentException>(() => mapper.Run(vertices, states, states.AsSpan(0, output.Length)));
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            MotionTests.Throws<OperationCanceledException>(() => mapper.Run(vertices, states, output, cancelled.Token));
            states[0] = float.NaN; MotionTests.Throws<ArgumentException>(() => mapper.Run(vertices, states, output));
        }
        Console.WriteLine("PASS: SAM mesh/joint landmark regression, metre/camera conversion, right-hand visibility and allocation-free reuse");
    }
}
