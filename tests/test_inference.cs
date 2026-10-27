using System.Diagnostics;
using HumanoidMocapX.Inference;
using HumanoidMocapX;

static class InferenceTests
{
    internal static void Run()
    {
        Linear();
        ParallelLinear();
        Attention();
        Sampling();
        Allocations();
        Console.WriteLine("PASS: linear buffers, attention masks/head isolation, twenty-step sampling and allocation checks");
    }

    // Large enough for the multithreaded path; it must match the serial layer bit for bit.
    static void ParallelLinear()
    {
        const int width = 640, channels = 96, rows = 70;
        var weight = new Tensor("big.weight", Enumerable.Range(0, width * channels).Select(i => MathF.Sin(i * .37f)).ToArray(), new[] { width, channels });
        var bias = new Tensor("big.bias", Enumerable.Range(0, channels).Select(i => MathF.Cos(i * .11f)).ToArray(), new[] { channels });
        var linear = new FlowLinear(new FlowWeights(new Dictionary<string, Tensor> { [weight.Name] = weight, [bias.Name] = bias }), "big", width, channels);
        var input = Enumerable.Range(0, rows * width).Select(i => MathF.Sin(i * .013f)).ToArray();
        float[] serial = new float[rows * channels], parallel = new float[serial.Length];
        linear.Run(input, rows, serial); linear.RunParallel(input, rows, parallel);
        MotionTests.Check(serial.Select(BitConverter.SingleToInt32Bits).SequenceEqual(parallel.Select(BitConverter.SingleToInt32Bits)),
            "Parallel linear differs from the serial layer");
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => linear.RunParallel(input, rows, parallel, cancelled.Token));
    }

    static void Linear()
    {
        var weight = new Tensor("test", new[] { 1f, 0f, -1f, 2f, -1f, .5f }, new[] { 3, 2 });
        var bias = new Tensor("bias", new[] { .25f, -.5f }, new[] { 2 });
        float[] input = { 1, 2, 3, -1, 0, 2 };
        var output = new float[4];
        FlowLinear.Run(input, 2, weight, bias, output);
        float[] expected = { -1.75f, 1f, -2.75f, -1.5f };
        for (var i = 0; i < output.Length; i++) MotionTests.Near(expected[i], output[i]);
        MotionTests.Throws<ArgumentException>(() => FlowLinear.Run(input, 1, weight, bias, output));
        var square = new Tensor("identity", new[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });
        MotionTests.Throws<ArgumentException>(() => FlowLinear.Run(output, 2, square, null, output));
    }

    static void Attention()
    {
        const int frames = 71, width = 8, heads = 2;
        foreach (var joint in new[] { false, true })
        {
            var tokens = frames * (joint ? 2 : 1);
            var length = tokens * width;
            var q = Enumerable.Range(0, length).Select(i => MathF.Sin(i * .13f)).ToArray();
            var k = Enumerable.Range(0, length).Select(i => MathF.Cos(i * .07f)).ToArray();
            var v = Enumerable.Range(0, length).Select(i => i % 19 - 9f).ToArray();
            var output = new float[length];
            var attention = new FlowAttention(frames, width, heads, joint);
            var ranges = new uint[tokens * 4]; attention.CopyKeyRanges(ranges);
            for (var row = 0; row < tokens; row++)
                for (var column = 0; column < tokens; column++)
                {
                    var allowed = !joint || (row < frames && column < frames && Math.Abs(row - column) <= 60)
                        || (column >= frames && Math.Abs(row % frames - column % frames) <= 3);
                    var inRange = column >= ranges[row * 4] && column < ranges[row * 4] + ranges[row * 4 + 1]
                        || column >= ranges[row * 4 + 2] && column < ranges[row * 4 + 2] + ranges[row * 4 + 3];
                    MotionTests.Equal(allowed, inRange);
                }
            MotionTests.Throws<ArgumentException>(() => attention.CopyKeyRanges(new uint[1]));
            attention.Run(q, k, v, output);
            var expected = DenseReference(q, k, v, frames, width, heads, joint);
            for (var i = 0; i < length; i++) MotionTests.Near(expected[i], output[i], .00001);
            MotionTests.Throws<ArgumentException>(() => attention.Run(q, k, v, q));
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            MotionTests.Throws<OperationCanceledException>(() => attention.Run(q, k, v, output, cancelled.Token));
        }

        // A video token must never see motion values, even when all attention scores are equal.
        var zero = new float[frames * 2];
        var impulse = new float[frames * 2]; impulse[60] = 1; impulse[61] = 100;
        impulse[frames + 3] = 10; impulse[frames + 4] = 1000;
        var result = new float[frames * 2];
        new FlowAttention(frames, 1, 1, true).Run(zero, zero, impulse, result);
        MotionTests.Near(11d / 65, result[0]);
        MotionTests.Near(10d / 4, result[frames]);
    }

    static float[] DenseReference(float[] q, float[] k, float[] v, int frames, int width, int heads, bool joint)
    {
        var count = q.Length / width; var headWidth = width / heads;
        var result = new float[q.Length];
        for (var row = 0; row < count; row++)
            for (var head = 0; head < heads; head++)
            {
                var probabilities = new double[count]; double sum = 0;
                for (var col = 0; col < count; col++)
                {
                    var allowed = !joint || (row < frames && col < frames && Math.Abs(row - col) <= 60)
                        || (col >= frames && Math.Abs(row % frames - col % frames) <= 3);
                    if (!allowed) continue;
                    double score = 0;
                    for (var channel = 0; channel < headWidth; channel++)
                        score += q[row * width + head * headWidth + channel] * k[col * width + head * headWidth + channel];
                    probabilities[col] = Math.Exp(score / Math.Sqrt(headWidth)); sum += probabilities[col];
                }
                for (var channel = 0; channel < headWidth; channel++)
                {
                    double value = 0;
                    for (var col = 0; col < count; col++) value += probabilities[col] / sum * v[col * width + head * headWidth + channel];
                    result[row * width + head * headWidth + channel] = (float)value;
                }
            }
        return result;
    }

    static void Sampling()
    {
        var state = Enumerable.Repeat(-2f, FlowSampler.Channels).ToArray();
        var scratch = new float[state.Length]; int calls = 0;
        void Predict(ReadOnlySpan<float> values, float time, Span<float> output, CancellationToken token)
        {
            MotionTests.Near(calls / 20d, time);
            MotionTests.Near(-2 + 6 * time, values[0]);
            output.Fill(4); calls++;
        }
        FlowSampler.Sample(state, scratch, Predict);
        MotionTests.Equal(20, calls);
        foreach (var value in state) MotionTests.Near(4, value);
        MotionTests.Throws<ArgumentException>(() => FlowSampler.Sample(state, state, Predict));
        MotionTests.Throws<ArgumentException>(() => FlowSampler.Sample(new float[1], new float[1], Predict));
        state[0] = float.NaN;
        MotionTests.Throws<ArithmeticException>(() => FlowSampler.Sample(state, scratch, Predict));
        state[0] = 0;
        MotionTests.Throws<ArithmeticException>(() => FlowSampler.Sample(state, scratch,
            static (_, _, output, _) => output.Fill(float.PositiveInfinity)));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => FlowSampler.Sample(state, scratch, Predict, cancelled.Token));
        MotionTests.Equal(20, calls);
    }

    static void Allocations()
    {
        var weight = new Tensor("benchmark", Enumerable.Repeat(.125f, 256 * 128).ToArray(), new[] { 256, 128 });
        var input = new float[8 * 256]; var output = new float[8 * 128];
        var attention = new FlowAttention(16, 8, 2, true);
        var qkv = new float[32 * 8]; var attended = new float[qkv.Length];
        var state = new float[FlowSampler.Channels]; var predicted = new float[state.Length];
        FlowPrediction predictor = static (_, _, result, _) => result.Fill(1);
        NeuralOps.Linear(input, 8, weight, null);
        FlowLinear.Run(input, 8, weight, null, output);
        attention.Run(qkv, qkv, qkv, attended);
        FlowSampler.Sample(state, predicted, predictor);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var start = Stopwatch.GetTimestamp();
        for (var i = 0; i < 10; i++) NeuralOps.Linear(input, 8, weight, null);
        var originalTime = Stopwatch.GetElapsedTime(start);
        var originalBytes = GC.GetAllocatedBytesForCurrentThread() - before;
        before = GC.GetAllocatedBytesForCurrentThread(); start = Stopwatch.GetTimestamp();
        for (var i = 0; i < 10; i++) FlowLinear.Run(input, 8, weight, null, output);
        var bufferedTime = Stopwatch.GetElapsedTime(start);
        var bufferedBytes = GC.GetAllocatedBytesForCurrentThread() - before;
        before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 10; i++) { attention.Run(qkv, qkv, qkv, attended); FlowSampler.Sample(state, predicted, predictor); }
        var loopBytes = GC.GetAllocatedBytesForCurrentThread() - before;
        MotionTests.Equal(0L, bufferedBytes); MotionTests.Equal(0L, loopBytes);
        Console.WriteLine($"Linear check, 10 x [8,256]→128: original {originalTime.TotalMilliseconds:F2} ms/{originalBytes} caller-thread bytes; buffered {bufferedTime.TotalMilliseconds:F2} ms/{bufferedBytes} bytes");
    }
}
