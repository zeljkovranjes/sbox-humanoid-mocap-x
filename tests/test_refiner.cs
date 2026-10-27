using HumanoidMocapX.Inference;

static class RefinerTests
{
    const int Width = 8, Heads = 2;
    const string Prefix = "text_refiner.";

    internal static void Run()
    {
        foreach (var frames in new[] { 1, 4 })
        {
            var tensors = Fixture();
            var refiner = new FlowRefiner(new FlowWeights(tensors), frames, Width, Heads);
            var input = Enumerable.Range(0, frames * Width).Select(i => MathF.Sin(i * .7f) + i * .03f).ToArray();
            var output = new float[input.Length];
            foreach (var time in new[] { 0f, .37f, 1f })
            {
                refiner.Run(input, time, output);
                var expected = Reference(tensors, input.Select(v => (double)v).ToArray(), time);
                for (var i = 0; i < output.Length; i++) MotionTests.Near(expected[i], output[i], .000005);
            }
            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < 5; i++) refiner.Run(input, .3f, output);
            MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
            MotionTests.Throws<ArgumentException>(() => refiner.Run(input, .5f, input));
            MotionTests.Throws<ArgumentOutOfRangeException>(() => refiner.Run(input, float.NaN, output));
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            MotionTests.Throws<OperationCanceledException>(() => refiner.Run(input, .5f, output, cancelled.Token));
            input[0] = float.PositiveInfinity;
            MotionTests.Throws<ArgumentException>(() => refiner.Run(input, .5f, output));
        }
        var broken = Fixture(); broken.Remove(Prefix + "input_embedder.bias");
        MotionTests.Throws<InvalidDataException>(() => new FlowRefiner(new FlowWeights(broken), 1, Width, Heads));
        broken = Fixture(); broken[Prefix + "input_embedder.weight"].Data[0] = float.NaN;
        MotionTests.Throws<InvalidDataException>(() => new FlowWeights(broken));
        var embedding = new float[5]; FlowConditioning.Time(.125f, 1000, embedding);
        MotionTests.Near(Math.Cos(125), embedding[0]); MotionTests.Near(Math.Cos(1.25), embedding[1]);
        MotionTests.Near(Math.Sin(125), embedding[2]); MotionTests.Near(Math.Sin(1.25), embedding[3]);
        MotionTests.Near(0, embedding[4]);
        Console.WriteLine("PASS: two-block video refiner against independent double-precision reference, malformed weights and zero-allocation execution");
    }

    internal static Dictionary<string, Tensor> Fixture(int width = Width, int heads = Heads)
    {
        var tensors = new Dictionary<string, Tensor>(); int serial = 1;
        var amplitude = .2f * MathF.Sqrt(Width / (float)width);
        void Add(string name, int[] shape, bool norm = false)
        {
            var count = shape.Aggregate(1, (a, b) => a * b);
            var values = new float[count];
            for (var i = 0; i < count; i++) values[i] = (norm ? .8f : 0) + amplitude * MathF.Sin((serial++ % 103) * .57f);
            tensors.Add(name, new Tensor(name, values, shape));
        }
        void Linear(string name, int inputs, int outputs)
        { Add(name + ".weight", new[] { inputs, outputs }); Add(name + ".bias", new[] { outputs }); }
        void Norm(string name, int width)
        { Add(name + ".weight", new[] { width }, true); Add(name + ".bias", new[] { width }); }
        Linear(Prefix + "input_embedder", width, width);
        foreach (var name in new[] { "context_encoder.linears", "timestep_encoder.blocks" })
        { Linear(Prefix + name + ".0", width, width); Linear(Prefix + name + ".2", width, width); }
        for (var block = 0; block < 2; block++)
        {
            var name = Prefix + "individual_token_refiner.blocks." + block;
            Linear(name + ".self_attn_qkv", width, 3 * width); Linear(name + ".self_attn_proj", width, width);
            Linear(name + ".mlp.fc1", width, 4 * width); Linear(name + ".mlp.fc2", 4 * width, width);
            Linear(name + ".adaLN_modulation.linear", width, 2 * width);
            Norm(name + ".norm1", width); Norm(name + ".norm2", width);
            Norm(name + ".self_attn_q_norm", width / heads); Norm(name + ".self_attn_k_norm", width / heads);
        }
        return tensors;
    }

    internal static double[] Reference(Dictionary<string, Tensor> weights, double[] input, double time)
    {
        var frames = input.Length / Width;
        var mean = new double[Width];
        for (var channel = 0; channel < Width; channel++) mean[channel] = Enumerable.Range(0, frames).Average(f => input[f * Width + channel]);
        var context = Mlp(weights, mean, Prefix + "context_encoder.linears.0", Prefix + "context_encoder.linears.2");
        var embedding = Enumerable.Range(0, Width).Select(i => i < Width / 2
            ? Math.Cos(time * Math.Pow(10000, -2d * i / Width))
            : Math.Sin(time * Math.Pow(10000, -2d * (i - Width / 2) / Width))).ToArray();
        var condition = Mlp(weights, embedding, Prefix + "timestep_encoder.blocks.0", Prefix + "timestep_encoder.blocks.2");
        for (var i = 0; i < Width; i++) condition[i] += context[i];
        var x = Linear(weights, input, Prefix + "input_embedder");
        for (var block = 0; block < 2; block++)
        {
            var name = Prefix + "individual_token_refiner.blocks." + block;
            var gates = Linear(weights, condition.Select(Silu).ToArray(), name + ".adaLN_modulation.linear");
            var qkv = Linear(weights, Normalize(weights, x, name + ".norm1"), name + ".self_attn_qkv");
            var query = new double[x.Length]; var key = new double[x.Length]; var value = new double[x.Length];
            for (var frame = 0; frame < frames; frame++)
                for (var c = 0; c < Width; c++)
                { query[frame * Width + c] = qkv[frame * Width * 3 + c]; key[frame * Width + c] = qkv[frame * Width * 3 + Width + c]; value[frame * Width + c] = qkv[frame * Width * 3 + Width * 2 + c]; }
            query = Normalize(weights, query, name + ".self_attn_q_norm");
            key = Normalize(weights, key, name + ".self_attn_k_norm");
            var attended = Attend(query, key, value);
            var delta = Linear(weights, attended, name + ".self_attn_proj");
            for (var i = 0; i < x.Length; i++) x[i] += delta[i] * gates[i % Width];
            delta = Mlp(weights, Normalize(weights, x, name + ".norm2"), name + ".mlp.fc1", name + ".mlp.fc2");
            for (var i = 0; i < x.Length; i++) x[i] += delta[i] * gates[Width + i % Width];
        }
        return x;
    }

    internal static double[] Linear(Dictionary<string, Tensor> weights, double[] input, string name)
    {
        var weight = weights[name + ".weight"]; var bias = weights[name + ".bias"].Data;
        var rows = input.Length / weight.Cols; var output = new double[rows * weight.Rows];
        for (var row = 0; row < rows; row++)
            for (var channel = 0; channel < weight.Rows; channel++)
            {
                double sum = bias[channel];
                for (var c = 0; c < weight.Cols; c++) sum += input[row * weight.Cols + c] * weight.Data[channel * weight.Cols + c];
                output[row * weight.Rows + channel] = sum;
            }
        return output;
    }

    static double Silu(double x) => x / (1 + Math.Exp(-x));

    static double[] Mlp(Dictionary<string, Tensor> weights, double[] input, string first, string second)
        => Linear(weights, Linear(weights, input, first).Select(Silu).ToArray(), second);

    static double[] Normalize(Dictionary<string, Tensor> weights, double[] input, string name)
    {
        var scale = weights[name + ".weight"].Data; var bias = weights[name + ".bias"].Data;
        var output = new double[input.Length];
        for (var offset = 0; offset < input.Length; offset += scale.Length)
        {
            var row = input.Skip(offset).Take(scale.Length).ToArray();
            var mean = row.Average(); var variance = row.Average(x => (x - mean) * (x - mean));
            for (var i = 0; i < row.Length; i++) output[offset + i] = (row[i] - mean) / Math.Sqrt(variance + 1e-5) * scale[i] + bias[i];
        }
        return output;
    }

    static double[] Attend(double[] query, double[] key, double[] value)
    {
        var frames = query.Length / Width; const int headWidth = Width / Heads;
        var result = new double[query.Length];
        for (var head = 0; head < Heads; head++)
            for (var frame = 0; frame < frames; frame++)
            {
                var scores = new double[frames];
                for (var other = 0; other < frames; other++)
                {
                    double dot = 0;
                    for (var c = 0; c < headWidth; c++) dot += query[frame * Width + head * headWidth + c] * key[other * Width + head * headWidth + c];
                    scores[other] = Math.Exp(dot / Math.Sqrt(headWidth));
                }
                var denominator = scores.Sum();
                for (var c = 0; c < headWidth; c++)
                    for (var other = 0; other < frames; other++)
                        result[frame * Width + head * headWidth + c] += scores[other] / denominator * value[other * Width + head * headWidth + c];
            }
        return result;
    }
}
