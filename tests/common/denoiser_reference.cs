using HumanoidMocap.EditorTools.MotionBricks;
using HumanoidMocapX.Inference;

/// <summary>Allocation-heavy, double-precision equations independent of production operators.</summary>
static class DenoiserReference
{
    const int Width = DenoiserTests.Width, Heads = DenoiserTests.Heads, HeadWidth = Width / Heads;

    internal static double[] Predict(Dictionary<string, Tensor> weights, double[] state, float[] body, float[] camera, float time)
    {
        double[] Linear(double[] x, string name) => RefinerTests.Linear(weights, x, name);
        double[] Mlp(double[] x, string name) => Linear(Linear(x, name + ".0").Select(Silu).ToArray(), name + ".2");
        var context = Add(Linear(body.Select(v => (double)v).ToArray(), "ctxt_encoder.feature"),
            Linear(camera.Select(v => (double)v).ToArray(), "ctxt_encoder.camera_R"));
        var frequencies = Enumerable.Range(0, Width / 2).Select(i => (double)time * 1000 * Math.Pow(10000, -2d * i / Width)).ToArray();
        var embeddedTime = frequencies.Select(Math.Cos).Concat(frequencies.Select(Math.Sin)).ToArray();
        var adapter = Add(Mlp(embeddedTime, "timestep_encoder.blocks"), Mlp(new double[FlowDenoiser.TextChannels], "vtxt_encoder.linears"));
        var streams = new[] { Linear(state, "input_encoder"), RefinerTests.Reference(weights, context, time) };
        var frames = streams[0].Length / Width;
        for (var index = 0; index < FlowDenoiser.DoubleBlocks; index++)
        {
            var q = new double[2 * frames * Width]; var k = new double[q.Length]; var v = new double[q.Length];
            var gates = new double[2][];
            for (var branch = 0; branch < 2; branch++)
            {
                var name = "double_blocks." + index + (branch == 0 ? ".motion" : ".text");
                gates[branch] = Linear(adapter.Select(Silu).ToArray(), name + "_mod.linear");
                var modulated = Modulate(Normalize(streams[branch], Width, false, 1e-6), gates[branch], 0, 1);
                var qkv = Linear(modulated, name + "_qkv");
                SplitQkv(weights, qkv, frames, 3 * Width, name + "_q_norm.weight", name + "_k_norm.weight",
                    q, k, v, branch * frames * Width);
            }
            var attended = Attention(q, k, v, frames);
            for (var branch = 0; branch < 2; branch++)
            {
                var name = "double_blocks." + index + (branch == 0 ? ".motion" : ".text");
                var projected = Linear(attended.Skip(branch * frames * Width).Take(frames * Width).ToArray(), name + "_out_proj");
                streams[branch] = AddGate(streams[branch], projected, gates[branch], 2);
                var normalized = Modulate(Normalize(streams[branch], Width, false, 1e-6), gates[branch], 3, 4);
                projected = Linear(Linear(normalized, name + "_mlp.fc1").Select(Gelu).ToArray(), name + "_mlp.fc2");
                streams[branch] = AddGate(streams[branch], projected, gates[branch], 5);
            }
        }
        var features = streams[0].Concat(streams[1]).ToArray();
        for (var index = 0; index < FlowDenoiser.SingleBlocks; index++)
        {
            var name = "single_blocks." + index;
            var gates = Linear(adapter.Select(Silu).ToArray(), name + ".modulation.linear");
            var modulated = Modulate(Normalize(features, Width, false, 1e-6), gates, 0, 1);
            var packed = Linear(modulated, name + ".linear1");
            var q = new double[features.Length]; var k = new double[q.Length]; var v = new double[q.Length];
            SplitQkv(weights, packed, 2 * frames, 7 * Width, name + ".q_norm.weight", name + ".k_norm.weight", q, k, v, 0);
            var attended = Attention(q, k, v, frames);
            var joined = new double[2 * frames * 5 * Width];
            for (var row = 0; row < 2 * frames; row++)
            {
                Array.Copy(attended, row * Width, joined, row * 5 * Width, Width);
                for (var c = 0; c < 4 * Width; c++) joined[row * 5 * Width + Width + c] = Gelu(packed[row * 7 * Width + 3 * Width + c]);
            }
            features = AddGate(features, Linear(joined, name + ".linear2"), gates, 2);
        }
        var finalGates = Linear(adapter.Select(Silu).ToArray(), "final_layer.adaLN_modulation.linear");
        var final = Modulate(Normalize(features.Take(frames * Width).ToArray(), Width, false, 1e-5), finalGates, 0, 1);
        return Linear(final, "final_layer.linear");
    }

    static void SplitQkv(Dictionary<string, Tensor> weights, double[] packed, int rows, int stride, string qName, string kName,
        double[] query, double[] key, double[] value, int destination)
    {
        var q = new double[rows * Width]; var k = new double[q.Length];
        for (var row = 0; row < rows; row++)
            for (var c = 0; c < Width; c++)
            { q[row * Width + c] = packed[row * stride + c]; k[row * Width + c] = packed[row * stride + Width + c]; value[destination + row * Width + c] = packed[row * stride + 2 * Width + c]; }
        q = Normalize(q, HeadWidth, true, 1e-6); k = Normalize(k, HeadWidth, true, 1e-6);
        for (var i = 0; i < q.Length; i++)
        { query[destination + i] = q[i] * weights[qName].Data[i % HeadWidth]; key[destination + i] = k[i] * weights[kName].Data[i % HeadWidth]; }
    }

    static double[] Attention(double[] q, double[] k, double[] v, int frames)
    {
        var tokens = frames * 2;
        foreach (var values in new[] { q, k })
            for (var token = 0; token < tokens; token++)
                for (var c = 0; c < Width; c += 2)
                {
                    var angle = (token % frames) * Math.Pow(10000, -(double)(c % HeadWidth) / HeadWidth);
                    var at = token * Width + c;
                    var rotated = new System.Numerics.Complex(values[at], values[at + 1]) * System.Numerics.Complex.FromPolarCoordinates(1, angle);
                    values[at] = rotated.Real; values[at + 1] = rotated.Imaginary;
                }
        var result = new double[q.Length];
        for (var row = 0; row < tokens; row++)
            for (var head = 0; head < Heads; head++)
            {
                var probabilities = new double[tokens];
                for (var column = 0; column < tokens; column++)
                {
                    var allowed = row < frames && column < frames && Math.Abs(row - column) <= 60
                        || column >= frames && Math.Abs(row % frames - column % frames) <= 3;
                    if (!allowed) continue;
                    double dot = 0;
                    for (var c = 0; c < HeadWidth; c++) dot += q[row * Width + head * HeadWidth + c] * k[column * Width + head * HeadWidth + c];
                    probabilities[column] = Math.Exp(dot / Math.Sqrt(HeadWidth));
                }
                var sum = probabilities.Sum();
                for (var c = 0; c < HeadWidth; c++)
                    for (var column = 0; column < tokens; column++) result[row * Width + head * HeadWidth + c] += probabilities[column] / sum * v[column * Width + head * HeadWidth + c];
            }
        return result;
    }

    static double[] Normalize(double[] x, int width, bool rms, double epsilon)
    {
        var result = new double[x.Length];
        for (var row = 0; row < x.Length / width; row++)
        {
            var values = x.Skip(row * width).Take(width).ToArray(); var mean = rms ? 0 : values.Average();
            var inverse = 1 / Math.Sqrt(values.Average(v => (v - mean) * (v - mean)) + epsilon);
            for (var c = 0; c < width; c++) result[row * width + c] = (values[c] - mean) * inverse;
        }
        return result;
    }

    static double Silu(double x) => x / (1 + Math.Exp(-x));
    static double Gelu(double x) => .5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + .044715 * x * x * x)));
    static double[] Add(double[] x, double[] y) => x.Zip(y, (a, b) => a + b).ToArray();
    static double[] AddGate(double[] x, double[] delta, double[] gates, int group)
        => x.Select((value, i) => value + delta[i] * gates[group * Width + i % Width]).ToArray();
    static double[] Modulate(double[] x, double[] gates, int shift, int scale)
        => x.Select((value, i) => value * (1 + gates[scale * Width + i % Width]) + gates[shift * Width + i % Width]).ToArray();
}
