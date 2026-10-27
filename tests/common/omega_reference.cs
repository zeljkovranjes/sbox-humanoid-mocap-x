using System.Numerics;
using HumanoidMocapX.EditorTools.MotionBricks;

static class OmegaReference
{
    internal static double Gelu(double value)
    {
        if (Math.Abs(value) >= 10) return Math.Max(0, value);
        const int intervals = 512;
        var step = Math.Abs(value) / intervals; double integral = 0;
        for (var i = 0; i <= intervals; i++)
            integral += (i == 0 || i == intervals ? 1 : i % 2 == 0 ? 2 : 4) * Math.Exp(-.5 * i * step * i * step);
        var cdf = .5 + Math.Sign(value) * integral * step / (3 * Math.Sqrt(2 * Math.PI));
        return value * cdf;
    }

    internal static double[] Normalize(Dictionary<string, Tensor> weights, double[] values, string name, double epsilon = 1e-5)
    {
        var scale = weights[name + ".weight"].Data; var bias = weights[name + ".bias"].Data;
        var result = new double[values.Length];
        for (var offset = 0; offset < values.Length; offset += scale.Length)
        {
            var row = values.Skip(offset).Take(scale.Length).ToArray(); var mean = row.Average();
            var deviation = Math.Sqrt(row.Average(value => (value - mean) * (value - mean)) + epsilon);
            for (var c = 0; c < scale.Length; c++) result[offset + c] = (row[c] - mean) / deviation * scale[c] + bias[c];
        }
        return result;
    }

    internal static double[] Block(Dictionary<string, Tensor> weights, string name, double[] values, int width, int heads, bool normalizeQk, float[] factors = null, bool swiglu = false)
    {
        var rows = values.Length / width; var headWidth = width / heads;
        var packed = RefinerTests.Linear(weights, Normalize(weights, values, name + ".norm1"), name + ".attn.qkv");
        var mask = weights[name + ".attn.qkv.bias_mask"].Data; var bias = weights[name + ".attn.qkv.bias"].Data;
        for (var i = 0; i < packed.Length; i++) packed[i] -= bias[i % bias.Length] * (1d - mask[i % mask.Length]);
        var q = new double[values.Length]; var k = new double[q.Length]; var v = new double[q.Length];
        for (var row = 0; row < rows; row++)
            for (var channel = 0; channel < width; channel++)
            {
                q[row * width + channel] = packed[row * width * 3 + channel];
                k[row * width + channel] = packed[row * width * 3 + width + channel];
                v[row * width + channel] = packed[row * width * 3 + 2 * width + channel];
            }
        if (normalizeQk) { q = Normalize(weights, q, name + ".attn.q_norm"); k = Normalize(weights, k, name + ".attn.k_norm"); }
        if (factors is not null)
            foreach (var tensor in new[] { q, k })
                for (var row = 0; row < rows; row++)
                    for (var head = 0; head < heads; head++)
                        for (var pair = 0; pair < headWidth / 2; pair++)
                        {
                            var index = row * width + head * headWidth + pair; var other = index + headWidth / 2;
                            var rotated = new Complex(tensor[index], tensor[other]) * new Complex(factors[row * headWidth + pair * 2], factors[row * headWidth + pair * 2 + 1]);
                            tensor[index] = rotated.Real; tensor[other] = rotated.Imaginary;
                        }
        var attended = new double[values.Length];
        for (var row = 0; row < rows; row++)
            for (var head = 0; head < heads; head++)
            {
                var scores = new double[rows];
                for (var other = 0; other < rows; other++)
                    for (var c = 0; c < headWidth; c++) scores[other] += q[row * width + head * headWidth + c] * k[other * width + head * headWidth + c] / Math.Sqrt(headWidth);
                var max = scores.Max(); var probabilities = scores.Select(score => Math.Exp(score - max)).ToArray(); var sum = probabilities.Sum();
                for (var other = 0; other < rows; other++)
                    for (var c = 0; c < headWidth; c++) attended[row * width + head * headWidth + c] += probabilities[other] / sum * v[other * width + head * headWidth + c];
            }
        var projected = RefinerTests.Linear(weights, attended, name + ".attn.proj");
        var result = values.Select((value, index) => value + projected[index] * weights[name + ".ls1.gamma"].Data[index % width]).ToArray();
        var normalized = Normalize(weights, result, name + ".norm2");
        var hidden = RefinerTests.Linear(weights, normalized, name + (swiglu ? ".mlp.w1" : ".mlp.fc1"));
        hidden = swiglu ? hidden.Zip(RefinerTests.Linear(weights, normalized, name + ".mlp.w2"), (value, gate) => value / (1 + Math.Exp(-value)) * gate).ToArray()
            : hidden.Select(Gelu).ToArray();
        projected = RefinerTests.Linear(weights, hidden, name + (swiglu ? ".mlp.w3" : ".mlp.fc2"));
        return result.Select((value, index) => value + projected[index] * weights[name + ".ls2.gamma"].Data[index % width]).ToArray();
    }

    internal static double[] Camera(Dictionary<string, Tensor> weights, float[] input, int frames, int width, int heads)
    {
        var values = Normalize(weights, input.Select(value => (double)value).ToArray(), "camera_head.token_norm");
        for (var index = 0; index < 4; index++) values = Block(weights, "camera_head.trunk." + index, values, width, heads, false);
        var camera = Enumerable.Range(0, frames).SelectMany(frame => values.Skip(frame * 17 * width).Take(width)).ToArray();
        var hidden = RefinerTests.Linear(weights, Normalize(weights, camera, "camera_head.trunk_norm"), "camera_head.camera_branch.0").Select(Gelu).ToArray();
        var output = RefinerTests.Linear(weights, hidden, "camera_head.camera_branch.2");
        for (var frame = 0; frame < frames; frame++)
            for (var c = 7; c < 9; c++) output[frame * 9 + c] = Math.Max(0, output[frame * 9 + c]) + .01;
        return output;
    }
}
