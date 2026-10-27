using HumanoidMocap.EditorTools.MotionBricks;

static class SamReference
{
    internal static double[] Block(Dictionary<string, Tensor> weights, string name, double[] tokens, double[] image,
        double[] tokenPosition, double[] imagePosition, int heads, bool firstLayer)
    {
        double[] Norm(double[] values, string suffix) => OmegaReference.Normalize(weights, values, name + suffix, 1e-6);
        static double[] Add(double[] a, double[] b) => a.Zip(b, (x, y) => x + y).ToArray();
        var tokenPe = Norm(tokenPosition, ".ln_pe_1"); var imagePe = Norm(imagePosition, ".ln_pe_2");
        var normalized = Norm(tokens, ".ln1"); var positioned = firstLayer ? normalized : Add(normalized, tokenPe);
        var values = Add(tokens, Attention(weights, name + ".self_attn", positioned, positioned, normalized, heads));
        var normalizedImage = Norm(image, ".ln2_2");
        values = Add(values, Attention(weights, name + ".cross_attn", Add(Norm(values, ".ln2_1"), tokenPe),
            Add(normalizedImage, imagePe), normalizedImage, heads));
        var hidden = RefinerTests.Linear(weights, Norm(values, ".ln3"), name + ".ffn.layers.0.0").Select(OmegaReference.Gelu).ToArray();
        return Add(values, RefinerTests.Linear(weights, hidden, name + ".ffn.layers.1"));
    }

    static double[] Attention(Dictionary<string, Tensor> weights, string name, double[] query, double[] key, double[] value, int heads)
    {
        var q = RefinerTests.Linear(weights, query, name + ".q_proj"); var k = RefinerTests.Linear(weights, key, name + ".k_proj");
        var v = RefinerTests.Linear(weights, value, name + ".v_proj"); var width = weights[name + ".q_proj.weight"].Rows;
        var headWidth = width / heads; var result = new double[q.Length];
        for (var row = 0; row < q.Length / width; row++)
            for (var head = 0; head < heads; head++)
            {
                var scores = Enumerable.Range(0, k.Length / width).Select(other => Enumerable.Range(0, headWidth)
                    .Sum(c => q[row * width + head * headWidth + c] * k[other * width + head * headWidth + c]) / Math.Sqrt(headWidth)).ToArray();
                var maximum = scores.Max(); var probabilities = scores.Select(score => Math.Exp(score - maximum)).ToArray(); var sum = probabilities.Sum();
                for (var c = 0; c < headWidth; c++) result[row * width + head * headWidth + c] = Enumerable.Range(0, probabilities.Length)
                    .Sum(other => probabilities[other] / sum * v[other * width + head * headWidth + c]);
            }
        return RefinerTests.Linear(weights, result, name + ".proj");
    }
}
