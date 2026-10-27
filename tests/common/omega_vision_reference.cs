using HumanoidMocap.EditorTools.MotionBricks;

static class OmegaVisionReference
{
    internal static double[] Image(Dictionary<string, Tensor> weights, float[] input, int height, int width, int channels, int heads, bool sam = false)
    {
        var name = sam ? "backbone.encoder" : "aggregator.patch_embed";
        var patches = height / 16 * (width / 16); var values = new double[(patches + 5) * channels];
        Array.Copy(weights[name + ".cls_token"].Data.Select(v => (double)v).ToArray(), values, channels);
        Array.Copy(weights[name + ".storage_tokens"].Data.Select(v => (double)v).ToArray(), 0, values, channels, channels * 4);
        var filter = weights[name + ".patch_embed.proj.weight"].Data; var bias = weights[name + ".patch_embed.proj.bias"].Data;
        double[] mean = { .485, .456, .406 }, deviation = { .229, .224, .225 };
        for (var y = 0; y < height / 16; y++)
            for (var x = 0; x < width / 16; x++)
                for (var output = 0; output < channels; output++)
                {
                    double sum = bias[output];
                    for (var c = 0; c < 3; c++)
                        for (var ky = 0; ky < 16; ky++)
                            for (var kx = 0; kx < 16; kx++)
                            {
                                var pixel = input[((y * 16 + ky) * width + x * 16 + kx) * 3 + c];
                                sum += (pixel - mean[c]) / deviation[c] * filter[((output * 3 + c) * 16 + ky) * 16 + kx];
                            }
                    values[((y * (width / 16) + x) + 5) * channels + output] = sum;
                }
        var positions = Positions(height / 16, width / 16, 5, weights[name + ".rope_embed.periods"].Data, sam);
        for (var i = 0; i < (sam ? 32 : 24); i++) values = OmegaReference.Block(weights, name + ".blocks." + i, values, channels, heads, false, positions, sam);
        return OmegaReference.Normalize(weights, values.Skip(channels * 5).ToArray(), name + ".norm");
    }

    internal static double[] Aggregate(Dictionary<string, Tensor> weights, double[][] patches, int height, int width, int channels, int heads)
    {
        var tokensPerFrame = height / 16 * (width / 16) + 17;
        var values = new double[patches.Length * tokensPerFrame * channels];
        for (var frame = 0; frame < patches.Length; frame++)
        {
            var prefix = weights["aggregator.camera_token"].Data.Skip(frame == 0 ? 0 : channels).Take(channels)
                .Concat(weights["aggregator.register_token"].Data.Skip(frame == 0 ? 0 : 16 * channels).Take(16 * channels)).Select(v => (double)v).ToArray();
            Array.Copy(prefix, 0, values, frame * tokensPerFrame * channels, prefix.Length);
            Array.Copy(patches[frame], 0, values, (frame * tokensPerFrame + 17) * channels, patches[frame].Length);
        }
        var positions = Positions(height / 16, width / 16, 17, weights["aggregator.rope_embed.periods"].Data);
        double[] lastFrame = null;
        for (var layer = 0; layer < 24; layer++)
        {
            for (var frame = 0; frame < patches.Length; frame++)
            {
                var frameValues = values.Skip(frame * tokensPerFrame * channels).Take(tokensPerFrame * channels).ToArray();
                var result = OmegaReference.Block(weights, "aggregator.frame_blocks." + layer, frameValues, channels, heads, true, positions);
                Array.Copy(result, 0, values, frame * tokensPerFrame * channels, result.Length);
            }
            if (layer == 23) lastFrame = Prefixes();
            if (new[] { 2, 6, 9, 14, 20 }.Contains(layer))
            {
                var result = OmegaReference.Block(weights, "aggregator.inter_frame_blocks." + layer, Prefixes(), channels, heads, true);
                for (var frame = 0; frame < patches.Length; frame++) Array.Copy(result, frame * 17 * channels, values, frame * tokensPerFrame * channels, 17 * channels);
            }
            else values = OmegaReference.Block(weights, "aggregator.inter_frame_blocks." + layer, values, channels, heads, true);
        }
        var lastGlobal = Prefixes();
        return Enumerable.Range(0, patches.Length * 17).SelectMany(token => lastFrame.Skip(token * channels).Take(channels)
            .Concat(lastGlobal.Skip(token * channels).Take(channels))).ToArray();

        double[] Prefixes() => Enumerable.Range(0, patches.Length).SelectMany(frame => values.Skip(frame * tokensPerFrame * channels).Take(17 * channels)).ToArray();
    }

    static float[] Positions(int height, int width, int prefix, float[] periods, bool separateAxes = false)
    {
        var channels = periods.Length * 4; var result = new float[(height * width + prefix) * channels];
        for (var row = 0; row < height * width + prefix; row++)
            for (var pair = 0; pair < channels / 2; pair++)
            {
                var coordinate = pair < periods.Length ? (row - prefix) / width : (row - prefix) % width;
                var extent = separateAxes ? (pair < periods.Length ? height : width) : Math.Max(height, width);
                var angle = row < prefix ? 0 : 2 * Math.PI * (2 * (coordinate + .5) / extent - 1) / periods[pair % periods.Length];
                result[row * channels + pair * 2] = (float)Math.Cos(angle); result[row * channels + pair * 2 + 1] = (float)Math.Sin(angle);
            }
        return result;
    }
}
