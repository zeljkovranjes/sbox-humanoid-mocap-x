using HumanoidMocapX.Inference;
using Sandbox;

namespace HumanoidMocapX.Checks;

/// <summary>Checks a complete YOLOX-L GPU pass using sparse synthetic filters and an
/// independent CPU evaluation. Does not validate published checkpoint accuracy.</summary>
public static class YoloGpuChecks
{
    /// <summary>Compares all 8400 prediction rows, tests input rejection and cancellation,
    /// and throws on disagreement. Run through trusted automation in a live editor.</summary>
    public static async Task<object> Run(CancellationToken cancellation)
    {
        var network = new YoloNetwork();
        var input = new float[640 * 640 * 3];
        for (var i = 0; i < input.Length; i++) input[i] = 128 + 100 * MathF.Sin(i * .013f);
        var filters = new YoloFilter[network.Nodes.Length];
        for (var i = 0; i < filters.Length; i++)
        {
            var node = network.Nodes[i];
            if (node.Operation != YoloOperation.Convolution) continue;
            var channels = network.Nodes[node.First].Channels;
            var weights = new float[node.Channels * channels * node.Kernel * node.Kernel];
            var bias = new float[node.Channels];
            for (var c = 0; c < node.Channels; c++)
            {
                weights[WeightIndex(node, channels, c)] = .2f + .01f * MathF.Sin(c);
                bias[c] = .01f * MathF.Cos(c + i);
            }
            filters[i] = new(weights, bias);
        }
        var expected = await Task.Run(() => Reference(network, filters, input, cancellation), cancellation);
        await GameTask.MainThread();
        var detector = new YoloGpuDetector(network, filters);
        try
        {
            var actual = new float[expected.Length];
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            var observed = false;
            try { await detector.PredictAsync(input, actual, cancelled.Token); }
            catch (OperationCanceledException) { observed = true; }
            if (!observed) throw new Exception("YOLOX ignored pre-cancellation.");
            try { await detector.PredictAsync(input, new float[1], cancellation); throw new Exception("YOLOX accepted a short output."); }
            catch (ArgumentException) { }
            await detector.PredictAsync(input, actual, cancellation);
            double largestError = 0;
            for (var i = 0; i < actual.Length; i++)
            {
                var error = Math.Abs((double)actual[i] - expected[i]); largestError = Math.Max(largestError, error);
                if (!float.IsFinite(actual[i]) || error > 1e-4 * Math.Max(1, Math.Abs(expected[i])))
                    throw new Exception($"YOLOX output {i}: expected {expected[i]:R}, got {actual[i]:R}.");
            }
            return new { Predictions = YoloNetwork.Predictions, LargestAbsoluteError = largestError };
        }
        finally { await GameTask.MainThread(); detector.Dispose(); }
    }

    static int WeightIndex(YoloNode node, int channels, int output)
        => ((output * channels + (output * 7 + 3) % channels) * node.Kernel + node.Kernel / 2) * node.Kernel + node.Kernel / 2;

    static float[] Reference(YoloNetwork network, YoloFilter[] filters, float[] input, CancellationToken cancellation)
    {
        var slots = network.Capacities.Select(count => new float[count]).ToArray();
        input.CopyTo(slots[network.Slots[0]], 0);
        for (var index = 1; index < network.Nodes.Length; index++)
        {
            cancellation.ThrowIfCancellationRequested();
            var node = network.Nodes[index]; var source = network.Nodes[node.First];
            var first = slots[network.Slots[node.First]]; var destination = slots[network.Slots[index]];
            var second = node.Second >= 0 ? slots[network.Slots[node.Second]] : null;
            for (var y = 0; y < node.Size; y++)
                for (var x = 0; x < node.Size; x++)
                    for (var channel = 0; channel < node.Channels; channel++)
                    {
                        var at = (y * node.Size + x) * node.Channels + channel;
                        switch (node.Operation)
                        {
                            case YoloOperation.Convolution:
                                var filter = filters[index];
                                var from = ((y * node.Stride) * source.Size + x * node.Stride) * source.Channels + (channel * 7 + 3) % source.Channels;
                                destination[at] = (float)((double)first[from] * filter.Weight[WeightIndex(node, source.Channels, channel)] + filter.Bias[channel]); break;
                            case YoloOperation.Silu: destination[at] = (float)(first[at] / (1 + Math.Exp(-first[at]))); break;
                            case YoloOperation.Add: destination[at] = first[at] + second[at]; break;
                            case YoloOperation.Concat:
                                var secondChannels = node.Channels - source.Channels;
                                destination[at] = channel < source.Channels ? first[(y * node.Size + x) * source.Channels + channel]
                                    : second[(y * node.Size + x) * secondChannels + channel - source.Channels]; break;
                            case YoloOperation.Focus:
                                var slice = channel / source.Channels;
                                destination[at] = first[((y * 2 + slice % 2) * source.Size + x * 2 + slice / 2) * source.Channels + channel % source.Channels]; break;
                            case YoloOperation.Upsample:
                                destination[at] = first[((y / 2) * source.Size + x / 2) * source.Channels + channel]; break;
                            case YoloOperation.MaxPool:
                                var maximum = float.NegativeInfinity;
                                for (var dy = -node.Kernel / 2; dy <= node.Kernel / 2; dy++)
                                    for (var dx = -node.Kernel / 2; dx <= node.Kernel / 2; dx++)
                                        if (x + dx >= 0 && x + dx < source.Size && y + dy >= 0 && y + dy < source.Size)
                                            maximum = Math.Max(maximum, first[((y + dy) * source.Size + x + dx) * source.Channels + channel]);
                                destination[at] = maximum; break;
                            default: throw new Exception("Unrecognized reference operation.");
                        }
                    }
        }
        var output = new float[YoloNetwork.Predictions * YoloNetwork.PredictionChannels]; var offset = 0;
        foreach (var head in network.Heads)
        {
            var length = network.Nodes[head].Elements;
            slots[network.Slots[head]].AsSpan(0, length).CopyTo(output.AsSpan(offset)); offset += length;
        }
        return output;
    }
}
