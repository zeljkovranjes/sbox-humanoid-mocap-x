using System;
using System.IO;
using System.Threading;
using HumanoidMocap.EditorTools.Inference;
using HumanoidMocapX.Inference;

namespace HumanoidMocapX.EditorTools;

/// <summary>Loads YOLOX-L tensors with the shared non-executing checkpoint reader.
/// Call off the editor thread; verify the downloaded file before loading it.</summary>
internal static class YoloCheckpoint
{
    internal static YoloFilter[] Load(string path, YoloNetwork network, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        using var checkpoint = new TorchCheckpoint(path, "model");
        var result = new YoloFilter[network.Nodes.Length];
        for (var index = 0; index < network.Nodes.Length; index++)
        {
            cancellation.ThrowIfCancellationRequested();
            var node = network.Nodes[index];
            if (node.Operation != YoloOperation.Convolution) continue;
            result[index] = LoadFilter(checkpoint, node, network.Nodes[node.First].Channels, cancellation);
        }
        return result;
    }

    internal static YoloFilter LoadFilter(TorchCheckpoint checkpoint, YoloNode node, int inputs, CancellationToken cancellation)
    {
        var prefix = node.Weight;
        var weights = Read(prefix + (node.Normalize ? ".conv.weight" : ".weight"), node.Channels, inputs, node.Kernel, node.Kernel);
        var bias = node.Normalize ? YoloWeights.Fold(weights,
            Read(prefix + ".bn.weight", node.Channels), Read(prefix + ".bn.bias", node.Channels),
            Read(prefix + ".bn.running_mean", node.Channels), Read(prefix + ".bn.running_var", node.Channels), cancellation)
            : Read(prefix + ".bias", node.Channels);
        return new(weights, bias);

        float[] Read(string name, params int[] shape)
        {
            if (!checkpoint.Tensors.TryGetValue(name, out var tensor) || !tensor.Shape.AsSpan().SequenceEqual(shape))
                throw new InvalidDataException("Missing or incorrectly shaped YOLOX-L tensor: " + name);
            var values = checkpoint.ReadFloat(name, cancellation);
            foreach (var value in values)
                if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite YOLOX-L tensor: " + name);
            return values;
        }
    }
}
