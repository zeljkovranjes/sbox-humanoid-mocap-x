using System;
using System.Collections.Generic;
using System.IO;

namespace HumanoidMocapX.Inference;

/// <summary>Validates named tensors once, before inference. Tensors remain caller-owned
/// and must not be changed after binding them to a model.</summary>
internal sealed class FlowWeights
{
    readonly IReadOnlyDictionary<string, Tensor> tensors;

    internal FlowWeights(IReadOnlyDictionary<string, Tensor> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        this.tensors = tensors;
        foreach (var pair in tensors)
        {
            if (pair.Value is null) throw new InvalidDataException("Missing inference tensor: " + pair.Key);
            foreach (var value in pair.Value.Data)
                if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite inference tensor: " + pair.Key);
        }
    }

    internal Tensor Get(string name, params int[] shape)
    {
        if (!tensors.TryGetValue(name, out var tensor) || !tensor.Shape.AsSpan().SequenceEqual(shape))
            throw new InvalidDataException("Missing or incorrectly shaped inference tensor: " + name);
        return tensor;
    }
}
