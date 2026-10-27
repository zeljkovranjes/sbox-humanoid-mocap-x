using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using HumanoidMocapX.Inference;

namespace HumanoidMocapX.EditorTools;

/// <summary>Reads inference tensors through Humanoid Mocap's restricted checkpoint
/// reader. Loading is CPU/disk work and must run off the editor thread.</summary>
internal static class FlowCheckpoint
{
    const string Prefix = "motion_transformer.";
    internal const int TensorCount = 304;
    const long MaximumElements = 600_000_000;

    /// <summary>Loads named weights without executing pickle code. The caller must first
    /// verify the checkpoint with ModelStore. Shape checks happen when binding a denoiser.</summary>
    internal static FlowWeights Load(string path, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        using var checkpoint = new TorchCheckpoint(path, "model_state_dict");
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        long elements = 0; int count = 0;
        foreach (var pair in checkpoint.Tensors)
        {
            if (!pair.Key.StartsWith(Prefix, StringComparison.Ordinal)) continue;
            var shape = pair.Value.Shape;
            if (shape.Length is < 1 or > 2) throw new InvalidDataException("Unexpected FlowHMR parameter rank: " + pair.Key);
            long size = 1;
            foreach (var dimension in shape)
            {
                if (dimension < 1) throw new InvalidDataException("Empty FlowHMR parameter: " + pair.Key);
                size = checked(size * dimension);
            }
            elements = checked(elements + size); count++;
        }
        if (count != TensorCount || elements > MaximumElements)
            throw new InvalidDataException("FlowHMR checkpoint does not contain the expected 304 denoiser tensors within the size limit.");
        foreach (var pair in checkpoint.Tensors)
        {
            cancellation.ThrowIfCancellationRequested();
            if (!pair.Key.StartsWith(Prefix, StringComparison.Ordinal)) continue;
            var name = pair.Key[Prefix.Length..];
            var shape = pair.Value.Shape.AsSpan().ToArray(); Array.Reverse(shape);
            tensors.Add(name, new Tensor(name, checkpoint.ReadFloat(pair.Key, cancellation), shape));
        }
        cancellation.ThrowIfCancellationRequested();
        return new FlowWeights(tensors);
    }

    /// <summary>Loads normalization from a verified checkpoint and regresses joints from
    /// the separate SMPL-H NPZ asset. Rejects missing statistics or incompatible arrays.</summary>
    internal static FlowDecoder LoadDecoder(string path, string bodyPath, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        using var checkpoint = new TorchCheckpoint(path, "model_state_dict");
        var mean = new float[FlowSampler.Channels]; var deviation = new float[FlowSampler.Channels];
        CopyStatistics("smooth_root_vel", 0, new[] { 1, 1, 3 }, new[] { 0, 2 });
        CopyStatistics("smooth_root_pos", 2, new[] { 1, 1, 3 }, new[] { 1 });
        CopyStatistics("local_joints_positions", FlowDecoder.JointOffset, new[] { 1, 1, 52, 3 });
        CopyStatistics("local_rot_data", FlowDecoder.RotationOffset, new[] { 1, 1, 52, 6 });
        CopyStatistics("rep_shapes", FlowDecoder.ShapeOffset, new[] { 1, 1, 16 });
        CopyStatistics("rep_foot_contacts", FlowDecoder.ContactOffset, new[] { 1, 1, 4 });
        var joints = SmplModel.Load(bodyPath, cancellation);
        cancellation.ThrowIfCancellationRequested();
        return new FlowDecoder(mean, deviation, joints.Template, joints.Directions);

        void CopyStatistics(string name, int destination, int[] shape, int[] selection = null)
        {
            var sourceMean = ReadTensor(checkpoint, name + "_mean", cancellation, shape);
            var sourceDeviation = ReadTensor(checkpoint, name + "_std", cancellation, shape);
            var count = selection?.Length ?? sourceMean.Length;
            for (var index = 0; index < count; index++)
            {
                var source = selection is null ? index : selection[index];
                mean[destination + index] = sourceMean[source];
                deviation[destination + index] = sourceDeviation[source];
            }
        }
    }

    static float[] ReadTensor(TorchCheckpoint checkpoint, string name, CancellationToken cancellation, params int[] shape)
    {
        if (!checkpoint.Tensors.TryGetValue(name, out var tensor) || !tensor.Shape.AsSpan().SequenceEqual(shape))
            throw new InvalidDataException("Missing or incorrectly shaped FlowHMR decoder buffer: " + name);
        var values = checkpoint.ReadFloat(name, cancellation);
        foreach (var value in values)
            if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite FlowHMR decoder buffer: " + name);
        return values;
    }
}
