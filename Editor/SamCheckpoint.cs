using System;
using System.IO;
using System.Threading;
using HumanoidMocapX.Inference;

namespace HumanoidMocapX.EditorTools;

internal static class SamCheckpoint
{
    /// <summary>Binds one complete SAM pose head to a separately loaded MHR body.
    /// Reads only numeric checkpoint data. Run off the editor thread.</summary>
    internal static SamPoseHead LoadPoseHead(string path, MhrBody body, bool hand, CancellationToken cancellation = default)
    {
        var weights = VisionCheckpoint.LoadSamHead(path, hand, cancellation);
        var parameters = LoadParameters(path, weights, hand, cancellation);
        cancellation.ThrowIfCancellationRequested();
        return new(new(weights, hand), parameters, new(weights, hand), body);
    }

    /// <summary>Reads exact integer hand indices and binds pose conversion weights.
    /// Rejects missing, incompatible or out-of-range checkpoint indices.</summary>
    internal static SamMhrParameters LoadParameters(string path, FlowWeights weights, bool hand, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        using var checkpoint = new TorchCheckpoint(path);
        var name = hand ? "head_pose_hand." : "head_pose.";
        var left = Read("hand_joint_idxs_left", 27, SamMhrParameters.PoseParameters);
        var right = Read("hand_joint_idxs_right", 27, SamMhrParameters.PoseParameters);
        var nonhand = Read("nonhand_param_idxs", 145, SamMhrParameters.ModelParameters);
        cancellation.ThrowIfCancellationRequested();
        return new(weights, left, right, nonhand, hand);

        int[] Read(string suffix, int count, int limit)
        {
            var key = name + suffix;
            if (!checkpoint.Tensors.TryGetValue(key, out var info) || info.Shape.Length != 1 || info.Shape[0] != count)
                throw new InvalidDataException("Missing or incorrectly shaped SAM index tensor: " + key);
            var values = checkpoint.ReadInt64(key, cancellation); var indices = new int[count];
            for (var i = 0; i < count; i++)
            {
                if (values[i] < 0 || values[i] >= limit) throw new InvalidDataException("SAM parameter index is out of range: " + key);
                indices[i] = (int)values[i];
            }
            return indices;
        }
    }
}
