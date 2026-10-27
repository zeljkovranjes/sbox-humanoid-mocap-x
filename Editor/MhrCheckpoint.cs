using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using HumanoidMocap.EditorTools.Inference;
using HumanoidMocap.EditorTools.MotionBricks;
using HumanoidMocapX.Inference;

namespace HumanoidMocapX.EditorTools;

internal static class MhrCheckpoint
{
    internal const int Joints = 127, Parameters = 249, Vertices = 18439, Identities = 45, Expressions = 72, CorrectiveHidden = 3000;

    /// <summary>Loads the complete numeric MHR body model without executing archive
    /// code. Run off the editor thread; needs the full corrective-weight allocation.</summary>
    internal static MhrBody LoadBody(string path, CancellationToken cancellation = default) => LoadBody(path, null, cancellation);

    /// <summary>Like LoadBody, but reuses <paramref name="shared"/>'s immutable corrective projection
    /// (~664 MB) instead of reading another copy; null reads it. Each body keeps its own buffers.</summary>
    internal static MhrBody LoadBody(string path, MhrCorrectives shared, CancellationToken cancellation)
        => new(LoadSkeleton(path, cancellation), LoadSkinning(path, cancellation), LoadBlendShapes(path, cancellation),
            shared is null ? LoadCorrectives(path, cancellation) : new MhrCorrectives(shared));

    /// <summary>Reads the released MHR skeleton's numeric data without executing
    /// TorchScript. Run off the editor thread. Rejects incompatible shapes and parents.</summary>
    internal static MhrSkeleton LoadSkeleton(string path, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        using var checkpoint = TorchCheckpoint.OpenScriptData(path, "__torch__.MHRDemo");
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        Read("parameter_transform.parameter_transform", Joints * MhrSkeleton.JointParameters, Parameters);
        Read("skeleton.joint_translation_offsets", Joints, 3);
        Read("skeleton.joint_prerotations", Joints, 4);
        var parentName = MhrSkeleton.Prefix + "skeleton.joint_parents";
        RequireShape(checkpoint, parentName, Joints);
        var source = checkpoint.ReadInt64(parentName, cancellation); var parents = new int[Joints];
        for (var joint = 0; joint < Joints; joint++)
        {
            if (source[joint] < -1 || source[joint] >= joint) throw new InvalidDataException("MHR joint parent is out of order or range.");
            parents[joint] = (int)source[joint];
        }
        cancellation.ThrowIfCancellationRequested();
        return new(new(tensors), parents, Parameters);

        void Read(string suffix, int rows, int columns)
        {
            var name = MhrSkeleton.Prefix + suffix;
            RequireShape(checkpoint, name, rows, columns);
            tensors.Add(name, new(name, checkpoint.ReadFloat(name, cancellation), new[] { columns, rows }));
        }
    }

    /// <summary>Reads MHR's inverse bind states and exact integer influence indices.
    /// Rejects incompatible or out-of-range data; run off the editor thread.</summary>
    internal static MhrSkinning LoadSkinning(string path, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        using var checkpoint = TorchCheckpoint.OpenScriptData(path, "__torch__.MHRDemo");
        const string prefix = MhrSkeleton.Prefix + "linear_blend_skinning.";
        var weightName = prefix + "skin_weights_flattened";
        if (!checkpoint.Tensors.TryGetValue(weightName, out var weights) || weights.Shape.Length != 1
            || weights.Shape[0] < 1 || weights.Shape[0] > Vertices * 8)
            throw new InvalidDataException("MHR skinning influence count is invalid.");
        RequireShape(checkpoint, prefix + "inverse_bind_pose", Joints, MhrState.Elements);
        var joints = Indices("skin_indices_flattened", Joints); var vertices = Indices("vert_indices_flattened", Vertices);
        var bind = checkpoint.ReadFloat(prefix + "inverse_bind_pose", cancellation);
        var values = checkpoint.ReadFloat(weightName, cancellation);
        cancellation.ThrowIfCancellationRequested();
        return new(bind, joints, vertices, values, Vertices);

        int[] Indices(string suffix, int limit)
        {
            var name = prefix + suffix; RequireShape(checkpoint, name, weights.Shape[0]);
            var source = checkpoint.ReadInt64(name, cancellation); var indices = new int[source.Length];
            for (var i = 0; i < source.Length; i++)
            {
                if (source[i] < 0 || source[i] >= limit) throw new InvalidDataException("MHR skinning index is out of range: " + name);
                indices[i] = (int)source[i];
            }
            return indices;
        }
    }

    /// <summary>Reads the released identity, expression and base-mesh tensors.
    /// Rejects incompatible shapes; run off the editor thread.</summary>
    internal static MhrBlendShapes LoadBlendShapes(string path, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        using var checkpoint = TorchCheckpoint.OpenScriptData(path, "__torch__.MHRDemo");
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        ReadTensor(checkpoint, tensors, MhrSkeleton.Prefix + "blend_shape.base_shape", cancellation, Vertices, 3);
        ReadTensor(checkpoint, tensors, MhrSkeleton.Prefix + "blend_shape.shape_vectors", cancellation, Identities, Vertices, 3);
        ReadTensor(checkpoint, tensors, "face_expressions_model.shape_vectors", cancellation, Expressions, Vertices, 3);
        cancellation.ThrowIfCancellationRequested();
        return new(new(tensors), Vertices, Identities, Expressions);
    }

    /// <summary>Reads the sparse pose layer and full dense corrective projection.
    /// Requires about 664 MB for projection weights; run off the editor thread.</summary>
    internal static MhrCorrectives LoadCorrectives(string path, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        using var checkpoint = TorchCheckpoint.OpenScriptData(path, "__torch__.MHRDemo");
        const string prefix = MhrCorrectives.Prefix;
        var indexName = prefix + "0.sparse_indices";
        if (!checkpoint.Tensors.TryGetValue(indexName, out var info) || info.Shape.Length != 2 || info.Shape[0] != 2
            || info.Shape[1] < 1 || info.Shape[1] > CorrectiveHidden * (Joints - 2) * 6)
            throw new InvalidDataException("MHR sparse corrective dimensions are invalid.");
        var indices = checkpoint.ReadInt64(indexName, cancellation);
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        ReadTensor(checkpoint, tensors, prefix + "0.sparse_weight", cancellation, info.Shape[1]);
        ReadTensor(checkpoint, tensors, prefix + "2.weight", cancellation, Vertices * 3, CorrectiveHidden);
        cancellation.ThrowIfCancellationRequested();
        return new(new(tensors), indices, Joints, CorrectiveHidden, Vertices);
    }

    static void ReadTensor(TorchCheckpoint checkpoint, Dictionary<string, Tensor> tensors, string name, CancellationToken cancellation, params int[] shape)
    {
        RequireShape(checkpoint, name, shape);
        var dimensions = shape.AsSpan().ToArray(); Array.Reverse(dimensions);
        tensors.Add(name, new(name, checkpoint.ReadFloat(name, cancellation), dimensions));
    }

    static void RequireShape(TorchCheckpoint checkpoint, string name, params int[] shape)
    {
        if (!checkpoint.Tensors.TryGetValue(name, out var tensor) || !tensor.Shape.AsSpan().SequenceEqual(shape))
            throw new InvalidDataException("Missing or incorrectly shaped MHR tensor: " + name);
    }
}
