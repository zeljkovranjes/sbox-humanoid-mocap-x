using System;
using System.IO;
using System.Threading;
using HumanoidMocap.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Computes MHR's sparse pose features, ReLU and dense corrective
/// projection. Uses joint rotations after the first two root joints.</summary>
internal sealed class MhrCorrectives
{
    internal const string Prefix = "pose_correctives_model.pose_dirs_predictor.";
    readonly int[] rows, columns;
    readonly float[] weights, features, hidden;
    int[] activeRows;
    internal int Joints { get; }
    internal int Hidden => hidden.Length;
    internal Tensor Projection { get; }
    internal int Coordinates => Projection.Rows;

    internal MhrCorrectives(FlowWeights tensors, ReadOnlySpan<long> indices, int joints, int hidden, int vertices)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        if (joints < 3 || hidden < 1 || vertices < 1 || indices.Length == 0 || indices.Length % 2 != 0)
            throw new ArgumentException("MHR pose corrective dimensions are invalid.");
        var count = indices.Length / 2; Joints = joints;
        Projection = tensors.Get(Prefix + "2.weight", hidden, checked(vertices * 3));
        weights = tensors.Get(Prefix + "0.sparse_weight", count).Data;
        features = new float[checked((joints - 2) * 6)]; this.hidden = new float[hidden];
        rows = new int[count]; columns = new int[count];
        for (var i = 0; i < count; i++)
        {
            if (indices[i] < 0 || indices[i] >= hidden || indices[count + i] < 0 || indices[count + i] >= features.Length)
                throw new InvalidDataException("MHR sparse pose index is out of range.");
            rows[i] = (int)indices[i]; columns[i] = (int)indices[count + i];
        }
    }

    /// <summary>Another instance over <paramref name="source"/>'s immutable sparse weights, indices
    /// and ~664 MB projection, with its own buffers and no restriction, so capture lanes keep one
    /// copy of the projection instead of one each. Instances may run concurrently.</summary>
    internal MhrCorrectives(MhrCorrectives source)
    {
        ArgumentNullException.ThrowIfNull(source);
        rows = source.rows; columns = source.columns; weights = source.weights; Projection = source.Projection; Joints = source.Joints;
        features = new float[source.features.Length]; hidden = new float[source.hidden.Length];
    }

    /// <summary>Writes XYZ pose corrections. Rejects invalid inputs and aliasing;
    /// cancellation may leave partial output. One call at a time per instance.</summary>
    internal void Run(ReadOnlySpan<float> jointParameters, Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (output.Length != Coordinates || jointParameters.Overlaps(output) || output.Overlaps(weights) || output.Overlaps(Projection.Data))
            throw new ArgumentException("MHR corrective output has invalid dimensions or overlaps inputs.");
        Prepare(jointParameters, hidden, cancellation);
        if (activeRows is null) FlowLinear.Run(hidden, 1, Projection, null, output, cancellation);
        else
        {
            // Same per-row dot product and zero bias as FlowLinear, so every active row is identical.
            output.Clear();
            foreach (var row in activeRows) output[row] = 0f + NeuralOps.Dot(Projection.Row(row), hidden);
        }
        foreach (var value in output)
            if (!float.IsFinite(value)) throw new ArithmeticException("MHR correctives produced a non-finite vertex.");
    }

    /// <summary>Limits later runs to the XYZ corrections of the given vertices and writes
    /// zero for the rest; the released projection is ~664 MB, so a full run is memory-bound.
    /// Callers that read only these vertices get results identical to a full run.
    /// Rejects vertices outside the model.</summary>
    internal void Restrict(ReadOnlySpan<int> vertices)
    {
        var active = new int[vertices.Length * 3];
        for (var i = 0; i < vertices.Length; i++)
        {
            if ((uint)vertices[i] >= (uint)(Coordinates / 3)) throw new ArgumentOutOfRangeException(nameof(vertices), "MHR corrective vertex is outside the model.");
            for (var axis = 0; axis < 3; axis++) active[i * 3 + axis] = vertices[i] * 3 + axis;
        }
        activeRows = active;
    }

    /// <summary>Writes the sparse layer's rectified output for the CPU or GPU dense
    /// projection. Rejects invalid dimensions, non-finite inputs and aliasing.</summary>
    internal void Prepare(ReadOnlySpan<float> jointParameters, Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (jointParameters.Length != Joints * MhrSkeleton.JointParameters || output.Length != Hidden)
            throw new ArgumentException("MHR corrective input dimensions do not match the model.");
        if (jointParameters.Overlaps(output) || output.Overlaps(weights) || output.Overlaps(Projection.Data))
            throw new ArgumentException("MHR corrective features must not overlap inputs or model weights.");
        foreach (var value in jointParameters)
            if (!float.IsFinite(value)) throw new ArgumentException("MHR joint parameters must be finite.", nameof(jointParameters));
        for (var joint = 2; joint < Joints; joint++)
        {
            var start = joint * MhrSkeleton.JointParameters + 3;
            var sx = MathF.Sin(jointParameters[start]); var cx = MathF.Cos(jointParameters[start]);
            var sy = MathF.Sin(jointParameters[start + 1]); var cy = MathF.Cos(jointParameters[start + 1]);
            var sz = MathF.Sin(jointParameters[start + 2]); var cz = MathF.Cos(jointParameters[start + 2]);
            var offset = (joint - 2) * 6;
            features[offset] = cy * cz - 1; features[offset + 1] = cy * sz; features[offset + 2] = -sy;
            features[offset + 3] = -cx * sz + sx * sy * cz; features[offset + 4] = cx * cz + sx * sy * sz - 1; features[offset + 5] = sx * cy;
        }
        output.Clear();
        for (var i = 0; i < weights.Length; i++)
        {
            if ((i & 255) == 0) cancellation.ThrowIfCancellationRequested();
            output[rows[i]] += weights[i] * features[columns[i]];
        }
        NeuralOps.Relu(output);
        foreach (var value in output)
            if (!float.IsFinite(value)) throw new ArithmeticException("MHR sparse pose projection overflowed.");
    }
}
