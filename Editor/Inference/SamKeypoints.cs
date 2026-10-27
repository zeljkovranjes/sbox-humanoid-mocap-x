// C# adaptation of SAM 3D Body's MHR keypoint mapping. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Collections.Generic;
using System.Threading;
using HumanoidMocap.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Regresses the 70 decoder keypoints from the full deformed MHR mesh and joints.</summary>
internal sealed class SamKeypoints
{
    internal const int Count = 70, Vertices = 18439, Joints = 127;
    readonly Tensor mapping;
    readonly float[] column = new float[Vertices + Joints];
    readonly bool isHand;

    internal SamKeypoints(FlowWeights weights, bool isHand = false)
    {
        this.isHand = isHand;
        var source = weights.Get((isHand ? "head_pose_hand." : "head_pose.") + "keypoint_mapping", Vertices + Joints, 308);
        // Forward returns only the first 70 rows; the remaining landmark definitions are unused.
        mapping = new(source.Name, source.Data.AsSpan(0, Count * column.Length).ToArray(), new[] { column.Length, Count });
    }

    /// <summary>Returns the ascending vertices with any nonzero (including negative zero)
    /// mapping weight. Other vertices add exactly zero to every keypoint.</summary>
    internal int[] MappedVertices()
    {
        var used = new List<int>();
        for (var vertex = 0; vertex < Vertices; vertex++)
            for (var point = 0; point < Count; point++)
                if (BitConverter.SingleToInt32Bits(mapping.Row(point)[vertex]) != 0) { used.Add(vertex); break; }
        return used.ToArray();
    }

    /// <summary>Consumes vertices and joint states in MHR units. Writes XYZ in
    /// metres with camera Y/Z flipped. The hand head exposes only points 21–41.</summary>
    internal void Run(ReadOnlySpan<float> vertices, ReadOnlySpan<float> states, Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (vertices.Length != Vertices * 3 || states.Length != Joints * MhrState.Elements || output.Length != Count * 3
            || vertices.Overlaps(output) || states.Overlaps(output)) throw new ArgumentException("SAM keypoint tensors have invalid dimensions or overlap.");
        SamDecoder.Finite(vertices); SamDecoder.Finite(states);
        Span<float> values = stackalloc float[Count];
        for (var axis = 0; axis < 3; axis++)
        {
            for (var vertex = 0; vertex < Vertices; vertex++) column[vertex] = vertices[vertex * 3 + axis] / 100;
            for (var joint = 0; joint < Joints; joint++) column[Vertices + joint] = states[joint * MhrState.Elements + axis] / 100;
            FlowLinear.Run(column, 1, mapping, null, values, cancellation);
            for (var point = 0; point < Count; point++)
                output[point * 3 + axis] = isHand && (point < 21 || point >= 42) ? 0 : values[point] * (axis == 0 ? 1 : -1);
        }
        SamDecoder.Finite(output);
    }
}
