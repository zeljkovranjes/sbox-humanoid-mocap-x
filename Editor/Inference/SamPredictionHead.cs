// C# adaptation of SAM 3D Body's body/hand prediction heads. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;
using HumanoidMocap.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Predicts pose or camera parameters from one normalized decoder token.
/// Each layer adds the fixed learned initialization, not the preceding prediction.</summary>
internal sealed class SamPredictionHead
{
    readonly FlowLinear first, second;
    readonly Tensor initial;
    readonly float[] hidden;
    readonly int width;
    internal int Parameters => initial.Data.Length;

    internal SamPredictionHead(FlowWeights weights, bool hand = false, bool camera = false, int width = 1024)
    {
        if (width < 1) throw new ArgumentOutOfRangeException(nameof(width));
        this.width = width;
        var part = camera ? "camera" : "pose"; var suffix = hand ? "_hand" : "";
        var name = "head_" + part + suffix + ".proj.layers.";
        var count = camera ? 3 : SamMhrParameters.Predictions;
        first = new(weights, name + "0.0", width, width); second = new(weights, name + "1", width, count);
        initial = weights.Get("init_" + part + suffix + ".weight", count, 1);
        hidden = new float[width];
    }

    /// <summary>Writes the complete prediction. Rejects non-finite tokens, bad
    /// dimensions or overlapping buffers. Reuse sequentially; cancellation propagates.</summary>
    internal void Run(ReadOnlySpan<float> token, Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (token.Length != width || output.Length != Parameters || token.Overlaps(output) || output.Overlaps(initial.Data))
            throw new ArgumentException("SAM prediction head input/output dimensions are invalid or overlap.");
        SamDecoder.Finite(token);
        first.Run(token, 1, hidden, cancellation); NeuralOps.Relu(hidden);
        second.Run(hidden, 1, output, cancellation); NeuralOps.AddInPlace(output, initial.Data);
        SamDecoder.Finite(output);
    }

    internal void CopyInitial(Span<float> output)
    {
        if (output.Length != Parameters) throw new ArgumentException("SAM initial prediction output has the wrong size.");
        initial.Data.AsSpan().CopyTo(output);
    }
}
