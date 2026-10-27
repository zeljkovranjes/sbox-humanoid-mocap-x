// C# adaptation of VGGT-Omega's camera head. See LICENSE-VGGT and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
using System;
using System.Threading;
using HumanoidMocap.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Four camera/register transformer blocks followed by nine camera parameters.
/// Consumes the final concatenated frame/global features, excluding image patches.</summary>
internal sealed class OmegaCameraHead
{
    internal const int PrefixTokens = 17, Blocks = 4;
    readonly int frames, width;
    readonly OmegaBlock[] blocks = new OmegaBlock[Blocks];
    readonly OmegaWorkspace workspace;
    readonly FlowAttention attention;
    readonly Tensor tokenWeight, tokenBias, outputWeight, outputBias;
    readonly FlowLinear first, second;
    readonly float[] features, pooled, hidden;

    internal OmegaCameraHead(FlowWeights weights, int frames, int width = 2048, int heads = 16)
    {
        if (frames < 1 || frames > FlowMotion.MaximumFrames / PrefixTokens || width < 8 || heads < 1 || width % heads != 0 || width / heads % 4 != 0)
            throw new ArgumentException("Invalid Omega camera head dimensions.");
        this.frames = frames; this.width = width;
        var tokens = frames * PrefixTokens;
        workspace = new(tokens, width); attention = new(tokens, width, heads, false);
        features = new float[checked(tokens * width)]; pooled = new float[frames * width]; hidden = new float[frames * width / 2];
        tokenWeight = weights.Get("camera_head.token_norm.weight", width); tokenBias = weights.Get("camera_head.token_norm.bias", width);
        outputWeight = weights.Get("camera_head.trunk_norm.weight", width); outputBias = weights.Get("camera_head.trunk_norm.bias", width);
        for (var index = 0; index < Blocks; index++) blocks[index] = new(weights, "camera_head.trunk." + index, width, heads, false);
        first = new(weights, "camera_head.camera_branch.0", width, width / 2);
        second = new(weights, "camera_head.camera_branch.2", width / 2, 9);
    }

    /// <summary>Writes [frames,9] translation, XYZW quaternion and positive FoV values.
    /// Rejects malformed/non-finite features and overlapping tensors. Not thread safe.</summary>
    internal void Run(ReadOnlySpan<float> input, Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (input.Length != features.Length || output.Length != frames * 9 || input.Overlaps(output))
            throw new ArgumentException("Omega camera head needs separate [frames,17,width] and [frames,9] tensors.");
        foreach (var value in input)
            if (!float.IsFinite(value)) throw new ArgumentException("Non-finite Omega camera feature.", nameof(input));
        input.CopyTo(features); NeuralOps.LayerNorm(features, frames * PrefixTokens, width, tokenWeight, tokenBias);
        foreach (var block in blocks) block.Run(features, workspace, attention, default, cancellation);
        for (var frame = 0; frame < frames; frame++) features.AsSpan(frame * PrefixTokens * width, width).CopyTo(pooled.AsSpan(frame * width, width));
        NeuralOps.LayerNorm(pooled, frames, width, outputWeight, outputBias);
        first.Run(pooled, frames, hidden, cancellation); OmegaActivation.Gelu(hidden);
        second.Run(hidden, frames, output, cancellation);
        Activate(output);
    }

    internal static void Activate(Span<float> poses)
    {
        if (poses.Length % 9 != 0) throw new ArgumentException("Omega camera output needs nine values per frame.");
        foreach (var value in poses)
            if (!float.IsFinite(value)) throw new ArithmeticException("Omega camera head produced non-finite values.");
        for (var offset = 0; offset < poses.Length; offset += 9)
        { poses[offset + 7] = Math.Max(0, poses[offset + 7]) + .01f; poses[offset + 8] = Math.Max(0, poses[offset + 8]) + .01f; }
    }
}
