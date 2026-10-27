// C# adaptation of VGGT-Omega's alternating encoder. See LICENSE-VGGT and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
using System;
using System.Threading;
using HumanoidMocap.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

internal sealed class OmegaAggregator
{
    readonly OmegaShape shape;
    readonly Tensor cameraToken, registerToken;
    readonly OmegaBlock[] frameBlocks = new OmegaBlock[OmegaShape.Depth], interBlocks = new OmegaBlock[OmegaShape.Depth];
    readonly OmegaWorkspace workspace;
    readonly FlowAttention frameAttention, globalAttention, registerAttention;
    readonly float[] features, registers, finalFrame, positions;

    internal OmegaAggregator(FlowWeights weights, OmegaShape shape)
    {
        if (shape.Frames == 0) throw new ArgumentException("Construct a valid Omega shape.", nameof(shape));
        this.shape = shape; var width = shape.Channels;
        cameraToken = weights.Get("aggregator.camera_token", width, 1, 2, 1);
        registerToken = weights.Get("aggregator.register_token", width, 16, 2, 1);
        for (var i = 0; i < frameBlocks.Length; i++)
        {
            frameBlocks[i] = new(weights, "aggregator.frame_blocks." + i, width, shape.Heads, true);
            interBlocks[i] = new(weights, "aggregator.inter_frame_blocks." + i, width, shape.Heads, true);
        }
        workspace = new(shape.Tokens, width); frameAttention = new(shape.FrameTokens, width, shape.Heads, false);
        globalAttention = new(shape.Tokens, width, shape.Heads, false); registerAttention = new(shape.Frames * OmegaShape.FramePrefix, width, shape.Heads, false);
        features = new float[shape.Tokens * width]; registers = new float[shape.Frames * OmegaShape.FramePrefix * width]; finalFrame = new float[registers.Length];
        positions = OmegaPosition.Factors(shape.Height / 16, shape.Width / 16, OmegaShape.FramePrefix,
            weights.Get("aggregator.rope_embed.periods", width / shape.Heads / 4).Data);
    }

    /// <summary>Combines [frames,patches,width] features and writes [frames,17,2*width]
    /// camera/register tokens. CPU reference path. Rejects malformed/aliased tensors.</summary>
    internal void Run(ReadOnlySpan<float> patches, Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (patches.Length != shape.Frames * shape.Patches * shape.Channels || output.Length != shape.CameraFeatures || patches.Overlaps(output))
            throw new ArgumentException("Omega aggregator tensors have invalid dimensions or overlap.");
        foreach (var value in patches) if (!float.IsFinite(value)) throw new ArgumentException("Non-finite Omega image feature.");
        Initialize(patches);
        for (var layer = 0; layer < OmegaShape.Depth; layer++)
        {
            cancellation.ThrowIfCancellationRequested();
            for (var frame = 0; frame < shape.Frames; frame++)
                frameBlocks[layer].Run(features.AsSpan(frame * shape.FrameTokens * shape.Channels, shape.FrameTokens * shape.Channels), workspace, frameAttention, positions, cancellation);
            if (layer == OmegaShape.Depth - 1) Gather(finalFrame);
            if (OmegaShape.RegistersOnly(layer))
            {
                Gather(registers); interBlocks[layer].Run(registers, workspace, registerAttention, default, cancellation);
                for (var frame = 0; frame < shape.Frames; frame++)
                    registers.AsSpan(frame * OmegaShape.FramePrefix * shape.Channels, OmegaShape.FramePrefix * shape.Channels)
                        .CopyTo(features.AsSpan(frame * shape.FrameTokens * shape.Channels));
            }
            else interBlocks[layer].Run(features, workspace, globalAttention, default, cancellation);
        }
        Gather(registers);
        for (var token = 0; token < shape.Frames * OmegaShape.FramePrefix; token++)
        {
            finalFrame.AsSpan(token * shape.Channels, shape.Channels).CopyTo(output.Slice(token * shape.Channels * 2, shape.Channels));
            registers.AsSpan(token * shape.Channels, shape.Channels).CopyTo(output.Slice(token * shape.Channels * 2 + shape.Channels, shape.Channels));
        }
    }

    void Initialize(ReadOnlySpan<float> patches)
    {
        for (var frame = 0; frame < shape.Frames; frame++)
        {
            var destination = features.AsSpan(frame * shape.FrameTokens * shape.Channels, shape.FrameTokens * shape.Channels);
            cameraToken.Data.AsSpan((frame == 0 ? 0 : 1) * shape.Channels, shape.Channels).CopyTo(destination);
            registerToken.Data.AsSpan((frame == 0 ? 0 : 1) * 16 * shape.Channels, 16 * shape.Channels).CopyTo(destination[shape.Channels..]);
            patches.Slice(frame * shape.Patches * shape.Channels, shape.Patches * shape.Channels).CopyTo(destination[(OmegaShape.FramePrefix * shape.Channels)..]);
        }
    }

    void Gather(Span<float> destination)
    {
        for (var frame = 0; frame < shape.Frames; frame++)
            features.AsSpan(frame * shape.FrameTokens * shape.Channels, OmegaShape.FramePrefix * shape.Channels)
                .CopyTo(destination.Slice(frame * OmegaShape.FramePrefix * shape.Channels, OmegaShape.FramePrefix * shape.Channels));
    }
}
