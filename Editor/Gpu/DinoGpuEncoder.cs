// C# adaptation of the Omega and SAM DINO encoders. See LICENSE-VGGT, LICENSE-DINOV3 and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
using System;
using System.Threading;
using Sandbox;
using TensorInference;

namespace HumanoidMocapX.Inference;

/// <summary>Shared DINO encoder for Omega images and SAM body/hand crops.
/// The caller owns storage and submits Run inside a graphics batch.</summary>
internal sealed class DinoGpuEncoder
{
    readonly DinoShape shape;
    readonly OmegaGpuBlock[] blocks;
    readonly ConvolutionShape projection;
    readonly GpuBuffer<float> features, patches, weight, bias, classToken, registers, norm, normBias, positions;
    readonly GpuBuffer<uint> ranges;

    internal DinoGpuEncoder(FlowGpuStorage storage, FlowWeights weights, DinoShape shape)
    {
        if (shape.Channels == 0 || shape.Channels / shape.Heads != 64) throw new ArgumentException("GPU DINO requires 64-channel heads.");
        this.shape = shape; var width = shape.Channels; var name = shape.Name;
        blocks = new OmegaGpuBlock[shape.Depth];
        features = storage.Create<float>(shape.Tokens * width); patches = storage.Create<float>(shape.Patches * width);
        weight = storage.Weight(weights, name + ".patch_embed.proj.weight", 16, 16, 3, width);
        bias = storage.Weight(weights, name + ".patch_embed.proj.bias", width);
        classToken = storage.Weight(weights, name + ".cls_token", width, 1, 1); registers = storage.Weight(weights, name + ".storage_tokens", width, 4, 1);
        norm = storage.Weight(weights, name + ".norm.weight", width); normBias = storage.Weight(weights, name + ".norm.bias", width);
        positions = storage.Upload<float>(OmegaPosition.Factors(shape.Height / 16, shape.Width / 16, DinoShape.PrefixTokens,
            weights.Get(name + ".rope_embed.periods", width / shape.Heads / 4).Data, shape.IsSam));
        var keys = new uint[shape.Tokens * 4];
        for (var row = 0; row < shape.Tokens; row++) keys[row * 4 + 1] = (uint)shape.Tokens;
        ranges = storage.Upload<uint>(keys);
        for (var i = 0; i < blocks.Length; i++) blocks[i] = new(storage, weights, name + ".blocks." + i, width, shape.Heads, false, shape.IsSam);
        projection = new(shape.Height, shape.Width, 3, width, 16, 16);
    }

    /// <summary>Consumes ImageNet-normalized HWC RGB and writes [patches,channels].
    /// Output may be consumed by a later GPU stage without readback. The workspace
    /// must fit all tokens and have SwiGLU storage for SAM. Discard output on failure.</summary>
    internal void Run(OmegaGpuWorkspace workspace, GpuBuffer<float> image, GpuBuffer<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (image is null || !image.IsValid || image.ElementCount != shape.ImageElements
            || output is null || !output.IsValid || output.ElementCount != shape.Patches * shape.Channels
            || ReferenceEquals(image, output)) throw new ArgumentException("DINO GPU images/features have invalid dimensions or overlap.");
        var width = shape.Channels; var kernels = workspace.Kernels;
        classToken.CopyTo(features, 0, 0, width); registers.CopyTo(features, 0, width, 4 * width);
        kernels.Convolve(image, weight, bias, patches, projection);
        patches.CopyTo(features, 0, DinoShape.PrefixTokens * width, shape.Patches * width);
        foreach (var block in blocks)
        { cancellation.ThrowIfCancellationRequested(); block.Run(workspace, features, ranges, shape.Tokens, positions, shape.Tokens, sharedKeyRanges: true); }
        kernels.Normalize(features, norm, normBias, workspace.Normalized, shape.Tokens, width);
        workspace.Normalized.CopyTo(output, DinoShape.PrefixTokens * width, 0, shape.Patches * width);
    }
}
