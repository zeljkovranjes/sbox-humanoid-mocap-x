// C# adaptation of the Omega and SAM DINO encoders. See LICENSE-VGGT, LICENSE-DINOV3 and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
using System;
using System.Threading;
using HumanoidMocap.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

internal sealed class DinoImageEncoder
{
    readonly DinoShape shape;
    readonly Tensor weight, bias, classToken, registers, norm, normBias;
    readonly OmegaBlock[] blocks;
    readonly OmegaWorkspace workspace;
    readonly FlowAttention attention;
    readonly float[] features, positions, normalizedImage;

    internal DinoImageEncoder(FlowWeights weights, OmegaShape shape) : this(weights, DinoShape.Omega(shape)) { }

    internal DinoImageEncoder(FlowWeights weights, DinoShape shape)
    {
        if (shape.Channels == 0) throw new ArgumentException("Construct a valid DINO shape.", nameof(shape));
        this.shape = shape; var width = shape.Channels;
        var name = shape.Name; blocks = new OmegaBlock[shape.Depth];
        weight = weights.Get(name + ".patch_embed.proj.weight", 16, 16, 3, width);
        bias = weights.Get(name + ".patch_embed.proj.bias", width);
        classToken = weights.Get(name + ".cls_token", width, 1, 1); registers = weights.Get(name + ".storage_tokens", width, 4, 1);
        norm = weights.Get(name + ".norm.weight", width); normBias = weights.Get(name + ".norm.bias", width);
        positions = OmegaPosition.Factors(shape.Height / 16, shape.Width / 16, DinoShape.PrefixTokens,
            weights.Get(name + ".rope_embed.periods", width / shape.Heads / 4).Data, shape.IsSam);
        for (var i = 0; i < blocks.Length; i++) blocks[i] = new(weights, name + ".blocks." + i, width, shape.Heads, false, shape.IsSam);
        workspace = new(shape.Tokens, width, shape.IsSam); attention = new(shape.Tokens, width, shape.Heads, false);
        features = new float[shape.Tokens * width]; normalizedImage = new float[shape.ImageElements];
    }

    /// <summary>Encodes one resized HWC RGB image in [0,1] into patch features. CPU
    /// reference path; reuse the instance sequentially. Rejects invalid/aliased tensors.</summary>
    internal void Run(ReadOnlySpan<float> image, Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (image.Length != shape.ImageElements || output.Length != shape.Patches * shape.Channels || image.Overlaps(output))
            throw new ArgumentException("DINO image encoder tensors have invalid dimensions or overlap.");
        NormalizeImage(image, normalizedImage); cancellation.ThrowIfCancellationRequested();
        classToken.Data.CopyTo(features, 0); registers.Data.CopyTo(features, shape.Channels);
        Project(cancellation);
        foreach (var block in blocks) block.Run(features, workspace, attention, positions, cancellation);
        features.AsSpan(DinoShape.PrefixTokens * shape.Channels).CopyTo(output);
        NeuralOps.LayerNorm(output, shape.Patches, shape.Channels, norm, normBias);
    }

    internal static void NormalizeImage(ReadOnlySpan<float> image, Span<float> output)
    {
        if (image.Length != output.Length || image.Length % 3 != 0) throw new ArgumentException("DINO images need complete RGB pixels.");
        for (var i = 0; i < image.Length; i++)
        {
            var pixel = image[i];
            if (!float.IsFinite(pixel) || pixel < 0 || pixel > 1) throw new ArgumentException("DINO input pixels must be finite and between zero and one.");
            var mean = i % 3 == 0 ? .485f : i % 3 == 1 ? .456f : .406f;
            var deviation = i % 3 == 0 ? .229f : i % 3 == 1 ? .224f : .225f;
            output[i] = (pixel - mean) / deviation;
        }
    }

    void Project(CancellationToken cancellation)
    {
        for (var y = 0; y < shape.Height / 16; y++)
            for (var x = 0; x < shape.Width / 16; x++)
            {
                cancellation.ThrowIfCancellationRequested();
                var destination = ((y * (shape.Width / 16) + x) + DinoShape.PrefixTokens) * shape.Channels;
                for (var channel = 0; channel < shape.Channels; channel++)
                {
                    var sum = bias.Data[channel];
                    for (var color = 0; color < 3; color++)
                        for (var ky = 0; ky < 16; ky++)
                            for (var kx = 0; kx < 16; kx++)
                                sum += normalizedImage[((y * 16 + ky) * shape.Width + x * 16 + kx) * 3 + color]
                                    * weight.Data[((channel * 3 + color) * 16 + ky) * 16 + kx];
                    features[destination + channel] = sum;
                }
            }
    }
}
