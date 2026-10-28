// C# adaptation of SAM 3D Body's transformer decoder. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Inference;

internal sealed class SamGpuDecoderBlock
{
    readonly SamDecoderShape shape;
    readonly SamGpuNorm tokenPositionNorm, imagePositionNorm, selfNorm, crossTokenNorm, crossImageNorm, hiddenNorm;
    readonly SamGpuAttention self, cross;
    readonly FlowGpuLinear first, second;

    internal SamGpuDecoderBlock(FlowGpuStorage storage, FlowWeights weights, string name, SamDecoderShape shape)
    {
        if (shape.Tokens == 0 || shape.HeadWidth is not (64 or 128)) throw new ArgumentException("GPU SAM needs 64 or 128 channels per head.");
        this.shape = shape;
        tokenPositionNorm = new(storage, weights, name + ".ln_pe_1", shape.Width); imagePositionNorm = new(storage, weights, name + ".ln_pe_2", shape.ContextWidth);
        selfNorm = new(storage, weights, name + ".ln1", shape.Width); crossTokenNorm = new(storage, weights, name + ".ln2_1", shape.Width);
        crossImageNorm = new(storage, weights, name + ".ln2_2", shape.ContextWidth); hiddenNorm = new(storage, weights, name + ".ln3", shape.Width);
        self = new(storage, weights, name + ".self_attn", shape, false); cross = new(storage, weights, name + ".cross_attn", shape, true);
        first = new(storage, weights, name + ".ffn.layers.0.0", shape.Width, shape.HiddenWidth);
        second = new(storage, weights, name + ".ffn.layers.1", shape.HiddenWidth, shape.Width);
    }

    internal void Run(SamGpuDecoderWorkspace w, GpuBuffer<float> tokens, GpuBuffer<float> image,
        GpuBuffer<float> tokenPositions, GpuBuffer<float> imagePositions, bool firstLayer)
    {
        if (w.Shape != shape) throw new ArgumentException("SAM GPU decoder workspace has different dimensions.");
        var k = w.Kernels;
        tokenPositionNorm.Run(k, tokenPositions, w.TokenPositions, shape.Tokens); imagePositionNorm.Run(k, imagePositions, w.ImagePositions, shape.ImageTokens);
        selfNorm.Run(k, tokens, w.NormalizedTokens, shape.Tokens);
        if (firstLayer) w.NormalizedTokens.CopyTo(w.PositionedTokens, shape.TokenElements);
        else k.Add(w.NormalizedTokens, w.TokenPositions, w.PositionedTokens, shape.TokenElements);
        self.Run(w, w.PositionedTokens, w.PositionedTokens, w.NormalizedTokens, w.Projected);
        k.Add(tokens, w.Projected, w.Residual, shape.TokenElements);
        crossTokenNorm.Run(k, w.Residual, w.NormalizedTokens, shape.Tokens); crossImageNorm.Run(k, image, w.NormalizedImage, shape.ImageTokens);
        k.Add(w.NormalizedTokens, w.TokenPositions, w.PositionedTokens, shape.TokenElements);
        k.Add(w.NormalizedImage, w.ImagePositions, w.PositionedImage, shape.ImageElements);
        cross.Run(w, w.PositionedTokens, w.PositionedImage, w.NormalizedImage, w.Projected);
        k.Add(w.Residual, w.Projected, tokens, shape.TokenElements);
        hiddenNorm.Run(k, tokens, w.NormalizedTokens, shape.Tokens);
        first.Run(k, w.NormalizedTokens, w.Hidden, shape.Tokens);
        k.Activate(w.Hidden, w.Activated, shape.Tokens * shape.HiddenWidth, TensorActivation.GeluErf);
        second.Run(k, w.Activated, w.Projected, shape.Tokens);
        k.Add(tokens, w.Projected, w.Residual, shape.TokenElements); w.Residual.CopyTo(tokens, shape.TokenElements);
    }
}

internal readonly struct SamGpuNorm
{
    readonly GpuBuffer<float> weight, bias;
    readonly int width;

    internal SamGpuNorm(FlowGpuStorage storage, FlowWeights weights, string name, int width)
    { this.width = width; weight = storage.Weight(weights, name + ".weight", width); bias = storage.Weight(weights, name + ".bias", width); }

    internal void Run(TensorKernels kernels, GpuBuffer<float> input, GpuBuffer<float> output, int rows)
        => kernels.Normalize(input, weight, bias, output, rows, width, SamLayerNorm.Epsilon);
}

internal sealed class SamGpuAttention
{
    readonly SamDecoderShape shape;
    readonly bool isCross;
    readonly FlowGpuLinear query, key, value, projection;

    internal SamGpuAttention(FlowGpuStorage storage, FlowWeights weights, string name, SamDecoderShape shape, bool isCross)
    {
        this.shape = shape; this.isCross = isCross; var context = isCross ? shape.ContextWidth : shape.Width;
        query = new(storage, weights, name + ".q_proj", shape.Width, shape.AttentionWidth);
        key = new(storage, weights, name + ".k_proj", context, shape.AttentionWidth); value = new(storage, weights, name + ".v_proj", context, shape.AttentionWidth);
        projection = new(storage, weights, name + ".proj", shape.AttentionWidth, shape.Width);
    }

    internal void Run(SamGpuDecoderWorkspace w, GpuBuffer<float> queries, GpuBuffer<float> keys, GpuBuffer<float> values, GpuBuffer<float> output)
    {
        var count = isCross ? shape.ImageTokens : shape.Tokens;
        query.Run(w.Kernels, queries, w.Query, shape.Tokens); key.Run(w.Kernels, keys, w.Key, count); value.Run(w.Kernels, values, w.Value, count);
        w.Kernels.Attention(w.Query, w.Key, w.Value, isCross ? w.CrossRanges : w.SelfRanges, w.Attended,
            shape.Tokens, shape.Heads, shape.HeadWidth, count);
        projection.Run(w.Kernels, w.Attended, output, shape.Tokens);
    }
}

internal sealed class SamGpuDecoderWorkspace
{
    internal readonly SamDecoderShape Shape;
    internal readonly TensorKernels Kernels = new();
    internal readonly GpuBuffer<float> NormalizedTokens, PositionedTokens, TokenPositions, NormalizedImage, PositionedImage, ImagePositions;
    internal readonly GpuBuffer<float> Projected, Residual, Hidden, Activated, Query, Key, Value, Attended;
    internal readonly GpuBuffer<uint> SelfRanges, CrossRanges;

    internal SamGpuDecoderWorkspace(FlowGpuStorage storage, SamDecoderShape shape)
    {
        if (shape.Tokens == 0) throw new ArgumentException("Construct a valid SAM decoder shape.", nameof(shape));
        Shape = shape;
        NormalizedTokens = storage.Create<float>(shape.TokenElements); PositionedTokens = storage.Create<float>(shape.TokenElements); TokenPositions = storage.Create<float>(shape.TokenElements);
        NormalizedImage = storage.Create<float>(shape.ImageElements); PositionedImage = storage.Create<float>(shape.ImageElements); ImagePositions = storage.Create<float>(shape.ImageElements);
        Projected = storage.Create<float>(shape.TokenElements); Residual = storage.Create<float>(shape.TokenElements);
        Hidden = storage.Create<float>(shape.Tokens * shape.HiddenWidth); Activated = storage.Create<float>(Hidden.ElementCount);
        Query = storage.Create<float>(shape.Tokens * shape.AttentionWidth); Attended = storage.Create<float>(Query.ElementCount);
        Key = storage.Create<float>(Math.Max(shape.Tokens, shape.ImageTokens) * shape.AttentionWidth); Value = storage.Create<float>(Key.ElementCount);
        var ranges = new uint[shape.Tokens * 4];
        for (var row = 0; row < shape.Tokens; row++) ranges[row * 4 + 1] = (uint)shape.Tokens;
        SelfRanges = storage.Upload<uint>(ranges);
        for (var row = 0; row < shape.Tokens; row++) ranges[row * 4 + 1] = (uint)shape.ImageTokens;
        CrossRanges = storage.Upload<uint>(ranges);
    }
}
