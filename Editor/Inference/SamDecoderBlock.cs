// C# adaptation of SAM 3D Body's transformer decoder. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

internal sealed class SamDecoderBlock
{
    readonly SamDecoderShape shape;
    readonly SamLayerNorm tokenPositionNorm, imagePositionNorm, selfNorm, crossTokenNorm, crossImageNorm, hiddenNorm;
    readonly SamAttention self, cross;
    readonly FlowLinear first, second;

    internal SamDecoderBlock(FlowWeights weights, string name, SamDecoderShape shape)
    {
        if (shape.Tokens == 0) throw new ArgumentException("Construct a valid SAM decoder shape.", nameof(shape));
        this.shape = shape;
        tokenPositionNorm = new(weights, name + ".ln_pe_1", shape.Width); imagePositionNorm = new(weights, name + ".ln_pe_2", shape.ContextWidth);
        selfNorm = new(weights, name + ".ln1", shape.Width); crossTokenNorm = new(weights, name + ".ln2_1", shape.Width);
        crossImageNorm = new(weights, name + ".ln2_2", shape.ContextWidth); hiddenNorm = new(weights, name + ".ln3", shape.Width);
        self = new(weights, name + ".self_attn", shape, false); cross = new(weights, name + ".cross_attn", shape, true);
        first = new(weights, name + ".ffn.layers.0.0", shape.Width, shape.HiddenWidth);
        second = new(weights, name + ".ffn.layers.1", shape.HiddenWidth, shape.Width);
    }

    internal void Run(Span<float> tokens, ReadOnlySpan<float> image, ReadOnlySpan<float> tokenPositions,
        ReadOnlySpan<float> imagePositions, bool firstLayer, SamDecoderWorkspace workspace, CancellationToken cancellation)
    {
        if (workspace.Shape != shape || tokens.Length != shape.TokenElements || tokenPositions.Length != tokens.Length
            || image.Length != shape.ImageElements || imagePositions.Length != image.Length)
            throw new ArgumentException("SAM decoder tensors do not match their workspace.");
        cancellation.ThrowIfCancellationRequested(); var w = workspace;
        tokenPositionNorm.Run(tokenPositions, w.TokenPositions); imagePositionNorm.Run(imagePositions, w.ImagePositions);
        selfNorm.Run(tokens, w.NormalizedTokens);
        if (firstLayer) w.NormalizedTokens.AsSpan().CopyTo(w.PositionedTokens);
        else Add(w.NormalizedTokens, w.TokenPositions, w.PositionedTokens);
        self.Run(w, w.PositionedTokens, w.PositionedTokens, w.NormalizedTokens, w.Projected, cancellation);
        Add(tokens, w.Projected, tokens);
        crossTokenNorm.Run(tokens, w.NormalizedTokens); crossImageNorm.Run(image, w.NormalizedImage);
        Add(w.NormalizedTokens, w.TokenPositions, w.PositionedTokens); Add(w.NormalizedImage, w.ImagePositions, w.PositionedImage);
        cross.Run(w, w.PositionedTokens, w.PositionedImage, w.NormalizedImage, w.Projected, cancellation);
        Add(tokens, w.Projected, tokens);
        hiddenNorm.Run(tokens, w.NormalizedTokens);
        first.Run(w.NormalizedTokens, shape.Tokens, w.Hidden, cancellation); OmegaActivation.Gelu(w.Hidden);
        second.Run(w.Hidden, shape.Tokens, w.Projected, cancellation); Add(tokens, w.Projected, tokens);
    }

    static void Add(ReadOnlySpan<float> first, ReadOnlySpan<float> second, Span<float> output)
    { for (var i = 0; i < output.Length; i++) output[i] = first[i] + second[i]; }
}

internal readonly struct SamLayerNorm
{
    readonly Tensor weight, bias;
    internal const float Epsilon = 1e-6f;

    internal SamLayerNorm(FlowWeights weights, string name, int width)
    { weight = weights.Get(name + ".weight", width); bias = weights.Get(name + ".bias", width); }

    internal void Run(ReadOnlySpan<float> values, Span<float> output)
    {
        if (output.Length != values.Length || output.Length % weight.Data.Length != 0)
            throw new ArgumentException("SAM normalization needs equally sized complete rows.");
        values.CopyTo(output); NeuralOps.LayerNorm(output, output.Length / weight.Data.Length, weight.Data.Length, weight, bias, Epsilon);
    }
}

internal sealed class SamAttention
{
    readonly SamDecoderShape shape;
    readonly bool isCross;
    readonly FlowLinear query, key, value, projection;

    internal SamAttention(FlowWeights weights, string name, SamDecoderShape shape, bool isCross)
    {
        this.shape = shape; this.isCross = isCross;
        var context = isCross ? shape.ContextWidth : shape.Width;
        query = new(weights, name + ".q_proj", shape.Width, shape.AttentionWidth);
        key = new(weights, name + ".k_proj", context, shape.AttentionWidth); value = new(weights, name + ".v_proj", context, shape.AttentionWidth);
        projection = new(weights, name + ".proj", shape.AttentionWidth, shape.Width);
    }

    internal void Run(SamDecoderWorkspace workspace, ReadOnlySpan<float> queries, ReadOnlySpan<float> keys,
        ReadOnlySpan<float> values, Span<float> output, CancellationToken cancellation)
    {
        var count = isCross ? shape.ImageTokens : shape.Tokens;
        var k = workspace.Key.AsSpan(0, count * shape.AttentionWidth); var v = workspace.Value.AsSpan(0, k.Length);
        query.Run(queries, shape.Tokens, workspace.Query, cancellation); key.Run(keys, count, k, cancellation); value.Run(values, count, v, cancellation);
        (isCross ? workspace.Cross : workspace.Self).Run(workspace.Query, k, v, workspace.Attended, cancellation);
        projection.Run(workspace.Attended, shape.Tokens, output, cancellation);
    }
}

internal sealed class SamDecoderWorkspace
{
    internal readonly SamDecoderShape Shape;
    internal readonly FlowAttention Self, Cross;
    internal readonly float[] NormalizedTokens, PositionedTokens, TokenPositions, NormalizedImage, PositionedImage, ImagePositions;
    internal readonly float[] Projected, Hidden, Query, Key, Value, Attended;

    internal SamDecoderWorkspace(SamDecoderShape shape)
    {
        if (shape.Tokens == 0) throw new ArgumentException("Construct a valid SAM decoder shape.", nameof(shape));
        Shape = shape;
        NormalizedTokens = new float[shape.TokenElements]; PositionedTokens = new float[shape.TokenElements]; TokenPositions = new float[shape.TokenElements];
        NormalizedImage = new float[shape.ImageElements]; PositionedImage = new float[shape.ImageElements]; ImagePositions = new float[shape.ImageElements];
        Projected = new float[shape.TokenElements]; Hidden = new float[shape.Tokens * shape.HiddenWidth];
        Query = new float[shape.Tokens * shape.AttentionWidth]; Attended = new float[Query.Length];
        Key = new float[Math.Max(shape.Tokens, shape.ImageTokens) * shape.AttentionWidth]; Value = new float[Key.Length];
        Self = new(shape.Tokens, shape.AttentionWidth, shape.Heads, false);
        Cross = new(shape.Tokens, shape.ImageTokens, shape.AttentionWidth, shape.Heads);
    }
}
