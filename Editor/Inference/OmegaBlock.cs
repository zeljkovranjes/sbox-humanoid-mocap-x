// C# adaptation of VGGT-Omega's inference transformer. See LICENSE-VGGT and THIRD_PARTY_NOTICES.md.
// SAM's SwiGLU configuration also retains the DINOv3 terms in LICENSE-DINOV3.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
using System;
using System.IO;
using System.Threading;
using HumanoidMocapX.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>One Omega/DINO transformer block, using the existing motion library's
/// tensor operations and caller-owned workspace. All weights are immutable after binding.</summary>
internal sealed class OmegaBlock
{
    readonly int width, heads, headWidth;
    readonly Tensor norm1, bias1, norm2, bias2, qkv, qkvBias, scale1, scale2, queryNorm, queryBias, keyNorm, keyBias;
    readonly FlowLinear projection, first, second, gate;

    internal OmegaBlock(FlowWeights weights, string name, int width, int heads, bool normalizeQk, bool swiglu = false)
    {
        if (width < 4 || heads < 1 || width % heads != 0 || width / heads % 4 != 0)
            throw new ArgumentException("Omega transformer heads must have a width divisible by four.");
        this.width = width; this.heads = heads; headWidth = width / heads;
        norm1 = weights.Get(name + ".norm1.weight", width); bias1 = weights.Get(name + ".norm1.bias", width);
        norm2 = weights.Get(name + ".norm2.weight", width); bias2 = weights.Get(name + ".norm2.bias", width);
        scale1 = weights.Get(name + ".ls1.gamma", width); scale2 = weights.Get(name + ".ls2.gamma", width);
        qkv = weights.Get(name + ".attn.qkv.weight", width, width * 3);
        qkvBias = MaskedBias(weights, name + ".attn.qkv", width);
        projection = new(weights, name + ".attn.proj", width, width);
        first = new(weights, name + (swiglu ? ".mlp.w1" : ".mlp.fc1"), width, width * 4);
        second = new(weights, name + (swiglu ? ".mlp.w3" : ".mlp.fc2"), width * 4, width);
        if (swiglu) gate = new(weights, name + ".mlp.w2", width, width * 4);
        if (normalizeQk)
        {
            queryNorm = weights.Get(name + ".attn.q_norm.weight", headWidth); queryBias = weights.Get(name + ".attn.q_norm.bias", headWidth);
            keyNorm = weights.Get(name + ".attn.k_norm.weight", headWidth); keyBias = weights.Get(name + ".attn.k_norm.bias", headWidth);
        }
    }

    internal static Tensor MaskedBias(FlowWeights weights, string name, int width)
    {
        var bias = weights.Get(name + ".bias", width * 3); var mask = weights.Get(name + ".bias_mask", width * 3);
        var result = bias.Data.AsSpan().ToArray();
        for (var channel = 0; channel < result.Length; channel++)
        {
            var multiplier = mask.Data[channel];
            if (multiplier is not (0 or 1)) throw new InvalidDataException("Non-binary Omega QKV bias mask: " + name);
            // Checkpoints can disable every bias, overriding the module's K-only initialization.
            result[channel] *= multiplier;
        }
        return new(name + ".masked_bias", result, new[] { width * 3 });
    }

    internal void Run(Span<float> values, OmegaWorkspace workspace, FlowAttention attention,
        ReadOnlySpan<float> factors, CancellationToken cancellation = default)
    {
        if (values.Length % width != 0 || values.Length > workspace.Normalized.Length
            || gate is not null && (workspace.Gate is null || workspace.Gate.Length < (long)values.Length * 4))
            throw new ArgumentException("Omega transformer input exceeds its workspace or has incomplete rows.");
        var rows = values.Length / width;
        var norm = workspace.Normalized.AsSpan(0, values.Length); var packed = workspace.Packed.AsSpan(0, values.Length * 3);
        var query = workspace.Query.AsSpan(0, values.Length); var key = workspace.Key.AsSpan(0, values.Length);
        var value = workspace.Value.AsSpan(0, values.Length); var attended = workspace.Attended.AsSpan(0, values.Length);
        var projected = workspace.Projected.AsSpan(0, values.Length); var hidden = workspace.Hidden.AsSpan(0, values.Length * 4);
        values.CopyTo(norm); NeuralOps.LayerNorm(norm, rows, width, norm1, bias1);
        FlowLinear.Run(norm, rows, qkv, qkvBias, packed, cancellation);
        for (var row = 0; row < rows; row++)
        {
            packed.Slice(row * width * 3, width).CopyTo(query.Slice(row * width, width));
            packed.Slice(row * width * 3 + width, width).CopyTo(key.Slice(row * width, width));
            packed.Slice(row * width * 3 + width * 2, width).CopyTo(value.Slice(row * width, width));
        }
        if (queryNorm is not null)
        {
            NeuralOps.LayerNorm(query, rows * heads, headWidth, queryNorm, queryBias);
            NeuralOps.LayerNorm(key, rows * heads, headWidth, keyNorm, keyBias);
        }
        if (!factors.IsEmpty) { OmegaPosition.Apply(query, factors, width, headWidth); OmegaPosition.Apply(key, factors, width, headWidth); }
        attention.Run(query, key, value, attended, cancellation);
        projection.Run(attended, rows, projected, cancellation); FlowConditioning.AddGated(values, projected, scale1.Data);
        values.CopyTo(norm); NeuralOps.LayerNorm(norm, rows, width, norm2, bias2);
        first.Run(norm, rows, hidden, cancellation);
        if (gate is null) OmegaActivation.Gelu(hidden);
        else
        {
            var gated = workspace.Gate.AsSpan(0, hidden.Length);
            gate.Run(norm, rows, gated, cancellation); FlowConditioning.Silu(hidden);
            for (var i = 0; i < hidden.Length; i++) hidden[i] *= gated[i];
        }
        second.Run(hidden, rows, projected, cancellation); FlowConditioning.AddGated(values, projected, scale2.Data);
    }
}

internal sealed class OmegaWorkspace
{
    internal readonly float[] Normalized, Packed, Query, Key, Value, Attended, Projected, Hidden, Gate;

    internal OmegaWorkspace(int tokens, int width, bool swiglu = false)
    {
        if (tokens < 1 || width < 1 || (long)tokens * width > int.MaxValue / 4) throw new ArgumentException("Invalid Omega workspace dimensions.");
        var elements = tokens * width;
        Normalized = new float[elements]; Packed = new float[elements * 3]; Query = new float[elements];
        Key = new float[elements]; Value = new float[elements]; Attended = new float[elements];
        Projected = new float[elements]; Hidden = new float[elements * 4];
        if (swiglu) Gate = new float[elements * 4];
    }
}
