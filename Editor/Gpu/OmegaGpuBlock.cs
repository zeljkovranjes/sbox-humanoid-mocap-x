// C# adaptation of VGGT-Omega's inference transformer. See LICENSE-VGGT and THIRD_PARTY_NOTICES.md.
// SAM's SwiGLU configuration also retains the DINOv3 terms in LICENSE-DINOV3.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
using System;
using Sandbox;
using TensorInference;

namespace HumanoidMocapX.Inference;

internal sealed class OmegaGpuBlock
{
    readonly int width, heads, headWidth;
    readonly GpuBuffer<float> norm1, bias1, norm2, bias2, queryNorm, queryBias, keyNorm, keyBias, scale1, scale2, qkvWeight, qkvBias;
    readonly FlowGpuLinear projection, first, second, gate;

    internal OmegaGpuBlock(FlowGpuStorage storage, FlowWeights weights, string name, int width, int heads, bool normalizeQk, bool swiglu = false)
    {
        if (heads < 1 || width % heads != 0 || width / heads is not (64 or 128)) throw new ArgumentException("GPU Omega needs 64 or 128 channels per head.");
        this.width = width; this.heads = heads; headWidth = width / heads;
        norm1 = storage.Weight(weights, name + ".norm1.weight", width); bias1 = storage.Weight(weights, name + ".norm1.bias", width);
        norm2 = storage.Weight(weights, name + ".norm2.weight", width); bias2 = storage.Weight(weights, name + ".norm2.bias", width);
        scale1 = storage.Weight(weights, name + ".ls1.gamma", width); scale2 = storage.Weight(weights, name + ".ls2.gamma", width);
        qkvWeight = storage.Weight(weights, name + ".attn.qkv.weight", width, width * 3);
        qkvBias = storage.Upload<float>(OmegaBlock.MaskedBias(weights, name + ".attn.qkv", width).Data);
        projection = new(storage, weights, name + ".attn.proj", width, width);
        first = new(storage, weights, name + (swiglu ? ".mlp.w1" : ".mlp.fc1"), width, width * 4);
        second = new(storage, weights, name + (swiglu ? ".mlp.w3" : ".mlp.fc2"), width * 4, width);
        if (swiglu) gate = new(storage, weights, name + ".mlp.w2", width, width * 4);
        if (normalizeQk)
        {
            queryNorm = storage.Weight(weights, name + ".attn.q_norm.weight", headWidth); queryBias = storage.Weight(weights, name + ".attn.q_norm.bias", headWidth);
            keyNorm = storage.Weight(weights, name + ".attn.k_norm.weight", headWidth); keyBias = storage.Weight(weights, name + ".attn.k_norm.bias", headWidth);
        }
    }

    internal void Run(OmegaGpuWorkspace workspace, GpuBuffer<float> values, GpuBuffer<uint> ranges, int tokens,
        GpuBuffer<float> factors = null, int positions = 0, bool sharedKeyRanges = false)
    {
        var kernels = workspace.Kernels; var count = checked(tokens * width);
        if (gate is not null && (workspace.Gate is null || workspace.Gate.ElementCount < (long)count * 4))
            throw new ArgumentException("DINO SwiGLU needs a gated workspace.");
        kernels.Normalize(values, norm1, bias1, workspace.Normalized, tokens, width);
        kernels.Linear(workspace.Normalized, qkvWeight, qkvBias, workspace.Packed, tokens, width, width * 3);
        kernels.CopyColumns(workspace.Packed, workspace.Query, tokens, width * 3, 0, width);
        kernels.CopyColumns(workspace.Packed, workspace.Key, tokens, width * 3, width, width);
        kernels.CopyColumns(workspace.Packed, workspace.Value, tokens, width * 3, width * 2, width);
        if (queryNorm is not null)
        {
            kernels.Normalize(workspace.Query, queryNorm, queryBias, workspace.Normalized, tokens * heads, headWidth);
            workspace.Normalized.CopyTo(workspace.Query, count);
            kernels.Normalize(workspace.Key, keyNorm, keyBias, workspace.Normalized, tokens * heads, headWidth);
            workspace.Normalized.CopyTo(workspace.Key, count);
        }
        if (factors is not null)
        {
            kernels.Rotary(workspace.Query, factors, tokens, width, headWidth, positions, true);
            kernels.Rotary(workspace.Key, factors, tokens, width, headWidth, positions, true);
        }
        kernels.Attention(workspace.Query, workspace.Key, workspace.Value, ranges, workspace.Attended, tokens, heads, headWidth,
            sharedKeyRanges: sharedKeyRanges);
        projection.Run(kernels, workspace.Attended, workspace.Projected, tokens);
        kernels.GatedAdd(values, workspace.Projected, scale1, workspace.Residual, tokens, width, 0);
        kernels.Normalize(workspace.Residual, norm2, bias2, workspace.Normalized, tokens, width);
        first.Run(kernels, workspace.Normalized, workspace.Hidden, tokens);
        if (gate is null) kernels.Activate(workspace.Hidden, workspace.Activated, count * 4, TensorActivation.GeluErf);
        else
        {
            gate.Run(kernels, workspace.Normalized, workspace.Gate, tokens);
            kernels.SwiGlu(workspace.Hidden, workspace.Gate, workspace.Activated, count * 4);
        }
        second.Run(kernels, workspace.Activated, workspace.Projected, tokens);
        kernels.GatedAdd(workspace.Residual, workspace.Projected, scale2, values, tokens, width, 0);
    }
}

internal sealed class OmegaGpuWorkspace
{
    internal readonly TensorKernels Kernels = new();
    internal readonly GpuBuffer<float> Normalized, Packed, Query, Key, Value, Attended, Projected, Hidden, Activated, Residual, Gate;

    internal OmegaGpuWorkspace(FlowGpuStorage storage, int tokens, int width, bool swiglu = false)
    {
        if (tokens < 1 || width < 1 || (long)tokens * width > uint.MaxValue / (4L * sizeof(float))) throw new ArgumentException("Invalid GPU Omega workspace dimensions.");
        var count = tokens * width;
        Normalized = storage.Create<float>(count); Packed = storage.Create<float>(count * 3);
        Query = storage.Create<float>(count); Key = storage.Create<float>(count); Value = storage.Create<float>(count);
        Attended = storage.Create<float>(count); Projected = storage.Create<float>(count); Residual = storage.Create<float>(count);
        Hidden = storage.Create<float>(count * 4); Activated = storage.Create<float>(count * 4);
        if (swiglu) Gate = storage.Create<float>(count * 4);
    }
}
