// C# adaptation of VGGT-Omega's camera head. See LICENSE-VGGT and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
using System;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Inference;

internal sealed class OmegaGpuCameraHead : IDisposable
{
    readonly FlowGpuStorage storage = new();
    readonly OmegaGpuWorkspace workspace;
    readonly OmegaGpuBlock[] blocks = new OmegaGpuBlock[OmegaCameraHead.Blocks];
    readonly FlowGpuLinear first, second;
    readonly GpuBuffer<float> tokenWeight, tokenBias, outputWeight, outputBias, input, features, pooled, normalized, hidden, activated, poses;
    readonly GpuBuffer<uint> ranges;
    readonly TensorBatch batch;
    readonly Action record;
    readonly int frames, width, tokens;
    GpuBuffer<float> source;
    CancellationToken cancellation;
    bool running, disposed;

    internal OmegaGpuCameraHead(FlowWeights weights, int frames, int width = 2048, int heads = 16)
    {
        ThreadSafe.AssertIsMainThread();
        if (frames < 1 || frames > FlowMotion.MaximumFrames / OmegaCameraHead.PrefixTokens || heads < 1
            || width % heads != 0 || width / heads is not (64 or 128)) throw new ArgumentException("Invalid GPU Omega camera dimensions.");
        this.frames = frames; this.width = width; tokens = frames * OmegaCameraHead.PrefixTokens;
        try
        {
            workspace = new(storage, tokens, width);
            input = storage.Create<float>(checked(tokens * width)); features = storage.Create<float>(tokens * width);
            pooled = storage.Create<float>(frames * width); normalized = storage.Create<float>(frames * width);
            hidden = storage.Create<float>(frames * width / 2); activated = storage.Create<float>(frames * width / 2);
            poses = storage.Create<float>(frames * 9);
            tokenWeight = storage.Weight(weights, "camera_head.token_norm.weight", width); tokenBias = storage.Weight(weights, "camera_head.token_norm.bias", width);
            outputWeight = storage.Weight(weights, "camera_head.trunk_norm.weight", width); outputBias = storage.Weight(weights, "camera_head.trunk_norm.bias", width);
            for (var index = 0; index < blocks.Length; index++) blocks[index] = new(storage, weights, "camera_head.trunk." + index, width, heads, false);
            first = new(storage, weights, "camera_head.camera_branch.0", width, width / 2);
            second = new(storage, weights, "camera_head.camera_branch.2", width / 2, 9);
            var keyRanges = new uint[tokens * 4];
            for (var token = 0; token < tokens; token++) keyRanges[token * 4 + 1] = (uint)tokens;
            ranges = storage.Upload<uint>(keyRanges);
            source = input; record = Record; batch = new();
        }
        catch { storage.Dispose(); throw; }
    }

    /// <summary>Consumes GPU-resident final tokens. The caller retains the source
    /// until readback completes, including after cancellation.</summary>
    internal async Task RunGpuAsync(GpuBuffer<float> tokensInput, Memory<float> output, CancellationToken token = default)
    {
        ThreadSafe.AssertIsMainThread(); ObjectDisposedException.ThrowIf(disposed, this);
        if (running) throw new InvalidOperationException("Await the current Omega camera prediction before starting another.");
        token.ThrowIfCancellationRequested();
        if (tokensInput is null || !tokensInput.IsValid || tokensInput.ElementCount != tokens * width || output.Length != frames * 9)
            throw new ArgumentException("GPU Omega camera tensors have incorrect dimensions.");
        running = true; cancellation = token; source = tokensInput;
        try
        {
            await batch.RunAsync(record, poses, output, token);
            OmegaCameraHead.Activate(output.Span);
        }
        finally { await GameTask.MainThread(); source = input; running = false; cancellation = default; }
    }

    /// <summary>Writes [frames,9] poses from [frames,17,width] final aggregator tokens.
    /// One call at a time, on the main thread. Discard partial output after failure.</summary>
    internal async Task RunAsync(ReadOnlyMemory<float> tokensInput, Memory<float> output, CancellationToken token = default)
    {
        ThreadSafe.AssertIsMainThread(); ObjectDisposedException.ThrowIf(disposed, this);
        if (running) throw new InvalidOperationException("Await the current Omega camera prediction before starting another.");
        token.ThrowIfCancellationRequested();
        if (tokensInput.Length != tokens * width || output.Length != frames * 9)
            throw new ArgumentException("GPU Omega camera tensors have incorrect dimensions.");
        foreach (var value in tokensInput.Span)
            if (!float.IsFinite(value)) throw new ArgumentException("Non-finite Omega camera feature.", nameof(tokensInput));
        running = true; cancellation = token;
        try
        {
            input.SetData<float>(tokensInput.Span);
            await batch.RunAsync(record, poses, output, token);
            OmegaCameraHead.Activate(output.Span);
        }
        finally { await GameTask.MainThread(); running = false; cancellation = default; }
    }

    void Record()
    {
        var kernels = workspace.Kernels;
        kernels.Normalize(source, tokenWeight, tokenBias, features, tokens, width);
        foreach (var block in blocks) { cancellation.ThrowIfCancellationRequested(); block.Run(workspace, features, ranges, tokens); }
        for (var frame = 0; frame < frames; frame++) features.CopyTo(pooled, frame * OmegaCameraHead.PrefixTokens * width, frame * width, width);
        kernels.Normalize(pooled, outputWeight, outputBias, normalized, frames, width);
        first.Run(kernels, normalized, hidden, frames);
        kernels.Activate(hidden, activated, frames * width / 2, TensorActivation.GeluErf);
        second.Run(kernels, activated, poses, frames);
    }

    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        if (disposed) return;
        if (running) throw new InvalidOperationException("Await Omega camera readback before disposal.");
        disposed = true; batch.Dispose(); storage.Dispose();
    }
}
