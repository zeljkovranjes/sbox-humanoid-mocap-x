// C# adaptation of SAM 3D Body's promptable decoder. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Inference;

/// <summary>Runs the six decoder layers on the GPU and intermediate pose feedback
/// on a worker thread. Create, run and dispose on the main thread.</summary>
internal sealed class SamGpuDecoder : IDisposable
{
    readonly FlowGpuStorage storage;
    readonly SamDecoderShape shape;
    readonly SamGpuDecoderBlock[] layers = new SamGpuDecoderBlock[SamDecoderShape.Layers];
    readonly SamLayerNorm finalNorm;
    readonly SamGpuDecoderWorkspace workspace;
    readonly TensorBatch batch;
    readonly GpuBuffer<float> tokens, image, positions, imagePositions;
    readonly float[] cpuTokens, cpuPositions, normalized;
    readonly Action prepare, record, update;
    ReadOnlyMemory<float> inputTokens, inputImage, inputPositions, inputImagePositions;
    Memory<float> destination;
    SamPoseFeedback feedback;
    CancellationToken cancellation;
    int layer;
    bool running, disposed;

    internal SamGpuDecoder(FlowWeights weights, SamDecoderShape shape, bool hand = false, FlowGpuWeights shared = null)
    {
        ThreadSafe.AssertIsMainThread();
        if (shape.Tokens == 0 || shape.HeadWidth is not (64 or 128)) throw new ArgumentException("GPU SAM needs 64 or 128 channels per head.");
        this.shape = shape; var name = hand ? "decoder_hand" : "decoder"; storage = new(shared);
        try
        {
            workspace = new(storage, shape);
            tokens = storage.Create<float>(shape.TokenElements); positions = storage.Create<float>(shape.TokenElements);
            image = storage.Create<float>(shape.ImageElements); imagePositions = storage.Create<float>(shape.ImageElements);
            cpuTokens = new float[shape.TokenElements]; cpuPositions = new float[shape.TokenElements]; normalized = new float[shape.TokenElements];
            for (var index = 0; index < layers.Length; index++) layers[index] = new(storage, weights, name + ".layers." + index, shape);
            finalNorm = new(weights, name + ".norm_final", shape.Width);
            prepare = Prepare; record = Record; update = Update; batch = new();
        }
        catch { storage.Dispose(); throw; }
    }

    /// <summary>Writes final normalized tokens. Feedback must be pure CPU code; it
    /// receives normalized tokens after layers 0–4 and updates keypoint tokens and
    /// positions. Keep input memory unchanged until completion. One call at a time.</summary>
    internal async Task RunAsync(ReadOnlyMemory<float> tokenInput, ReadOnlyMemory<float> imageInput,
        ReadOnlyMemory<float> tokenPositionInput, ReadOnlyMemory<float> imagePositionInput, Memory<float> output,
        SamPoseFeedback poseFeedback, CancellationToken token = default)
    {
        ThreadSafe.AssertIsMainThread(); ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(poseFeedback);
        if (running) throw new InvalidOperationException("Await the current SAM decoder before starting another.");
        token.ThrowIfCancellationRequested(); running = true; cancellation = token; feedback = poseFeedback;
        inputTokens = tokenInput; inputImage = imageInput; inputPositions = tokenPositionInput; inputImagePositions = imagePositionInput; destination = output;
        try
        {
            await Task.Run(prepare, token); await GameTask.MainThread();
            image.SetData<float>(inputImage.Span); imagePositions.SetData<float>(inputImagePositions.Span);
            for (layer = 0; layer < layers.Length; layer++)
            {
                token.ThrowIfCancellationRequested();
                tokens.SetData<float>(cpuTokens); positions.SetData<float>(cpuPositions);
                await batch.RunAsync(record, tokens, cpuTokens, token);
                await Task.Run(update, token); await GameTask.MainThread();
            }
            token.ThrowIfCancellationRequested(); normalized.AsSpan().CopyTo(destination.Span);
        }
        finally
        {
            await GameTask.MainThread(); running = false; cancellation = default; feedback = null;
            inputTokens = inputImage = inputPositions = inputImagePositions = default; destination = default;
        }
    }

    void Prepare()
    {
        SamDecoder.Validate(shape, inputTokens.Span, inputImage.Span, inputPositions.Span, inputImagePositions.Span, destination.Span);
        inputTokens.Span.CopyTo(cpuTokens); inputPositions.Span.CopyTo(cpuPositions);
    }

    void Record()
    { cancellation.ThrowIfCancellationRequested(); layers[layer].Run(workspace, tokens, image, positions, imagePositions, layer == 0); }

    void Update()
    {
        cancellation.ThrowIfCancellationRequested(); finalNorm.Run(cpuTokens, normalized); SamDecoder.Finite(normalized);
        if (layer == layers.Length - 1) return;
        feedback(layer, normalized, cpuTokens, cpuPositions, cancellation); SamDecoder.Finite(cpuTokens); SamDecoder.Finite(cpuPositions);
    }

    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        if (disposed) return;
        if (running) throw new InvalidOperationException("Await SAM decoder work and readback before disposal.");
        disposed = true; batch.Dispose(); storage.Dispose();
    }
}
