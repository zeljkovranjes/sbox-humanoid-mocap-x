// C# adaptation of SAM 3D Body's image branch. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Inference;

/// <summary>Owns one shared DINOv3 encoder and separate body/hand image conditioning.
/// Create, run and dispose on the main thread; crop and ray preparation stay pure C#.</summary>
internal sealed class SamGpuImageEncoder : IDisposable
{
    readonly FlowGpuStorage storage;
    readonly DinoShape shape;
    readonly DinoGpuEncoder encoder;
    readonly OmegaGpuWorkspace workspace;
    readonly SamGpuImageConditioning bodyConditioning, handConditioning;
    readonly TensorBatch batch;
    readonly GpuBuffer<float> pixels, features, rays, context;
    readonly float[] normalized, bodyPositions, handPositions;
    readonly Action prepare, record;
    ReadOnlyMemory<float> inputImage, inputRays;
    Memory<float> destination;
    CancellationToken cancellation;
    bool isHand, running, disposed;

    internal SamGpuImageEncoder(FlowWeights backbone, FlowWeights body, FlowWeights hand, DinoShape shape, FlowGpuWeights shared = null)
    {
        ThreadSafe.AssertIsMainThread();
        if (!shape.IsSam) throw new ArgumentException("SAM image encoding requires a SAM DINO shape.");
        this.shape = shape; storage = new(shared);
        try
        {
            workspace = new(storage, shape.Tokens, shape.Channels, true); encoder = new(storage, backbone, shape);
            bodyConditioning = new(storage, body, shape); handConditioning = new(storage, hand, shape, true);
            bodyPositions = SamImageConditioning.Positions(body, shape, false); handPositions = SamImageConditioning.Positions(hand, shape, true);
            pixels = storage.Create<float>(shape.ImageElements); normalized = new float[shape.ImageElements];
            features = storage.Create<float>(shape.Patches * shape.Channels); context = storage.Create<float>(features.ElementCount);
            rays = storage.Create<float>(shape.Patches * SamRays.Channels);
            prepare = Prepare; record = Record; batch = new();
        }
        catch { storage.Dispose(); throw; }
    }

    /// <summary>Encodes RGB [0,1] and camera-ray features into the decoder context.
    /// Keeps all inputs unchanged until completion. Rejects invalid dimensions,
    /// aliasing, non-finite values, concurrent calls and use after disposal.</summary>
    internal async Task RunAsync(ReadOnlyMemory<float> image, ReadOnlyMemory<float> rayFeatures, Memory<float> output,
        bool hand = false, CancellationToken token = default)
    {
        ThreadSafe.AssertIsMainThread(); ObjectDisposedException.ThrowIf(disposed, this);
        if (running) throw new InvalidOperationException("Await the current SAM image encoding before starting another.");
        token.ThrowIfCancellationRequested(); running = true; cancellation = token; isHand = hand;
        inputImage = image; inputRays = rayFeatures; destination = output;
        try
        {
            await Task.Run(prepare, token); await GameTask.MainThread();
            pixels.SetData<float>(normalized); rays.SetData<float>(inputRays.Span);
            await batch.RunAsync(record, context, destination, token);
            SamDecoder.Finite(destination.Span);
        }
        finally
        {
            await GameTask.MainThread(); running = false; cancellation = default;
            inputImage = inputRays = default; destination = default;
        }
    }

    /// <summary>Copies fixed positional image context for the selected decoder.</summary>
    internal void CopyPositions(Span<float> output, bool hand = false)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (output.Length != bodyPositions.Length) throw new ArgumentException("SAM positional context has the wrong size.");
        (hand ? handPositions : bodyPositions).AsSpan().CopyTo(output);
    }

    void Prepare()
    {
        if (inputImage.Length != shape.ImageElements || inputRays.Length != shape.Patches * SamRays.Channels
            || destination.Length != shape.Patches * shape.Channels || inputImage.Span.Overlaps(destination.Span) || inputRays.Span.Overlaps(destination.Span))
            throw new ArgumentException("SAM image encoder inputs/output have invalid dimensions or overlap.");
        cancellation.ThrowIfCancellationRequested(); SamDecoder.Finite(inputRays.Span);
        DinoImageEncoder.NormalizeImage(inputImage.Span, normalized);
    }

    void Record()
    {
        encoder.Run(workspace, pixels, features, cancellation);
        (isHand ? handConditioning : bodyConditioning).Run(workspace.Kernels, features, rays, context, cancellation);
    }

    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        if (disposed) return;
        if (running) throw new InvalidOperationException("Await SAM image encoding and readback before disposal.");
        disposed = true; batch.Dispose(); storage.Dispose();
    }
}
