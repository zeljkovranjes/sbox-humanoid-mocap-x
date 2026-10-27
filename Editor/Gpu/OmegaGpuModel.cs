// C# adaptation of VGGT-Omega's image and alternating encoders. See LICENSE-VGGT and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
using System;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;
using TensorInference;

namespace HumanoidMocapX.Inference;

/// <summary>Camera-only Omega inference: 24 image blocks, 24 alternating frame/global
/// pairs and the four-block camera head. Loads one RGB image at a time and retains
/// features on the GPU. Create, run and dispose on the main thread.</summary>
internal sealed class OmegaGpuModel : IDisposable
{
    readonly FlowGpuStorage storage = new();
    readonly OmegaShape shape;
    readonly OmegaGpuWorkspace workspace;
    readonly OmegaGpuBlock[] frames = new OmegaGpuBlock[OmegaShape.Depth], interFrames = new OmegaGpuBlock[OmegaShape.Depth];
    readonly TensorBatch batch;
    readonly GpuBuffer<float> image, patches, features, registers, finalFrame, finalTokens;
    // Set by the build steps, so a model can be built part by part (LoadAsync).
    DinoGpuEncoder encoder;
    OmegaGpuCameraHead camera;
    GpuBuffer<float> framePositions, cameraToken, registerToken;
    readonly GpuBuffer<uint> frameRanges, globalRanges, registerRanges;
    readonly float[] input;
    readonly float[] fence = new float[1];
    readonly Action normalizeImage, recordImage, recordLayer;
    CancellationToken cancellation;
    int frame, layer;
    bool running, disposed;

    const string FrameBlocks = "aggregator.frame_blocks.", InterFrameBlocks = "aggregator.inter_frame_blocks.";
    static readonly string[] ImageParts = ["aggregator.patch_embed.", "aggregator.camera_token", "aggregator.register_token", "aggregator.rope_embed."];

    /// <summary>Builds the whole model from fully loaded weights.</summary>
    internal OmegaGpuModel(FlowWeights weights, OmegaShape shape) : this(shape)
    {
        try { BuildImage(weights); BuildBlocks(weights, frames, FrameBlocks); BuildBlocks(weights, interFrames, InterFrameBlocks); BuildHead(weights); }
        catch { Dispose(); throw; }
    }

    /// <summary>Builds the same model in four parts (image encoder, frame blocks, global blocks,
    /// camera head). <paramref name="load"/> reads one part's weights by name prefix and runs off
    /// the editor thread; each part is dropped once uploaded, so only one part's CPU weights is live
    /// at a time instead of the whole checkpoint (when they are collected is up to the GC).
    /// Failures and cancellation release everything built.</summary>
    internal static async Task<OmegaGpuModel> LoadAsync(Func<string[], FlowWeights> load, OmegaShape shape, CancellationToken token)
    {
        ThreadSafe.AssertIsMainThread(); ArgumentNullException.ThrowIfNull(load);
        token.ThrowIfCancellationRequested();
        var model = new OmegaGpuModel(shape);
        try
        {
            model.BuildImage(await Part(ImageParts));
            model.BuildBlocks(await Part([FrameBlocks]), model.frames, FrameBlocks);
            model.BuildBlocks(await Part([InterFrameBlocks]), model.interFrames, InterFrameBlocks);
            model.BuildHead(await Part(["camera_head."]));
            return model;
        }
        catch { await GameTask.MainThread(); model.Dispose(); throw; }

        async Task<FlowWeights> Part(string[] prefixes)
        {
            var part = await Task.Run(() => load(prefixes), token);
            await GameTask.MainThread(); token.ThrowIfCancellationRequested();
            return part;
        }
    }

    // Everything that needs no weights: shared GPU buffers, attention ranges and callbacks.
    OmegaGpuModel(OmegaShape shape)
    {
        ThreadSafe.AssertIsMainThread();
        if (shape.Frames == 0 || shape.Channels / shape.Heads != 64) throw new ArgumentException("GPU Omega image heads must have 64 channels.");
        this.shape = shape; var width = shape.Channels;
        input = new float[shape.ImageElements];
        try
        {
            workspace = new(storage, shape.Tokens, width);
            image = storage.Create<float>(input.Length); patches = storage.Create<float>(shape.Patches * width);
            features = storage.Create<float>(shape.Tokens * width);
            registers = storage.Create<float>(shape.Frames * OmegaShape.FramePrefix * width);
            finalFrame = storage.Create<float>(registers.ElementCount); finalTokens = storage.Create<float>(shape.CameraFeatures);
            frameRanges = Ranges(shape.Frames, shape.FrameTokens, true);
            globalRanges = Ranges(shape.Frames, shape.FrameTokens, false); registerRanges = Ranges(shape.Frames, OmegaShape.FramePrefix, false);
            normalizeImage = NormalizeImage; recordImage = RecordImage; recordLayer = RecordLayer;
            batch = new();
        }
        catch { storage.Dispose(); throw; }
    }

    void BuildImage(FlowWeights weights)
    {
        var width = shape.Channels;
        encoder = new(storage, weights, DinoShape.Omega(shape));
        cameraToken = storage.Weight(weights, "aggregator.camera_token", width, 1, 2, 1);
        registerToken = storage.Weight(weights, "aggregator.register_token", width, 16, 2, 1);
        framePositions = storage.Upload<float>(OmegaPosition.Factors(shape.Height / 16, shape.Width / 16, OmegaShape.FramePrefix,
            weights.Get("aggregator.rope_embed.periods", width / shape.Heads / 4).Data));
    }

    void BuildBlocks(FlowWeights weights, OmegaGpuBlock[] blocks, string prefix)
    {
        for (var i = 0; i < OmegaShape.Depth; i++) blocks[i] = new(storage, weights, prefix + i, shape.Channels, shape.Heads, true);
    }

    void BuildHead(FlowWeights weights) => camera = new(weights, shape.Frames, shape.Channels * 2, shape.Heads);

    /// <summary>Calls readImage for each sampled frame with a reusable HWC RGB [0,1]
    /// destination. The callback must fill the entire image before returning. Writes
    /// [frames,9] camera poses. Discard output on failure; instances allow one call at a time.</summary>
    internal async Task RunAsync(Func<int, Memory<float>, CancellationToken, Task> readImage, Memory<float> poses, CancellationToken token = default)
    {
        ThreadSafe.AssertIsMainThread(); ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(readImage);
        if (running) throw new InvalidOperationException("Await the current Omega inference before starting another.");
        if (poses.Length != shape.Frames * 9) throw new ArgumentException("Omega output needs nine camera values per frame.", nameof(poses));
        token.ThrowIfCancellationRequested(); running = true; cancellation = token;
        try
        {
            for (frame = 0; frame < shape.Frames; frame++)
            {
                token.ThrowIfCancellationRequested();
                // A partial reader must fail instead of reusing pixels from the preceding frame.
                input.AsSpan().Fill(float.NaN);
                await readImage(frame, input.AsMemory(), token);
                await Task.Run(normalizeImage, token);
                await GameTask.MainThread();
                image.SetData<float>(input);
                await batch.RunAsync(recordImage, features, fence, token);
                await GameTask.MainThread();
            }
            for (layer = 0; layer < OmegaShape.Depth; layer++)
            {
                token.ThrowIfCancellationRequested();
                await batch.RunAsync(recordLayer, features, fence, token);
                await GameTask.MainThread();
            }
            await camera.RunGpuAsync(finalTokens, poses, token);
        }
        finally { await GameTask.MainThread(); running = false; cancellation = default; }
    }

    void NormalizeImage() => DinoImageEncoder.NormalizeImage(input, input);

    void RecordImage()
    {
        var width = shape.Channels;
        encoder.Run(workspace, image, patches, cancellation);
        var start = frame * shape.FrameTokens * width;
        cameraToken.CopyTo(features, (frame == 0 ? 0 : 1) * width, start, width);
        registerToken.CopyTo(features, (frame == 0 ? 0 : 1) * 16 * width, start + width, 16 * width);
        patches.CopyTo(features, 0, start + OmegaShape.FramePrefix * width, shape.Patches * width);
    }

    void RecordLayer()
    {
        cancellation.ThrowIfCancellationRequested();
        frames[layer].Run(workspace, features, frameRanges, shape.Tokens, framePositions, shape.FrameTokens);
        if (layer == OmegaShape.Depth - 1) Gather(finalFrame);
        if (OmegaShape.RegistersOnly(layer))
        {
            Gather(registers); interFrames[layer].Run(workspace, registers, registerRanges, shape.Frames * OmegaShape.FramePrefix, sharedKeyRanges: true);
            for (var f = 0; f < shape.Frames; f++)
                registers.CopyTo(features, f * OmegaShape.FramePrefix * shape.Channels, f * shape.FrameTokens * shape.Channels, OmegaShape.FramePrefix * shape.Channels);
        }
        else interFrames[layer].Run(workspace, features, globalRanges, shape.Tokens, sharedKeyRanges: true);
        if (layer != OmegaShape.Depth - 1) return;
        Gather(registers);
        workspace.Kernels.ConcatColumns(finalFrame, registers, finalTokens, shape.Frames * OmegaShape.FramePrefix, shape.Channels, shape.Channels);
    }

    void Gather(GpuBuffer<float> destination)
    {
        for (var f = 0; f < shape.Frames; f++)
            features.CopyTo(destination, f * shape.FrameTokens * shape.Channels, f * OmegaShape.FramePrefix * shape.Channels, OmegaShape.FramePrefix * shape.Channels);
    }

    GpuBuffer<uint> Ranges(int count, int tokens, bool withinFrame)
    {
        var values = new uint[count * tokens * 4];
        for (var row = 0; row < count * tokens; row++)
        { values[row * 4] = withinFrame ? (uint)(row / tokens * tokens) : 0; values[row * 4 + 1] = (uint)(withinFrame ? tokens : count * tokens); }
        return storage.Upload<uint>(values);
    }

    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        if (disposed) return;
        if (running) throw new InvalidOperationException("Await Omega inference, including cancelled GPU work, before disposal.");
        disposed = true; batch?.Dispose(); camera?.Dispose(); storage.Dispose();
    }
}
