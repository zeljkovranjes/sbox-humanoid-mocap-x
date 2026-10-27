// C# adaptation of FlowHMR's HunyuanMotionMMDiT. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Batch-one, unpadded FlowHMR x1 predictor. Video conditioning must come
/// from SAM 3D Body and relative world-to-camera rotations. Instances are not thread safe.</summary>
internal sealed class FlowDenoiser
{
    internal const int FeatureChannels = 3072, CameraChannels = 9, TextChannels = 768;
    internal const int ModelWidth = 1024, ModelHeads = 16, DoubleBlocks = 6, SingleBlocks = 12;
    readonly int frames, width;
    readonly FlowLinear inputEncoder, featureEncoder, cameraEncoder, finalModulation, finalProjection;
    readonly FlowConditionMlp timeEncoder;
    readonly FlowRefiner refiner;
    readonly FlowDoubleBlock[] doubles;
    readonly FlowSingleBlock[] singles;
    readonly FlowBlockBuffers buffers;
    readonly FlowLayerNorm finalNorm;
    readonly float[] features, context, textCondition, timeEmbedding, adapter, activatedAdapter;
    bool hasVideo;

    internal FlowDenoiser(FlowWeights weights, int frames, int width = ModelWidth, int heads = ModelHeads)
    {
        this.frames = frames; this.width = width;
        buffers = new(frames, width, heads);
        inputEncoder = new(weights, "input_encoder", FlowSampler.Channels, width);
        featureEncoder = new(weights, "ctxt_encoder.feature", FeatureChannels, width);
        cameraEncoder = new(weights, "ctxt_encoder.camera_R", CameraChannels, width);
        timeEncoder = new(weights, "timestep_encoder.blocks", width, width);
        var textEncoder = new FlowConditionMlp(weights, "vtxt_encoder.linears", TextChannels, width);
        refiner = new(weights, frames, width, heads);
        finalModulation = new(weights, "final_layer.adaLN_modulation.linear", width, 2 * width);
        finalProjection = new(weights, "final_layer.linear", width, FlowSampler.Channels);
        finalNorm = new(width, 1e-5f);
        features = new float[2 * buffers.StreamElements]; context = new float[buffers.StreamElements];
        textCondition = new float[width]; timeEmbedding = new float[width]; adapter = new float[width]; activatedAdapter = new float[width];
        textEncoder.Run(new float[TextChannels], textCondition, default);
        doubles = new FlowDoubleBlock[DoubleBlocks]; singles = new FlowSingleBlock[SingleBlocks];
        for (var i = 0; i < doubles.Length; i++) doubles[i] = new(weights, i, width, heads);
        for (var i = 0; i < singles.Length; i++) singles[i] = new(weights, i, width, heads);
    }

    /// <summary>Binds [frames,3072] body features and [frames,9] camera rotations.
    /// Inputs are consumed immediately. Failure invalidates the previous video binding.</summary>
    internal void SetVideo(ReadOnlySpan<float> bodyFeatures, ReadOnlySpan<float> cameraRotations, CancellationToken cancellation = default)
    {
        hasVideo = false;
        cancellation.ThrowIfCancellationRequested();
        Validate(bodyFeatures, (long)frames * FeatureChannels, nameof(bodyFeatures));
        Validate(cameraRotations, (long)frames * CameraChannels, nameof(cameraRotations));
        featureEncoder.Run(bodyFeatures, frames, context, cancellation);
        var cameras = buffers.Normalized.AsSpan(0, context.Length);
        cameraEncoder.Run(cameraRotations, frames, cameras, cancellation);
        NeuralOps.AddInPlace(context, cameras);
        Validate(context, context.Length, "encoded video");
        hasVideo = true;
    }

    /// <summary>Predicts clean [frames,491] motion at time in [0,1]. Writes the output
    /// buffer without allocating; cancellation leaves partial output that must be discarded.</summary>
    internal void Predict(ReadOnlySpan<float> state, float time, Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (!hasVideo) throw new InvalidOperationException("Bind video features before FlowHMR prediction.");
        Validate(state, (long)frames * FlowSampler.Channels, nameof(state));
        if (output.Length != state.Length || state.Overlaps(output)) throw new ArgumentException("Prediction needs a separate output tensor with the state shape.");
        if (!float.IsFinite(time) || time < 0 || time > 1) throw new ArgumentOutOfRangeException(nameof(time));
        var count = buffers.StreamElements;
        inputEncoder.Run(state, frames, features.AsSpan(0, count), cancellation);
        FlowConditioning.Time(time, 1000, timeEmbedding);
        timeEncoder.Run(timeEmbedding, adapter, cancellation); NeuralOps.AddInPlace(adapter, textCondition);
        adapter.AsSpan().CopyTo(activatedAdapter); FlowConditioning.Silu(activatedAdapter);
        refiner.Run(context, time, features.AsSpan(count), cancellation);
        foreach (var block in doubles) block.Run(features, activatedAdapter, buffers, cancellation);
        foreach (var block in singles) block.Run(features, activatedAdapter, buffers, cancellation);
        var gates = buffers.Gates.AsSpan(0, 2 * width); var normalized = buffers.Normalized.AsSpan(0, count);
        finalModulation.Run(activatedAdapter, 1, gates, cancellation);
        finalNorm.Run(features.AsSpan(0, count), normalized);
        FlowConditioning.Modulate(normalized, gates[..width], gates[width..]);
        finalProjection.Run(normalized, frames, output, cancellation);
        Validate(output, output.Length, "predicted motion");
    }

    static void Validate(ReadOnlySpan<float> values, long length, string name)
    {
        if (values.Length != length) throw new ArgumentException("FlowHMR tensor shape mismatch: " + name);
        foreach (var value in values)
            if (!float.IsFinite(value)) throw new ArithmeticException("Non-finite FlowHMR tensor: " + name);
    }
}
