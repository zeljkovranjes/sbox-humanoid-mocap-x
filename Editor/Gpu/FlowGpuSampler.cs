// C# adaptation of FlowHMR's HunyuanMotionMMDiT and x1 sampler. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;
using TensorInference;

namespace HumanoidMocapX.Inference;

/// <summary>Runs all twenty FlowHMR steps with GPU-resident weights and working tensors.
/// Create, sample and dispose on the main thread. Instances support one capture at a time.</summary>
internal sealed class FlowGpuSampler : IDisposable
{
    readonly FlowGpuStorage storage = new();
    readonly TensorBatch batch;
    readonly FlowGpuWorkspace workspace;
    readonly FlowGpuLinear inputEncoder, featureEncoder, cameraEncoder, finalModulation, finalProjection;
    readonly FlowGpuMlp timeEncoder;
    readonly FlowGpuRefiner refiner;
    readonly FlowGpuDoubleBlock[] doubles;
    readonly FlowGpuSingleBlock[] singles;
    readonly GpuBuffer<float> body, camera, context, features, state, prediction, nextState;
    readonly GpuBuffer<float> mainTimes, refinerTimes, timeInput, timeCondition, textCondition, adapter, activatedAdapter;
    readonly Action recordStep;
    readonly float[] fence = new float[1];
    readonly int frames, width, latentElements;
    CancellationToken recordingCancellation;
    int step;
    bool running, disposed;

    internal FlowGpuSampler(FlowWeights weights, int frames, int width = FlowDenoiser.ModelWidth,
        int heads = FlowDenoiser.ModelHeads)
    {
        ThreadSafe.AssertIsMainThread();
        ArgumentNullException.ThrowIfNull(weights);
        if (frames < 1 || frames > FlowRotary.MaximumFrames || width < FlowGpuWorkspace.HeadWidth
            || width > FlowDenoiser.ModelWidth || heads < 1 || (long)heads * FlowGpuWorkspace.HeadWidth != width)
            throw new ArgumentException("GPU FlowHMR needs 1–5000 frames and 64-channel attention heads.");
        if (!Graphics.IsAvailable) throw new InvalidOperationException("GPU FlowHMR requires an initialized graphics device.");
        this.frames = frames; this.width = width; latentElements = checked(frames * FlowSampler.Channels);
        try
        {
            workspace = new(storage, frames, width, heads);
            inputEncoder = new(storage, weights, "input_encoder", FlowSampler.Channels, width);
            featureEncoder = new(storage, weights, "ctxt_encoder.feature", FlowDenoiser.FeatureChannels, width);
            cameraEncoder = new(storage, weights, "ctxt_encoder.camera_R", FlowDenoiser.CameraChannels, width);
            timeEncoder = new(storage, weights, "timestep_encoder.blocks", width);
            refiner = new(storage, weights, workspace);
            finalModulation = new(storage, weights, "final_layer.adaLN_modulation.linear", width, 2 * width);
            finalProjection = new(storage, weights, "final_layer.linear", width, FlowSampler.Channels);
            doubles = new FlowGpuDoubleBlock[FlowDenoiser.DoubleBlocks]; singles = new FlowGpuSingleBlock[FlowDenoiser.SingleBlocks];
            for (var i = 0; i < doubles.Length; i++) doubles[i] = new(storage, weights, i, width);
            for (var i = 0; i < singles.Length; i++) singles[i] = new(storage, weights, i, width);
            body = storage.Create<float>(frames * FlowDenoiser.FeatureChannels);
            camera = storage.Create<float>(frames * FlowDenoiser.CameraChannels);
            context = storage.Create<float>(workspace.StreamElements); features = storage.Create<float>(2 * workspace.StreamElements);
            state = storage.Create<float>(latentElements); prediction = storage.Create<float>(latentElements);
            nextState = storage.Create<float>(latentElements);
            mainTimes = Times(1000); refinerTimes = Times(1);
            timeInput = storage.Create<float>(width); timeCondition = storage.Create<float>(width);
            adapter = storage.Create<float>(width); activatedAdapter = storage.Create<float>(width);
            var text = new float[width];
            new FlowConditionMlp(weights, "vtxt_encoder.linears", FlowDenoiser.TextChannels, width)
                .Run(new float[FlowDenoiser.TextChannels], text, default);
            textCondition = storage.Upload<float>(text);
            recordStep = RecordStep;
            batch = new TensorBatch();
        }
        catch { storage.Dispose(); throw; }
    }

    /// <summary>Consumes [frames,3072] body features, [frames,9] relative camera rotations
    /// and [frames,491] initial noise. Writes sampled latents to output; discard output on
    /// failure or cancellation. Inputs are uploaded before the first asynchronous wait.</summary>
    internal async Task SampleAsync(ReadOnlyMemory<float> bodyFeatures, ReadOnlyMemory<float> cameraRotations,
        ReadOnlyMemory<float> initialNoise, Memory<float> output, CancellationToken cancellation = default)
    {
        ThreadSafe.AssertIsMainThread();
        ObjectDisposedException.ThrowIf(disposed, this);
        if (running) throw new InvalidOperationException("Await the current GPU capture before sampling again.");
        cancellation.ThrowIfCancellationRequested();
        Validate(bodyFeatures.Span, frames * FlowDenoiser.FeatureChannels, nameof(bodyFeatures));
        Validate(cameraRotations.Span, frames * FlowDenoiser.CameraChannels, nameof(cameraRotations));
        Validate(initialNoise.Span, latentElements, nameof(initialNoise));
        if (output.Length != latentElements) throw new ArgumentException("GPU output must have [frames,491] elements.", nameof(output));
        running = true; recordingCancellation = cancellation;
        Exception failure = null;
        try
        {
            body.SetData<float>(bodyFeatures.Span); camera.SetData<float>(cameraRotations.Span); state.SetData<float>(initialNoise.Span);
            for (step = 0; step < FlowSampler.Steps; step++)
            {
                cancellation.ThrowIfCancellationRequested();
                var destination = step == FlowSampler.Steps - 1 ? output : fence.AsMemory();
                // Fence each step to bound submissions and make cancellation observable without reading every tensor.
                await batch.RunAsync(recordStep, state, destination, cancellation);
                await GameTask.MainThread();
                Validate(destination.Span, destination.Length, "sampled motion");
            }
        }
        catch (Exception error) { failure = error; }
        await GameTask.MainThread();
        running = false; recordingCancellation = default;
        if (failure is OperationCanceledException && cancellation.IsCancellationRequested) cancellation.ThrowIfCancellationRequested();
        if (failure is not null) throw new InvalidOperationException("GPU FlowHMR sampling failed; discard the partial output.", failure);
    }

    GpuBuffer<float> Times(float factor)
    {
        var values = new float[FlowSampler.Steps * width];
        for (var index = 0; index < FlowSampler.Steps; index++)
            FlowConditioning.Time(index / (float)FlowSampler.Steps, factor, values.AsSpan(index * width, width));
        return storage.Upload<float>(values);
    }

    void RecordStep()
    {
        recordingCancellation.ThrowIfCancellationRequested();
        if (step == 0) SetVideo();
        var kernels = workspace.Kernels;
        inputEncoder.Run(kernels, state, features, frames);
        mainTimes.CopyTo(timeInput, step * width, 0, width);
        timeEncoder.Run(workspace, timeInput, timeCondition);
        kernels.Add(timeCondition, textCondition, adapter, width);
        kernels.Activate(adapter, activatedAdapter, width, TensorActivation.Silu);
        refinerTimes.CopyTo(timeInput, step * width, 0, width);
        refiner.Run(workspace, timeInput, features, recordingCancellation);
        foreach (var block in doubles)
        { recordingCancellation.ThrowIfCancellationRequested(); block.Run(workspace, features, activatedAdapter); }
        foreach (var block in singles)
        { recordingCancellation.ThrowIfCancellationRequested(); block.Run(workspace, features, activatedAdapter); }
        finalModulation.Run(kernels, activatedAdapter, workspace.MotionGates);
        kernels.Normalize(features, null, null, workspace.Normalized, frames, width, 1e-5f);
        kernels.Modulate(workspace.Normalized, workspace.MotionGates, workspace.Modified, frames, width, 0, width);
        finalProjection.Run(kernels, workspace.Modified, prediction, frames);
        var time = step / (float)FlowSampler.Steps;
        var delta = (step + 1) / (float)FlowSampler.Steps - time;
        kernels.Blend(state, prediction, nextState, latentElements, delta / Math.Max(1 - time, 1f / FlowSampler.Steps));
        nextState.CopyTo(state, latentElements);
    }

    void SetVideo()
    {
        featureEncoder.Run(workspace.Kernels, body, context, frames);
        cameraEncoder.Run(workspace.Kernels, camera, workspace.Normalized, frames);
        workspace.Kernels.Add(context, workspace.Normalized, workspace.Modified, workspace.StreamElements);
        workspace.Modified.CopyTo(context, workspace.StreamElements);
        refiner.SetVideo(workspace, context);
    }

    static void Validate(ReadOnlySpan<float> values, int length, string name)
    {
        if (values.Length != length) throw new ArgumentException("GPU FlowHMR tensor shape mismatch: " + name);
        foreach (var value in values)
            if (!float.IsFinite(value)) throw new ArithmeticException("Non-finite GPU FlowHMR tensor: " + name);
    }

    /// <summary>Releases weights, scratch storage and the offscreen view. Await any
    /// in-flight capture first, including one whose cancellation has been requested.</summary>
    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        if (disposed) return;
        if (running) throw new InvalidOperationException("Await GPU capture completion before disposing the sampler.");
        disposed = true; batch.Dispose(); storage.Dispose();
    }
}
