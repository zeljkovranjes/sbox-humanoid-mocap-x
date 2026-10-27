using System;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;
using TensorInference;

namespace HumanoidMocapX.Inference;

/// <summary>Executes YOLOX-L entirely through s&amp;box compute shaders. Construction,
/// inference and disposal require the main thread; one frame may be in flight.</summary>
internal sealed class YoloGpuDetector : IDisposable
{
    readonly FlowGpuStorage storage = new();
    readonly YoloNetwork network;
    readonly TensorKernels kernels = new();
    readonly TensorBatch batch;
    readonly GpuBuffer<float>[] values, weights, biases;
    readonly ConvolutionShape[] convolutions;
    readonly GpuBuffer<float> predictions;
    readonly Action record;
    readonly Action prepareImage, decodeBoxes;
    readonly float[] preparedImage = new float[YoloNetwork.ImageSize * YoloNetwork.ImageSize * 3];
    readonly float[] logits = new float[YoloNetwork.Predictions * YoloNetwork.PredictionChannels];
    readonly YoloDetection detection = new();
    ReadOnlyMemory<byte> sourceImage;
    Memory<YoloBox> destinationBoxes;
    int sourceWidth, sourceHeight, personCount;
    CancellationToken cancellation;
    bool running, disposed;

    internal YoloGpuDetector(YoloNetwork network, YoloFilter[] filters)
    {
        ThreadSafe.AssertIsMainThread();
        ArgumentNullException.ThrowIfNull(network); ArgumentNullException.ThrowIfNull(filters);
        if (filters.Length != network.Nodes.Length) throw new ArgumentException("YOLOX filters do not match the network.", nameof(filters));
        if (!Graphics.IsAvailable) throw new InvalidOperationException("YOLOX requires an initialized graphics device.");
        this.network = network;
        values = new GpuBuffer<float>[network.Nodes.Length]; weights = new GpuBuffer<float>[values.Length];
        biases = new GpuBuffer<float>[values.Length]; convolutions = new ConvolutionShape[values.Length];
        try
        {
            var slots = new GpuBuffer<float>[network.Capacities.Length];
            for (var index = 0; index < slots.Length; index++) slots[index] = storage.Create<float>(network.Capacities[index]);
            for (var index = 0; index < values.Length; index++)
            {
                values[index] = slots[network.Slots[index]];
                var node = network.Nodes[index];
                if (node.Operation != YoloOperation.Convolution) continue;
                var input = network.Nodes[node.First];
                convolutions[index] = new(input.Size, input.Size, input.Channels, node.Channels, node.Kernel, node.Stride, node.Kernel / 2);
                var filter = filters[index];
                if (filter is null) throw new ArgumentException("Missing YOLOX filter: " + node.Weight, nameof(filters));
                Validate(filter.Weight, convolutions[index].WeightElements); Validate(filter.Bias, node.Channels);
                weights[index] = storage.Upload<float>(filter.Weight); biases[index] = storage.Upload<float>(filter.Bias);
            }
            predictions = storage.Create<float>(YoloNetwork.Predictions * YoloNetwork.PredictionChannels);
            record = Record; prepareImage = PrepareImage; decodeBoxes = DecodeBoxes; batch = new();
        }
        catch { storage.Dispose(); throw; }
    }

    /// <summary>Detects people in an RGBA8 frame, writing original-pixel boxes into
    /// output (capacity at least 8400). CPU image conversion and suppression run off
    /// the editor thread. Keep source/output memory alive until the call completes.</summary>
    internal async Task<int> DetectAsync(ReadOnlyMemory<byte> rgba, int width, int height, Memory<YoloBox> output,
        CancellationToken token = default)
    {
        ThreadSafe.AssertIsMainThread(); ObjectDisposedException.ThrowIf(disposed, this);
        if (running) throw new InvalidOperationException("Await the previous YOLOX frame before submitting another.");
        token.ThrowIfCancellationRequested();
        if (output.Length < YoloNetwork.Predictions) throw new ArgumentException("YOLOX output needs capacity for 8400 boxes.", nameof(output));
        running = true; cancellation = token;
        sourceImage = rgba; sourceWidth = width; sourceHeight = height; destinationBoxes = output;
        try
        {
            await Task.Run(prepareImage, token);
            await GameTask.MainThread();
            await SubmitAsync(preparedImage, logits, token);
            await Task.Run(decodeBoxes, token);
            return personCount;
        }
        finally
        {
            await GameTask.MainThread();
            sourceImage = default; destinationBoxes = default; cancellation = default; running = false;
        }
    }

    void PrepareImage() => YoloImage.Prepare(sourceImage.Span, sourceWidth, sourceHeight, preparedImage, cancellation);
    void DecodeBoxes() => personCount = detection.Decode(logits, sourceWidth, sourceHeight, destinationBoxes.Span, cancellation);

    /// <summary>Consumes a 640×640 BGR HWC image with values in [0,255] and writes
    /// [8400,85] raw logits in stride 8/16/32 order. Discard output on cancellation/failure.</summary>
    internal async Task PredictAsync(ReadOnlyMemory<float> image, Memory<float> output, CancellationToken token = default)
    {
        ThreadSafe.AssertIsMainThread(); ObjectDisposedException.ThrowIf(disposed, this);
        if (running) throw new InvalidOperationException("Await the previous YOLOX frame before submitting another.");
        token.ThrowIfCancellationRequested(); Validate(image.Span, network.Nodes[0].Elements);
        foreach (var pixel in image.Span)
            if (pixel < 0 || pixel > 255) throw new ArgumentException("YOLOX input pixels must be between 0 and 255.", nameof(image));
        if (output.Length != predictions.ElementCount) throw new ArgumentException("YOLOX output must have [8400,85] elements.", nameof(output));
        running = true; cancellation = token;
        try
        {
            await SubmitAsync(image, output, token);
            Validate(output.Span, output.Length);
        }
        finally { await GameTask.MainThread(); running = false; cancellation = default; }
    }

    Task SubmitAsync(ReadOnlyMemory<float> image, Memory<float> output, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        values[0].SetData<float>(image.Span);
        return batch.RunAsync(record, predictions, output, token);
    }

    void Record()
    {
        for (var index = 1; index < network.Nodes.Length; index++)
        {
            cancellation.ThrowIfCancellationRequested();
            var node = network.Nodes[index]; var source = network.Nodes[node.First];
            var input = values[node.First]; var output = values[index];
            switch (node.Operation)
            {
                case YoloOperation.Convolution: kernels.Convolve(input, weights[index], biases[index], output, convolutions[index]); break;
                case YoloOperation.Silu: kernels.Activate(input, output, node.Elements, TensorActivation.Silu); break;
                case YoloOperation.Add: kernels.Add(input, values[node.Second], output, node.Elements); break;
                case YoloOperation.Concat:
                    kernels.ConcatColumns(input, values[node.Second], output, node.Size * node.Size, source.Channels, network.Nodes[node.Second].Channels); break;
                case YoloOperation.Focus: kernels.Focus(input, output, source.Size, source.Size, source.Channels); break;
                case YoloOperation.MaxPool: kernels.MaxPool(input, output, source.Size, source.Size, source.Channels, node.Kernel); break;
                case YoloOperation.Upsample: kernels.UpsampleNearest(input, output, source.Size, source.Size, source.Channels); break;
                default: throw new InvalidOperationException("Unsupported YOLOX operation.");
            }
        }
        var offset = 0;
        foreach (var head in network.Heads)
        {
            var length = network.Nodes[head].Elements;
            values[head].CopyTo(predictions, 0, offset, length); offset += length;
        }
    }

    static void Validate(ReadOnlySpan<float> values, int count)
    {
        if (values.Length != count) throw new ArgumentException("YOLOX tensor dimensions disagree.");
        foreach (var value in values)
            if (!float.IsFinite(value)) throw new ArithmeticException("YOLOX tensor contains non-finite values.");
    }

    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        if (disposed) return;
        if (running) throw new InvalidOperationException("Await YOLOX inference, including cancelled work, before disposing it.");
        disposed = true; batch.Dispose(); storage.Dispose();
    }
}
