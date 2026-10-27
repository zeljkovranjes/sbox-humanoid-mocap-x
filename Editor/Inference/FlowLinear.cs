#nullable enable annotations
using System;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

internal sealed class FlowLinear
{
    // Below this many multiply-adds, thread start-up costs more than it saves.
    const long ParallelMultiplies = 1 << 22;
    const int ChannelsPerBlock = 16;
    static readonly ParallelOptions Parallelism = new() { MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount - 1) };
    readonly Tensor weight, bias;

    internal FlowLinear(FlowWeights weights, string name, int inputs, int outputs)
    {
        weight = weights.Get(name + ".weight", inputs, outputs);
        bias = weights.Get(name + ".bias", outputs);
    }

    internal void Run(ReadOnlySpan<float> input, int rows, Span<float> output, CancellationToken cancellation = default)
        => Run(input, rows, weight, bias, output, cancellation);

    /// <summary>Same result as <see cref="Run(ReadOnlySpan{float}, int, Span{float}, CancellationToken)"/>,
    /// with output channels split across threads for large layers. Each output keeps the
    /// same dot product, so values are identical. Small layers run serially without allocating.</summary>
    internal void RunParallel(float[] input, int rows, float[] output, CancellationToken cancellation = default)
    {
        if ((long)rows * weight.Cols * weight.Rows < ParallelMultiplies) Run(input, rows, output, cancellation);
        else RunBlocks(input, rows, output, cancellation);
    }

    // Separate method: its lambda's captured state is allocated on entry, which the serial path must avoid.
    void RunBlocks(float[] input, int rows, float[] output, CancellationToken cancellation)
    {
        Validate(input, rows, weight, bias, output);
        int width = weight.Cols, channels = weight.Rows;
        // Cancellation is checked around the loop so callers see OperationCanceledException, not AggregateException.
        cancellation.ThrowIfCancellationRequested();
        Parallel.For(0, (channels + ChannelsPerBlock - 1) / ChannelsPerBlock, Parallelism, block =>
        {
            for (var channel = block * ChannelsPerBlock; channel < Math.Min(channels, (block + 1) * ChannelsPerBlock); channel++)
            {
                var weights = weight.Row(channel); var offset = bias?.Data[channel] ?? 0;
                for (var row = 0; row < rows; row++)
                    output[row * channels + channel] = offset + NeuralOps.Dot(weights, input.AsSpan(row * width, width));
            }
        });
        cancellation.ThrowIfCancellationRequested();
    }

    /// <summary>Uses the shared SIMD dot product with caller-owned output; rejects
    /// mismatched dimensions and input/output aliasing. Does not allocate.</summary>
    internal static void Run(ReadOnlySpan<float> input, int rows, Tensor weight, Tensor? bias, Span<float> output,
        CancellationToken cancellation = default)
    {
        Validate(input, rows, weight, bias, output);
        int width = weight.Cols, channels = weight.Rows;
        for (var channel = 0; channel < channels; channel++)
        {
            cancellation.ThrowIfCancellationRequested();
            var weights = weight.Row(channel);
            var offset = bias?.Data[channel] ?? 0;
            for (var row = 0; row < rows; row++)
                output[row * channels + channel] = offset + NeuralOps.Dot(weights, input.Slice(row * width, width));
        }
    }

    static void Validate(ReadOnlySpan<float> input, int rows, Tensor weight, Tensor? bias, Span<float> output)
    {
        ArgumentNullException.ThrowIfNull(weight);
        int width = weight.Cols, channels = weight.Rows;
        if (rows < 0 || input.Length != (long)rows * width || output.Length != (long)rows * channels)
            throw new ArgumentException("Linear input/output shape does not match " + weight.Name);
        if (bias is not null && bias.Data.Length != channels)
            throw new ArgumentException("Linear bias shape does not match " + weight.Name);
        if (input.Overlaps(output) || weight.Data.AsSpan().Overlaps(output) || (bias is not null && bias.Data.AsSpan().Overlaps(output)))
            throw new ArgumentException("Linear output must not overlap its inputs.");
    }
}
