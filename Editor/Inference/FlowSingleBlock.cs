// C# adaptation of FlowHMR's MMSingleStreamBlock. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;
using HumanoidMocapX.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

internal sealed class FlowSingleBlock
{
    readonly FlowLinear modulation, first, second;
    readonly Tensor queryScale, keyScale;

    internal FlowSingleBlock(FlowWeights weights, int index, int width, int heads)
    {
        var name = "single_blocks." + index;
        modulation = new(weights, name + ".modulation.linear", width, 3 * width);
        first = new(weights, name + ".linear1", width, 7 * width);
        second = new(weights, name + ".linear2", 5 * width, width);
        queryScale = weights.Get(name + ".q_norm.weight", width / heads);
        keyScale = weights.Get(name + ".k_norm.weight", width / heads);
    }

    internal void Run(Span<float> features, ReadOnlySpan<float> activatedAdapter, FlowBlockBuffers buffers, CancellationToken cancellation)
    {
        var width = buffers.Width; var rows = 2 * buffers.Frames;
        var gates = buffers.Gates.AsSpan(0, 3 * width);
        modulation.Run(activatedAdapter, 1, gates, cancellation);
        buffers.LayerNorm.Run(features, buffers.Normalized);
        FlowConditioning.Modulate(buffers.Normalized, gates[..width], gates.Slice(width, width));
        first.Run(buffers.Normalized, rows, buffers.Packed, cancellation);
        buffers.PrepareQkv(buffers.Packed, rows, 7 * width, 0, queryScale, keyScale);
        buffers.Attend(cancellation);
        for (var row = 0; row < rows; row++)
        {
            var joined = buffers.Joined.AsSpan(row * 5 * width, 5 * width);
            buffers.Attended.AsSpan(row * width, width).CopyTo(joined[..width]);
            buffers.Packed.AsSpan(row * 7 * width + 3 * width, 4 * width).CopyTo(joined[width..]);
            FlowConditioning.Gelu(joined[width..]);
        }
        second.Run(buffers.Joined, rows, buffers.Projected, cancellation);
        FlowConditioning.AddGated(features, buffers.Projected, gates.Slice(2 * width, width));
    }
}
