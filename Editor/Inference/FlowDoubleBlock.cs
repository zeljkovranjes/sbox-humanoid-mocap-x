// C# adaptation of FlowHMR's MMDoubleStreamBlock. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;
using HumanoidMocap.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

internal sealed class FlowDoubleBlock
{
    readonly Branch motion, video;

    internal FlowDoubleBlock(FlowWeights weights, int index, int width, int heads)
    {
        var name = "double_blocks." + index;
        motion = new(weights, name + ".motion", width, heads);
        video = new(weights, name + ".text", width, heads);
    }

    internal void Run(Span<float> features, ReadOnlySpan<float> activatedAdapter, FlowBlockBuffers buffers, CancellationToken cancellation)
    {
        var count = buffers.StreamElements; var width = buffers.Width;
        var motionGates = buffers.Gates.AsSpan(0, 6 * width); var videoGates = buffers.Gates.AsSpan(6 * width, 6 * width);
        motion.Prepare(features[..count], activatedAdapter, motionGates, buffers, 0, cancellation);
        video.Prepare(features[count..], activatedAdapter, videoGates, buffers, buffers.Frames, cancellation);
        buffers.Attend(cancellation);
        motion.Finish(features[..count], buffers.Attended.AsSpan(0, count), motionGates, buffers, cancellation);
        video.Finish(features[count..], buffers.Attended.AsSpan(count), videoGates, buffers, cancellation);
    }

    sealed class Branch
    {
        readonly FlowLinear modulation, qkv, projection, mlpFirst, mlpSecond;
        readonly Tensor queryScale, keyScale;

        internal Branch(FlowWeights weights, string name, int width, int heads)
        {
            modulation = new(weights, name + "_mod.linear", width, 6 * width);
            qkv = new(weights, name + "_qkv", width, 3 * width);
            projection = new(weights, name + "_out_proj", width, width);
            mlpFirst = new(weights, name + "_mlp.fc1", width, 4 * width);
            mlpSecond = new(weights, name + "_mlp.fc2", 4 * width, width);
            queryScale = weights.Get(name + "_q_norm.weight", width / heads);
            keyScale = weights.Get(name + "_k_norm.weight", width / heads);
        }

        internal void Prepare(ReadOnlySpan<float> features, ReadOnlySpan<float> adapter, Span<float> gates,
            FlowBlockBuffers buffers, int firstRow, CancellationToken cancellation)
        {
            var width = buffers.Width; var count = buffers.StreamElements;
            modulation.Run(adapter, 1, gates, cancellation);
            var normalized = buffers.Normalized.AsSpan(0, count);
            buffers.LayerNorm.Run(features, normalized);
            FlowConditioning.Modulate(normalized, gates[..width], gates.Slice(width, width));
            var packed = buffers.Packed.AsSpan(0, 3 * count);
            qkv.Run(normalized, buffers.Frames, packed, cancellation);
            buffers.PrepareQkv(packed, buffers.Frames, 3 * width, firstRow, queryScale, keyScale);
        }

        internal void Finish(Span<float> features, ReadOnlySpan<float> attended, ReadOnlySpan<float> gates,
            FlowBlockBuffers buffers, CancellationToken cancellation)
        {
            var width = buffers.Width; var count = buffers.StreamElements;
            var projected = buffers.Projected.AsSpan(0, count);
            projection.Run(attended, buffers.Frames, projected, cancellation);
            FlowConditioning.AddGated(features, projected, gates.Slice(2 * width, width));
            var normalized = buffers.Normalized.AsSpan(0, count);
            buffers.LayerNorm.Run(features, normalized);
            FlowConditioning.Modulate(normalized, gates.Slice(3 * width, width), gates.Slice(4 * width, width));
            var hidden = buffers.Packed.AsSpan(0, 4 * count);
            mlpFirst.Run(normalized, buffers.Frames, hidden, cancellation); FlowConditioning.Gelu(hidden);
            mlpSecond.Run(hidden, buffers.Frames, projected, cancellation);
            FlowConditioning.AddGated(features, projected, gates.Slice(5 * width, width));
        }
    }
}
