using System;
using System.Threading;
using HumanoidMocap.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>One scratch workspace shared by all eighteen denoiser blocks.</summary>
internal sealed class FlowBlockBuffers
{
    internal int Frames { get; }
    internal int Width { get; }
    internal int StreamElements { get; }
    internal readonly float[] Normalized, Packed, Joined, Projected, Gates, Attended;
    internal readonly FlowLayerNorm LayerNorm;
    readonly float[] query, key, value;
    readonly FlowRotary rotary;
    readonly FlowAttention attention;

    internal FlowBlockBuffers(int frames, int width, int heads)
    {
        rotary = new(frames, width, heads);
        attention = new(frames, width, heads, true);
        Frames = frames; Width = width; StreamElements = checked(frames * width);
        var both = checked(2 * StreamElements);
        Normalized = new float[both]; Packed = new float[checked(7 * both)]; Joined = new float[checked(5 * both)];
        Projected = new float[both]; Gates = new float[checked(12 * width)]; Attended = new float[both];
        query = new float[both]; key = new float[both]; value = new float[both];
        LayerNorm = new(width, 1e-6f);
    }

    internal void PrepareQkv(ReadOnlySpan<float> packed, int rows, int rowWidth, int firstRow, Tensor queryScale, Tensor keyScale)
    {
        var count = checked(rows * Width); var offset = checked(firstRow * Width);
        var q = query.AsSpan(offset, count); var k = key.AsSpan(offset, count); var v = value.AsSpan(offset, count);
        for (var row = 0; row < rows; row++)
        {
            var source = packed.Slice(row * rowWidth, 3 * Width);
            source[..Width].CopyTo(q.Slice(row * Width, Width));
            source.Slice(Width, Width).CopyTo(k.Slice(row * Width, Width));
            source.Slice(2 * Width, Width).CopyTo(v.Slice(row * Width, Width));
        }
        FlowNormalization.Rms(q, queryScale.Data); FlowNormalization.Rms(k, keyScale.Data);
    }

    internal void Attend(CancellationToken cancellation)
    {
        rotary.Apply(query.AsSpan(0, StreamElements), key.AsSpan(0, StreamElements));
        rotary.Apply(query.AsSpan(StreamElements), key.AsSpan(StreamElements));
        attention.Run(query, key, value, Attended, cancellation);
    }
}
