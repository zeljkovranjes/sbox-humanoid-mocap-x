using System;
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Inference;

/// <summary>Reuses one GPU workspace across the refiner and all eighteen denoiser blocks.</summary>
internal sealed class FlowGpuWorkspace
{
    internal readonly TensorKernels Kernels = new();
    internal readonly int Frames, Width, Heads, StreamElements;
    internal const int HeadWidth = 64;
    internal readonly GpuBuffer<float> Normalized, Modified, Packed, Hidden, Activated, Joined, Projected, Next;
    internal readonly GpuBuffer<float> Slice, Component, Query, Key, Value, Attended, MotionGates, VideoGates;
    internal readonly GpuBuffer<float> ConditionHidden, ConditionActivated;
    readonly GpuBuffer<float> rotary;
    readonly GpuBuffer<uint> fullRanges, jointRanges;

    internal FlowGpuWorkspace(FlowGpuStorage storage, int frames, int width, int heads)
    {
        Frames = frames; Width = width; Heads = heads; StreamElements = checked(frames * width);
        var both = checked(2 * StreamElements);
        Normalized = storage.Create<float>(both); Modified = storage.Create<float>(both);
        Packed = storage.Create<float>(7 * both); Hidden = storage.Create<float>(4 * both);
        Activated = storage.Create<float>(4 * both); Joined = storage.Create<float>(5 * both);
        Projected = storage.Create<float>(both); Next = storage.Create<float>(both);
        Slice = storage.Create<float>(both); Component = storage.Create<float>(both);
        Query = storage.Create<float>(both); Key = storage.Create<float>(both);
        Value = storage.Create<float>(both); Attended = storage.Create<float>(both);
        MotionGates = storage.Create<float>(6 * width); VideoGates = storage.Create<float>(6 * width);
        ConditionHidden = storage.Create<float>(width); ConditionActivated = storage.Create<float>(width);
        var factors = new float[frames * HeadWidth];
        new FlowRotary(frames, width, heads).CopyFactors(factors);
        rotary = storage.Upload<float>(factors);
        var ranges = new uint[8 * frames];
        new FlowAttention(frames, width, heads, true).CopyKeyRanges(ranges);
        jointRanges = storage.Upload<uint>(ranges);
        new FlowAttention(frames, width, heads, false).CopyKeyRanges(ranges.AsSpan(0, 4 * frames));
        fullRanges = storage.Upload<uint>(ranges.AsSpan(0, 4 * frames));
    }

    internal void PrepareQkv(int rows, int stride, int offset, GpuBuffer<float> queryScale,
        GpuBuffer<float> keyScale, GpuBuffer<float> queryBias = null, GpuBuffer<float> keyBias = null, bool rms = true)
    {
        var epsilon = rms ? 1e-6f : 1e-5f;
        Kernels.CopyColumns(Packed, Component, rows, stride, 0, Width);
        Kernels.Normalize(Component, queryScale, queryBias, Normalized, rows * Heads, HeadWidth, epsilon, rms);
        Normalized.CopyTo(Query, 0, offset, rows * Width);
        Kernels.CopyColumns(Packed, Component, rows, stride, Width, Width);
        Kernels.Normalize(Component, keyScale, keyBias, Normalized, rows * Heads, HeadWidth, epsilon, rms);
        Normalized.CopyTo(Key, 0, offset, rows * Width);
        Kernels.CopyColumns(Packed, Component, rows, stride, 2 * Width, Width);
        Component.CopyTo(Value, 0, offset, rows * Width);
    }

    internal void Attend(bool joint)
    {
        var rows = joint ? 2 * Frames : Frames;
        if (joint)
        {
            Kernels.Rotary(Query, rotary, rows, Width, HeadWidth, Frames);
            Kernels.Rotary(Key, rotary, rows, Width, HeadWidth, Frames);
        }
        Kernels.Attention(Query, Key, Value, joint ? jointRanges : fullRanges, Attended, rows, Heads);
    }
}
