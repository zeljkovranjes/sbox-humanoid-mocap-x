using System;
using System.Collections.Generic;
using Sandbox;

namespace HumanoidMocapX.Inference;

/// <summary>Owns all model buffers, including partially constructed models. Weights come
/// from <paramref name="shared"/> when given, so several instances upload them once.</summary>
internal sealed class FlowGpuStorage(FlowGpuWeights shared = null) : IDisposable
{
    readonly List<GpuBuffer> buffers = new();

    internal GpuBuffer<T> Create<T>(int count) where T : unmanaged
    {
        var buffer = new GpuBuffer<T>(count);
        buffers.Add(buffer);
        return buffer;
    }

    internal GpuBuffer<T> Upload<T>(ReadOnlySpan<T> values) where T : unmanaged
    {
        var buffer = Create<T>(values.Length);
        buffer.SetData<T>(values);
        return buffer;
    }

    internal GpuBuffer<float> Weight(FlowWeights weights, string name, params int[] shape)
    {
        var data = weights.Get(name, shape).Data;
        return shared is null ? Upload<float>(data) : shared.Get(data);
    }

    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        foreach (var buffer in buffers) buffer.Dispose();
        buffers.Clear();
    }
}
