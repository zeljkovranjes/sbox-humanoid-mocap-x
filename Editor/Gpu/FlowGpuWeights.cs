using System;
using System.Collections.Generic;
using Sandbox;

namespace HumanoidMocapX.Inference;

/// <summary>Weight buffers shared by several model instances: each checkpoint tensor is
/// uploaded once, however many instances use it. Weights are read-only on the GPU, so
/// instances recording in separate batches cannot race on them. Dispose on the main
/// thread after every instance that uses them.</summary>
internal sealed class FlowGpuWeights : IDisposable
{
    // Keyed by the tensor's array: FlowWeights returns the same instance on every lookup.
    readonly Dictionary<float[], GpuBuffer<float>> buffers = new();

    internal GpuBuffer<float> Get(float[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (buffers.TryGetValue(data, out var buffer)) return buffer;
        buffer = new GpuBuffer<float>(data.Length);
        buffers.Add(data, buffer);
        buffer.SetData<float>(data);
        return buffer;
    }

    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        foreach (var buffer in buffers.Values) buffer.Dispose();
        buffers.Clear();
    }
}
