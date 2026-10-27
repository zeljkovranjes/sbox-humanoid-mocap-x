using HumanoidMocapX.Inference;
using Sandbox;

namespace HumanoidMocapX.Checks;

/// <summary>Live-editor check of the ownership boundary between feature lanes and their
/// shared GPU weights.</summary>
public static class SharedWeightChecks
{
    /// <summary>Two storages share one weight upload; releasing one keeps the weight intact
    /// on the GPU while freeing its own buffers. Throws on any violation.</summary>
    public static async Task<object> Run(CancellationToken cancellation)
    {
        await GameTask.MainThread();
        var values = Enumerable.Range(0, 4096).Select(i => MathF.Sin(i * .1f)).ToArray();
        var weights = new FlowWeights(new Dictionary<string, Tensor> { ["w"] = new("w", values, new[] { values.Length }) });
        var shared = new FlowGpuWeights();
        var first = new FlowGpuStorage(shared); var second = new FlowGpuStorage(shared);
        var weight = first.Weight(weights, "w", values.Length);
        if (!ReferenceEquals(weight, second.Weight(weights, "w", values.Length))) throw new Exception("Lanes uploaded the same weight twice.");
        var scratch = first.Create<float>(16);
        first.Dispose();
        if (scratch.IsValid) throw new Exception("Releasing a lane kept its own buffer.");
        if (!weight.IsValid) throw new Exception("Releasing one lane freed the shared weights.");
        var readback = new float[values.Length];
        using (var batch = new TensorInference.TensorBatch())
        {
            try { await batch.RunAsync(() => { }, weight, readback, cancellation); }
            finally { await GameTask.MainThread(); }
        }
        if (!readback.AsSpan().SequenceEqual(values)) throw new Exception("Shared weights changed after a lane was released.");
        second.Dispose(); shared.Dispose();
        if (weight.IsValid) throw new Exception("Disposing the shared weights left them allocated.");
        return new { Passed = true, Elements = values.Length };
    }
}
