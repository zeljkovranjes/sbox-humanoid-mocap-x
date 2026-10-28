// C# adaptation of FlowHMR's MMSingleStreamBlock. See THIRD_PARTY_NOTICES.md.
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Inference;

internal sealed class FlowGpuSingleBlock
{
    readonly FlowGpuLinear modulation, first, second;
    readonly GpuBuffer<float> queryScale, keyScale;

    internal FlowGpuSingleBlock(FlowGpuStorage storage, FlowWeights weights, int index, int width)
    {
        var name = "single_blocks." + index;
        modulation = new(storage, weights, name + ".modulation.linear", width, 3 * width);
        first = new(storage, weights, name + ".linear1", width, 7 * width);
        second = new(storage, weights, name + ".linear2", 5 * width, width);
        queryScale = storage.Weight(weights, name + ".q_norm.weight", FlowGpuWorkspace.HeadWidth);
        keyScale = storage.Weight(weights, name + ".k_norm.weight", FlowGpuWorkspace.HeadWidth);
    }

    internal void Run(FlowGpuWorkspace w, GpuBuffer<float> features, GpuBuffer<float> adapter)
    {
        var k = w.Kernels; var rows = 2 * w.Frames; var width = w.Width;
        modulation.Run(k, adapter, w.MotionGates);
        k.Normalize(features, null, null, w.Normalized, rows, width, 1e-6f);
        k.Modulate(w.Normalized, w.MotionGates, w.Modified, rows, width, 0, width);
        first.Run(k, w.Modified, w.Packed, rows);
        w.PrepareQkv(rows, 7 * width, 0, queryScale, keyScale);
        w.Attend(true);
        k.CopyColumns(w.Packed, w.Hidden, rows, 7 * width, 3 * width, 4 * width);
        k.Activate(w.Hidden, w.Activated, rows * 4 * width, TensorActivation.Gelu);
        k.ConcatColumns(w.Attended, w.Activated, w.Joined, rows, width, 4 * width);
        second.Run(k, w.Joined, w.Projected, rows);
        k.GatedAdd(features, w.Projected, w.MotionGates, w.Next, rows, width, 2 * width);
        w.Next.CopyTo(features, rows * width);
    }
}
