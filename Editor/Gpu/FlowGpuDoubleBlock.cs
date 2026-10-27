// C# adaptation of FlowHMR's MMDoubleStreamBlock. See THIRD_PARTY_NOTICES.md.
using Sandbox;
using TensorInference;

namespace HumanoidMocapX.Inference;

internal sealed class FlowGpuDoubleBlock
{
    readonly Branch motion, video;

    internal FlowGpuDoubleBlock(FlowGpuStorage storage, FlowWeights weights, int index, int width)
    {
        var name = "double_blocks." + index;
        motion = new(storage, weights, name + ".motion", width);
        video = new(storage, weights, name + ".text", width);
    }

    internal void Run(FlowGpuWorkspace workspace, GpuBuffer<float> features, GpuBuffer<float> adapter)
    {
        motion.Prepare(workspace, features, adapter, workspace.MotionGates, 0);
        video.Prepare(workspace, features, adapter, workspace.VideoGates, workspace.StreamElements);
        workspace.Attend(true);
        motion.Finish(workspace, features, workspace.MotionGates, 0);
        video.Finish(workspace, features, workspace.VideoGates, workspace.StreamElements);
    }

    sealed class Branch
    {
        readonly FlowGpuLinear modulation, qkv, projection, first, second;
        readonly GpuBuffer<float> queryScale, keyScale;

        internal Branch(FlowGpuStorage storage, FlowWeights weights, string name, int width)
        {
            modulation = new(storage, weights, name + "_mod.linear", width, 6 * width);
            qkv = new(storage, weights, name + "_qkv", width, 3 * width);
            projection = new(storage, weights, name + "_out_proj", width, width);
            first = new(storage, weights, name + "_mlp.fc1", width, 4 * width);
            second = new(storage, weights, name + "_mlp.fc2", 4 * width, width);
            queryScale = storage.Weight(weights, name + "_q_norm.weight", FlowGpuWorkspace.HeadWidth);
            keyScale = storage.Weight(weights, name + "_k_norm.weight", FlowGpuWorkspace.HeadWidth);
        }

        internal void Prepare(FlowGpuWorkspace w, GpuBuffer<float> features, GpuBuffer<float> adapter,
            GpuBuffer<float> gates, int offset)
        {
            var k = w.Kernels; var rows = w.Frames; var width = w.Width;
            features.CopyTo(w.Slice, offset, 0, w.StreamElements);
            modulation.Run(k, adapter, gates);
            k.Normalize(w.Slice, null, null, w.Normalized, rows, width, 1e-6f);
            k.Modulate(w.Normalized, gates, w.Modified, rows, width, 0, width);
            qkv.Run(k, w.Modified, w.Packed, rows);
            w.PrepareQkv(rows, 3 * width, offset, queryScale, keyScale);
        }

        internal void Finish(FlowGpuWorkspace w, GpuBuffer<float> features, GpuBuffer<float> gates, int offset)
        {
            var k = w.Kernels; var rows = w.Frames; var width = w.Width;
            features.CopyTo(w.Slice, offset, 0, w.StreamElements);
            w.Attended.CopyTo(w.Component, offset, 0, w.StreamElements);
            projection.Run(k, w.Component, w.Projected, rows);
            k.GatedAdd(w.Slice, w.Projected, gates, w.Next, rows, width, 2 * width);
            w.Next.CopyTo(w.Slice, w.StreamElements);
            k.Normalize(w.Slice, null, null, w.Normalized, rows, width, 1e-6f);
            k.Modulate(w.Normalized, gates, w.Modified, rows, width, 3 * width, 4 * width);
            first.Run(k, w.Modified, w.Hidden, rows);
            k.Activate(w.Hidden, w.Activated, 4 * w.StreamElements, TensorActivation.Gelu);
            second.Run(k, w.Activated, w.Projected, rows);
            k.GatedAdd(w.Slice, w.Projected, gates, w.Next, rows, width, 5 * width);
            w.Next.CopyTo(features, 0, offset, w.StreamElements);
        }
    }
}
