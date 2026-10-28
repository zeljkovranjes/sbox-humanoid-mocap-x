// C# adaptation of FlowHMR's SingleTokenRefiner. See THIRD_PARTY_NOTICES.md.
using System.Threading;
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Inference;

internal sealed class FlowGpuRefiner
{
    readonly FlowGpuLinear embed;
    readonly FlowGpuMlp contextMlp, timeMlp;
    readonly Block[] blocks;
    readonly GpuBuffer<float> embedded, mean, context, time, condition, activated;

    internal FlowGpuRefiner(FlowGpuStorage storage, FlowWeights weights, FlowGpuWorkspace workspace)
    {
        var width = workspace.Width;
        const string prefix = "text_refiner.";
        embed = new(storage, weights, prefix + "input_embedder", width, width);
        contextMlp = new(storage, weights, prefix + "context_encoder.linears", width);
        timeMlp = new(storage, weights, prefix + "timestep_encoder.blocks", width);
        embedded = storage.Create<float>(workspace.StreamElements); mean = storage.Create<float>(width);
        context = storage.Create<float>(width); time = storage.Create<float>(width);
        condition = storage.Create<float>(width); activated = storage.Create<float>(width);
        blocks = new Block[2];
        for (var i = 0; i < blocks.Length; i++)
            blocks[i] = new(storage, weights, prefix + "individual_token_refiner.blocks." + i, width);
    }

    internal void SetVideo(FlowGpuWorkspace workspace, GpuBuffer<float> input)
    {
        workspace.Kernels.MeanRows(input, mean, workspace.Frames, workspace.Width);
        contextMlp.Run(workspace, mean, context);
        embed.Run(workspace.Kernels, input, embedded, workspace.Frames);
    }

    internal void Run(FlowGpuWorkspace workspace, GpuBuffer<float> embedding, GpuBuffer<float> features,
        CancellationToken cancellation)
    {
        timeMlp.Run(workspace, embedding, time);
        workspace.Kernels.Add(time, context, condition, workspace.Width);
        workspace.Kernels.Activate(condition, activated, workspace.Width, TensorActivation.Silu);
        embedded.CopyTo(workspace.Slice, workspace.StreamElements);
        foreach (var block in blocks)
        { cancellation.ThrowIfCancellationRequested(); block.Run(workspace, activated); }
        workspace.Slice.CopyTo(features, 0, workspace.StreamElements, workspace.StreamElements);
    }

    sealed class Block
    {
        readonly FlowGpuLinear qkv, projection, first, second, modulation;
        readonly GpuBuffer<float> norm1Scale, norm1Bias, norm2Scale, norm2Bias, queryScale, queryBias, keyScale, keyBias;

        internal Block(FlowGpuStorage storage, FlowWeights weights, string name, int width)
        {
            qkv = new(storage, weights, name + ".self_attn_qkv", width, 3 * width);
            projection = new(storage, weights, name + ".self_attn_proj", width, width);
            first = new(storage, weights, name + ".mlp.fc1", width, 4 * width);
            second = new(storage, weights, name + ".mlp.fc2", 4 * width, width);
            modulation = new(storage, weights, name + ".adaLN_modulation.linear", width, 2 * width);
            norm1Scale = storage.Weight(weights, name + ".norm1.weight", width);
            norm1Bias = storage.Weight(weights, name + ".norm1.bias", width);
            norm2Scale = storage.Weight(weights, name + ".norm2.weight", width);
            norm2Bias = storage.Weight(weights, name + ".norm2.bias", width);
            queryScale = storage.Weight(weights, name + ".self_attn_q_norm.weight", FlowGpuWorkspace.HeadWidth);
            queryBias = storage.Weight(weights, name + ".self_attn_q_norm.bias", FlowGpuWorkspace.HeadWidth);
            keyScale = storage.Weight(weights, name + ".self_attn_k_norm.weight", FlowGpuWorkspace.HeadWidth);
            keyBias = storage.Weight(weights, name + ".self_attn_k_norm.bias", FlowGpuWorkspace.HeadWidth);
        }

        internal void Run(FlowGpuWorkspace w, GpuBuffer<float> condition)
        {
            var k = w.Kernels; var rows = w.Frames; var width = w.Width;
            modulation.Run(k, condition, w.MotionGates);
            k.Normalize(w.Slice, norm1Scale, norm1Bias, w.Normalized, rows, width);
            qkv.Run(k, w.Normalized, w.Packed, rows);
            w.PrepareQkv(rows, 3 * width, 0, queryScale, keyScale, queryBias, keyBias, false);
            w.Attend(false);
            projection.Run(k, w.Attended, w.Projected, rows);
            k.GatedAdd(w.Slice, w.Projected, w.MotionGates, w.Next, rows, width, 0);
            w.Next.CopyTo(w.Slice, w.StreamElements);
            k.Normalize(w.Slice, norm2Scale, norm2Bias, w.Normalized, rows, width);
            first.Run(k, w.Normalized, w.Hidden, rows);
            k.Activate(w.Hidden, w.Activated, 4 * w.StreamElements, TensorActivation.Silu);
            second.Run(k, w.Activated, w.Projected, rows);
            k.GatedAdd(w.Slice, w.Projected, w.MotionGates, w.Next, rows, width, width);
            w.Next.CopyTo(w.Slice, w.StreamElements);
        }
    }
}
