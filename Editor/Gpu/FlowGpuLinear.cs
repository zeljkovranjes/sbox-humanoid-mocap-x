using Sandbox;
using TensorInference;

namespace HumanoidMocapX.Inference;

internal sealed class FlowGpuLinear
{
    readonly GpuBuffer<float> weight, bias;
    readonly int inputs, outputs;

    internal FlowGpuLinear(FlowGpuStorage storage, FlowWeights weights, string name, int inputs, int outputs)
    {
        this.inputs = inputs; this.outputs = outputs;
        weight = storage.Weight(weights, name + ".weight", inputs, outputs);
        bias = storage.Weight(weights, name + ".bias", outputs);
    }

    internal void Run(TensorKernels kernels, GpuBuffer<float> input, GpuBuffer<float> output, int rows = 1)
        => kernels.Linear(input, weight, bias, output, rows, inputs, outputs);
}

internal sealed class FlowGpuMlp
{
    readonly FlowGpuLinear first, second;

    internal FlowGpuMlp(FlowGpuStorage storage, FlowWeights weights, string name, int width)
    {
        first = new(storage, weights, name + ".0", width, width);
        second = new(storage, weights, name + ".2", width, width);
    }

    internal void Run(FlowGpuWorkspace workspace, GpuBuffer<float> input, GpuBuffer<float> output)
    {
        first.Run(workspace.Kernels, input, workspace.ConditionHidden);
        workspace.Kernels.Activate(workspace.ConditionHidden, workspace.ConditionActivated, workspace.Width, TensorActivation.Silu);
        second.Run(workspace.Kernels, workspace.ConditionActivated, output);
    }
}
