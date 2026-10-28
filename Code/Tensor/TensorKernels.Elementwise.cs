using System;
using Sandbox;

namespace HumanoidMocapX.TensorInference;

public sealed partial class TensorKernels
{
    readonly ComputeShader elementwise = new("shaders/tensor-inference/elementwise/elementwise_compute.shader");
    const int ElementsPerLine = 65536;
    enum ElementwiseOperation { Silu, Gelu, Add, Modulate, GatedAdd, Blend, GeluErf, SwiGlu }

    /// <summary>Applies an activation to count values. Rejects invalid activation,
    /// count, undersized buffers and overlapping input/output.</summary>
    public void Activate(GpuBuffer<float> input, GpuBuffer<float> output, int count, TensorActivation activation)
    {
        if (activation is not (TensorActivation.Silu or TensorActivation.Gelu or TensorActivation.GeluErf)) throw new ArgumentOutOfRangeException(nameof(activation));
        Elementwise(input, input, input, output, count, 1,
            activation == TensorActivation.Silu ? ElementwiseOperation.Silu
                : activation == TensorActivation.Gelu ? ElementwiseOperation.Gelu : ElementwiseOperation.GeluErf);
    }

    /// <summary>Adds equally sized tensors into a separate output. Rejects invalid
    /// count, undersized buffers and output aliasing.</summary>
    public void Add(GpuBuffer<float> first, GpuBuffer<float> second, GpuBuffer<float> output, int count)
        => Elementwise(first, second, first, output, count, 1, ElementwiseOperation.Add);

    /// <summary>Computes SiLU(input) * gate into a separate output. Rejects invalid
    /// count, undersized buffers and output aliasing.</summary>
    public void SwiGlu(GpuBuffer<float> input, GpuBuffer<float> gate, GpuBuffer<float> output, int count)
        => Elementwise(input, gate, input, output, count, 1, ElementwiseOperation.SwiGlu);

    /// <summary>Computes input * (1 + scale) + shift for [rows,width] values. Shift
    /// and scale are width-element slices of parameters. Rejects invalid slices,
    /// undersized buffers and output aliasing.</summary>
    public void Modulate(GpuBuffer<float> input, GpuBuffer<float> parameters, GpuBuffer<float> output,
        int rows, int width, int shiftOffset, int scaleOffset)
    {
        ParameterSlice(parameters, shiftOffset, width); ParameterSlice(parameters, scaleOffset, width);
        Elementwise(input, input, parameters, output, rows, width, ElementwiseOperation.Modulate, shiftOffset, scaleOffset);
    }

    /// <summary>Computes input + update * gate for [rows,width] values. Gate is a
    /// width-element slice of parameters. Rejects invalid shapes/slices, undersized
    /// buffers and output aliasing.</summary>
    public void GatedAdd(GpuBuffer<float> input, GpuBuffer<float> update, GpuBuffer<float> parameters,
        GpuBuffer<float> output, int rows, int width, int gateOffset)
    {
        ParameterSlice(parameters, gateOffset, width);
        Elementwise(input, update, parameters, output, rows, width, ElementwiseOperation.GatedAdd, gateOffset);
    }

    /// <summary>Computes input + amount * (target - input), with amount in [0,1].
    /// Rejects invalid amounts/counts, undersized buffers and output aliasing.</summary>
    public void Blend(GpuBuffer<float> input, GpuBuffer<float> target, GpuBuffer<float> output, int count, float amount)
    {
        if (!float.IsFinite(amount) || amount < 0 || amount > 1) throw new ArgumentOutOfRangeException(nameof(amount));
        Elementwise(input, target, input, output, count, 1, ElementwiseOperation.Blend, amount: amount);
    }

    void Elementwise(GpuBuffer<float> input, GpuBuffer<float> other, GpuBuffer<float> parameters,
        GpuBuffer<float> output, int rows, int width, ElementwiseOperation operation, int firstOffset = 0,
        int secondOffset = 0, float amount = 0)
    {
        var count = MatrixElements(rows, width);
        Require(input, count, nameof(input)); Require(other, count, nameof(other)); Require(output, count, nameof(output));
        if (ReferenceEquals(input, output) || ReferenceEquals(other, output) || ReferenceEquals(parameters, output))
            throw new ArgumentException("Elementwise output must not overlap an input or parameter buffer.");
        var attributes = elementwise.Attributes;
        attributes.Set("Input", input); attributes.Set("Other", other);
        attributes.Set("Parameters", parameters); attributes.Set("Output", output);
        attributes.Set("Count", count); attributes.Set("Width", width); attributes.Set("Operation", (int)operation);
        attributes.Set("FirstOffset", firstOffset); attributes.Set("SecondOffset", secondOffset); attributes.Set("Amount", amount);
        DispatchElements(elementwise, count);
        CompleteWrite(output);
    }

    static void ParameterSlice(GpuBuffer<float> parameters, int offset, int width)
    {
        if (offset < 0 || width < 1) throw new ArgumentException("Tensor parameter slices need a nonnegative offset and positive width.");
        Require(parameters, (long)offset + width, nameof(parameters));
    }

    static int MatrixElements(int rows, int width)
    {
        if (rows < 1 || width < 1 || (long)rows * width > int.MaxValue)
            throw new ArgumentException("Tensor dimensions must be positive and fit the buffer index range.");
        return rows * width;
    }

    static void DispatchElements(ComputeShader shader, int count)
        => shader.Dispatch(Math.Min(count, ElementsPerLine), (count - 1) / ElementsPerLine + 1, 1);
}
