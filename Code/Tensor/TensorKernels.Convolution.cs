using System;
using Sandbox;

namespace HumanoidMocapX.TensorInference;

public sealed partial class TensorKernels
{
    readonly ComputeShader convolution = new("shaders/tensor-inference/convolution/convolution_compute.shader");

    /// <summary>Convolves an HWC image with OIHW weights and optional output-channel
    /// bias, writing an HWC result. Zero padding, square kernels, one image and no
    /// dilation/groups. Rejects default shapes, undersized buffers and output aliasing.</summary>
    public void Convolve(GpuBuffer<float> input, GpuBuffer<float> weight, GpuBuffer<float> bias,
        GpuBuffer<float> output, ConvolutionShape shape)
    {
        if (shape.OutputElements == 0) throw new ArgumentException("Construct a valid convolution shape.", nameof(shape));
        Require(input, shape.InputElements, nameof(input)); Require(weight, shape.WeightElements, nameof(weight));
        Require(output, shape.OutputElements, nameof(output));
        if (bias is not null) Require(bias, shape.Outputs, nameof(bias));
        if (ReferenceEquals(input, output) || ReferenceEquals(weight, output) || ReferenceEquals(bias, output))
            throw new ArgumentException("Convolution output must not overlap its inputs.");
        var attributes = convolution.Attributes;
        attributes.Set("Input", input); attributes.Set("Weight", weight);
        attributes.Set("Bias", bias ?? weight); attributes.Set("Output", output);
        attributes.Set("Height", shape.Height); attributes.Set("Width", shape.Width);
        attributes.Set("Channels", shape.Channels); attributes.Set("Outputs", shape.Outputs);
        attributes.Set("Kernel", shape.Kernel); attributes.Set("Stride", shape.Stride); attributes.Set("Padding", shape.Padding);
        attributes.Set("OutputWidth", shape.OutputWidth); attributes.Set("OutputPixels", shape.OutputPixels);
        attributes.Set("KernelElements", shape.KernelElements); attributes.Set("HasBias", bias is null ? 0 : 1);
        // Same register tiling as the linear kernel: 4x4 outputs per thread.
        convolution.Dispatch((shape.Outputs - 1) / LinearValuesPerThread + 1, (shape.OutputPixels - 1) / LinearValuesPerThread + 1, 1);
        CompleteWrite(output);
    }
}
