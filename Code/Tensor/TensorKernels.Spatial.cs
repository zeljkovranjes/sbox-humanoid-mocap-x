using System;
using Sandbox;

namespace HumanoidMocapX.TensorInference;

public sealed partial class TensorKernels
{
    readonly ComputeShader spatial = new("shaders/tensor-inference/spatial/spatial_compute.shader");

    /// <summary>Packs each 2×2 HWC block into channels in top-left, bottom-left,
    /// top-right, bottom-right order. Rejects odd dimensions, short buffers and aliasing.</summary>
    public void Focus(GpuBuffer<float> input, GpuBuffer<float> output, int height, int width, int channels)
    {
        var count = ImageElements(height, width, channels);
        if (height % 2 != 0 || width % 2 != 0) throw new ArgumentException("Focus requires even image dimensions.");
        Spatial(input, output, height, width, channels, count, count, 0, 1);
    }

    /// <summary>Repeats each HWC pixel twice on both axes. Rejects invalid shapes,
    /// buffer bounds and aliasing.</summary>
    public void UpsampleNearest(GpuBuffer<float> input, GpuBuffer<float> output, int height, int width, int channels)
    {
        var count = ImageElements(height, width, channels);
        if (count > int.MaxValue / 4) throw new ArgumentException("Upsampled image exceeds the buffer index range.");
        Spatial(input, output, height, width, channels, count, count * 4, 1, 1);
    }

    /// <summary>Computes an HWC maximum with stride one and negative-infinity padding.
    /// Kernel must be odd and between 1 and 31. Rejects invalid shapes/buffers and aliasing.</summary>
    public void MaxPool(GpuBuffer<float> input, GpuBuffer<float> output, int height, int width, int channels, int kernel)
    {
        var count = ImageElements(height, width, channels);
        if (kernel < 1 || kernel > 31 || kernel % 2 == 0) throw new ArgumentException("Max pooling requires an odd kernel from 1 to 31.");
        Spatial(input, output, height, width, channels, count, count, 2, kernel);
    }

    static int ImageElements(int height, int width, int channels)
    {
        if (height < 1 || width < 1 || channels < 1 || height > int.MaxValue - 31 || width > int.MaxValue - 31
            || (long)height * width > int.MaxValue / channels)
            throw new ArgumentException("Image dimensions must be positive and fit GPU index limits.");
        return height * width * channels;
    }

    void Spatial(GpuBuffer<float> input, GpuBuffer<float> output, int height, int width, int channels,
        int inputCount, int outputCount, int operation, int kernel)
    {
        Require(input, inputCount, nameof(input)); Require(output, outputCount, nameof(output));
        if (ReferenceEquals(input, output)) throw new ArgumentException("Spatial output must not overlap its input.");
        var attributes = spatial.Attributes;
        attributes.Set("Input", input); attributes.Set("Output", output);
        attributes.Set("Height", height); attributes.Set("Width", width); attributes.Set("Channels", channels);
        attributes.Set("Count", outputCount); attributes.Set("Operation", operation); attributes.Set("Kernel", kernel);
        DispatchElements(spatial, outputCount); CompleteWrite(output);
    }
}
