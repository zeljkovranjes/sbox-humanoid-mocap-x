using System;

namespace HumanoidMocapX.TensorInference;

/// <summary>Dimensions for one square, ungrouped 2D convolution. Images use
/// [height,width,channels] storage and weights use [outputs,channels,kernel,kernel].</summary>
public readonly struct ConvolutionShape
{
    public int Height { get; }
    public int Width { get; }
    public int Channels { get; }
    public int Outputs { get; }
    public int Kernel { get; }
    public int Stride { get; }
    public int Padding { get; }
    public int OutputHeight { get; }
    public int OutputWidth { get; }
    public int InputElements { get; }
    public int WeightElements { get; }
    public int OutputElements { get; }
    public int KernelElements { get; }
    public int OutputPixels => OutputHeight * OutputWidth;
    const int MaximumTileDimension = 65535 * 16;

    /// <summary>Validates positive image/filter dimensions, nonnegative zero padding
    /// and the GPU buffer/dispatch limits. Rejects kernels leaving no output pixels.</summary>
    public ConvolutionShape(int height, int width, int channels, int outputs, int kernel, int stride = 1, int padding = 0)
    {
        if (height < 1 || width < 1 || channels < 1 || outputs < 1 || kernel < 1 || stride < 1 || padding < 0)
            throw new ArgumentException("Convolution dimensions and stride must be positive; padding must be nonnegative.");
        var paddedHeight = (long)height + 2L * padding;
        var paddedWidth = (long)width + 2L * padding;
        if (paddedHeight > uint.MaxValue || paddedWidth > uint.MaxValue)
            throw new ArgumentException("Padded convolution dimensions exceed GPU coordinate limits.");
        if (paddedHeight < kernel || paddedWidth < kernel)
            throw new ArgumentException("Convolution kernel is larger than the padded image.");
        var outputHeight = (paddedHeight - kernel) / stride + 1;
        var outputWidth = (paddedWidth - kernel) / stride + 1;
        if (outputHeight > MaximumTileDimension || outputWidth > MaximumTileDimension
            || outputHeight * outputWidth > MaximumTileDimension || outputs > MaximumTileDimension)
            throw new ArgumentException("Convolution output exceeds GPU dispatch limits.");
        InputElements = Elements(height, width, channels);
        KernelElements = Elements(kernel, kernel, channels);
        WeightElements = Elements(KernelElements, outputs, 1);
        OutputElements = Elements((int)outputHeight, (int)outputWidth, outputs);
        Height = height; Width = width; Channels = channels; Outputs = outputs;
        Kernel = kernel; Stride = stride; Padding = padding;
        OutputHeight = (int)outputHeight; OutputWidth = (int)outputWidth;
    }

    static int Elements(int height, int width, int channels)
    {
        if ((long)height * width > int.MaxValue / channels)
            throw new ArgumentException("Convolution tensor exceeds the supported buffer index range.");
        return height * width * channels;
    }
}
