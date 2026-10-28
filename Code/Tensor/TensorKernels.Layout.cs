using System;
using Sandbox;

namespace HumanoidMocapX.TensorInference;

public sealed partial class TensorKernels
{
    readonly ComputeShader layout = new("shaders/tensor-inference/layout/layout_compute.shader");
    readonly ComputeShader rotary = new("shaders/tensor-inference/rotary/rotary_compute.shader");
    readonly ComputeShader mean = new("shaders/tensor-inference/reduction/mean_compute.shader");

    /// <summary>Extracts columns starting at firstColumn from input[rows,inputWidth]
    /// into output[rows,columns]. Rejects invalid slices, undersized buffers and aliasing.</summary>
    public void CopyColumns(GpuBuffer<float> input, GpuBuffer<float> output, int rows, int inputWidth, int firstColumn, int columns)
    {
        var count = MatrixElements(rows, columns);
        if (firstColumn < 0 || inputWidth < 1 || (long)firstColumn + columns > inputWidth)
            throw new ArgumentException("Requested columns fall outside the input matrix.");
        Require(input, (long)rows * inputWidth, nameof(input)); Require(output, count, nameof(output));
        Layout(input, input, output, count, inputWidth, 0, firstColumn, columns);
    }

    /// <summary>Concatenates first[rows,firstWidth] and second[rows,secondWidth] by row.
    /// Rejects invalid dimensions, undersized buffers and output aliasing.</summary>
    public void ConcatColumns(GpuBuffer<float> first, GpuBuffer<float> second, GpuBuffer<float> output,
        int rows, int firstWidth, int secondWidth)
    {
        Require(first, MatrixElements(rows, firstWidth), nameof(first));
        Require(second, MatrixElements(rows, secondWidth), nameof(second));
        var columns = checked(firstWidth + secondWidth); var count = MatrixElements(rows, columns);
        Require(output, count, nameof(output));
        Layout(first, second, output, count, firstWidth, secondWidth, 0, columns);
    }

    void Layout(GpuBuffer<float> first, GpuBuffer<float> second, GpuBuffer<float> output, int count,
        int firstWidth, int secondWidth, int firstColumn, int columns)
    {
        if (ReferenceEquals(first, output) || ReferenceEquals(second, output))
            throw new ArgumentException("Layout output must not overlap its inputs.");
        var attributes = layout.Attributes;
        attributes.Set("First", first); attributes.Set("Second", second); attributes.Set("Output", output);
        attributes.Set("Count", count); attributes.Set("FirstWidth", firstWidth); attributes.Set("SecondWidth", secondWidth);
        attributes.Set("FirstColumn", firstColumn); attributes.Set("Columns", columns);
        DispatchElements(layout, count);
        CompleteWrite(output);
    }

    /// <summary>Rotates adjacent pairs in values[rows,width] in place. Factors are
    /// [positions,headWidth] interleaved cos/sin values; positions repeat per stream.
    /// splitHalves pairs the first and second half of each head instead of neighbors.
    /// Rejects inconsistent dimensions, undersized buffers and factor/value aliasing.</summary>
    public void Rotary(GpuBuffer<float> values, GpuBuffer<float> factors, int rows, int width, int headWidth, int positions, bool splitHalves = false)
    {
        var count = MatrixElements(rows, width);
        if (headWidth < 2 || headWidth % 2 != 0 || width % headWidth != 0 || positions < 1 || rows % positions != 0)
            throw new ArgumentException("Rotary embedding needs even heads and a position table covering each stream.");
        Require(values, count, nameof(values)); Require(factors, (long)positions * headWidth, nameof(factors));
        if (ReferenceEquals(values, factors)) throw new ArgumentException("Rotary factors must not overlap the values being rotated.");
        var attributes = rotary.Attributes;
        attributes.Set("Values", values); attributes.Set("Factors", factors); attributes.Set("Pairs", count / 2);
        attributes.Set("Width", width); attributes.Set("HeadWidth", headWidth); attributes.Set("Positions", positions);
        attributes.Set("SplitHalves", splitHalves ? 1 : 0);
        DispatchElements(rotary, count / 2);
        CompleteWrite(values);
    }

    /// <summary>Computes the mean of input[rows,width] into output[width]. Rejects
    /// invalid dimensions, undersized buffers and input/output aliasing.</summary>
    public void MeanRows(GpuBuffer<float> input, GpuBuffer<float> output, int rows, int width)
    {
        var count = MatrixElements(rows, width);
        Require(input, count, nameof(input)); Require(output, width, nameof(output));
        if (ReferenceEquals(input, output)) throw new ArgumentException("Mean output must not overlap its input.");
        var attributes = mean.Attributes;
        attributes.Set("Input", input); attributes.Set("Output", output); attributes.Set("Rows", rows); attributes.Set("Width", width);
        DispatchElements(mean, width);
        CompleteWrite(output);
    }
}
