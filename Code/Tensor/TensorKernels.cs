using System;
using Sandbox;

namespace HumanoidMocapX.TensorInference;

/// <summary>Dispatches tensor operations through s&amp;box's graphics context.
/// Call on the engine's render/main thread. Callers own all input/output buffers.</summary>
public sealed partial class TensorKernels
{
    readonly ComputeShader linear = new("shaders/tensor-inference/linear/linear_compute.shader");
    readonly ComputeShader normalization = new("shaders/tensor-inference/normalization/normalization_compute.shader");
    readonly ComputeShader attention = new("shaders/tensor-inference/attention/attention_compute.shader");
    readonly ComputeShader attentionPerQuery = new("shaders/tensor-inference/attention/attention_query_compute.shader");
    const int NormalizationLanes = 64, RowsPerGroupLine = 256;
    const int AttentionHeadWidth = 64, MaximumDispatchGroups = 65535, QueriesPerGroup = 64;
    const int LinearValuesPerThread = 4, LinearTile = 16 * LinearValuesPerThread;

    /// <summary>Computes input[rows,inputs] × weight[outputs,inputs]ᵀ plus optional bias[outputs].
    /// Writes output[rows,outputs]; rejects undersized buffers and output aliasing.</summary>
    public void Linear(GpuBuffer<float> input, GpuBuffer<float> weight, GpuBuffer<float> bias,
        GpuBuffer<float> output, int rows, int inputs, int outputs)
    {
        if (rows < 1 || inputs < 1 || outputs < 1 || rows > MaximumDispatchGroups * LinearTile || outputs > MaximumDispatchGroups * LinearTile)
            throw new ArgumentException("Linear dimensions must be positive and fit the GPU dispatch limits.");
        Require(input, (long)rows * inputs, nameof(input));
        Require(weight, (long)outputs * inputs, nameof(weight));
        Require(output, (long)rows * outputs, nameof(output));
        if (bias is not null) Require(bias, outputs, nameof(bias));
        if (ReferenceEquals(input, output) || ReferenceEquals(weight, output) || ReferenceEquals(bias, output))
            throw new ArgumentException("Linear output must not overlap its inputs.");
        var attributes = linear.Attributes;
        attributes.Set("Input", input); attributes.Set("Weight", weight);
        attributes.Set("Bias", bias ?? weight); attributes.Set("Output", output);
        attributes.Set("Rows", rows); attributes.Set("Inputs", inputs); attributes.Set("Outputs", outputs);
        attributes.Set("HasBias", bias is null ? 0 : 1);
        linear.Dispatch((outputs - 1) / LinearValuesPerThread + 1,
            (rows - 1) / LinearValuesPerThread + 1, 1);
        CompleteWrite(output);
    }

    /// <summary>Normalizes each row of input[rows,width] using population variance,
    /// or mean square when rms is true. Optional scale/bias have [width] elements.
    /// Rejects invalid dimensions, epsilon, buffers and output aliasing.</summary>
    public void Normalize(GpuBuffer<float> input, GpuBuffer<float> scale, GpuBuffer<float> bias,
        GpuBuffer<float> output, int rows, int width, float epsilon = 1e-5f, bool rms = false)
    {
        if (rows < 1 || rows > MaximumDispatchGroups * RowsPerGroupLine || width < 1 || !float.IsFinite(epsilon) || epsilon <= 0)
            throw new ArgumentException("Normalization needs positive dimensions and finite positive epsilon.");
        Require(input, (long)rows * width, nameof(input)); Require(output, (long)rows * width, nameof(output));
        if (scale is not null) Require(scale, width, nameof(scale));
        if (bias is not null) Require(bias, width, nameof(bias));
        if (ReferenceEquals(input, output) || ReferenceEquals(scale, output) || ReferenceEquals(bias, output))
            throw new ArgumentException("Normalization output must not overlap its inputs.");
        var attributes = normalization.Attributes;
        attributes.Set("Input", input); attributes.Set("Scale", scale ?? input);
        attributes.Set("Bias", bias ?? input); attributes.Set("Output", output);
        attributes.Set("Rows", rows); attributes.Set("Width", width); attributes.Set("Epsilon", epsilon);
        attributes.Set("HasScale", scale is null ? 0 : 1); attributes.Set("HasBias", bias is null ? 0 : 1);
        attributes.Set("Rms", rms ? 1 : 0);
        normalization.Dispatch(Math.Min(rows, RowsPerGroupLine) * NormalizationLanes,
            (rows - 1) / RowsPerGroupLine + 1, 1);
        CompleteWrite(output);
    }

    /// <summary>Computes scaled dot-product attention for [tokens,heads,headWidth] tensors.
    /// Head width is 64 or 128. keyTokens defaults to tokens; supply a different
    /// positive count for cross-attention K/V tensors.
    /// keyRanges[tokens,4] contains two (start,count) pairs per query. Ranges are
    /// clipped to the key count; empty rows produce zero. Rejects invalid buffers,
    /// dimensions and output aliasing. No dense attention matrix is allocated.
    /// sharedKeyRanges promises every query has the same ranges as query 0; with 64-channel heads
    /// it selects a kernel that gives each query its own thread, with bit-identical results.</summary>
    public void Attention(GpuBuffer<float> query, GpuBuffer<float> key, GpuBuffer<float> value,
        GpuBuffer<uint> keyRanges, GpuBuffer<float> output, int tokens, int heads, int headWidth = AttentionHeadWidth, int keyTokens = 0,
        bool sharedKeyRanges = false)
    {
        if (keyTokens == 0) keyTokens = tokens;
        if (tokens < 1 || tokens > MaximumDispatchGroups * RowsPerGroupLine || heads < 1 || heads > MaximumDispatchGroups
            || keyTokens < 1 || keyTokens > MaximumDispatchGroups * RowsPerGroupLine || headWidth is not (64 or 128))
            throw new ArgumentException("Attention needs positive tokens and a supported head count.");
        var elements = (long)tokens * heads * headWidth;
        Require(query, elements, nameof(query)); Require(key, (long)keyTokens * heads * headWidth, nameof(key));
        Require(value, (long)keyTokens * heads * headWidth, nameof(value)); Require(output, elements, nameof(output));
        Require(keyRanges, (long)tokens * 4, nameof(keyRanges));
        if (ReferenceEquals(query, output) || ReferenceEquals(key, output) || ReferenceEquals(value, output))
            throw new ArgumentException("Attention output must not overlap its inputs.");
        var perQuery = sharedKeyRanges && headWidth == AttentionHeadWidth;
        var shader = perQuery ? attentionPerQuery : attention;
        var attributes = shader.Attributes;
        attributes.Set("Query", query); attributes.Set("Key", key); attributes.Set("Value", value);
        attributes.Set("KeyRanges", keyRanges); attributes.Set("Output", output);
        attributes.Set("Tokens", tokens); attributes.Set("Heads", heads); attributes.Set("HeadWidth", headWidth);
        attributes.Set("KeyTokens", keyTokens);
        // The per-query kernel XORs products with this to force rounding; it must stay 0.
        if (perQuery) attributes.Set("Zero", 0);
        if (perQuery) shader.Dispatch((tokens - 1) / QueriesPerGroup * QueriesPerGroup + QueriesPerGroup, 1, heads);
        else shader.Dispatch(Math.Min(tokens, RowsPerGroupLine) * AttentionHeadWidth, (tokens - 1) / RowsPerGroupLine + 1, heads);
        CompleteWrite(output);
    }

    static void CompleteWrite(GpuBuffer output)
    {
        if (Graphics.IsActive) Graphics.UavBarrier(output);
    }

    static void Require<T>(GpuBuffer<T> buffer, long elements, string name) where T : unmanaged
    {
        if (buffer is null || !buffer.IsValid || buffer.ElementCount < elements)
            throw new ArgumentException("Tensor buffer is missing, disposed or smaller than its declared shape.", name);
    }
}
