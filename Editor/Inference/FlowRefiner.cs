// C# adaptation of FlowHMR's SingleTokenRefiner. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;
using HumanoidMocapX.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Refines unpadded, batch-one video features. Uses two full-attention blocks,
/// including the refiner's unscaled timestep embedding. Instances are not thread safe.</summary>
internal sealed class FlowRefiner
{
    readonly int frames, width;
    readonly FlowLinear embed;
    readonly FlowConditionMlp contextMlp, timeMlp;
    readonly RefinerBlock[] blocks;
    readonly float[] mean, time, context, condition;

    internal FlowRefiner(FlowWeights weights, int frames, int width = 1024, int heads = 16)
    {
        if (frames < 1 || frames > FlowMotion.MaximumFrames || width < 2 || heads < 1 || width % heads != 0)
            throw new ArgumentException("Invalid FlowHMR refiner dimensions.");
        this.frames = frames; this.width = width;
        const string prefix = "text_refiner.";
        embed = new(weights, prefix + "input_embedder", width, width);
        contextMlp = new(weights, prefix + "context_encoder.linears", width, width);
        timeMlp = new(weights, prefix + "timestep_encoder.blocks", width, width);
        mean = new float[width]; time = new float[width]; context = new float[width]; condition = new float[width];
        blocks = new RefinerBlock[2];
        for (var i = 0; i < blocks.Length; i++)
            blocks[i] = new(weights, prefix + "individual_token_refiner.blocks." + i, frames, width, heads);
    }

    internal void Run(ReadOnlySpan<float> input, float timestep, Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (input.Length != (long)frames * width || output.Length != input.Length || input.Overlaps(output))
            throw new ArgumentException("Refiner needs separate [frames,width] input and output tensors.");
        if (!float.IsFinite(timestep) || timestep < 0 || timestep > 1)
            throw new ArgumentOutOfRangeException(nameof(timestep));
        foreach (var value in input)
            if (!float.IsFinite(value)) throw new ArgumentException("Refiner input contains non-finite video features.", nameof(input));
        mean.AsSpan().Clear();
        for (var frame = 0; frame < frames; frame++)
            for (var channel = 0; channel < width; channel++) mean[channel] += input[frame * width + channel];
        for (var channel = 0; channel < width; channel++) mean[channel] /= frames;
        contextMlp.Run(mean, context, cancellation);
        FlowConditioning.Time(timestep, 1, time);
        timeMlp.Run(time, condition, cancellation);
        NeuralOps.AddInPlace(condition, context);
        embed.Run(input, frames, output, cancellation);
        foreach (var block in blocks) block.Run(output, condition, cancellation);
    }

    sealed class RefinerBlock
    {
        readonly int frames, width, heads, headWidth;
        readonly FlowLinear qkvProjection, projection, mlpFirst, mlpSecond, modulation;
        readonly Tensor norm1Weight, norm1Bias, norm2Weight, norm2Bias, queryWeight, queryBias, keyWeight, keyBias;
        readonly FlowAttention attention;
        readonly float[] norm, qkv, query, key, value, attended, projected, hidden, activatedCondition, gates;

        internal RefinerBlock(FlowWeights weights, string name, int frames, int width, int heads)
        {
            this.frames = frames; this.width = width; this.heads = heads; headWidth = width / heads;
            qkvProjection = new(weights, name + ".self_attn_qkv", width, 3 * width);
            projection = new(weights, name + ".self_attn_proj", width, width);
            mlpFirst = new(weights, name + ".mlp.fc1", width, 4 * width);
            mlpSecond = new(weights, name + ".mlp.fc2", 4 * width, width);
            modulation = new(weights, name + ".adaLN_modulation.linear", width, 2 * width);
            norm1Weight = weights.Get(name + ".norm1.weight", width); norm1Bias = weights.Get(name + ".norm1.bias", width);
            norm2Weight = weights.Get(name + ".norm2.weight", width); norm2Bias = weights.Get(name + ".norm2.bias", width);
            queryWeight = weights.Get(name + ".self_attn_q_norm.weight", headWidth); queryBias = weights.Get(name + ".self_attn_q_norm.bias", headWidth);
            keyWeight = weights.Get(name + ".self_attn_k_norm.weight", headWidth); keyBias = weights.Get(name + ".self_attn_k_norm.bias", headWidth);
            var size = checked(frames * width);
            norm = new float[size]; qkv = new float[checked(size * 3)];
            query = new float[size]; key = new float[size]; value = new float[size];
            attended = new float[size]; projected = new float[size]; hidden = new float[checked(size * 4)];
            activatedCondition = new float[width]; gates = new float[width * 2];
            attention = new(frames, width, heads, false);
        }

        internal void Run(Span<float> features, ReadOnlySpan<float> condition, CancellationToken cancellation)
        {
            condition.CopyTo(activatedCondition); FlowConditioning.Silu(activatedCondition);
            modulation.Run(activatedCondition, 1, gates, cancellation);
            features.CopyTo(norm); NeuralOps.LayerNorm(norm, frames, width, norm1Weight, norm1Bias);
            qkvProjection.Run(norm, frames, qkv, cancellation);
            for (var frame = 0; frame < frames; frame++)
            {
                var source = qkv.AsSpan(frame * 3 * width, 3 * width);
                source[..width].CopyTo(query.AsSpan(frame * width, width));
                source.Slice(width, width).CopyTo(key.AsSpan(frame * width, width));
                source.Slice(2 * width, width).CopyTo(value.AsSpan(frame * width, width));
            }
            NeuralOps.LayerNorm(query, frames * heads, headWidth, queryWeight, queryBias);
            NeuralOps.LayerNorm(key, frames * heads, headWidth, keyWeight, keyBias);
            attention.Run(query, key, value, attended, cancellation);
            projection.Run(attended, frames, projected, cancellation);
            FlowConditioning.AddGated(features, projected, gates.AsSpan(0, width));
            features.CopyTo(norm); NeuralOps.LayerNorm(norm, frames, width, norm2Weight, norm2Bias);
            mlpFirst.Run(norm, frames, hidden, cancellation); FlowConditioning.Silu(hidden);
            mlpSecond.Run(hidden, frames, projected, cancellation);
            FlowConditioning.AddGated(features, projected, gates.AsSpan(width, width));
        }
    }
}
