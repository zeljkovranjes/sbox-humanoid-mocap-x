// C# adaptation of SAM 3D Body's promptable decoder. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Predicts an intermediate pose from normalized tokens and updates the
/// keypoint tokens/positions in place. Called after layers 0–4; never retain these spans.</summary>
internal delegate void SamPoseFeedback(int layer, ReadOnlySpan<float> normalized, Span<float> tokens,
    Span<float> positions, CancellationToken cancellation);

/// <summary>Six-layer body/hand decoder with required intermediate pose feedback.
/// Inputs already include camera conditioning and prompt embeddings.</summary>
internal sealed class SamDecoder
{
    readonly SamDecoderShape shape;
    readonly SamDecoderBlock[] layers = new SamDecoderBlock[SamDecoderShape.Layers];
    readonly SamLayerNorm finalNorm;
    readonly SamDecoderWorkspace workspace;
    readonly float[] tokens, positions, normalized;

    internal SamDecoder(FlowWeights weights, SamDecoderShape shape, bool hand = false)
    {
        if (shape.Tokens == 0) throw new ArgumentException("Construct a valid SAM decoder shape.", nameof(shape));
        this.shape = shape; var name = hand ? "decoder_hand" : "decoder";
        for (var layer = 0; layer < layers.Length; layer++) layers[layer] = new(weights, name + ".layers." + layer, shape);
        finalNorm = new(weights, name + ".norm_final", shape.Width); workspace = new(shape);
        tokens = new float[shape.TokenElements]; positions = new float[shape.TokenElements]; normalized = new float[shape.TokenElements];
    }

    /// <summary>Writes all final normalized tokens, including the first pose token.
    /// The caller predicts the final pose from this output separately. Intermediate
    /// feedback is mandatory; no unconditioned fallback is supplied. Reuse sequentially.</summary>
    internal void Run(ReadOnlySpan<float> input, ReadOnlySpan<float> image, ReadOnlySpan<float> tokenPositions,
        ReadOnlySpan<float> imagePositions, Span<float> output, SamPoseFeedback feedback, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(feedback);
        Validate(shape, input, image, tokenPositions, imagePositions, output);
        input.CopyTo(tokens); tokenPositions.CopyTo(positions);
        for (var layer = 0; layer < layers.Length; layer++)
        {
            layers[layer].Run(tokens, image, positions, imagePositions, layer == 0, workspace, cancellation);
            finalNorm.Run(tokens, normalized); Finite(normalized);
            if (layer == layers.Length - 1) break;
            feedback(layer, normalized, tokens, positions, cancellation); Finite(tokens); Finite(positions);
        }
        cancellation.ThrowIfCancellationRequested(); normalized.AsSpan().CopyTo(output);
    }

    internal static void Validate(SamDecoderShape shape, ReadOnlySpan<float> input, ReadOnlySpan<float> image,
        ReadOnlySpan<float> tokenPositions, ReadOnlySpan<float> imagePositions, Span<float> output)
    {
        if (input.Length != shape.TokenElements || tokenPositions.Length != input.Length || output.Length != input.Length
            || image.Length != shape.ImageElements || imagePositions.Length != image.Length || input.Overlaps(output)
            || tokenPositions.Overlaps(output) || image.Overlaps(output) || imagePositions.Overlaps(output))
            throw new ArgumentException("SAM decoder tensors have invalid dimensions or overlap.");
        Finite(input); Finite(image); Finite(tokenPositions); Finite(imagePositions);
    }

    internal static void Finite(ReadOnlySpan<float> values)
    { foreach (var value in values) if (!float.IsFinite(value)) throw new ArgumentException("Non-finite SAM decoder feature or pose feedback."); }
}
