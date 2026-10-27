// C# adaptation of SAM 3D Body's camera and prompt encoding. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;
using HumanoidMocapX.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Prepares body or hand decoder image context for inference without a segmentation mask.</summary>
internal sealed class SamImageConditioning
{
    readonly DinoShape shape;
    readonly Tensor projection, noMask;
    readonly SamLayerNorm norm;
    readonly float[] combined, positions;

    internal SamImageConditioning(FlowWeights weights, DinoShape shape, bool hand = false)
    {
        if (!shape.IsSam) throw new ArgumentException("Image conditioning requires a SAM shape.", nameof(shape));
        this.shape = shape; var name = hand ? "ray_cond_emb_hand" : "ray_cond_emb";
        var weight = weights.Get(name + ".conv.weight", 1, 1, shape.Channels + SamRays.Channels, shape.Channels);
        projection = new(weight.Name, weight.Data, new[] { shape.Channels + SamRays.Channels, shape.Channels });
        noMask = weights.Get("prompt_encoder.no_mask_embed.weight", shape.Channels, 1);
        norm = new(weights, name + ".norm", shape.Channels);
        positions = Positions(weights, shape, hand);
        combined = new float[shape.Patches * (shape.Channels + SamRays.Channels)];
    }

    /// <summary>Consumes raw DINO patch features and encoded camera rays; writes
    /// normalized decoder context. Rejects malformed/overlapping inputs. Discard output on failure.</summary>
    internal void Run(ReadOnlySpan<float> image, ReadOnlySpan<float> rays, Span<float> output, CancellationToken cancellation = default)
    {
        if (image.Length != shape.Patches * shape.Channels || output.Length != image.Length || rays.Length != shape.Patches * SamRays.Channels
            || image.Overlaps(output) || rays.Overlaps(output)) throw new ArgumentException("SAM image context tensors have invalid dimensions or overlap.");
        cancellation.ThrowIfCancellationRequested(); SamDecoder.Finite(image); SamDecoder.Finite(rays);
        var width = shape.Channels;
        for (var patch = 0; patch < shape.Patches; patch++)
        {
            var row = combined.AsSpan(patch * (width + SamRays.Channels), width + SamRays.Channels);
            for (var i = 0; i < width; i++) row[i] = image[patch * width + i] + noMask.Data[i];
            rays.Slice(patch * SamRays.Channels, SamRays.Channels).CopyTo(row.Slice(width));
        }
        FlowLinear.Run(combined, shape.Patches, projection, null, output, cancellation);
        norm.Run(output, output); SamDecoder.Finite(output);
    }

    /// <summary>Copies the fixed row-major positional context for this body or hand decoder.</summary>
    internal void CopyPositions(Span<float> output)
    {
        if (output.Length != positions.Length) throw new ArgumentException("SAM image position output has the wrong size.", nameof(output));
        positions.AsSpan().CopyTo(output);
    }

    internal static float[] Positions(FlowWeights weights, DinoShape shape, bool hand)
    {
        if (!shape.IsSam) throw new ArgumentException("Image positions require a SAM shape.", nameof(shape));
        var half = shape.Channels / 2;
        var matrix = weights.Get((hand ? "hand_pe_layer" : "prompt_encoder.pe_layer") + ".positional_encoding_gaussian_matrix", half, 2).Data;
        var output = new float[shape.Patches * shape.Channels]; var height = shape.Height / 16; var width = shape.Width / 16;
        for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
                for (var i = 0; i < half; i++)
                {
                    var angle = 2 * MathF.PI * ((2 * ((x + .5f) / width) - 1) * matrix[i] + (2 * ((y + .5f) / height) - 1) * matrix[half + i]);
                    var at = (y * width + x) * shape.Channels + i;
                    output[at] = MathF.Sin(angle); output[at + half] = MathF.Cos(angle);
                }
        SamDecoder.Finite(output); return output;
    }
}
