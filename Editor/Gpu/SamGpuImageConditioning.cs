// C# adaptation of SAM 3D Body's camera and prompt encoding. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Inference;

/// <summary>Conditions GPU DINO patch features for one body/hand decoder. Caller owns
/// storage and submits Run inside a graphics batch; no image readback is required.</summary>
internal sealed class SamGpuImageConditioning
{
    readonly DinoShape shape;
    readonly GpuBuffer<float> masked, combined, projected, weight, mask, norm, bias;
    internal GpuBuffer<float> Positions { get; }

    internal SamGpuImageConditioning(FlowGpuStorage storage, FlowWeights weights, DinoShape shape, bool hand = false)
    {
        if (!shape.IsSam) throw new ArgumentException("GPU image conditioning requires a SAM shape.", nameof(shape));
        this.shape = shape; var width = shape.Channels; var name = hand ? "ray_cond_emb_hand" : "ray_cond_emb";
        weight = storage.Weight(weights, name + ".conv.weight", 1, 1, width + SamRays.Channels, width);
        norm = storage.Weight(weights, name + ".norm.weight", width); bias = storage.Weight(weights, name + ".norm.bias", width);
        var parameters = new float[width * 2]; weights.Get("prompt_encoder.no_mask_embed.weight", width, 1).Data.CopyTo(parameters, 0);
        mask = storage.Upload<float>(parameters);
        masked = storage.Create<float>(shape.Patches * width); projected = storage.Create<float>(shape.Patches * width);
        combined = storage.Create<float>(shape.Patches * (width + SamRays.Channels));
        Positions = storage.Upload<float>(SamImageConditioning.Positions(weights, shape, hand));
    }

    /// <summary>Consumes finite raw DINO features and SamRays output; writes normalized
    /// context. Rejects invalid buffers or aliasing. Positional context remains in Positions.</summary>
    internal void Run(TensorKernels kernels, GpuBuffer<float> image, GpuBuffer<float> rays,
        GpuBuffer<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (image is null || !image.IsValid || image.ElementCount != shape.Patches * shape.Channels
            || rays is null || !rays.IsValid || rays.ElementCount != shape.Patches * SamRays.Channels
            || output is null || !output.IsValid || output.ElementCount != image.ElementCount
            || ReferenceEquals(image, output) || ReferenceEquals(rays, output) || ReferenceEquals(Positions, output))
            throw new ArgumentException("SAM GPU image context buffers have invalid dimensions or overlap.");
        kernels.Modulate(image, mask, masked, shape.Patches, shape.Channels, 0, shape.Channels);
        kernels.ConcatColumns(masked, rays, combined, shape.Patches, shape.Channels, SamRays.Channels);
        kernels.Linear(combined, weight, null, projected, shape.Patches, shape.Channels + SamRays.Channels, shape.Channels);
        kernels.Normalize(projected, norm, bias, output, shape.Patches, shape.Channels, SamLayerNorm.Epsilon);
    }
}
