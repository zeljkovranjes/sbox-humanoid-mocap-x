// C# adaptation of SAM 3D Body's initial decoder tokens. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Prepares the released model's first body/hand pass with one invalid
/// point prompt, camera conditioning and learned pose/camera initialization.</summary>
internal sealed class SamInitialTokens
{
    internal const int Count = 145, HandBoxStart = 3, KeypointStart = 5;
    readonly int width;
    readonly FlowLinear initialProjection;
    readonly float[] input = new float[525], previous, prompt;
    readonly Tensor handBoxes, keypoints, spatialKeypoints;

    internal SamInitialTokens(FlowWeights weights, bool hand = false, int width = 1024, int contextWidth = 1280)
    {
        if (width < 1 || contextWidth < 1) throw new ArgumentException("SAM initial token dimensions must be positive.");
        this.width = width; var suffix = hand ? "_hand" : "";
        weights.Get("init_pose" + suffix + ".weight", SamMhrParameters.Predictions, 1).Data.AsSpan().CopyTo(input.AsSpan(3));
        weights.Get("init_camera" + suffix + ".weight", 3, 1).Data.AsSpan().CopyTo(input.AsSpan(522));
        initialProjection = new(weights, "init_to_token_mhr" + suffix, input.Length, width);
        previous = new float[width]; prompt = new float[width];
        new FlowLinear(weights, "prev_to_token_mhr" + suffix, 522, width).Run(input.AsSpan(3), 1, previous);
        var invalid = weights.Get("prompt_encoder.invalid_point_embed.weight", contextWidth, 1);
        new FlowLinear(weights, "prompt_to_token", contextWidth, width).Run(invalid.Data, 1, prompt);
        handBoxes = weights.Get("hand_box_embedding.weight", width, 2);
        keypoints = weights.Get("keypoint_embedding" + suffix + ".weight", width, SamKeypoints.Count);
        spatialKeypoints = weights.Get("keypoint3d_embedding" + suffix + ".weight", width, SamKeypoints.Count);
    }

    /// <summary>Writes all 145 tokens and their initial positions. The first token
    /// depends on the crop; previous-pose and invalid-prompt embeddings also supply
    /// their positional vectors. Rejects malformed or overlapping outputs.</summary>
    internal void Run(SamProjection projection, Span<float> tokens, Span<float> positions, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (tokens.Length != Count * width || positions.Length != tokens.Length || tokens.Overlaps(positions))
            throw new ArgumentException("SAM initial token outputs have invalid dimensions or overlap.");
        projection.CopyCondition(input.AsSpan(0, 3));
        initialProjection.Run(input, 1, tokens.Slice(0, width), cancellation);
        previous.AsSpan().CopyTo(tokens.Slice(width)); prompt.AsSpan().CopyTo(tokens.Slice(width * 2));
        handBoxes.Data.AsSpan().CopyTo(tokens.Slice(width * HandBoxStart));
        keypoints.Data.AsSpan().CopyTo(tokens.Slice(width * KeypointStart));
        spatialKeypoints.Data.AsSpan().CopyTo(tokens.Slice(width * (KeypointStart + SamKeypoints.Count)));
        positions.Clear(); previous.AsSpan().CopyTo(positions.Slice(width)); prompt.AsSpan().CopyTo(positions.Slice(width * 2));
        SamDecoder.Finite(tokens); SamDecoder.Finite(positions);
    }
}
