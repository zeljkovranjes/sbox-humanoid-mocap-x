// C# adaptation of SAM 3D Body's intermediate keypoint updates. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;
using HumanoidMocapX.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Updates 2D/3D keypoint positions and samples the conditioned image
/// context with zero padding and align_corners=false.</summary>
internal sealed class SamKeypointFeedback
{
    readonly SamDecoderShape shape;
    readonly int imageWidth, imageHeight, keypointStart;
    readonly FlowLinear positionFirst, positionSecond, spatialFirst, spatialSecond, imageProjection;
    readonly float[] hidden, embedded, features, projected, centered = new float[SamKeypoints.Count * 3];
    readonly bool[] valid = new bool[SamKeypoints.Count];

    internal SamKeypointFeedback(FlowWeights weights, SamDecoderShape shape, int imageWidth, int imageHeight, bool hand = false)
    {
        if (shape.Tokens < 145 || imageWidth < 1 || imageHeight < 1 || (long)imageWidth * imageHeight != shape.ImageTokens)
            throw new ArgumentException("SAM feedback needs 70 2D/3D tokens and matching image patch dimensions.");
        this.shape = shape; this.imageWidth = imageWidth; this.imageHeight = imageHeight;
        keypointStart = shape.Tokens - SamKeypoints.Count * 2;
        var suffix = hand ? "_hand" : "";
        positionFirst = new(weights, "keypoint_posemb_linear" + suffix + ".layers.0.0", 2, shape.Width);
        positionSecond = new(weights, "keypoint_posemb_linear" + suffix + ".layers.1", shape.Width, shape.Width);
        spatialFirst = new(weights, "keypoint3d_posemb_linear" + suffix + ".layers.0.0", 3, shape.Width);
        spatialSecond = new(weights, "keypoint3d_posemb_linear" + suffix + ".layers.1", shape.Width, shape.Width);
        imageProjection = new(weights, "keypoint_feat_linear" + suffix, shape.ContextWidth, shape.Width);
        hidden = new float[SamKeypoints.Count * shape.Width]; embedded = new float[hidden.Length]; projected = new float[hidden.Length];
        features = new float[SamKeypoints.Count * shape.ContextWidth];
    }

    /// <summary>Consumes crop XY coordinates, camera depths and camera-space XYZ
    /// keypoints. Changes only the final 140 token positions and the 70 2D tokens.
    /// Rejects malformed or aliased buffers; reuses storage sequentially.</summary>
    internal void Run(ReadOnlySpan<float> image, ReadOnlySpan<float> cropped, ReadOnlySpan<float> depths, ReadOnlySpan<float> points,
        Span<float> tokens, Span<float> positions, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (image.Length != shape.ImageElements || cropped.Length != SamKeypoints.Count * 2 || depths.Length != SamKeypoints.Count
            || points.Length != centered.Length || tokens.Length != shape.TokenElements || positions.Length != tokens.Length
            || image.Overlaps(tokens) || image.Overlaps(positions) || cropped.Overlaps(tokens) || cropped.Overlaps(positions)
            || depths.Overlaps(tokens) || depths.Overlaps(positions) || points.Overlaps(tokens) || points.Overlaps(positions) || tokens.Overlaps(positions))
            throw new ArgumentException("SAM keypoint feedback tensors have invalid dimensions or overlap.");
        SamDecoder.Finite(image); SamDecoder.Finite(cropped); SamDecoder.Finite(depths); SamDecoder.Finite(points);
        SamDecoder.Finite(tokens); SamDecoder.Finite(positions);
        var position2d = positions.Slice(keypointStart * shape.Width, hidden.Length);
        positionFirst.Run(cropped, SamKeypoints.Count, hidden, cancellation); NeuralOps.Relu(hidden);
        positionSecond.RunParallel(hidden, SamKeypoints.Count, embedded, cancellation); embedded.CopyTo(position2d);
        Sample(image, cropped, depths);
        for (var point = 0; point < SamKeypoints.Count; point++) if (!valid[point]) position2d.Slice(point * shape.Width, shape.Width).Clear();
        imageProjection.RunParallel(features, SamKeypoints.Count, projected, cancellation);
        // Invalid samples are zeroed before projection, so its learned bias still applies.
        NeuralOps.AddInPlace(tokens.Slice(keypointStart * shape.Width, hidden.Length), projected);
        for (var point = 0; point < SamKeypoints.Count; point++)
            for (var axis = 0; axis < 3; axis++) centered[point * 3 + axis] = points[point * 3 + axis] - (points[9 * 3 + axis] + points[10 * 3 + axis]) / 2;
        spatialFirst.Run(centered, SamKeypoints.Count, hidden, cancellation); NeuralOps.Relu(hidden);
        spatialSecond.RunParallel(hidden, SamKeypoints.Count, embedded, cancellation);
        embedded.CopyTo(positions.Slice((keypointStart + SamKeypoints.Count) * shape.Width));
        SamDecoder.Finite(tokens); SamDecoder.Finite(positions);
    }

    void Sample(ReadOnlySpan<float> image, ReadOnlySpan<float> cropped, ReadOnlySpan<float> depths)
    {
        features.AsSpan().Clear();
        for (var point = 0; point < SamKeypoints.Count; point++)
        {
            var x = cropped[point * 2]; var y = cropped[point * 2 + 1];
            valid[point] = x >= -.5f && x <= .5f && y >= -.5f && y <= .5f && depths[point] >= 1e-5f;
            if (!valid[point]) continue;
            x = (x + .5f) * imageWidth - .5f; y = (y + .5f) * imageHeight - .5f;
            var left = (int)MathF.Floor(x); var top = (int)MathF.Floor(y); var dx = x - left; var dy = y - top;
            var output = features.AsSpan(point * shape.ContextWidth, shape.ContextWidth);
            Accumulate(image, left, top, (1 - dx) * (1 - dy), output);
            Accumulate(image, left + 1, top, dx * (1 - dy), output);
            Accumulate(image, left, top + 1, (1 - dx) * dy, output);
            Accumulate(image, left + 1, top + 1, dx * dy, output);
        }
    }

    void Accumulate(ReadOnlySpan<float> image, int x, int y, float weight, Span<float> output)
    {
        if (x < 0 || x >= imageWidth || y < 0 || y >= imageHeight) return;
        var start = (y * imageWidth + x) * shape.ContextWidth;
        for (var channel = 0; channel < output.Length; channel++) output[channel] += image[start + channel] * weight;
    }
}
