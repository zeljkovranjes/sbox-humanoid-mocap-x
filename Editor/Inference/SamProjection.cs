// C# adaptation of SAM 3D Body's perspective camera head. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;

namespace HumanoidMocapX.Inference;

/// <summary>Projects camera-space predictions into one axis-aligned SAM crop,
/// using the released model's intrinsic-centre CLIFF convention.</summary>
internal readonly struct SamProjection
{
    readonly float focalX, focalY, centerX, centerY, boxX, boxY, boxSize, scaleFactor;
    readonly float affineX, affineY, offsetX, offsetY;
    readonly int width, height;

    internal SamProjection(ReadOnlySpan<float> intrinsics, ReadOnlySpan<float> affine,
        float boxX, float boxY, float boxSize, DinoShape shape, bool hand = false)
    {
        if (intrinsics.Length != 9 || affine.Length != 6 || !shape.IsSam)
            throw new ArgumentException("SAM projection requires full-image intrinsics, a crop affine and SAM image dimensions.");
        SamDecoder.Finite(intrinsics); SamDecoder.Finite(affine);
        if (intrinsics[0] <= 0 || intrinsics[4] <= 0 || intrinsics[1] != 0 || intrinsics[3] != 0
            || intrinsics[6] != 0 || intrinsics[7] != 0 || intrinsics[8] != 1
            || affine[0] <= 0 || affine[4] <= 0 || affine[1] != 0 || affine[3] != 0
            || !float.IsFinite(boxX) || !float.IsFinite(boxY) || !float.IsFinite(boxSize) || boxSize <= 0)
            throw new ArgumentException("SAM projection camera or crop is invalid, rotated or skewed.");
        focalX = intrinsics[0]; focalY = intrinsics[4]; centerX = intrinsics[2]; centerY = intrinsics[5];
        affineX = affine[0]; affineY = affine[4]; offsetX = affine[2]; offsetY = affine[5];
        this.boxX = boxX; this.boxY = boxY; this.boxSize = boxSize;
        width = shape.Width; height = shape.Height; scaleFactor = hand ? 10 : 1;
    }

    /// <summary>Copies the crop's CLIFF condition: centre offset and size divided
    /// by the original image's horizontal focal length.</summary>
    internal void CopyCondition(Span<float> output)
    {
        if (width == 0 || output.Length != 3) throw new ArgumentException("SAM camera conditioning requires a valid crop and three output values.");
        output[0] = (boxX - centerX) / focalX; output[1] = (boxY - centerY) / focalX; output[2] = boxSize / focalX;
        SamDecoder.Finite(output);
    }

    /// <summary>Writes crop coordinates centred on zero, signed camera depths and
    /// camera translation in metres. Behind-camera points remain signed for masking.
    /// Rejects aliasing/non-finite data; undefined projections fail explicitly.</summary>
    internal void Run(ReadOnlySpan<float> points, ReadOnlySpan<float> camera, Span<float> cropped, Span<float> depths, Span<float> translation)
    {
        if (width == 0 || points.Length == 0 || points.Length % 3 != 0 || camera.Length != 3
            || depths.Length != points.Length / 3 || cropped.Length != depths.Length * 2 || translation.Length != 3
            || points.Overlaps(cropped) || points.Overlaps(depths) || points.Overlaps(translation)
            || camera.Overlaps(cropped) || camera.Overlaps(depths) || camera.Overlaps(translation)
            || cropped.Overlaps(depths) || cropped.Overlaps(translation) || depths.Overlaps(translation))
            throw new ArgumentException("SAM projection tensors have invalid dimensions or overlap.");
        SamDecoder.Finite(points); SamDecoder.Finite(camera);
        var denominator = boxSize * -camera[0] * scaleFactor + 1e-8f;
        translation[0] = camera[1] + 2 * (boxX - centerX) / denominator;
        translation[1] = -camera[2] + 2 * (boxY - centerY) / denominator;
        translation[2] = 2 * focalX / denominator;
        for (var point = 0; point < depths.Length; point++)
        {
            var x = points[point * 3] + translation[0]; var y = points[point * 3 + 1] + translation[1];
            var depth = points[point * 3 + 2] + translation[2]; depths[point] = depth;
            var imageX = focalX * (x / depth) + centerX; var imageY = focalY * (y / depth) + centerY;
            cropped[point * 2] = (affineX * imageX + offsetX) / width - .5f;
            cropped[point * 2 + 1] = (affineY * imageY + offsetY) / height - .5f;
        }
        SamDecoder.Finite(translation); SamDecoder.Finite(cropped); SamDecoder.Finite(depths);
    }
}
