// C# adaptation of SAM 3D Body crop transforms and OpenCV affine sampling.
// See LICENSE-SAM, LICENSE-OPENCV and THIRD_PARTY_NOTICES.md.
using System;
using System.Runtime.InteropServices;
using System.Threading;
using HumanoidMocapX.EditorTools.Video;

namespace HumanoidMocapX.Inference;

/// <summary>Describes one square SAM body/hand crop in original-image coordinates.
/// Affine image sampling retains OpenCV's fixed-point rounding and black border.</summary>
internal readonly struct SamCrop
{
    const float PriorAspect = .75f, BodyPadding = 1.25f, HandPadding = .9f;
    const int AffineBits = 10, SampleBits = 5, AffineScale = 1 << AffineBits, RoundDelta = 1 << (AffineBits - SampleBits - 1);
    readonly double scaleX, scaleY, offsetX, offsetY, inverseX, inverseY, sourceX, sourceY;
    internal DinoShape Shape { get; }
    internal float CenterX { get; }
    internal float CenterY { get; }
    internal float Size { get; }

    internal SamCrop(float left, float top, float right, float bottom, DinoShape shape, bool hand = false)
    {
        if (!shape.IsSam || shape.Width != shape.Height || !float.IsFinite(left) || !float.IsFinite(top)
            || !float.IsFinite(right) || !float.IsFinite(bottom) || right <= left || bottom <= top)
            throw new ArgumentException("SAM cropping requires a finite, positive XYXY box and square SAM image dimensions.");
        Shape = shape; CenterX = (left + right) * .5f; CenterY = (top + bottom) * .5f;
        var padding = hand ? HandPadding : BodyPadding; var width = (right - left) * padding; var height = (bottom - top) * padding;
        Size = width > height * PriorAspect ? width / PriorAspect : height;
        // getAffineTransform receives three float32 source points, even though it returns doubles.
        var upper = (float)(CenterY - (double)Size * .5); var horizontal = CenterX - (CenterY - upper);
        var halfWidth = (double)CenterX - horizontal; var halfHeight = (double)CenterY - upper;
        if (!float.IsFinite(Size) || Size <= 0 || halfWidth <= 0 || halfHeight <= 0)
            throw new ArgumentException("SAM crop box cannot be represented by distinct affine source points.");
        scaleX = shape.Width * .5 / halfWidth; scaleY = shape.Height * .5 / halfHeight;
        offsetX = shape.Width * .5 - scaleX * CenterX; offsetY = shape.Height * .5 - scaleY * CenterY;
        var reciprocalDeterminant = 1 / (scaleX * scaleY);
        inverseX = scaleY * reciprocalDeterminant; inverseY = scaleX * reciprocalDeterminant;
        sourceX = -inverseX * offsetX; sourceY = -inverseY * offsetY;
        _ = Fixed(sourceX); _ = Fixed(sourceY);
        _ = checked(Fixed(sourceX) + RoundDelta + Fixed(inverseX * (shape.Width - 1)));
        _ = checked(Fixed(sourceY + inverseY * (shape.Height - 1)) + RoundDelta);
    }

    /// <summary>Copies the float32 full-image-to-crop affine used by SAM's camera
    /// conditioning. Image resampling itself retains the double-precision matrix.</summary>
    internal void CopyAffine(Span<float> output)
    {
        if (!Shape.IsSam || output.Length != 6) throw new ArgumentException("SAM affine output needs six values and a valid crop.");
        output[0] = (float)scaleX; output[1] = 0; output[2] = (float)offsetX;
        output[3] = 0; output[4] = (float)scaleY; output[5] = (float)offsetY;
        SamDecoder.Finite(output);
    }

    /// <summary>Converts centred crop XY pairs to original-image pixels in place.
    /// A mirrored crop stays in mirrored-image coordinates; its caller unflips X.</summary>
    internal void ToImage(Span<float> points)
    {
        if (!Shape.IsSam || points.Length % 2 != 0) throw new ArgumentException("SAM image projection requires a valid crop and XY pairs.");
        SamDecoder.Finite(points);
        for (var i = 0; i < points.Length; i += 2)
        {
            points[i] = ((points[i] + .5f) * Shape.Width - (float)offsetX) / (float)scaleX;
            points[i + 1] = ((points[i + 1] + .5f) * Shape.Height - (float)offsetY) / (float)scaleY;
        }
        SamDecoder.Finite(points);
    }

    /// <summary>Writes HWC RGB floats in [0,1] from RGBA8, ignoring alpha. The input
    /// is sampled mirrored when flipHorizontal is set, using a crop in that
    /// mirrored image's coordinates. Rejects invalid dimensions,
    /// aliasing and OpenCV-incompatible image sizes. Discard output on cancellation.</summary>
    internal void Prepare(ReadOnlySpan<byte> rgba, int width, int height, Span<float> output, CancellationToken cancellation = default,
        bool flipHorizontal = false)
    {
        cancellation.ThrowIfCancellationRequested();
        if (!Shape.IsSam || width >= short.MaxValue || height >= short.MaxValue || output.Length != Shape.ImageElements
            || rgba.Overlaps(MemoryMarshal.AsBytes(output))) throw new ArgumentException("SAM crop buffers have invalid dimensions or overlap.");
        var image = new LinearRgbImage(rgba, width, height);
        Span<int> horizontal = stackalloc int[Shape.Width];
        var startX = checked(Fixed(sourceX) + RoundDelta);
        for (var x = 0; x < horizontal.Length; x++)
        {
            var coordinate = Coordinate(checked(startX + Fixed(inverseX * x)));
            horizontal[x] = flipHorizontal ? ((width - 1) << SampleBits) - coordinate : coordinate;
        }
        for (var y = 0; y < Shape.Height; y++)
        {
            cancellation.ThrowIfCancellationRequested();
            var vertical = Coordinate(checked(Fixed(sourceY + inverseY * y) + RoundDelta));
            for (var x = 0; x < horizontal.Length; x++)
            {
                var pixel = image.Sample(horizontal[x], vertical); var at = (y * Shape.Width + x) * 3;
                output[at] = pixel.X; output[at + 1] = pixel.Y; output[at + 2] = pixel.Z;
            }
        }
    }

    static int Fixed(double coordinate)
    {
        var rounded = Math.Round(coordinate * AffineScale);
        if (!double.IsFinite(rounded) || rounded < int.MinValue || rounded > int.MaxValue - RoundDelta)
            throw new ArgumentException("SAM crop exceeds the supported fixed-point image coordinate range.");
        return (int)rounded;
    }

    static int Coordinate(int affine)
    {
        var sample = affine >> (AffineBits - SampleBits);
        return (Math.Clamp(sample >> SampleBits, short.MinValue, short.MaxValue) << SampleBits) | (sample & 31);
    }
}
