// C# adaptation of VGGT-Omega image preparation and Pillow bicubic resampling.
// See LICENSE-VGGT, LICENSE-PILLOW and THIRD_PARTY_NOTICES.md.
using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Crops unsupported aspect ratios and performs byte-rounded, antialiased
/// bicubic resizing for Omega's max-size 512 input. Buffers are reused per video.</summary>
internal sealed class OmegaImage
{
    const int Resolution = 512, PatchSize = 16, Precision = 22;
    readonly int sourceWidth, sourceHeight, left, top, cropWidth, cropHeight;
    readonly Filter horizontal, vertical;
    readonly byte[] intermediate;
    readonly bool verticalFirst;
    internal int Width { get; }
    internal int Height { get; }

    internal OmegaImage(int width, int height)
    {
        if (width < 1 || height < 1 || (long)width * height > int.MaxValue / 4)
            throw new ArgumentException("Omega image dimensions must fit an RGBA pixel buffer.");
        sourceWidth = width; sourceHeight = height;
        cropWidth = (int)Math.Min(width, 2L * height); cropHeight = (int)Math.Min(height, 2L * width);
        left = (width - cropWidth) / 2; top = (height - cropHeight) / 2;
        Width = cropWidth >= cropHeight ? Resolution : Round(Resolution * cropWidth / (double)cropHeight);
        Height = cropHeight >= cropWidth ? Resolution : Round(Resolution * cropHeight / (double)cropWidth);
        horizontal = new(cropWidth, Width); vertical = new(cropHeight, Height);
        verticalFirst = cropHeight > Height && cropHeight - Height > 2 * (cropWidth - Width);
        intermediate = new byte[checked((verticalFirst ? cropWidth * Height : Width * cropHeight) * 3)];
    }

    /// <summary>Writes HWC RGB floats in [0,1]. Transparent pixels composite onto white
    /// before resizing. Rejects malformed/overlapping buffers; discard output on cancellation.</summary>
    internal void Prepare(ReadOnlySpan<byte> rgba, Span<float> output, CancellationToken cancellation = default)
    {
        if (rgba.Length != (long)sourceWidth * sourceHeight * 4 || output.Length != Width * Height * 3
            || rgba.Overlaps(MemoryMarshal.AsBytes(output))) throw new ArgumentException("Omega image tensors have invalid dimensions or overlap.");
        cancellation.ThrowIfCancellationRequested();
        if (verticalFirst) VerticalThenHorizontal(rgba, output, cancellation);
        else HorizontalThenVertical(rgba, output, cancellation);
    }

    void HorizontalThenVertical(ReadOnlySpan<byte> rgba, Span<float> output, CancellationToken cancellation)
    {
        for (var y = 0; y < cropHeight; y++)
        {
            cancellation.ThrowIfCancellationRequested();
            for (var x = 0; x < Width; x++)
                for (var channel = 0; channel < 3; channel++)
                {
                    long sum = 1 << (Precision - 1);
                    for (var k = 0; k < horizontal.Count[x]; k++)
                        sum += Pixel(rgba, horizontal.Start[x] + k, y, channel) * (long)horizontal.Weight[x * horizontal.Kernel + k];
                    intermediate[(y * Width + x) * 3 + channel] = Clip(sum);
                }
        }
        for (var y = 0; y < Height; y++)
        {
            cancellation.ThrowIfCancellationRequested();
            for (var x = 0; x < Width; x++)
                for (var channel = 0; channel < 3; channel++)
                {
                    long sum = 1 << (Precision - 1);
                    for (var k = 0; k < vertical.Count[y]; k++)
                        sum += intermediate[((vertical.Start[y] + k) * Width + x) * 3 + channel] * (long)vertical.Weight[y * vertical.Kernel + k];
                    output[(y * Width + x) * 3 + channel] = Clip(sum) / 255f;
                }
        }
    }

    void VerticalThenHorizontal(ReadOnlySpan<byte> rgba, Span<float> output, CancellationToken cancellation)
    {
        for (var y = 0; y < Height; y++)
        {
            cancellation.ThrowIfCancellationRequested();
            for (var x = 0; x < cropWidth; x++)
                for (var channel = 0; channel < 3; channel++)
                {
                    long sum = 1 << (Precision - 1);
                    for (var k = 0; k < vertical.Count[y]; k++)
                        sum += Pixel(rgba, x, vertical.Start[y] + k, channel) * (long)vertical.Weight[y * vertical.Kernel + k];
                    intermediate[(y * cropWidth + x) * 3 + channel] = Clip(sum);
                }
        }
        for (var y = 0; y < Height; y++)
        {
            cancellation.ThrowIfCancellationRequested();
            for (var x = 0; x < Width; x++)
                for (var channel = 0; channel < 3; channel++)
                {
                    long sum = 1 << (Precision - 1);
                    for (var k = 0; k < horizontal.Count[x]; k++)
                        sum += intermediate[(y * cropWidth + horizontal.Start[x] + k) * 3 + channel] * (long)horizontal.Weight[x * horizontal.Kernel + k];
                    output[(y * Width + x) * 3 + channel] = Clip(sum) / 255f;
                }
        }
    }

    int Pixel(ReadOnlySpan<byte> rgba, int x, int y, int channel)
    {
        var at = ((y + top) * sourceWidth + x + left) * 4; var alpha = rgba[at + 3];
        return (rgba[at + channel] * alpha + 255 * (255 - alpha) + 127) / 255;
    }

    static byte Clip(long value) => (byte)Math.Clamp(value >> Precision, 0L, 255L);
    static int Round(double value) => Math.Max(PatchSize, (int)Math.Round(value / PatchSize) * PatchSize);

    sealed class Filter
    {
        internal readonly int[] Start, Count, Weight;
        internal readonly int Kernel;

        internal Filter(int source, int target)
        {
            var scale = source / (double)target; var filterScale = Math.Max(1, scale); var support = 2 * filterScale;
            Kernel = (int)Math.Ceiling(support) * 2 + 1;
            Start = new int[target]; Count = new int[target]; Weight = new int[checked(target * Kernel)];
            var temporary = new double[Kernel];
            for (var pixel = 0; pixel < target; pixel++)
            {
                var center = (pixel + .5) * scale;
                Start[pixel] = Math.Max(0, (int)(center - support + .5));
                Count[pixel] = Math.Min(source, (int)(center + support + .5)) - Start[pixel];
                double total = 0;
                for (var k = 0; k < Count[pixel]; k++) total += temporary[k] = Cubic((k + Start[pixel] - center + .5) / filterScale);
                for (var k = 0; k < Count[pixel]; k++) Weight[pixel * Kernel + k] = (int)Math.Round(temporary[k] / total * (1 << Precision), MidpointRounding.AwayFromZero);
            }
        }

        static double Cubic(double value)
        {
            var x = Math.Abs(value);
            return x < 1 ? (1.5 * x - 2.5) * x * x + 1 : x < 2 ? -.5 * (((x - 5) * x + 8) * x - 4) : 0;
        }
    }
}
