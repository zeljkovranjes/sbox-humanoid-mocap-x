using System;

namespace HumanoidMocapX.EditorTools.Video;
using Vector3 = System.Numerics.Vector3;

/// <summary>Reads RGBA8 images with byte-rounded bilinear RGB sampling. Coordinates
/// use five fractional bits; alpha is ignored. The image is borrowed, never copied.</summary>
public readonly ref struct LinearRgbImage
{
    readonly ReadOnlySpan<byte> pixels;
    readonly int width, height;

    /// <summary>Validates the borrowed image dimensions. Throws for empty images,
    /// incomplete pixels or dimensions that cannot fit the supplied buffer.</summary>
    public LinearRgbImage(ReadOnlySpan<byte> rgba, int width, int height)
    {
        if (width < 1 || height < 1 || (long)width * height * 4 != rgba.Length)
            throw new ArgumentException("RGB sampling requires a complete, nonempty RGBA8 image.");
        pixels = rgba; this.width = width; this.height = height;
    }

    /// <summary>Returns RGB in [0,1] at coordinates measured in 1/32 pixels.
    /// Outside neighbours are black unless replicateBorder selects the nearest edge.
    /// A default/uninitialized image is rejected.</summary>
    public Vector3 Sample(int x32, int y32, bool replicateBorder = false)
    {
        if (width == 0) throw new InvalidOperationException("Construct a valid RGB image before sampling.");
        var x = x32 >> 5; var y = y32 >> 5; var fx = x32 & 31; var fy = y32 & 31;
        var a = Offset(x, y, replicateBorder); var b = Offset(x + 1, y, replicateBorder);
        var c = Offset(x, y + 1, replicateBorder); var d = Offset(x + 1, y + 1, replicateBorder);
        var wa = (32 - fx) * (32 - fy); var wb = fx * (32 - fy); var wc = (32 - fx) * fy; var wd = fx * fy;
        return new(Channel(0, a, b, c, d, wa, wb, wc, wd), Channel(1, a, b, c, d, wa, wb, wc, wd), Channel(2, a, b, c, d, wa, wb, wc, wd));
    }

    float Channel(int channel, int a, int b, int c, int d, int wa, int wb, int wc, int wd)
    {
        var value = (a < 0 ? 0 : pixels[a + channel]) * wa + (b < 0 ? 0 : pixels[b + channel]) * wb
            + (c < 0 ? 0 : pixels[c + channel]) * wc + (d < 0 ? 0 : pixels[d + channel]) * wd;
        return ((value + 512) >> 10) / 255f;
    }

    int Offset(int x, int y, bool replicate)
    {
        if (replicate) { x = Math.Clamp(x, 0, width - 1); y = Math.Clamp(y, 0, height - 1); }
        return x < 0 || y < 0 || x >= width || y >= height ? -1 : (y * width + x) * 4;
    }
}
