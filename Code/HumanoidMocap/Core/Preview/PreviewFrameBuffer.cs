using System;

namespace HumanoidMocapX.Core.Preview;

/// <summary>Owns a bounded RGBA preview frame. Reuses storage until the preview size
/// changes; callers must synchronize concurrent readers and writers.</summary>
public sealed class PreviewFrameBuffer
{
    const int MaximumWidth = 960, MaximumHeight = 540, PixelBytes = 4;
    byte[] pixels = Array.Empty<byte>();
    /// <summary>Width of the owned preview in pixels; zero before the first frame.</summary>
    public int Width { get; private set; }
    /// <summary>Height of the owned preview in pixels; zero before the first frame.</summary>
    public int Height { get; private set; }
    /// <summary>Increases after each successful copy so consumers can skip unchanged frames.</summary>
    public long Version { get; private set; }
    /// <summary>Borrowed pixels, valid until the next Copy call.</summary>
    public ReadOnlySpan<byte> Pixels => pixels;

    /// <summary>Copies a packed RGBA frame, reducing it to fit 960×540 without changing
    /// aspect ratio. Rejects nonpositive dimensions and an incorrect byte count.</summary>
    public void Copy(ReadOnlySpan<byte> source, int width, int height)
    {
        if (width < 1 || height < 1 || (long)width * height * PixelBytes != source.Length)
            throw new ArgumentException("Video preview requires a complete packed RGBA frame.");
        var scale = Math.Min(1, Math.Min((double)MaximumWidth / width, (double)MaximumHeight / height));
        var targetWidth = Math.Max(1, (int)(width * scale));
        var targetHeight = Math.Max(1, (int)(height * scale));
        var length = targetWidth * targetHeight * PixelBytes;
        if (pixels.Length != length) pixels = new byte[length];
        if (targetWidth == width && targetHeight == height) source.CopyTo(pixels);
        else
            for (var y = 0; y < targetHeight; y++)
                for (var x = 0; x < targetWidth; x++)
                {
                    var from = ((long)y * height / targetHeight * width + (long)x * width / targetWidth) * PixelBytes;
                    var to = (y * targetWidth + x) * PixelBytes;
                    source.Slice((int)from, PixelBytes).CopyTo(pixels.AsSpan(to, PixelBytes));
                }
        Width = targetWidth; Height = targetHeight; Version++;
    }
}
