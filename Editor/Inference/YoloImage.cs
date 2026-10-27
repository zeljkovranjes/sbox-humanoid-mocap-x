// C# adaptation of YOLOX's non-legacy validation transform. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

internal static class YoloImage
{
    /// <summary>Resizes RGBA8 into the upper-left of a 640×640 BGR HWC float image,
    /// padding with 114. Bilinear samples are rounded to bytes; alpha is ignored.
    /// Rejects malformed/overlapping buffers and unrepresentable aspect ratios.</summary>
    internal static void Prepare(ReadOnlySpan<byte> rgba, int width, int height, Span<float> output,
        CancellationToken cancellation = default)
    {
        if (width < 1 || height < 1 || (long)width * height > int.MaxValue / 4 || rgba.Length != (long)width * height * 4
            || output.Length != YoloNetwork.ImageSize * YoloNetwork.ImageSize * 3
            || rgba.Overlaps(System.Runtime.InteropServices.MemoryMarshal.AsBytes(output)))
            throw new ArgumentException("YOLOX preprocessing needs RGBA8 pixels and a separate [640,640,3] output.");
        cancellation.ThrowIfCancellationRequested();
        var ratio = Math.Min(YoloNetwork.ImageSize / (double)width, YoloNetwork.ImageSize / (double)height);
        var resizedWidth = (int)(width * ratio); var resizedHeight = (int)(height * ratio);
        if (resizedWidth < 1 || resizedHeight < 1) throw new ArgumentException("Image aspect ratio leaves no pixels after resizing.");
        output.Fill(114);
        for (var y = 0; y < resizedHeight; y++)
        {
            cancellation.ThrowIfCancellationRequested();
            var sourceY = Math.Clamp((y + .5) * height / resizedHeight - .5, 0, height - 1);
            var top = (int)sourceY; var bottom = Math.Min(top + 1, height - 1); var vertical = sourceY - top;
            for (var x = 0; x < resizedWidth; x++)
            {
                var sourceX = Math.Clamp((x + .5) * width / resizedWidth - .5, 0, width - 1);
                var left = (int)sourceX; var right = Math.Min(left + 1, width - 1); var horizontal = sourceX - left;
                for (var channel = 0; channel < 3; channel++)
                {
                    var a = rgba[(top * width + left) * 4 + 2 - channel]; var b = rgba[(top * width + right) * 4 + 2 - channel];
                    var c = rgba[(bottom * width + left) * 4 + 2 - channel]; var d = rgba[(bottom * width + right) * 4 + 2 - channel];
                    var upper = a + (b - a) * horizontal; var lower = c + (d - c) * horizontal;
                    output[(y * YoloNetwork.ImageSize + x) * 3 + channel] = (float)Math.Floor(upper + (lower - upper) * vertical + .5);
                }
            }
        }
    }
}
