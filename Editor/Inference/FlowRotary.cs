using System;

namespace HumanoidMocapX.Inference;

/// <summary>Adjacent-pair rotary embedding with the HY-Motion backbone's base-10000
/// frequencies. Apply separately to motion and video so both start at position zero.</summary>
internal sealed class FlowRotary
{
    internal const int MaximumFrames = CaptureOptions.MaximumFrames;
    readonly int frames, width, headWidth;
    readonly float[] cosines, sines;

    internal FlowRotary(int frames, int width, int heads)
    {
        if (frames < 1 || frames > MaximumFrames || heads < 1 || width < 1 || width % heads != 0 || width / heads % 2 != 0)
            throw new ArgumentException("Rotary embedding needs 1–5000 frames and even, equally sized heads.");
        this.frames = frames; this.width = width; headWidth = width / heads;
        var pairs = headWidth / 2;
        cosines = new float[checked(frames * pairs)]; sines = new float[cosines.Length];
        for (var frame = 0; frame < frames; frame++)
            for (var pair = 0; pair < pairs; pair++)
            {
                var angle = frame / MathF.Pow(10000, 2f * pair / headWidth);
                cosines[frame * pairs + pair] = MathF.Cos(angle);
                sines[frame * pairs + pair] = MathF.Sin(angle);
            }
    }

    internal void Apply(Span<float> query, Span<float> key)
    {
        if (query.Length != (long)frames * width || key.Length != query.Length || query.Overlaps(key))
            throw new ArgumentException("Rotary embedding needs separate [frames,width] query and key tensors.");
        for (var frame = 0; frame < frames; frame++)
            for (var channel = 0; channel < width; channel += 2)
            {
                var angle = frame * (headWidth / 2) + channel % headWidth / 2;
                var at = frame * width + channel;
                Rotate(query, at, cosines[angle], sines[angle]);
                Rotate(key, at, cosines[angle], sines[angle]);
            }
    }

    /// <summary>Copies the same position table as interleaved cosine/sine pairs for GPU execution.</summary>
    internal void CopyFactors(Span<float> output)
    {
        if (output.Length != (long)frames * headWidth)
            throw new ArgumentException("Rotary factors need [frames,headWidth] elements.");
        for (var i = 0; i < cosines.Length; i++)
        { output[2 * i] = cosines[i]; output[2 * i + 1] = sines[i]; }
    }

    static void Rotate(Span<float> values, int offset, float cosine, float sine)
    {
        var real = values[offset]; var imaginary = values[offset + 1];
        values[offset] = real * cosine - imaginary * sine;
        values[offset + 1] = real * sine + imaginary * cosine;
    }
}
