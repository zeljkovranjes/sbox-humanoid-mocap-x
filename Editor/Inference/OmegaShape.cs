using System;

namespace HumanoidMocapX.Inference;

/// <summary>Validated dimensions for camera-only Omega inference on resized RGB images.</summary>
internal readonly record struct OmegaShape
{
    internal const int PatchSize = 16, Depth = 24, ImagePrefix = 5, FramePrefix = 17;
    internal int Frames { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal int Channels { get; }
    internal int Heads { get; }
    internal int Patches => Height / PatchSize * (Width / PatchSize);
    internal int ImageTokens => Patches + ImagePrefix;
    internal int FrameTokens => Patches + FramePrefix;
    internal int Tokens => Frames * FrameTokens;
    internal int ImageElements => Height * Width * 3;
    internal int CameraFeatures => Frames * FramePrefix * Channels * 2;

    internal OmegaShape(int frames, int height, int width, int channels = 1024, int heads = 16)
    {
        if (frames < 1 || frames > CaptureOptions.MaximumFrames || height < PatchSize || width < PatchSize
            || height > 512 || width > 512 || height % PatchSize != 0 || width % PatchSize != 0
            || channels < 4 || channels > 1024 || heads < 1 || channels % heads != 0 || channels / heads % 4 != 0)
            throw new ArgumentException("Omega needs 1–5000 frames, patch-aligned images up to 512 pixels, and heads divisible by four.");
        var tokens = (long)frames * (height / PatchSize * (width / PatchSize) + FramePrefix);
        // The widest MLP buffer has four float32 values per feature; s&box caps a buffer at UInt32.MaxValue bytes.
        if (tokens * channels > uint.MaxValue / (4L * sizeof(float)))
            throw new ArgumentException("Omega working tensors exceed supported buffer dimensions; increase the camera interval.");
        Frames = frames; Height = height; Width = width; Channels = channels; Heads = heads;
    }

    internal static bool RegistersOnly(int layer) => layer is 2 or 6 or 9 or 14 or 20;
}
