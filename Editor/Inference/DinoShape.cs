using System;

namespace HumanoidMocapX.Inference;

/// <summary>Dimensions for the two DINO image encoders used by camera and body inference.</summary>
internal readonly record struct DinoShape
{
    internal const int PatchSize = 16, PrefixTokens = 5;
    internal int Height { get; }
    internal int Width { get; }
    internal int Channels { get; }
    internal int Heads { get; }
    internal bool IsSam { get; }
    internal int Depth => IsSam ? 32 : 24;
    internal string Name => IsSam ? "backbone.encoder" : "aggregator.patch_embed";
    internal int Patches => Height / PatchSize * (Width / PatchSize);
    internal int Tokens => Patches + PrefixTokens;
    internal int ImageElements => Height * Width * 3;

    DinoShape(int height, int width, int channels, int heads, bool isSam)
    {
        if (height < PatchSize || width < PatchSize || height > 512 || width > 512
            || height % PatchSize != 0 || width % PatchSize != 0 || channels < 4 || channels > 1280
            || heads < 1 || channels % heads != 0 || channels / heads % 4 != 0)
            throw new ArgumentException("DINO needs patch-aligned images up to 512 pixels and heads divisible by four.");
        Height = height; Width = width; Channels = channels; Heads = heads; IsSam = isSam;
    }

    internal static DinoShape Omega(OmegaShape shape) => new(shape.Height, shape.Width, shape.Channels, shape.Heads, false);
    internal static DinoShape Sam(int height = 512, int width = 512, int channels = 1280, int heads = 20)
        => new(height, width, channels, heads, true);
}
