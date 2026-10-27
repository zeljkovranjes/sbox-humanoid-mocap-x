using System;

namespace HumanoidMocapX.Inference;

internal readonly record struct SamDecoderShape
{
    internal const int Layers = 6;
    internal int Tokens { get; }
    internal int ImageTokens { get; }
    internal int Width { get; }
    internal int ContextWidth { get; }
    internal int Heads { get; }
    internal int HeadWidth { get; }
    internal int HiddenWidth { get; }
    internal int AttentionWidth => Heads * HeadWidth;
    internal int TokenElements => Tokens * Width;
    internal int ImageElements => ImageTokens * ContextWidth;

    internal SamDecoderShape(int tokens, int imageTokens, int width = 1024, int contextWidth = 1280,
        int heads = 8, int headWidth = 64, int hiddenWidth = 1024)
    {
        if (tokens < 1 || tokens > 512 || imageTokens < 1 || imageTokens > 1024 || width < 1 || width > 2048
            || contextWidth < 1 || contextWidth > 2048 || heads < 1 || heads > 32 || headWidth < 1 || headWidth > 128
            || hiddenWidth < 1 || hiddenWidth > 8192) throw new ArgumentException("Invalid SAM decoder dimensions.");
        Tokens = tokens; ImageTokens = imageTokens; Width = width; ContextWidth = contextWidth;
        Heads = heads; HeadWidth = headWidth; HiddenWidth = hiddenWidth;
    }
}
