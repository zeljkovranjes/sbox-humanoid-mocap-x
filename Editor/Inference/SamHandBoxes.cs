// C# adaptation of SAM 3D Body's hand detection and crop selection.
// Copyright (c) Meta Platforms, Inc. and affiliates. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;
using HumanoidMocap.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Regresses left/right square hand boxes from the body's two hand tokens.
/// Boxes use full-image coordinates and may extend beyond the image border.</summary>
internal sealed class SamHandBoxes
{
    readonly int width;
    readonly FlowLinear first, second, last;
    readonly float[] hidden, next;

    internal SamHandBoxes(FlowWeights weights, int width = 1024)
    {
        if (width < 1) throw new ArgumentOutOfRangeException(nameof(width));
        this.width = width;
        first = new(weights, "bbox_embed.layers.0", width, width);
        second = new(weights, "bbox_embed.layers.1", width, width);
        last = new(weights, "bbox_embed.layers.2", width, 4);
        hidden = new float[width * 2]; next = new float[hidden.Length];
    }

    /// <summary>Writes left XYXY followed by right XYXY from exactly two normalized
    /// hand tokens and the body's crop. Rejects invalid or overlapping inputs;
    /// discard output on failure or cancellation. Reuse sequentially.</summary>
    internal void Run(ReadOnlySpan<float> tokens, SamCrop bodyCrop, Span<float> boxes, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (tokens.Length != width * 2 || boxes.Length != 8 || tokens.Overlaps(boxes))
            throw new ArgumentException("SAM hand boxes require two tokens and eight separate output coordinates.");
        Span<float> affine = stackalloc float[6]; bodyCrop.CopyAffine(affine); SamDecoder.Finite(tokens);
        first.Run(tokens, 2, hidden, cancellation); NeuralOps.Relu(hidden);
        second.Run(hidden, 2, next, cancellation); NeuralOps.Relu(next);
        last.Run(next, 2, boxes, cancellation); SamDecoder.Finite(boxes);
        for (var i = 0; i < boxes.Length; i++) boxes[i] = 1 / (1 + MathF.Exp(-boxes[i])) * bodyCrop.Shape.Width;
        for (var hand = 0; hand < 2; hand++)
        {
            var at = hand * 4;
            var size = Math.Max(boxes[at + 2], boxes[at + 3]) / affine[0];
            var x = (boxes[at] - affine[2]) / affine[0]; var y = (boxes[at + 1] - affine[5]) / affine[0];
            boxes[at] = x - size * .5f; boxes[at + 1] = y - size * .5f;
            boxes[at + 2] = x + size * .5f; boxes[at + 3] = y + size * .5f;
            if (boxes[at + 2] <= boxes[at] || boxes[at + 3] <= boxes[at + 1])
                throw new ArgumentException("SAM predicted an empty hand box.");
        }
        SamDecoder.Finite(boxes);
    }

    /// <summary>Constructs one hand crop. Left-hand coordinates are reflected into
    /// the horizontally flipped input; sample that crop with flipHorizontal=true.</summary>
    internal static SamCrop Crop(ReadOnlySpan<float> box, DinoShape shape, int imageWidth, bool left)
    {
        if (box.Length != 4 || imageWidth < 1 || imageWidth >= short.MaxValue)
            throw new ArgumentException("SAM hand crop needs four coordinates and a valid image width.");
        return left ? new(imageWidth - box[2] - 1, box[1], imageWidth - box[0] - 1, box[3], shape, true)
            : new(box[0], box[1], box[2], box[3], shape, true);
    }
}
