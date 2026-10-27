// C# adaptation of YOLOX decoding and FlowHMR's person filtering. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

internal readonly record struct YoloBox(float Left, float Top, float Right, float Bottom, float Score);

/// <summary>Decodes raw YOLOX-L heads and applies class-agnostic suppression before
/// selecting people, as required by FlowHMR. Reuses candidate storage across frames.</summary>
internal sealed class YoloDetection
{
    readonly Candidate[] candidates = new Candidate[YoloNetwork.Predictions];
    readonly bool[] suppressed = new bool[YoloNetwork.Predictions];
    const float ConfidenceThreshold = .5f, OverlapThreshold = .3f;

    readonly record struct Candidate(YoloBox Box, int Class, int Anchor) : IComparable<Candidate>
    {
        public int CompareTo(Candidate other)
        {
            var order = other.Box.Score.CompareTo(Box.Score);
            return order == 0 ? Anchor.CompareTo(other.Anchor) : order;
        }
    }

    /// <summary>Writes original-image pixel boxes to output (capacity at least 8400),
    /// returning the count. No detections returns zero; no fallback person is invented.
    /// Rejects invalid shapes and non-finite predictions. Instances are not thread safe.</summary>
    internal int Decode(ReadOnlySpan<float> logits, int width, int height, Span<YoloBox> output,
        CancellationToken cancellation = default)
    {
        if (width < 1 || height < 1 || logits.Length != YoloNetwork.Predictions * YoloNetwork.PredictionChannels
            || output.Length < YoloNetwork.Predictions)
            throw new ArgumentException("YOLOX decoding needs positive image dimensions, [8400,85] logits and capacity for 8400 boxes.");
        cancellation.ThrowIfCancellationRequested();
        var ratio = Math.Min(YoloNetwork.ImageSize / (double)width, YoloNetwork.ImageSize / (double)height);
        var count = 0; var anchor = 0;
        for (var stride = 8; stride <= 32; stride *= 2)
        {
            var size = YoloNetwork.ImageSize / stride;
            for (var y = 0; y < size; y++)
                for (var x = 0; x < size; x++, anchor++)
                {
                    if ((anchor & 255) == 0) cancellation.ThrowIfCancellationRequested();
                    var row = logits.Slice(anchor * YoloNetwork.PredictionChannels, YoloNetwork.PredictionChannels);
                    foreach (var value in row)
                        if (!float.IsFinite(value)) throw new ArithmeticException("YOLOX prediction contains non-finite values.");
                    var bestClass = 0; var bestScore = Sigmoid(row[5]);
                    for (var category = 1; category < YoloNetwork.Classes; category++)
                    {
                        var score = Sigmoid(row[5 + category]);
                        if (score > bestScore) { bestScore = score; bestClass = category; }
                    }
                    var confidence = Sigmoid(row[4]) * bestScore;
                    if (confidence < ConfidenceThreshold) continue;
                    var centerX = (row[0] + x) * stride; var centerY = (row[1] + y) * stride;
                    var boxWidth = MathF.Exp(row[2]) * stride; var boxHeight = MathF.Exp(row[3]) * stride;
                    var box = new YoloBox((float)((centerX - boxWidth / 2) / ratio), (float)((centerY - boxHeight / 2) / ratio),
                        (float)((centerX + boxWidth / 2) / ratio), (float)((centerY + boxHeight / 2) / ratio), confidence);
                    if (!float.IsFinite(box.Left) || !float.IsFinite(box.Top) || !float.IsFinite(box.Right) || !float.IsFinite(box.Bottom))
                        throw new ArithmeticException("YOLOX box coordinates overflowed.");
                    if (box.Right > box.Left && box.Bottom > box.Top) candidates[count++] = new(box, bestClass, anchor);
                }
        }
        Array.Sort(candidates, 0, count); Array.Clear(suppressed, 0, count);
        var people = 0;
        for (var index = 0; index < count; index++)
        {
            cancellation.ThrowIfCancellationRequested();
            if (suppressed[index]) continue;
            var candidate = candidates[index];
            if (candidate.Class == 0) output[people++] = candidate.Box;
            for (var next = index + 1; next < count; next++)
                if (!suppressed[next] && Overlap(candidate.Box, candidates[next].Box) > OverlapThreshold) suppressed[next] = true;
        }
        return people;
    }

    static float Sigmoid(float value) => 1 / (1 + MathF.Exp(-value));

    static double Overlap(YoloBox first, YoloBox second)
    {
        var intersection = Math.Max(0, (double)Math.Min(first.Right, second.Right) - Math.Max(first.Left, second.Left))
            * Math.Max(0, (double)Math.Min(first.Bottom, second.Bottom) - Math.Max(first.Top, second.Top));
        var firstArea = ((double)first.Right - first.Left) * ((double)first.Bottom - first.Top);
        var secondArea = ((double)second.Right - second.Left) * ((double)second.Bottom - second.Top);
        return intersection / (firstArea + secondArea - intersection);
    }
}
