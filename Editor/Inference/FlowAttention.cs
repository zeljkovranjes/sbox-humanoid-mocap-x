// C# adaptation of FlowHMR's narrowband_v2m attention. See THIRD_PARTY_NOTICES.md.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Batch-one attention for motion, image and pose transformers.
/// Buffers belong to the instance; concurrent calls are unsupported.</summary>
internal sealed class FlowAttention
{
    const int MotionWindow = 60;
    const int VideoWindow = 3;
    readonly int tokens, keyTokens, width, heads, headWidth;
    readonly float scale;
    readonly float[] scores;
    readonly (int Start, int Count, int VideoStart, int VideoCount)[] ranges;

    /// <param name="frames">Frames per stream, without padding.</param>
    /// <param name="jointStreams">True for concatenated motion/video; false for the full context-refiner attention.</param>
    internal FlowAttention(int frames, int width, int heads, bool jointStreams)
    {
        if (frames < 1 || frames > FlowMotion.MaximumFrames || width < 1 || heads < 1 || width % heads != 0)
            throw new ArgumentException("Invalid FlowHMR attention frame count or head dimensions.");
        tokens = checked(frames * (jointStreams ? 2 : 1));
        keyTokens = tokens;
        this.width = width; this.heads = heads; headWidth = width / heads;
        scale = 1f / MathF.Sqrt(headWidth);
        scores = new float[jointStreams ? Math.Min(frames, MotionWindow * 2 + 1) + Math.Min(frames, VideoWindow * 2 + 1) : frames];
        ranges = new (int, int, int, int)[tokens];
        for (var query = 0; query < tokens; query++)
        {
            if (!jointStreams) { ranges[query] = (0, frames, 0, 0); continue; }
            var isMotion = query < frames;
            var frame = query % frames;
            var motionStart = Math.Max(0, frame - MotionWindow);
            var motionEnd = Math.Min(frames, frame + MotionWindow + 1);
            var videoStart = Math.Max(0, frame - VideoWindow);
            var videoEnd = Math.Min(frames, frame + VideoWindow + 1);
            ranges[query] = (motionStart, isMotion ? motionEnd - motionStart : 0,
                frames + videoStart, videoEnd - videoStart);
        }
    }

    /// <summary>Full cross-attention with independently sized query and context sequences.</summary>
    internal FlowAttention(int queries, int keys, int width, int heads)
    {
        if (queries < 1 || keys < 1 || queries > FlowMotion.MaximumFrames || keys > FlowMotion.MaximumFrames
            || width < 1 || heads < 1 || width % heads != 0 || (long)Math.Max(queries, keys) * width > int.MaxValue)
            throw new ArgumentException("Invalid cross-attention token count or head dimensions.");
        tokens = queries; keyTokens = keys; this.width = width; this.heads = heads; headWidth = width / heads;
        scale = 1f / MathF.Sqrt(headWidth); scores = new float[keys]; ranges = new (int, int, int, int)[queries];
        for (var query = 0; query < queries; query++) ranges[query] = (0, keys, 0, 0);
    }

    /// <summary>Copies two (start,count) ranges per query for the GPU attention kernel.</summary>
    internal void CopyKeyRanges(Span<uint> output)
    {
        if (output.Length != (long)tokens * 4) throw new ArgumentException("Attention key ranges need four integers per token.");
        for (var token = 0; token < tokens; token++)
        {
            var range = ranges[token];
            output[token * 4] = (uint)range.Start; output[token * 4 + 1] = (uint)range.Count;
            output[token * 4 + 2] = (uint)range.VideoStart; output[token * 4 + 3] = (uint)range.VideoCount;
        }
    }

    /// <summary>Accepts [token,width] Q/K/V after normalization and position encoding.
    /// Cross-attention uses its context count for K/V. Writes [query,width] output,
    /// rejecting shape mismatches and input/output aliasing.</summary>
    internal void Run(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> value,
        Span<float> output, CancellationToken cancellation = default)
    {
        if (query.Length != (long)tokens * width || key.Length != (long)keyTokens * width || value.Length != key.Length || output.Length != query.Length)
            throw new ArgumentException("Attention tensors do not match their query/context dimensions.");
        if (query.Overlaps(output) || key.Overlaps(output) || value.Overlaps(output))
            throw new ArgumentException("Attention output must not overlap its inputs.");
        for (var token = 0; token < tokens; token++)
        {
            cancellation.ThrowIfCancellationRequested();
            var range = ranges[token];
            var count = range.Count + range.VideoCount;
            for (var head = 0; head < heads; head++)
            {
                var q = query.Slice(token * width + head * headWidth, headWidth);
                for (var i = 0; i < count; i++)
                {
                    var source = i < range.Count ? range.Start + i : range.VideoStart + i - range.Count;
                    scores[i] = NeuralOps.Dot(q, key.Slice(source * width + head * headWidth, headWidth)) * scale;
                }
                NeuralOps.SoftmaxInPlace(scores.AsSpan(0, count));
                var destination = output.Slice(token * width + head * headWidth, headWidth);
                destination.Clear();
                for (var i = 0; i < count; i++)
                {
                    var source = i < range.Count ? range.Start + i : range.VideoStart + i - range.Count;
                    var v = value.Slice(source * width + head * headWidth, headWidth);
                    var probability = scores[i];
                    for (var channel = 0; channel < headWidth; channel++) destination[channel] += probability * v[channel];
                }
            }
        }
    }
}
