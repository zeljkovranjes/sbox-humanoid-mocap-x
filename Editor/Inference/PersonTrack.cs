using System;

namespace HumanoidMocapX.Inference;

/// <summary>Tracks the initially largest person through overlapping detections.
/// Rejects ambiguous associations and long gaps instead of silently changing people.</summary>
internal sealed class PersonTrack
{
    // shortcut: overlap tracking needs a clearly separated subject; add appearance tracking for crossings or long occlusions.
    const int MaximumGap = 15;
    const float MinimumOverlap = .2f, AmbiguityRatio = .9f;
    readonly float[] boxes;
    YoloBox previous;
    int frame, lastMatched = -1;

    internal PersonTrack(int frames)
    {
        if (frames < 1 || frames > CaptureOptions.MaximumFrames) throw new ArgumentOutOfRangeException(nameof(frames));
        boxes = new float[frames * 4];
    }

    /// <summary>Consumes all person detections for the next frame. The first frame
    /// must contain a person. Short internal gaps are linearly interpolated; a trailing
    /// gap or an uncertain identity fails rather than exporting invented motion.</summary>
    internal void Add(ReadOnlySpan<YoloBox> detections)
    {
        if (frame >= boxes.Length / 4) throw new InvalidOperationException("All capture frames already have detections.");
        var best = -1; double score = 0, second = 0;
        for (var i = 0; i < detections.Length; i++)
        {
            var box = detections[i]; Validate(box);
            var current = frame == 0 ? ((double)box.Right - box.Left) * ((double)box.Bottom - box.Top) : Overlap(previous, box);
            if (current > score) { second = score; score = current; best = i; }
            else second = Math.Max(second, current);
        }
        if (frame == 0 && best < 0) throw new InvalidOperationException("No person was detected at the start of the selected clip.");
        if (frame > 0 && score >= MinimumOverlap && second >= score * AmbiguityRatio)
            throw new InvalidOperationException($"Person tracking is ambiguous at frame {frame}. Use a clip containing one clearly separated person.");
        if (frame > 0 && (best < 0 || score < MinimumOverlap))
        {
            if (frame - lastMatched > MaximumGap) throw new InvalidOperationException($"The selected person was lost at frame {frame}. Trim the clip to keep the person visible.");
            frame++; return;
        }
        var selected = detections[best]; var output = boxes.AsSpan(frame * 4, 4);
        output[0] = selected.Left; output[1] = selected.Top; output[2] = selected.Right; output[3] = selected.Bottom;
        if (lastMatched >= 0)
            for (var gap = lastMatched + 1; gap < frame; gap++)
            {
                var fraction = (gap - lastMatched) / (float)(frame - lastMatched);
                for (var axis = 0; axis < 4; axis++) boxes[gap * 4 + axis] = boxes[lastMatched * 4 + axis] * (1 - fraction) + output[axis] * fraction;
            }
        previous = selected; lastMatched = frame++;
    }

    /// <summary>Returns borrowed XYXY boxes after every frame has a reliable track.
    /// Fails for an incomplete timeline or a missing final detection.</summary>
    internal ReadOnlyMemory<float> Complete()
    {
        if (frame != boxes.Length / 4 || lastMatched != frame - 1)
            throw new InvalidOperationException("Person tracking did not cover the complete clip; trim any ending without a visible person.");
        return boxes;
    }

    static void Validate(YoloBox box)
    {
        if (!float.IsFinite(box.Left) || !float.IsFinite(box.Top) || !float.IsFinite(box.Right) || !float.IsFinite(box.Bottom)
            || !float.IsFinite(box.Score) || box.Score < 0 || box.Score > 1 || box.Right <= box.Left || box.Bottom <= box.Top)
            throw new ArgumentException("Person detection contains an invalid box or confidence.");
    }

    static double Overlap(YoloBox a, YoloBox b)
    {
        var area = Math.Max(0, (double)Math.Min(a.Right, b.Right) - Math.Max(a.Left, b.Left))
            * Math.Max(0, (double)Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Top, b.Top));
        return area / (((double)a.Right - a.Left) * ((double)a.Bottom - a.Top)
            + ((double)b.Right - b.Left) * ((double)b.Bottom - b.Top) - area);
    }
}
