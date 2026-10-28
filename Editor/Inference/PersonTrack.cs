using System;

namespace HumanoidMocapX.Inference;

/// <summary>Tracks the largest person, from the first frame they appear in, through
/// overlapping detections until they are lost. Rejects ambiguous associations instead of
/// silently changing people.</summary>
internal sealed class PersonTrack
{
    // shortcut: overlap tracking needs a clearly separated subject; add appearance tracking for crossings or long occlusions.
    const int MaximumGap = 15;
    const float MinimumOverlap = .2f, AmbiguityRatio = .9f;
    readonly float[] boxes;
    YoloBox previous;
    int frame, first = -1, lastMatched = -1;
    bool ended;

    internal PersonTrack(int frames)
    {
        if (frames < 1 || frames > CaptureOptions.MaximumFrames) throw new ArgumentOutOfRangeException(nameof(frames));
        boxes = new float[frames * 4];
    }

    /// <summary>Consumes all person detections for the next frame. Tracking starts at the
    /// first frame with a person and ends after a gap longer than half a second; short gaps
    /// are linearly interpolated. An uncertain identity fails rather than exporting invented
    /// motion.</summary>
    internal void Add(ReadOnlySpan<YoloBox> detections)
    {
        if (frame >= boxes.Length / 4) throw new InvalidOperationException("All capture frames already have detections.");
        var best = -1; double score = 0, second = 0;
        for (var i = 0; i < detections.Length; i++)
        {
            var box = detections[i]; Validate(box);
            var current = first < 0 ? ((double)box.Right - box.Left) * ((double)box.Bottom - box.Top) : Overlap(previous, box);
            if (current > score) { second = score; score = current; best = i; }
            else second = Math.Max(second, current);
        }
        // Frames before the person appears and after they are lost are left out of the capture.
        if (ended || first < 0 && best < 0) { frame++; return; }
        if (first >= 0 && score >= MinimumOverlap && second >= score * AmbiguityRatio)
            throw new InvalidOperationException($"Person tracking is ambiguous at frame {frame}. Use a clip containing one clearly separated person.");
        if (first >= 0 && (best < 0 || score < MinimumOverlap))
        {
            if (frame - lastMatched > MaximumGap) ended = true;
            frame++; return;
        }
        if (first < 0) first = frame;
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

    /// <summary>The frames from the person's first to last detection, with their boxes.
    /// Fails for an incomplete timeline or a clip without any person.</summary>
    internal PersonSpan Complete()
    {
        if (frame != boxes.Length / 4) throw new InvalidOperationException("Person tracking did not receive every frame.");
        if (first < 0) throw new InvalidOperationException("No person was detected in the selected clip.");
        var count = lastMatched - first + 1;
        return new PersonSpan(boxes.AsSpan(first * 4, count * 4).ToArray(), first, count);
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
