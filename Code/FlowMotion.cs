using System;
using System.Linq;

namespace HumanoidMocapX;

/// <summary>Numeric SMPL-H motion in metres and Y-up coordinates.</summary>
public sealed class FlowMotion
{
    public const int JointCount = 52;
    public const int MaximumFrames = 108000;
    public double Fps { get; set; } = 30;
    public double StartTime { get; set; }
    public string Video { get; set; } = "";
    public string Name { get; set; } = "capture";
    public FlowModel Model { get; set; } = FlowModel.Latest;
    public int[] Parents { get; set; } = Array.Empty<int>();
    public float[][] Rest { get; set; } = Array.Empty<float[]>();
    public float[][] Poses { get; set; } = Array.Empty<float[]>();
    public float[][] Translations { get; set; } = Array.Empty<float[]>();
    public float[][] Contacts { get; set; } = Array.Empty<float[]>();

    const int ContactCount = 4;
    static readonly int[] SmplParents = {
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
        20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
        21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50
    };

    /// <summary>The supported SMPL-H parent indices, with -1 for the pelvis.</summary>
    public static ReadOnlySpan<int> JointParents => SmplParents;

    /// <summary>Rejects malformed shapes, non-finite data and unsupported SMPL-H skeletons.</summary>
    public void Validate()
    {
        if (Model is not (FlowModel.Base or FlowModel.Latest)) throw new FormatException("Unknown FlowHMR model.");
        if (!double.IsFinite(Fps) || Fps != 30 || !double.IsFinite(StartTime) || StartTime < 0)
            throw new FormatException("FlowHMR requires 30 fps and a finite, non-negative start time.");
        if (Parents is null || !Parents.SequenceEqual(SmplParents))
            throw new FormatException("Expected the 52-joint SMPL-H hierarchy.");
        CheckRows(Rest, JointCount, 3, nameof(Rest));
        if (Poses is null || Poses.Length is < 1 or > MaximumFrames)
            throw new FormatException("FlowHMR motion is empty or exceeds the frame budget.");
        CheckRows(Poses, Poses.Length, JointCount * 3, nameof(Poses));
        CheckRows(Translations, Poses.Length, 3, nameof(Translations));
        if (Contacts is null) throw new FormatException("Foot-contact predictions cannot be null.");
        if (Contacts.Length == 0) return;
        CheckRows(Contacts, Poses.Length, ContactCount, nameof(Contacts));
        if (Contacts.Any(row => row.Any(p => p < 0 || p > 1)))
            throw new FormatException("Foot-contact probabilities must lie between zero and one.");
    }

    static void CheckRows(float[][] values, int rows, int columns, string name)
    {
        if (values is null || values.Length != rows || values.Any(row => row is null
            || row.Length != columns || row.Any(v => !float.IsFinite(v))))
            throw new FormatException($"Invalid {name}: expected {rows} rows of {columns} finite numbers.");
    }
}
