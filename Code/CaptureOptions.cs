using System;

namespace HumanoidMocapX;

/// <summary>FlowHMR settings selected in the editor.</summary>
public sealed class CaptureOptions
{
    /// <summary>Maximum frames supported by the released model's position table.</summary>
    public const int MaximumFrames = 5000;

    public FlowModel Model { get; set; } = FlowModel.Latest;
    public int CameraInterval { get; set; } = 10;
    /// <summary>Frames processed at once by local capture: <see cref="CaptureLanes.Auto"/> or a
    /// fixed count, which is still capped to what the GPU's memory allows.</summary>
    public int Lanes { get; set; } = CaptureLanes.Auto;

    /// <summary>Rejects unsupported models, camera sampling intervals and lane counts.</summary>
    public void Validate()
    {
        if (Model is not (FlowModel.Base or FlowModel.Latest))
            throw new ArgumentOutOfRangeException(nameof(Model));
        ValidateCameraInterval(CameraInterval);
        CaptureLanes.Validate(Lanes);
    }

    internal static void ValidateCameraInterval(int interval)
    {
        if (interval is not (1 or 10 or 30))
            throw new ArgumentOutOfRangeException(nameof(CameraInterval), "Camera interval must be 1, 10 or 30 frames.");
    }
}

/// <summary>The two released FlowHMR checkpoints.</summary>
public enum FlowModel { Base, Latest }
