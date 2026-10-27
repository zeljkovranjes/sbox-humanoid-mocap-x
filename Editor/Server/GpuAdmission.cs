using System;
using System.Collections.Generic;
using HumanoidMocapX;

namespace HumanoidMocapHost;

/// <summary>Reserves each running job's peak before another starts. Graphics readings
/// are supplied by the editor thread; queue checks use only the cached numbers.</summary>
internal sealed class GpuAdmission(int lanes)
{
    const ulong GiB = 1UL << 30, MiB = 1UL << 20;
    const ulong Headroom = 512 * MiB;
    readonly object gate = new();
    ulong budget, used, physical, idleUsed;
    DateTime updated;

    internal void Update(ulong budgetBytes, ulong usedBytes, ulong physicalBytes)
    {
        lock (gate)
        {
            budget = budgetBytes > 0 ? budgetBytes : physicalBytes;
            used = usedBytes; physical = physicalBytes; updated = DateTime.UtcNow;
        }
    }

    /// <summary>Returns false while current memory pressure or a stale editor reading
    /// prevents a start. Throws a descriptive limit error for settings beyond this GPU.</summary>
    internal bool CanStart(HostedCaptureOptions next, IReadOnlyList<HostedCaptureOptions> running)
    {
        var required = RequiredBytes(next, lanes);
        lock (gate)
        {
            if (DateTime.UtcNow - updated > TimeSpan.FromSeconds(5)) return false;
            if (budget == 0) throw new LimitExceededException("The GPU reports no memory budget. Restart the host after the graphics device is ready.");
            if (required + Headroom > (physical > 0 ? physical : budget))
                throw new LimitExceededException("This capture exceeds the GPU's memory capacity. Choose a shorter range or a larger camera interval.");
            if (running.Count == 0) idleUsed = used;
            ulong reserved = idleUsed;
            foreach (var active in running) reserved = checked(reserved + RequiredBytes(active, lanes));
            // Count the peak reservation or observed use, whichever is larger; adding both
            // would charge an active job twice and unnecessarily prevent concurrency.
            var occupied = Math.Max(used, reserved);
            return occupied <= budget && required + Headroom <= budget - occupied;
        }
    }

    internal static ulong RequiredBytes(HostedCaptureOptions options, int lanes)
    {
        options.Validate();
        if (lanes is < 1 or > CaptureLanes.Maximum) throw new ArgumentOutOfRangeException(nameof(lanes));
        if (options.End is not { } end) throw new ArgumentException("GPU admission needs the inspected video end.");
        var count = Math.Ceiling((end - options.Start) * 30 - 0.0000001);
        if (count is < 1 or > CaptureOptions.MaximumFrames)
            throw new LimitExceededException("Choose a capture range of at most 5,000 frames (166.66 seconds at 30 fps).");
        var frames = (int)count; var samples = FlowCamera.SampleCount(frames, options.CameraInterval);
        // Worst-case 512-square Omega: 1,041 tokens x 1,024 channels. Its widest
        // buffer is four floats per feature; the engine has a 32-bit byte-size limit.
        if ((ulong)samples * 1041 * 1024 * 4 * sizeof(float) > uint.MaxValue)
            throw new LimitExceededException("Camera tensors exceed the engine buffer limit. Choose a shorter range or a larger camera interval.");
        // Checkpoint sizes bound resident weights. Rounded workspace bounds include
        // the image encoder/camera head (Omega) and refiner/latent buffers (Flow).
        var camera = 5 * GiB + 64 * MiB + (ulong)samples * 96 * MiB;
        var features = 4 * GiB + (ulong)(lanes - 1) * 512 * MiB;
        var generation = 2 * GiB + (ulong)frames * 512 * 1024;
        return Math.Max(camera, Math.Max(features, generation));
    }
}
