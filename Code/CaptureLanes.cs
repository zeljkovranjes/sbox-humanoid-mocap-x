using System;

namespace HumanoidMocapX;

/// <summary>Chooses how many frames local capture processes at once. Each extra lane lets
/// one frame's CPU work overlap another's GPU work, and costs a little more video memory.</summary>
public static class CaptureLanes
{
    /// <summary>Largest lane count offered; requests are also capped by the GPU memory budget.</summary>
    public const int Maximum = 6;
    /// <summary>Requested value meaning "as many as this GPU allows".</summary>
    public const int Auto = 0;
    // Measured: shared SAM weights plus one lane used ~3.2 GB and each extra lane ~0.15 GB,
    // so these reserves leave headroom for the editor and other applications.
    const ulong FirstLaneBytes = 4UL << 30, ExtraLaneBytes = 512UL << 20;

    /// <summary>Lanes this GPU can safely run, from the OS video memory budget and the
    /// memory already in use. An unknown budget (0) allows a single lane.</summary>
    public static int Capacity(ulong budgetBytes, ulong usedBytes)
    {
        if (budgetBytes <= usedBytes) return 1;
        var free = budgetBytes - usedBytes;
        if (free <= FirstLaneBytes) return 1;
        return (int)Math.Min(Maximum, 1 + (free - FirstLaneBytes) / ExtraLaneBytes);
    }

    /// <summary>Applies the user's choice: Auto takes the capacity; a fixed count is capped
    /// to it. Rejects requests outside Auto or 1 to <see cref="Maximum"/>.</summary>
    public static int Resolve(int requested, ulong budgetBytes, ulong usedBytes)
    {
        Validate(requested);
        var capacity = Capacity(budgetBytes, usedBytes);
        return requested == Auto ? capacity : Math.Min(requested, capacity);
    }

    internal static void Validate(int requested)
    {
        if (requested is < Auto or > Maximum)
            throw new ArgumentOutOfRangeException(nameof(requested), $"Lanes must be Auto (0) or 1 to {Maximum}.");
    }
}
