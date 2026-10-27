using HumanoidMocapX;

static class CaptureLaneTests
{
    const ulong Gib = 1UL << 30, Mib = 1UL << 20;

    internal static void Run()
    {
        // Unknown or exhausted budgets still allow one lane.
        MotionTests.Equal(1, CaptureLanes.Capacity(0, 0));
        MotionTests.Equal(1, CaptureLanes.Capacity(8 * Gib, 9 * Gib));
        MotionTests.Equal(1, CaptureLanes.Capacity(4 * Gib, 0));
        // 4 GiB for the first lane, then 512 MiB per extra lane, up to the maximum.
        MotionTests.Equal(2, CaptureLanes.Capacity(4 * Gib + 512 * Mib, 0));
        MotionTests.Equal(3, CaptureLanes.Capacity(16 * Gib, 11 * Gib));
        MotionTests.Equal(CaptureLanes.Maximum, CaptureLanes.Capacity(16 * Gib, 2 * Gib));
        MotionTests.Equal(CaptureLanes.Maximum, CaptureLanes.Resolve(CaptureLanes.Auto, 24 * Gib, 0));
        MotionTests.Equal(2, CaptureLanes.Resolve(2, 24 * Gib, 0));
        MotionTests.Equal(1, CaptureLanes.Resolve(4, 4 * Gib, 0));
        MotionTests.Throws<ArgumentOutOfRangeException>(() => CaptureLanes.Resolve(-1, 24 * Gib, 0));
        MotionTests.Throws<ArgumentOutOfRangeException>(() => CaptureLanes.Resolve(CaptureLanes.Maximum + 1, 24 * Gib, 0));
        MotionTests.Throws<ArgumentOutOfRangeException>(() => new CaptureOptions { Lanes = CaptureLanes.Maximum + 1 }.Validate());
        new CaptureOptions { Lanes = 3 }.Validate(); new CaptureOptions { Lanes = CaptureLanes.Maximum }.Validate();
        Console.WriteLine("PASS: capture lanes from video memory, Auto, capped manual choices and validation");
    }
}
