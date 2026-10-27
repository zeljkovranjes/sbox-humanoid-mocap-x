using System;
using System.Threading;
using System.Threading.Tasks;

namespace HumanoidMocapX.Inference;

/// <summary>Random access to a fixed-size, 30 fps capture timeline. The owner keeps
/// the source alive until inference finishes; readers must reject incomplete frames.</summary>
internal interface ICaptureFrames
{
    int Width { get; }
    int Height { get; }
    int Count { get; }

    /// <summary>Fills the complete RGBA8 destination for the requested timeline
    /// index. Calls are sequential; failures and cancellation propagate.</summary>
    ValueTask ReadAsync(int frame, Memory<byte> rgba, CancellationToken cancellation);
}
