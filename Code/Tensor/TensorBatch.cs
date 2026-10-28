using System;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;

namespace HumanoidMocapX.TensorInference;

/// <summary>Records tensor work in an engine graphics context and reads back its result.
/// Create, run and dispose on the main thread. One batch may be in flight per instance.</summary>
public sealed class TensorBatch : IDisposable
{
    readonly Scene scene;
    readonly CameraComponent camera;
    readonly SceneCustomObject renderObject;
    readonly Texture target;
    readonly Action<ReadOnlySpan<float>> receive;
    TaskCompletionSource<bool> completion;
    Action commands;
    GpuBuffer<float> result;
    Memory<float> destination;
    Exception recordingFailure;
    int recorded, readScheduled;
    bool disposed;

    /// <summary>Creates an isolated offscreen context. Throws when graphics is unavailable
    /// or engine resources cannot be created.</summary>
    public TensorBatch()
    {
        ThreadSafe.AssertIsMainThread();
        if (!Graphics.IsAvailable) throw new InvalidOperationException("Tensor inference requires an initialized graphics device.");
        scene = Scene.CreateEditorScene();
        try
        {
            camera = scene.CreateObject().Components.Create<CameraComponent>();
            camera.IsMainCamera = false; camera.EnablePostProcessing = false;
            renderObject = new SceneCustomObject(scene.SceneWorld);
            renderObject.Flags.IsOpaque = false; renderObject.Flags.IsTranslucent = true; renderObject.Flags.CastShadows = false;
            renderObject.RenderOverride = Record;
            // The public GPU readback API needs a render scope; standalone scopes are internal.
            target = Texture.CreateRenderTarget().WithSize(8, 8).Create();
            receive = Receive;
        }
        catch { scene.Destroy(); throw; }
    }

    /// <summary>Runs commands on the render thread, then copies the result prefix into
    /// destination. Keep buffers alive until completion. Cancellation after submission
    /// waits for readback before throwing; recording/readback failures also propagate.</summary>
    public async Task RunAsync(Action commands, GpuBuffer<float> result, Memory<float> destination,
        CancellationToken cancellation = default)
    {
        ThreadSafe.AssertIsMainThread();
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(commands);
        if (completion is not null) throw new InvalidOperationException("Await the current tensor batch before starting another.");
        if (result is null || !result.IsValid || destination.Length < 1 || destination.Length > result.ElementCount)
            throw new ArgumentException("Readback requires a valid buffer and a nonempty destination within its bounds.");
        cancellation.ThrowIfCancellationRequested();
        this.commands = commands; this.result = result; this.destination = destination;
        recordingFailure = null; recorded = 0; readScheduled = 0;
        completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Exception failure = null;
        try
        {
            if (!camera.RenderToTexture(target)) throw new InvalidOperationException("The engine could not submit the tensor batch view.");
        }
        catch (Exception error) { failure = error; }
        if (Interlocked.CompareExchange(ref readScheduled, 0, 0) != 0)
        {
            try { await completion.Task; }
            catch (Exception error) { failure ??= error; }
        }
        await GameTask.MainThread();
        completion = null; this.commands = null; this.result = null; this.destination = default;
        if (failure is not null) throw new InvalidOperationException("Tensor batch submission or readback failed.", failure);
        if (recordingFailure is OperationCanceledException && cancellation.IsCancellationRequested) cancellation.ThrowIfCancellationRequested();
        if (recordingFailure is not null) throw new InvalidOperationException("Tensor batch recording failed.", recordingFailure);
        if (Interlocked.CompareExchange(ref recorded, 0, 0) == 0)
            throw new InvalidOperationException("The engine did not execute the tensor batch render callback.");
        cancellation.ThrowIfCancellationRequested();
    }

    void Record(SceneObject unused)
    {
        if (Interlocked.Exchange(ref recorded, 1) != 0) return;
        try { commands(); }
        catch (Exception error) { recordingFailure = error; }
        try
        {
            // Even failed recording must finish queued GPU work before its buffers are disposed.
            result.GetDataAsync(receive, 0, destination.Length);
            Interlocked.Exchange(ref readScheduled, 1);
        }
        catch (Exception error) { recordingFailure ??= error; }
    }

    void Receive(ReadOnlySpan<float> values)
    {
        try { values.CopyTo(destination.Span); completion.TrySetResult(true); }
        catch (Exception error) { completion.TrySetException(error); }
    }

    /// <summary>Releases the isolated scene and target. Rejects disposal while a batch
    /// is in flight; await its completion even after cancellation.</summary>
    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        if (disposed) return;
        if (completion is not null) throw new InvalidOperationException("Await GPU readback before disposing the tensor batch.");
        disposed = true; renderObject.RenderOverride = null;
        scene.Destroy(); target.Dispose();
    }
}
