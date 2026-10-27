using System;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;

namespace HumanoidMocapX.Inference;

/// <summary>Owns one body/hand decoder, learned initialization and complete pose
/// feedback. Reuses all tensor storage. Construct, run and dispose on the main thread.</summary>
internal sealed class SamGpuPoseDecoder : IDisposable
{
    readonly SamGpuDecoder decoder;
    readonly SamInitialTokens initial;
    readonly SamDecoderFeedback feedback;
    readonly float[] tokens, positions, decoded;
    readonly float[] projected = new float[SamKeypoints.Count * 2], depths = new float[SamKeypoints.Count], translation = new float[3];
    readonly Action prepare, finish;
    readonly SamPoseFeedback update;
    SamProjection projection;
    SamCrop crop;
    ReadOnlyMemory<float> image;
    CancellationToken cancellation;
    bool running, disposed;
    internal ReadOnlySpan<float> Tokens => decoded;
    internal ReadOnlySpan<float> ImageKeypoints => projected;

    internal SamGpuPoseDecoder(FlowWeights weights, SamDecoderShape shape, SamInitialTokens initial, SamDecoderFeedback feedback, bool hand,
        FlowGpuWeights shared = null)
    {
        ThreadSafe.AssertIsMainThread(); ArgumentNullException.ThrowIfNull(initial); ArgumentNullException.ThrowIfNull(feedback);
        this.initial = initial; this.feedback = feedback;
        tokens = new float[shape.TokenElements]; positions = new float[tokens.Length]; decoded = new float[tokens.Length];
        prepare = Prepare; finish = Finish; update = feedback.Update;
        decoder = new(weights, shape, hand, shared);
    }

    /// <summary>Updates final tokens and image keypoints. The caller supplies a crop,
    /// matching camera projection and conditioned image context. Outputs expire on
    /// the next call and must be discarded after failure; input stays borrowed until completion.</summary>
    internal async Task RunAsync(SamCrop crop, SamProjection projection, ReadOnlyMemory<float> image, ReadOnlyMemory<float> imagePositions,
        CancellationToken token = default)
    {
        ThreadSafe.AssertIsMainThread(); ObjectDisposedException.ThrowIf(disposed, this);
        if (running) throw new InvalidOperationException("Await the current SAM pose before starting another.");
        token.ThrowIfCancellationRequested(); running = true;
        this.crop = crop; this.projection = projection; this.image = image; cancellation = token;
        try
        {
            await Task.Run(prepare, token); await GameTask.MainThread();
            await decoder.RunAsync(tokens, image, positions, imagePositions, decoded, update, token);
            await Task.Run(finish, token);
        }
        finally { await GameTask.MainThread(); running = false; this.image = default; cancellation = default; }
    }

    void Prepare()
    { initial.Run(projection, tokens, positions, cancellation); feedback.Prepare(projection, image); }

    void Finish()
    {
        feedback.Finish(decoded, cancellation);
        projection.Run(feedback.Pose.Keypoints, feedback.CameraParameters, projected, depths, translation);
        crop.ToImage(projected);
    }

    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        if (disposed) return;
        if (running) throw new InvalidOperationException("Await SAM pose work before disposal.");
        disposed = true; decoder.Dispose();
    }
}
