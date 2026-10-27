using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;

namespace HumanoidMocapX.Inference;

/// <summary>Extracts FlowHMR's body, left-hand and right-hand features from one
/// tracked person. Owns the shared image encoder and both pose decoders.</summary>
internal sealed class SamGpuFeatures : IDisposable
{
    internal const int FeatureWidth = 1024, Elements = FeatureWidth * 3;
    readonly SamGpuImageEncoder encoder;
    readonly SamGpuPoseDecoder body, hand;
    readonly SamHandBoxes handBoxes;
    readonly DinoShape shape;
    readonly SamRays rayEncoder;
    readonly float[] pixels, rays, context, positions, affine = new float[6], camera = new float[9], boxes = new float[8];
    readonly float[] points = new float[3 * SamKeypoints.Count * 2];
    readonly Action prepare, detectHands;
    ReadOnlyMemory<byte> source;
    SamCrop crop;
    SamProjection projection;
    CancellationToken cancellation;
    int width, height, pass;
    bool running, disposed;
    internal ReadOnlySpan<float> HandBoxes => boxes;
    /// <summary>Original-image XY landmarks for body, mirrored-left and right
    /// passes. Hand passes use landmarks 21–41; other hand landmarks are masked.</summary>
    internal ReadOnlySpan<float> ImageKeypoints => points;

    internal SamGpuFeatures(SamGpuImageEncoder encoder, SamGpuPoseDecoder body, SamGpuPoseDecoder hand, SamHandBoxes handBoxes, DinoShape shape)
    {
        ThreadSafe.AssertIsMainThread(); ArgumentNullException.ThrowIfNull(encoder); ArgumentNullException.ThrowIfNull(body);
        ArgumentNullException.ThrowIfNull(hand); ArgumentNullException.ThrowIfNull(handBoxes);
        if (!shape.IsSam || shape.Width != shape.Height || shape.Channels != 1280) throw new ArgumentException("FlowHMR requires square SAM crops with 1280 image channels.");
        this.encoder = encoder; this.body = body; this.hand = hand; this.handBoxes = handBoxes; this.shape = shape;
        rayEncoder = new(shape); pixels = new float[shape.ImageElements]; rays = new float[rayEncoder.Elements];
        context = new float[shape.Patches * shape.Channels]; positions = new float[context.Length];
        prepare = Prepare; detectHands = DetectHands;
    }

    /// <summary>Writes 3072 channels in body/left/right order, matching FlowHMR's
    /// first-pass decoder hooks. Requires original RGBA8 pixels, XYXY person box and
    /// original-pixel intrinsics. Discard output on failure; keep inputs unchanged
    /// until completion. No download or external process is used.</summary>
    internal async Task RunAsync(ReadOnlyMemory<byte> rgba, int width, int height, ReadOnlyMemory<float> personBox,
        ReadOnlyMemory<float> intrinsics, Memory<float> output, CancellationToken token = default)
    {
        ThreadSafe.AssertIsMainThread(); ObjectDisposedException.ThrowIf(disposed, this);
        if (running) throw new InvalidOperationException("Await current SAM features before starting another frame.");
        token.ThrowIfCancellationRequested();
        if (width < 1 || height < 1 || width >= short.MaxValue || height >= short.MaxValue || rgba.Length != (long)width * height * 4
            || personBox.Length != 4 || intrinsics.Length != 9 || output.Length != Elements || personBox.Span.Overlaps(output.Span)
            || intrinsics.Span.Overlaps(output.Span) || rgba.Span.Overlaps(MemoryMarshal.AsBytes(output.Span)))
            throw new ArgumentException("SAM feature frame, box, intrinsics or output dimensions are invalid or overlap.");
        var box = personBox.Span;
        crop = new(box[0], box[1], box[2], box[3], shape); intrinsics.Span.CopyTo(camera);
        source = rgba; this.width = width; this.height = height; cancellation = token; running = true;
        try
        {
            for (pass = 0; pass < 3; pass++)
            {
                await Task.Run(prepare, token); await GameTask.MainThread();
                await encoder.RunAsync(pixels, rays, context, pass != 0, token); encoder.CopyPositions(positions, pass != 0);
                var decoder = pass == 0 ? body : hand;
                await decoder.RunAsync(crop, projection, context, positions, token);
                decoder.Tokens.Slice(0, FeatureWidth).CopyTo(output.Span.Slice(pass * FeatureWidth, FeatureWidth));
                var pointOutput = points.AsSpan(pass * SamKeypoints.Count * 2, SamKeypoints.Count * 2);
                decoder.ImageKeypoints.CopyTo(pointOutput);
                if (pass == 1) for (var i = 0; i < pointOutput.Length; i += 2) pointOutput[i] = width - pointOutput[i] - 1;
                if (pass == 0) { await Task.Run(detectHands, token); await GameTask.MainThread(); }
            }
        }
        finally { await GameTask.MainThread(); running = false; source = default; cancellation = default; }
    }

    void Prepare()
    {
        if (pass != 0) crop = SamHandBoxes.Crop(boxes.AsSpan((pass - 1) * 4, 4), shape, width, pass == 1);
        crop.CopyAffine(affine); projection = new(camera, affine, crop.CenterX, crop.CenterY, crop.Size, shape, pass != 0);
        crop.Prepare(source.Span, width, height, pixels, cancellation, pass == 1);
        // Upstream keeps the original camera intrinsics even for the mirrored left image.
        rayEncoder.Prepare(camera, affine, rays, cancellation);
    }

    void DetectHands() => handBoxes.Run(body.Tokens.Slice(SamInitialTokens.HandBoxStart * FeatureWidth, FeatureWidth * 2), crop, boxes, cancellation);

    public void Dispose()
    {
        ThreadSafe.AssertIsMainThread();
        if (disposed) return;
        if (running) throw new InvalidOperationException("Await SAM frame inference before disposal.");
        disposed = true; hand.Dispose(); body.Dispose(); encoder.Dispose();
    }
}

