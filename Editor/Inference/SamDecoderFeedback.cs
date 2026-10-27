using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Connects real body/hand predictions to the decoder's five intermediate
/// updates and final prediction. Reuse sequentially for one crop at a time.</summary>
internal sealed class SamDecoderFeedback
{
    readonly SamDecoderShape shape;
    readonly SamPoseHead pose;
    readonly SamPredictionHead camera;
    readonly SamKeypointFeedback keypoints;
    readonly float[] cameraParameters = new float[3], translation = new float[3];
    readonly float[] cropped = new float[SamKeypoints.Count * 2], depths = new float[SamKeypoints.Count];
    ReadOnlyMemory<float> image;
    SamProjection projection;
    internal SamPoseHead Pose => pose;
    internal ReadOnlySpan<float> CameraParameters => cameraParameters;
    internal ReadOnlySpan<float> CameraTranslation => translation;

    internal SamDecoderFeedback(SamDecoderShape shape, SamPoseHead pose, SamPredictionHead camera, SamKeypointFeedback keypoints)
    {
        ArgumentNullException.ThrowIfNull(pose); ArgumentNullException.ThrowIfNull(camera); ArgumentNullException.ThrowIfNull(keypoints);
        if (camera.Parameters != 3 || shape.Tokens < 145) throw new ArgumentException("SAM feedback requires a camera head and complete keypoint tokens.");
        this.shape = shape; this.pose = pose; this.camera = camera; this.keypoints = keypoints;
    }

    /// <summary>Binds this crop's projection and conditioned image. The image must
    /// remain unchanged until decoding finishes. No image copy is made.</summary>
    internal void Prepare(SamProjection projection, ReadOnlyMemory<float> image)
    {
        if (image.Length != shape.ImageElements) throw new ArgumentException("SAM feedback image dimensions do not match the decoder.");
        SamDecoder.Finite(image.Span); this.image = image; this.projection = projection;
    }

    internal void Update(int layer, ReadOnlySpan<float> normalized, Span<float> tokens, Span<float> positions, CancellationToken cancellation)
    {
        if (layer < 0 || layer >= SamDecoderShape.Layers - 1) throw new ArgumentOutOfRangeException(nameof(layer));
        Finish(normalized, cancellation);
        keypoints.Run(image.Span, cropped, depths, pose.Keypoints, tokens, positions, cancellation);
    }

    /// <summary>Evaluates the final normalized tokens without applying another
    /// feedback update. Missing crop preparation and incompatible tensors fail.</summary>
    internal void Finish(ReadOnlySpan<float> normalized, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (image.Length != shape.ImageElements || normalized.Length != shape.TokenElements)
            throw new ArgumentException("Prepare the SAM feedback crop and supply complete normalized decoder tokens.");
        var token = normalized.Slice(0, shape.Width); pose.Run(token, cancellation); camera.Run(token, cameraParameters, cancellation);
        projection.Run(pose.Keypoints, cameraParameters, cropped, depths, translation);
    }
}
