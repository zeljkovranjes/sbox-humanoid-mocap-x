// Adapted from FlowHMR camera interpolation and VGGT-Omega pose decoding.
// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
// VGGT-Omega adaptations are subject to LICENSE-VGGT. See THIRD_PARTY_NOTICES.md.
using System;
using System.Numerics;
using System.Threading;

namespace HumanoidMocapX;

using Vector3 = System.Numerics.Vector3;

/// <summary>Converts VGGT-Omega predictions to FlowHMR rotation features and SAM
/// intrinsics. This decodes camera predictions; it does not estimate them from images.</summary>
public static class FlowCamera
{
    public const int PoseChannels = 9;
    public const int MatrixChannels = 9;

    /// <summary>Counts samples at the selected interval, including both endpoints.
    /// Rejects clips outside 1–5000 frames or intervals other than 1, 10 and 30.</summary>
    public static int SampleCount(int frames, int interval)
    {
        if (frames < 1 || frames > CaptureOptions.MaximumFrames)
            throw new ArgumentOutOfRangeException(nameof(frames), "Camera clips need 1–5000 frames.");
        CaptureOptions.ValidateCameraInterval(interval);
        return 1 + (frames - 1 + interval - 1) / interval;
    }

    /// <summary>Returns the clip-relative frame for a sample. Always includes the last
    /// frame, even when it falls between intervals. Rejects invalid sample indices.</summary>
    public static int SampleFrame(int sampleIndex, int frames, int interval)
    {
        var count = SampleCount(frames, interval);
        if ((uint)sampleIndex >= (uint)count) throw new ArgumentOutOfRangeException(nameof(sampleIndex));
        return Math.Min(sampleIndex * interval, frames - 1);
    }

    /// <summary>Decodes [samples,9] VGGT poses (translation, XYZW quaternion, vertical
    /// and horizontal field of view in radians). Writes row-major [frames,3,3] rotations
    /// relative to the first camera and intrinsics in original-frame pixels. All spans
    /// must be disjoint. Rejects invalid shapes, non-finite poses, zero quaternions and
    /// invalid field of view. Discard outputs after any failure or cancellation.</summary>
    public static void Decode(ReadOnlySpan<float> poses, int frames, int interval, int originalWidth,
        int originalHeight, Span<float> rotations, Span<float> intrinsics, CancellationToken cancellation = default)
    {
        var samples = SampleCount(frames, interval);
        if (originalWidth <= 0 || originalHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(originalWidth), "Original video dimensions must be positive.");
        if (poses.Length != samples * PoseChannels || rotations.Length != frames * MatrixChannels
            || intrinsics.Length != rotations.Length)
            throw new ArgumentException("Camera tensors need [samples,9] poses and two [frames,3,3] outputs.");
        if (poses.Overlaps(rotations) || poses.Overlaps(intrinsics) || rotations.Overlaps(intrinsics))
            throw new ArgumentException("Camera input and output buffers must not overlap.");
        cancellation.ThrowIfCancellationRequested();
        for (var sample = 0; sample < samples; sample++)
        {
            cancellation.ThrowIfCancellationRequested();
            _ = ReadPose(poses.Slice(sample * PoseChannels, PoseChannels), originalWidth, originalHeight, sample);
        }

        var left = ReadPose(poses[..PoseChannels], originalWidth, originalHeight, 0);
        var firstInverse = Quaternion.Conjugate(left.Rotation);
        var right = left;
        var nextSample = 0;
        var leftFrame = 0; var rightFrame = 0;
        for (var frame = 0; frame < frames; frame++)
        {
            cancellation.ThrowIfCancellationRequested();
            if (frame > rightFrame)
            {
                left = right; leftFrame = rightFrame; nextSample++;
                rightFrame = Math.Min(nextSample * interval, frames - 1);
                right = ReadPose(poses.Slice(nextSample * PoseChannels, PoseChannels), originalWidth, originalHeight, nextSample);
            }
            var fraction = rightFrame == leftFrame ? 0 : (frame - leftFrame) / (float)(rightFrame - leftFrame);
            var rotation = Quaternion.Normalize(Quaternion.Slerp(left.Rotation, right.Rotation, fraction));
            // The shared OpenCV-to-Y-up world-axis flip cancels in (R_t D)(R_0 D)^T.
            WriteRotation(Quaternion.Normalize(rotation * firstInverse), rotations.Slice(frame * MatrixChannels, MatrixChannels));
            var matrix = intrinsics.Slice(frame * MatrixChannels, MatrixChannels);
            matrix.Clear();
            // Upstream interpolates focal lengths after decoding, rather than interpolating FoV.
            matrix[0] = (float)((1d - fraction) * left.Fx + fraction * (double)right.Fx);
            matrix[4] = (float)((1d - fraction) * left.Fy + fraction * (double)right.Fy);
            matrix[2] = originalWidth * .5f; matrix[5] = originalHeight * .5f; matrix[8] = 1;
        }
    }

    /// <summary>The largest distance of any sampled camera centre from the first, in VGGT's
    /// scene-normalized units, where the mean distance of the scene from the first camera is
    /// about 1. A tripod gives nearly 0; a camera that slides with the subject does not. Rejects
    /// the same invalid poses as <see cref="Decode"/>.</summary>
    public static float Travel(ReadOnlySpan<float> poses)
    {
        if (poses.Length == 0 || poses.Length % PoseChannels != 0)
            throw new ArgumentException("Camera travel needs [samples,9] poses.");
        Vector3 first = default; float travel = 0;
        for (var sample = 0; sample < poses.Length / PoseChannels; sample++)
        {
            var pose = poses.Slice(sample * PoseChannels, PoseChannels);
            // Poses are camera-from-world, so the centre is -R^T t.
            var (rotation, _, _) = ReadPose(pose, 1, 1, sample);
            var centre = -Vector3.Transform(new Vector3(pose[0], pose[1], pose[2]), Quaternion.Conjugate(rotation));
            if (sample == 0) first = centre;
            travel = MathF.Max(travel, Vector3.Distance(centre, first));
        }
        return travel;
    }

    static (Quaternion Rotation, float Fx, float Fy) ReadPose(ReadOnlySpan<float> pose, int width, int height, int sample)
    {
        foreach (var value in pose)
            if (!float.IsFinite(value)) throw new ArgumentException($"Camera sample {sample} contains a non-finite pose.");
        var largest = Math.Max(Math.Max(Math.Abs(pose[3]), Math.Abs(pose[4])), Math.Max(Math.Abs(pose[5]), Math.Abs(pose[6])));
        if (largest == 0) throw new ArgumentException($"Camera sample {sample} has a zero quaternion.");
        // Scaling first avoids overflow/underflow when normalizing a finite prediction.
        var rotation = Quaternion.Normalize(new Quaternion(pose[3] / largest, pose[4] / largest, pose[5] / largest, pose[6] / largest));
        if (pose[7] <= 0 || pose[7] >= Math.PI || pose[8] <= 0 || pose[8] >= Math.PI)
            throw new ArgumentException($"Camera sample {sample} field of view must lie strictly between 0 and pi radians.");
        var fx = (float)(width * .5 / Math.Tan(pose[8] * .5));
        var fy = (float)(height * .5 / Math.Tan(pose[7] * .5));
        if (!float.IsFinite(fx) || !float.IsFinite(fy) || fx <= 0 || fy <= 0)
            throw new ArgumentException($"Camera sample {sample} produces invalid focal lengths.");
        return (rotation, fx, fy);
    }

    static void WriteRotation(Quaternion rotation, Span<float> output)
    {
        // System.Numerics uses row vectors; the network consumes column-vector matrices.
        var matrix = Matrix4x4.CreateFromQuaternion(rotation);
        output[0] = matrix.M11; output[1] = matrix.M21; output[2] = matrix.M31;
        output[3] = matrix.M12; output[4] = matrix.M22; output[5] = matrix.M32;
        output[6] = matrix.M13; output[7] = matrix.M23; output[8] = matrix.M33;
    }
}
