using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;
using Sandbox;

namespace HumanoidMocapX.Checks;

/// <summary>Compares released SAM decoder weights on CPU/GPU with complete real
/// MHR pose feedback and learned initialization. Image features are synthetic, not video.</summary>
public static class SamReleasedChecks
{
    /// <summary>Loads the installed checkpoints, runs both heads/decoders and
    /// compares final tokens and camera-space keypoints. Throws on any mismatch.</summary>
    public static async Task<object> Run(string samPath, string mhrPath, CancellationToken cancellation)
    {
        var body = await Task.Run(() => MhrCheckpoint.LoadBody(mhrPath, cancellation), cancellation);
        var bodyResult = await Decoder(samPath, body, false, cancellation);
        var handResult = await Decoder(samPath, body, true, cancellation);
        return new { Passed = true, SyntheticImageFeatures = true, LearnedInitialTokens = true, Body = bodyResult, Hand = handResult };
    }

    static async Task<object> Decoder(string path, MhrBody body, bool hand, CancellationToken cancellation)
    {
        var shape = new SamDecoderShape(145, 6);
        var weights = await Task.Run(() => VisionCheckpoint.LoadSamDecoder(path, hand, cancellation), cancellation);
        var feedback = await Task.Run(() =>
        {
            var pose = SamCheckpoint.LoadPoseHead(path, body, hand, cancellation);
            var camera = new SamPredictionHead(VisionCheckpoint.LoadSamHead(path, hand, cancellation), hand, true);
            var update = new SamKeypointFeedback(VisionCheckpoint.LoadSamFeedback(path, hand, cancellation), shape, 3, 2, hand);
            return new SamDecoderFeedback(shape, pose, camera, update);
        }, cancellation);
        var image = DenoiserTests.Wave(shape.ImageElements, .017f); var tokens = new float[shape.TokenElements];
        var positions = new float[tokens.Length]; var imagePositions = new float[image.Length];
        var expected = new float[tokens.Length]; var actual = new float[tokens.Length];
        var projection = new SamProjection(new float[] { 700, 0, 320, 0, 700, 240, 0, 0, 1 },
            new float[] { .1f, 0, -8, 0, .1f, -8 }, 320, 240, 480, DinoShape.Sam(32, 48), hand);
        await Task.Run(() => new SamInitialTokens(VisionCheckpoint.LoadSamInitialTokens(path, hand, cancellation), hand)
            .Run(projection, tokens, positions, cancellation), cancellation);
        feedback.Prepare(projection, image); SamPoseFeedback updatePose = feedback.Update;
        var expectedPoints = await Task.Run(() =>
        {
            new SamDecoder(weights, shape, hand).Run(tokens, image, positions, imagePositions, expected, updatePose, cancellation);
            feedback.Finish(expected, cancellation); return feedback.Pose.Keypoints.ToArray();
        }, cancellation);
        await GameTask.MainThread(); var gpu = new SamGpuDecoder(weights, shape, hand);
        double tokenError = 0, keypointError = 0;
        var previousTokens = new float[actual.Length]; var previousPoints = new float[expectedPoints.Length];
        try
        {
            for (var pass = 0; pass < 2; pass++)
            {
                await GameTask.MainThread();
                await gpu.RunAsync(tokens, image, positions, imagePositions, actual, updatePose, cancellation);
                await Task.Run(() => feedback.Finish(actual, cancellation), cancellation);
                tokenError = Math.Max(tokenError, Compare(expected, actual, 6e-4));
                keypointError = Math.Max(keypointError, Compare(expectedPoints, feedback.Pose.Keypoints, 3e-4));
                if (pass > 0 && (!previousTokens.AsSpan().SequenceEqual(actual) || !previousPoints.AsSpan().SequenceEqual(feedback.Pose.Keypoints)))
                    throw new Exception("Repeated SAM GPU decoding changed its tokens or final keypoints.");
                actual.AsSpan().CopyTo(previousTokens); feedback.Pose.Keypoints.CopyTo(previousPoints);
            }
        }
        finally { await GameTask.MainThread(); gpu.Dispose(); }
        return new { Layers = 6, PoseFeedbackUpdates = 5, MaximumTokenError = tokenError, MaximumKeypointErrorMetres = keypointError, ExactRepeat = true };
    }

    static double Compare(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, double tolerance)
    {
        double maximum = 0;
        for (var i = 0; i < expected.Length; i++)
        {
            var error = Math.Abs((double)expected[i] - actual[i]); maximum = Math.Max(maximum, error);
            if (!float.IsFinite(actual[i]) || error > tolerance * Math.Max(1, Math.Abs(expected[i])))
                throw new Exception($"Released SAM CPU/GPU output {i}: expected {expected[i]:R}, got {actual[i]:R}.");
        }
        return maximum;
    }
}
