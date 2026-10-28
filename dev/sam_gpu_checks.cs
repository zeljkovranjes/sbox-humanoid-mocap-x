using HumanoidMocapX.Inference;
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Checks;

/// <summary>Compares all 32 SAM DINOv3 image blocks on CPU and GPU with synthetic
/// weights, plus both pose decoders with synthetic feedback. Requires a live
/// editor; MHR inference and released weights are not covered.</summary>
public static class SamGpuChecks
{
    /// <summary>Checks repeated image encoding, SwiGLU, per-axis positions and patch
    /// output layout. Throws on mismatch; this does not validate released weights.</summary>
    public static async Task<object> Run(CancellationToken cancellation)
    {
        var shape = DinoShape.Sam(32, 48, 64, 1); var weights = new FlowWeights(OmegaFixture.Sam(shape.Channels, shape.Heads));
        var pixels = Enumerable.Range(0, shape.ImageElements).Select(i => .5f + .4f * MathF.Sin(i * .013f)).ToArray();
        var expected = new float[shape.Patches * shape.Channels]; var actual = new float[expected.Length]; var normalized = new float[pixels.Length];
        var cpu = new DinoImageEncoder(weights, shape);
        await GameTask.MainThread();
        var storage = new FlowGpuStorage(); TensorBatch batch = null; double maximumError = 0;
        try
        {
            batch = new(); var workspace = new OmegaGpuWorkspace(storage, shape.Tokens, shape.Channels, true);
            var gpu = new DinoGpuEncoder(storage, weights, shape);
            var image = storage.Create<float>(pixels.Length); var output = storage.Create<float>(actual.Length);
            for (var pass = 0; pass < 2; pass++)
            {
                if (pass > 0) Array.Reverse(pixels);
                await Task.Run(() => { cpu.Run(pixels, expected, cancellation); DinoImageEncoder.NormalizeImage(pixels, normalized); }, cancellation);
                await GameTask.MainThread(); image.SetData<float>(normalized);
                await batch.RunAsync(() => gpu.Run(workspace, image, output, cancellation), output, actual, cancellation);
                for (var i = 0; i < actual.Length; i++)
                {
                    var error = Math.Abs((double)expected[i] - actual[i]); maximumError = Math.Max(maximumError, error);
                    if (!float.IsFinite(actual[i]) || error > 2e-4 * Math.Max(1, Math.Abs(expected[i])))
                        throw new Exception($"SAM DINO GPU value {i}: expected {expected[i]:R}, got {actual[i]:R}.");
                }
            }
        }
        finally { await GameTask.MainThread(); batch?.Dispose(); storage.Dispose(); }
        foreach (var hand in new[] { false, true })
        {
            maximumError = Math.Max(maximumError, await Conditioning(hand, cancellation));
            maximumError = Math.Max(maximumError, await Decoder(hand, cancellation));
        }
        return new { Passed = true, LargestAbsoluteError = maximumError };
    }

    static async Task<double> Conditioning(bool hand, CancellationToken cancellation)
    {
        var shape = DinoShape.Sam(32, 48, 64, 1); var weights = new FlowWeights(SamFixture.ImageConditioning(shape.Channels, hand));
        var cpu = new SamImageConditioning(weights, shape, hand); var rays = new SamRays(shape); var camera = new float[rays.Elements];
        rays.Prepare(new float[] { 700, 0, 320, 0, 530, 240, 0, 0, 1 }, new float[] { .5f, 0, 120, 0, .7f, -13 }, camera, cancellation);
        var features = DenoiserTests.Wave(shape.Patches * shape.Channels, .017f); var expected = new float[features.Length];
        var actual = new float[features.Length]; double maximumError = 0;
        await GameTask.MainThread(); var storage = new FlowGpuStorage(); TensorBatch batch = null;
        try
        {
            batch = new(); var gpu = new SamGpuImageConditioning(storage, weights, shape, hand); var kernels = new TensorKernels();
            var image = storage.Upload<float>(features); var rayBuffer = storage.Upload<float>(camera); var output = storage.Create<float>(actual.Length);
            for (var pass = 0; pass < 2; pass++)
            {
                if (pass > 0) Array.Reverse(features);
                await Task.Run(() => cpu.Run(features, camera, expected, cancellation), cancellation);
                await GameTask.MainThread(); image.SetData<float>(features);
                await batch.RunAsync(() => gpu.Run(kernels, image, rayBuffer, output, cancellation), output, actual, cancellation);
                Compare();
            }
            cpu.CopyPositions(expected);
            await batch.RunAsync(() => gpu.Positions.CopyTo(output), output, actual, cancellation); Compare();
        }
        finally { await GameTask.MainThread(); batch?.Dispose(); storage.Dispose(); }
        return maximumError;

        void Compare()
        {
            for (var i = 0; i < actual.Length; i++)
            {
                var error = Math.Abs((double)expected[i] - actual[i]); maximumError = Math.Max(maximumError, error);
                if (!float.IsFinite(actual[i]) || error > 3e-4 * Math.Max(1, Math.Abs(expected[i])))
                    throw new Exception($"SAM image conditioning GPU value {i}: expected {expected[i]:R}, got {actual[i]:R}.");
            }
        }
    }

    static async Task<double> Decoder(bool hand, CancellationToken cancellation)
    {
        var shape = new SamDecoderShape(5, 7, 32, 48, 1, 64, 40);
        var weights = new FlowWeights(SamFixture.Decoder(shape, hand)); var cpu = new SamDecoder(weights, shape, hand);
        var tokens = DenoiserTests.Wave(shape.TokenElements, .017f); var image = DenoiserTests.Wave(shape.ImageElements, .023f);
        var tokenPe = DenoiserTests.Wave(tokens.Length, .007f); var imagePe = DenoiserTests.Wave(image.Length, .011f);
        var expected = new float[tokens.Length]; var actual = new float[tokens.Length]; double maximumError = 0;
        SamPoseFeedback feedback = (layer, normalized, current, positions, token) =>
        { token.ThrowIfCancellationRequested(); SamFixture.Feedback(shape.Width, layer, normalized, current, positions); };
        await GameTask.MainThread(); var gpu = new SamGpuDecoder(weights, shape, hand);
        try
        {
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel(); var observed = false;
            try { await gpu.RunAsync(tokens, image, tokenPe, imagePe, actual, feedback, cancelled.Token); }
            catch (OperationCanceledException) { observed = true; }
            if (!observed) throw new Exception("SAM decoder ignored cancellation.");
            observed = false;
            SamPoseFeedback invalid = (_, _, current, _, _) => current[0] = float.NaN;
            try { await gpu.RunAsync(tokens, image, tokenPe, imagePe, actual, invalid, cancellation); }
            catch (ArgumentException) { observed = true; }
            if (!observed) throw new Exception("SAM decoder accepted non-finite pose feedback.");
            for (var pass = 0; pass < 2; pass++)
            {
                if (pass > 0) Array.Reverse(image);
                await Task.Run(() => cpu.Run(tokens, image, tokenPe, imagePe, expected, feedback, cancellation), cancellation);
                await GameTask.MainThread(); await gpu.RunAsync(tokens, image, tokenPe, imagePe, actual, feedback, cancellation);
                for (var i = 0; i < actual.Length; i++)
                {
                    var error = Math.Abs((double)expected[i] - actual[i]); maximumError = Math.Max(maximumError, error);
                    if (!float.IsFinite(actual[i]) || error > 3e-4 * Math.Max(1, Math.Abs(expected[i])))
                        throw new Exception($"SAM decoder GPU value {i}: expected {expected[i]:R}, got {actual[i]:R}.");
                }
            }
        }
        finally { await GameTask.MainThread(); gpu.Dispose(); }
        return maximumError;
    }
}
