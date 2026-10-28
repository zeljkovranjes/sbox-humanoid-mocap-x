using HumanoidMocapX.Inference;
using Sandbox;
using HumanoidMocapX.TensorInference;

namespace HumanoidMocapX.Checks;

/// <summary>Compares synthetic Omega image-to-camera inference on CPU and GPU.
/// Requires a live editor; published weights are not tested here.</summary>
public static class OmegaGpuChecks
{
    /// <summary>Exercises 64/128-channel heads, masked bias, Q/K normalization,
    /// split-half positions and the complete image-to-camera path. Throws on mismatch.</summary>
    public static async Task<object> Run(CancellationToken cancellation)
    {
        double largestError = 0;
        foreach (var headWidth in new[] { 64, 128 })
        {
            const int frames = 2, heads = 2;
            var width = headWidth * heads;
            var tensors = OmegaFixture.Create(width, heads); var weights = new FlowWeights(tensors);
            var input = DenoiserTests.Wave(frames * 17 * width, .013f); var expected = new float[frames * 9];
            await Task.Run(() => new OmegaCameraHead(weights, frames, width, heads).Run(input, expected, cancellation), cancellation);
            await GameTask.MainThread();
            var head = new OmegaGpuCameraHead(weights, frames, width, heads);
            try
            {
                var actual = new float[expected.Length];
                using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
                var observed = false;
                try { await head.RunAsync(input, actual, cancelled.Token); } catch (OperationCanceledException) { observed = true; }
                if (!observed) throw new Exception("Omega camera ignored cancellation.");
                await head.RunAsync(input, actual, cancellation); Compare(expected, actual);
            }
            finally { await GameTask.MainThread(); head.Dispose(); }

            const int tokens = 9;
            var periods = Enumerable.Range(0, headWidth / 4).Select(i => MathF.Pow(100, i / (float)(headWidth / 4))).ToArray();
            var factors = OmegaPosition.Factors(2, 3, 3, periods);
            var blockInput = DenoiserTests.Wave(tokens * width, .029f);
            foreach (var qk in new[] { false, true })
            {
                var name = qk ? "normalized" : "plain";
                OmegaFixture.Block(tensors, name, width, heads, qk); weights = new(tensors);
                var blockExpected = (float[])blockInput.Clone();
                var cpu = new OmegaBlock(weights, name, width, heads, qk);
                var cpuWorkspace = new OmegaWorkspace(tokens, width); var attention = new FlowAttention(tokens, width, heads, false);
                await Task.Run(() => cpu.Run(blockExpected, cpuWorkspace, attention, factors, cancellation), cancellation);
                await GameTask.MainThread();
                var storage = new FlowGpuStorage(); var batch = new TensorBatch();
                try
                {
                    var workspace = new OmegaGpuWorkspace(storage, tokens, width); var gpu = new OmegaGpuBlock(storage, weights, name, width, heads, qk);
                    var values = storage.Upload<float>(blockInput); var rotary = storage.Upload<float>(factors);
                    var keys = new uint[tokens * 4];
                    for (var row = 0; row < tokens; row++) keys[row * 4 + 1] = tokens;
                    var ranges = storage.Upload<uint>(keys); var actual = new float[blockExpected.Length];
                    await batch.RunAsync(() => gpu.Run(workspace, values, ranges, tokens, rotary, tokens), values, actual, cancellation);
                    Compare(blockExpected, actual);
                }
                finally { await GameTask.MainThread(); batch.Dispose(); storage.Dispose(); }
            }
        }
        await FullModel(Compare, cancellation);
        return new { Passed = true, LargestAbsoluteError = largestError };

        void Compare(float[] expected, float[] actual)
        {
            for (var i = 0; i < actual.Length; i++)
            {
                var error = Math.Abs((double)expected[i] - actual[i]); largestError = Math.Max(largestError, error);
                if (!float.IsFinite(actual[i]) || error > 1e-4 * Math.Max(1, Math.Abs(expected[i])))
                    throw new Exception($"Omega GPU output {i}: expected {expected[i]:R}, got {actual[i]:R}.");
            }
        }
    }

    static async Task FullModel(Action<float[], float[]> compare, CancellationToken cancellation)
    {
        var shape = new OmegaShape(2, 16, 32, 64, 1);
        var weights = new FlowWeights(OmegaFixture.Full(shape.Channels, shape.Heads));
        var images = Enumerable.Range(0, shape.Frames).Select(frame => Enumerable.Range(0, shape.ImageElements)
            .Select(i => .5f + .4f * MathF.Sin(i * .013f + frame)).ToArray()).ToArray();
        var expected = new float[shape.Frames * 9]; var actual = new float[expected.Length];
        await GameTask.MainThread();
        var model = new OmegaGpuModel(weights, shape);
        try
        {
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            var observed = false;
            try { await model.RunAsync((_, _, _) => throw new Exception("Cancelled Omega read an image."), actual, cancelled.Token); }
            catch (OperationCanceledException) { observed = true; }
            if (!observed) throw new Exception("Omega model ignored cancellation.");
            observed = false;
            try { await model.RunAsync((_, _, _) => Task.CompletedTask, actual, cancellation); }
            catch (ArgumentException) { observed = true; }
            if (!observed) throw new Exception("Omega accepted an incomplete image reader.");
            for (var run = 0; run < 2; run++)
            {
                if (run > 0) Array.Reverse(images);
                await Task.Run(() =>
                {
                    var encoder = new DinoImageEncoder(weights, shape); var aggregator = new OmegaAggregator(weights, shape);
                    var patches = new float[shape.Frames * shape.Patches * shape.Channels]; var combined = new float[shape.CameraFeatures];
                    for (var frame = 0; frame < shape.Frames; frame++)
                        encoder.Run(images[frame], patches.AsSpan(frame * shape.Patches * shape.Channels, shape.Patches * shape.Channels), cancellation);
                    aggregator.Run(patches, combined, cancellation);
                    new OmegaCameraHead(weights, shape.Frames, shape.Channels * 2, shape.Heads).Run(combined, expected, cancellation);
                }, cancellation);
                await GameTask.MainThread(); var nextFrame = 0;
                await model.RunAsync((frame, destination, token) =>
                {
                    token.ThrowIfCancellationRequested();
                    if (frame != nextFrame++) throw new Exception("Omega image requests are not sequential.");
                    images[frame].AsSpan().CopyTo(destination.Span); return Task.CompletedTask;
                }, actual, cancellation);
                if (nextFrame != shape.Frames) throw new Exception("Omega skipped an image.");
                compare(expected, actual);
            }
        }
        finally { await GameTask.MainThread(); model.Dispose(); }
    }
}
