using HumanoidMocapX.EditorTools.MotionBricks;
using HumanoidMocapX;
using HumanoidMocapX.Inference;

static class DenoiserTests
{
    internal const int Width = 8, Heads = 2;

    internal static void Run()
    {
        Rotary();
        var weights = Fixture();
        MotionTests.Equal(304, weights.Count);
        const int frames = 3;
        var model = new FlowDenoiser(new FlowWeights(weights), frames, Width, Heads);
        var state = Wave(frames * FlowSampler.Channels, .019f);
        var body = Wave(frames * FlowDenoiser.FeatureChannels, .037f);
        var camera = Wave(frames * FlowDenoiser.CameraChannels, .17f);
        var output = new float[state.Length];
        MotionTests.Throws<InvalidOperationException>(() => model.Predict(state, .3f, output));
        model.SetVideo(body, camera);
        foreach (var time in new[] { 0f, .37f, .95f })
        {
            model.Predict(state, time, output);
            var expected = DenoiserReference.Predict(weights, state.Select(v => (double)v).ToArray(), body, camera, time);
            for (var i = 0; i < output.Length; i++) MotionTests.Near(expected[i], output[i], .00003);
        }
        var first = output.ToArray();
        body.AsSpan().Clear(); camera.AsSpan().Clear();
        model.Predict(state, .95f, output);
        for (var i = 0; i < output.Length; i++) MotionTests.Near(first[i], output[i], 0);
        model.Predict(state, .5f, output);
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 3; i++) model.Predict(state, .5f, output);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        MotionTests.Throws<ArgumentException>(() => model.Predict(state, .5f, state));
        MotionTests.Throws<ArgumentOutOfRangeException>(() => model.Predict(state, float.NaN, output));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => model.Predict(state, .5f, output, cancelled.Token));
        MotionTests.Throws<OperationCanceledException>(() => model.SetVideo(body, camera, cancelled.Token));
        MotionTests.Throws<InvalidOperationException>(() => model.Predict(state, .5f, output));
        model.SetVideo(body, camera);
        var expectedState = state.Select(v => (double)v).ToArray();
        for (var step = 0; step < FlowSampler.Steps; step++)
        {
            var time = step / (float)FlowSampler.Steps;
            var prediction = DenoiserReference.Predict(weights, expectedState, body, camera, time);
            var dt = (step + 1) / (float)FlowSampler.Steps - time;
            for (var i = 0; i < expectedState.Length; i++) expectedState[i] += (prediction[i] - expectedState[i]) / Math.Max(1f - time, .05f) * dt;
        }
        FlowSampler.Sample(state, output, model.Predict);
        for (var i = 0; i < state.Length; i++) MotionTests.Near(expectedState[i], state[i], .00005);
        Console.WriteLine("PASS: all eighteen denoiser blocks, conditioning/final projection, sampler integration and zero-allocation prediction");
        GpuFixtures();
    }

    static void GpuFixtures()
    {
        foreach (var (frames, heads) in new[] { (1, 1), (3, 2), (67, 1) })
        {
            var width = 64 * heads;
            var model = new FlowDenoiser(new FlowWeights(Fixture(width, heads)), frames, width, heads);
            var body = Wave(frames * FlowDenoiser.FeatureChannels, .037f);
            var camera = Wave(frames * FlowDenoiser.CameraChannels, .17f);
            var initial = Wave(frames * FlowSampler.Channels, .019f);
            var state = (float[])initial.Clone(); var prediction = new float[state.Length];
            model.SetVideo(body, camera); FlowSampler.Sample(state, prediction, model.Predict);
            MotionTests.Check(!state.AsSpan().SequenceEqual(initial), "GPU-check reference fixture must generate nontrivial motion");
            var first = (float[])state.Clone();
            for (var i = 0; i < body.Length; i++) body[i] *= -.4f;
            for (var i = 0; i < camera.Length; i++) camera[i] *= .7f;
            initial.CopyTo(state, 0);
            model.SetVideo(body, camera); FlowSampler.Sample(state, prediction, model.Predict);
            MotionTests.Check(!state.AsSpan().SequenceEqual(first), "GPU-check reference fixture must respond to changed video conditioning");
        }
        Console.WriteLine("PASS: CPU reference fixtures for GPU comparisons, 64-channel heads and 67-frame attention windows");
    }

    static void Rotary()
    {
        const int frames = 5, width = 8, heads = 2, headWidth = width / heads;
        var q = Wave(frames * width, .3f); var k = Wave(q.Length, .2f); var original = q.ToArray();
        var rotary = new FlowRotary(frames, width, heads);
        rotary.Apply(q, k);
        var factors = new float[frames * headWidth]; rotary.CopyFactors(factors);
        MotionTests.Throws<ArgumentException>(() => rotary.CopyFactors(new float[1]));
        for (var frame = 0; frame < frames; frame++)
            for (var head = 0; head < heads; head++)
                for (var pair = 0; pair < headWidth / 2; pair++)
                {
                    var at = frame * width + head * headWidth + pair * 2;
                    var angle = frame * Math.Pow(10000, -2d * pair / headWidth);
                    MotionTests.Near(Math.Cos(angle), factors[frame * headWidth + pair * 2]);
                    MotionTests.Near(Math.Sin(angle), factors[frame * headWidth + pair * 2 + 1]);
                    var expected = new System.Numerics.Complex(original[at], original[at + 1]) * System.Numerics.Complex.FromPolarCoordinates(1, angle);
                    MotionTests.Near(expected.Real, q[at]); MotionTests.Near(expected.Imaginary, q[at + 1]);
                }
        MotionTests.Throws<ArgumentException>(() => new FlowRotary(5001, width, heads));
        MotionTests.Throws<ArgumentException>(() => new FlowRotary(1, 6, 2));
    }

    internal static float[] Wave(int count, float frequency) => Enumerable.Range(0, count).Select(i => MathF.Sin(i * frequency)).ToArray();

    internal static Dictionary<string, Tensor> Fixture(int width = Width, int heads = Heads)
    {
        var weights = RefinerTests.Fixture(width, heads); int serial = 37;
        var amplitude = .07f * MathF.Sqrt(Width / (float)width);
        void Add(string name, int[] shape, bool norm = false)
        {
            var count = shape.Aggregate(1, (a, b) => a * b); var data = new float[count];
            for (var i = 0; i < count; i++) data[i] = (norm ? 1 : 0) + amplitude * MathF.Sin((serial++ % 503) * .73f);
            weights.Add(name, new Tensor(name, data, shape));
        }
        void Linear(string name, int inputs, int outputs)
        { Add(name + ".weight", new[] { inputs, outputs }); Add(name + ".bias", new[] { outputs }); }
        Linear("input_encoder", FlowSampler.Channels, width);
        Linear("ctxt_encoder.feature", FlowDenoiser.FeatureChannels, width);
        Linear("ctxt_encoder.camera_R", FlowDenoiser.CameraChannels, width);
        Linear("timestep_encoder.blocks.0", width, width); Linear("timestep_encoder.blocks.2", width, width);
        Linear("vtxt_encoder.linears.0", FlowDenoiser.TextChannels, width); Linear("vtxt_encoder.linears.2", width, width);
        Linear("final_layer.adaLN_modulation.linear", width, 2 * width); Linear("final_layer.linear", width, FlowSampler.Channels);
        for (var index = 0; index < FlowDenoiser.DoubleBlocks; index++)
            foreach (var branch in new[] { "motion", "text" })
            {
                var name = "double_blocks." + index + "." + branch;
                Linear(name + "_mod.linear", width, 6 * width); Linear(name + "_qkv", width, 3 * width);
                Linear(name + "_out_proj", width, width); Linear(name + "_mlp.fc1", width, 4 * width); Linear(name + "_mlp.fc2", 4 * width, width);
                Add(name + "_q_norm.weight", new[] { width / heads }, true); Add(name + "_k_norm.weight", new[] { width / heads }, true);
            }
        for (var index = 0; index < FlowDenoiser.SingleBlocks; index++)
        {
            var name = "single_blocks." + index;
            Linear(name + ".modulation.linear", width, 3 * width); Linear(name + ".linear1", width, 7 * width); Linear(name + ".linear2", 5 * width, width);
            Add(name + ".q_norm.weight", new[] { width / heads }, true); Add(name + ".k_norm.weight", new[] { width / heads }, true);
        }
        return weights;
    }
}
