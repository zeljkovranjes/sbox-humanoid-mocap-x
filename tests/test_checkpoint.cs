using System.IO.Compression;
using System.Runtime.InteropServices;
using HumanoidMocapX.Inference;
using HumanoidMocapX;
using HumanoidMocapX.EditorTools;

static class CheckpointTests
{
    internal static void Run()
    {
        var folder = Path.Combine(Path.GetTempPath(), "flow-checkpoints-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(folder);
        try
        {
            FlatStorage(folder);
            TensorViews(folder);
            var tensors = DenoiserTests.Fixture();
            var valid = Path.Combine(folder, "valid.ckpt"); CheckpointFixture.Write(valid, tensors);
            var weights = FlowCheckpoint.Load(valid);
            foreach (var pair in tensors)
            {
                var actual = weights.Get(pair.Key, pair.Value.Shape);
                MotionTests.Check(actual.Data.AsSpan().SequenceEqual(pair.Value.Data), "Checkpoint values or matrix orientation changed: " + pair.Key);
            }
            var model = new FlowDenoiser(weights, 1, DenoiserTests.Width, DenoiserTests.Heads);
            model.SetVideo(new float[FlowDenoiser.FeatureChannels], new float[FlowDenoiser.CameraChannels]);
            var state = new float[FlowSampler.Channels]; var output = new float[state.Length];
            model.Predict(state, .5f, output);
            MotionTests.Check(output.Any(v => v != 0), "Loaded model did not produce its learned output");
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            MotionTests.Throws<OperationCanceledException>(() => FlowCheckpoint.Load(valid, cancelled.Token));

            var wrongRoot = Path.Combine(folder, "wrong-root.ckpt"); CheckpointFixture.Write(wrongRoot, tensors, "other_state");
            MotionTests.Throws<InvalidDataException>(() => FlowCheckpoint.Load(wrongRoot));
            var shortModel = Path.Combine(folder, "missing.ckpt"); tensors.Remove("input_encoder.bias"); CheckpointFixture.Write(shortModel, tensors);
            MotionTests.Throws<InvalidDataException>(() => FlowCheckpoint.Load(shortModel));
            tensors = DenoiserTests.Fixture();
            var weight = tensors["input_encoder.weight"];
            tensors[weight.Name] = new Tensor(weight.Name, weight.Data, weight.Shape.Reverse().ToArray());
            var wrongShape = Path.Combine(folder, "shape.ckpt"); CheckpointFixture.Write(wrongShape, tensors);
            var shapeWeights = FlowCheckpoint.Load(wrongShape);
            MotionTests.Throws<InvalidDataException>(() => new FlowDenoiser(shapeWeights, 1, DenoiserTests.Width, DenoiserTests.Heads));
            tensors = DenoiserTests.Fixture(); tensors["input_encoder.weight"].Data[0] = float.NaN;
            var nan = Path.Combine(folder, "nan.ckpt"); CheckpointFixture.Write(nan, tensors);
            MotionTests.Throws<InvalidDataException>(() => FlowCheckpoint.Load(nan));
            var executable = Path.Combine(folder, "executable.ckpt");
            using (var zip = ZipFile.Open(executable, ZipArchiveMode.Create))
            using (var payload = zip.CreateEntry("fixture/data.pkl").Open())
                payload.Write(new byte[] { 0x80, 2, (byte)'c' }.Concat("fixture\nunsupported\n."u8.ToArray()).ToArray());
            MotionTests.Throws<InvalidDataException>(() => FlowCheckpoint.Load(executable));
            Console.WriteLine("PASS: checkpoint bit preservation, bounded read allocation, tensor views, malformed weights, executable-pickle rejection and cancellation");
        }
        finally { Directory.Delete(folder, true); }
    }

    static void FlatStorage(string folder)
    {
        // Cross two 64 KiB read boundaries, preserving signed zero and NaN payloads.
        var values = new float[32_777];
        int[] bits = { 0, int.MinValue, 1, 0x7f800000, unchecked((int)0xff800000), 0x7fc12345, unchecked((int)0xffc54321) };
        for (var i = 0; i < values.Length; i++) values[i] = BitConverter.Int32BitsToSingle(bits[i % bits.Length]);
        var path = Path.Combine(folder, "flat.ckpt");
        WriteView(path, values, 0, new[] { values.Length }, new[] { 1 });
        using var checkpoint = new TorchCheckpoint(path);
        _ = checkpoint.ReadFloat("weight");
        var before = GC.GetAllocatedBytesForCurrentThread();
        var actual = checkpoint.ReadFloat("weight");
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        MotionTests.Check(MemoryMarshal.AsBytes(actual.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(values.AsSpan())), "FP32 tensor bits changed.");
        const int StreamOverheadBudget = 64 * 1024;
        MotionTests.Check(allocated < values.Length * sizeof(float) + StreamOverheadBudget, $"Flat checkpoint read duplicated its storage: {allocated} allocated bytes.");
        before = GC.GetAllocatedBytesForCurrentThread();
        MotionTests.Throws<OperationCanceledException>(() => checkpoint.ReadFloat("weight", new CancellationToken(true)));
        MotionTests.Check(GC.GetAllocatedBytesForCurrentThread() - before < StreamOverheadBudget, "Cancelled read allocated tensor storage.");
    }

    static void TensorViews(string folder)
    {
        Check("offset", new float[] { 10, 20, 30, 40, 50, 60 }, 2, new[] { 2 }, new[] { 1 }, new float[] { 30, 40 });
        Check("transpose", new float[] { 10, 20, 30, 40, 50, 60 }, 0, new[] { 2, 3 }, new[] { 1, 2 }, new float[] { 10, 30, 50, 20, 40, 60 });
        Check("singleton", new float[] { 10, 20, 30, 40 }, 0, new[] { 1, 4 }, new[] { 999, 1 }, new float[] { 10, 20, 30, 40 });
        Check("empty", Array.Empty<float>(), 0, new[] { 0 }, new[] { 1 }, Array.Empty<float>());
        Check("scalar", new float[] { 42 }, 0, Array.Empty<int>(), Array.Empty<int>(), new float[] { 42 });

        void Check(string name, float[] storage, int offset, int[] shape, int[] stride, float[] expected)
        {
            var path = Path.Combine(folder, name + ".ckpt");
            WriteView(path, storage, offset, shape, stride);
            using var checkpoint = new TorchCheckpoint(path);
            MotionTests.Check(checkpoint.ReadFloat("weight").AsSpan().SequenceEqual(expected), "Tensor view changed: " + name);
        }
    }

    static void WriteView(string path, float[] values, int offset, int[] shape, int[] stride)
    {
        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);
        using (var storage = archive.CreateEntry("fixture/data/0", CompressionLevel.NoCompression).Open())
            storage.Write(MemoryMarshal.AsBytes(values.AsSpan()));
        using var writer = new BinaryWriter(archive.CreateEntry("fixture/data.pkl").Open());
        writer.Write((byte)0x80); writer.Write((byte)2); writer.Write((byte)'}');
        CheckpointFixture.Text(writer, "weight"); CheckpointFixture.Symbol(writer, "torch._utils", "_rebuild_tensor");
        writer.Write((byte)'('); writer.Write((byte)'('); CheckpointFixture.Text(writer, "storage");
        CheckpointFixture.Symbol(writer, "torch", "FloatStorage");
        CheckpointFixture.Text(writer, "0"); CheckpointFixture.Text(writer, "cpu");
        CheckpointFixture.Integer(writer, values.Length); writer.Write((byte)'t'); writer.Write((byte)'Q');
        CheckpointFixture.Integer(writer, offset); CheckpointFixture.Tuple(writer, shape); CheckpointFixture.Tuple(writer, stride);
        writer.Write((byte)'t'); writer.Write((byte)'R'); writer.Write((byte)'s'); writer.Write((byte)'.');
    }
}
