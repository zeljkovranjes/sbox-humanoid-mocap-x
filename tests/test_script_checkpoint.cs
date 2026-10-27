using System.IO.Compression;
using System.Text;
using HumanoidMocapX.EditorTools.Inference;
using HumanoidMocapX.EditorTools.MotionBricks;

static class ScriptCheckpointTests
{
    internal static void Run()
    {
        var folder = Path.Combine(Path.GetTempPath(), "script-checkpoint-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            var path = Path.Combine(folder, "model.pt");
            var expected = new float[] { 1, -2, .25f, 4 };
            CheckpointFixture.Write(path, new Dictionary<string, Tensor> { ["weight"] = new("weight", expected, new[] { 2, 2 }) }, null, "");
            byte[] state;
            using (var archive = ZipFile.OpenRead(path))
            using (var source = archive.GetEntry("fixture/data.pkl").Open())
            using (var buffer = new MemoryStream()) { source.CopyTo(buffer); state = buffer.ToArray()[2..^1]; }
            Replace(path, writer =>
            {
                Object(writer, "__torch__", "MHRDemo"); writer.Write((byte)'}'); writer.Write((byte)'(');
                CheckpointFixture.Text(writer, "joint_names_list");
                CheckpointFixture.Symbol(writer, "torch.jit._pickle", "restore_type_tag");
                writer.Write((byte)']'); writer.Write((byte)'(');
                CheckpointFixture.Text(writer, "root"); CheckpointFixture.Text(writer, "left_hand"); writer.Write((byte)'e');
                CheckpointFixture.Text(writer, "List[str]"); writer.Write((byte)0x86); writer.Write((byte)'R');
                CheckpointFixture.Text(writer, "character"); Object(writer, "__torch__.fixture", "Character");
                writer.Write((byte)'q'); writer.Write((byte)1); writer.Write(state); writer.Write((byte)'b');
                CheckpointFixture.Text(writer, "alias"); writer.Write((byte)'h'); writer.Write((byte)1);
                writer.Write((byte)'u'); writer.Write((byte)'b');
            });
            using (var parsed = TorchCheckpoint.OpenScriptData(path, "__torch__.MHRDemo"))
            {
                MotionTests.Equal(2, parsed.Tensors.Count);
                MotionTests.Check(parsed.ReadFloat("character.weight").SequenceEqual(expected), "Script tensor bytes changed");
                MotionTests.Check(parsed.ReadFloat("alias.weight").SequenceEqual(expected), "Shared script object lost data");
                MotionTests.Check(parsed.StringLists["joint_names_list"].SequenceEqual(new[] { "root", "left_hand" }), "Script joint names changed");
                MotionTests.Throws<InvalidDataException>(() => parsed.ReadInt64("character.weight"));
                using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
                MotionTests.Throws<OperationCanceledException>(() => parsed.ReadFloat("character.weight", cancelled.Token));
            }
            MotionTests.Throws<InvalidDataException>(() => new TorchCheckpoint(path));
            MotionTests.Throws<InvalidDataException>(() => TorchCheckpoint.OpenScriptData(path, "__torch__.Other"));
            Reject(path, w => { Object(w, "__torch__", "MHRDemo"); });
            Reject(path, w =>
            {
                Object(w, "__torch__", "MHRDemo"); w.Write((byte)'q'); w.Write((byte)0);
                w.Write((byte)'}'); CheckpointFixture.Text(w, "cycle"); w.Write((byte)'h'); w.Write((byte)0); w.Write((byte)'s'); w.Write((byte)'b');
            });
            Reject(path, w =>
            {
                CheckpointFixture.Symbol(w, "__torch__", "MHRDemo"); w.Write((byte)')'); w.Write((byte)'R');
            });
            Reject(path, w => { CheckpointFixture.Symbol(w, "os", "system"); w.Write((byte)')'); w.Write((byte)'R'); });
            Reject(path, w =>
            {
                Object(w, "__torch__", "MHRDemo"); w.Write((byte)'}');
                CheckpointFixture.Text(w, "bad.name"); w.Write((byte)'N'); w.Write((byte)'s'); w.Write((byte)'b');
            });
            Reject(path, w =>
            {
                for (var i = 0; i < 65; i++)
                {
                    Object(w, "__torch__", "MHRDemo"); w.Write((byte)'}');
                    if (i < 64) CheckpointFixture.Text(w, "child");
                }
                w.Write((byte)'b');
                for (var i = 0; i < 64; i++) { w.Write((byte)'s'); w.Write((byte)'b'); }
            });
            Integers(path);
            Console.WriteLine("PASS: inert TorchScript tensor data, joint-name lists, shared objects, default rejection, forbidden calls, cycles and nesting limits");
        }
        finally { Directory.Delete(folder, true); }
    }

    static void Integers(string path)
    {
        foreach (var dtype in new[] { "UInt32Storage", "LongStorage", "IntStorage", "ByteStorage", "BoolStorage" })
        {
            long[] values = dtype switch
            {
                "UInt32Storage" => new long[] { 0, 16_777_217, uint.MaxValue, 1 },
                "LongStorage" => new long[] { long.MinValue, 16_777_217, uint.MaxValue, long.MaxValue },
                "IntStorage" => new long[] { int.MinValue, 16_777_217, -1, int.MaxValue },
                _ => new long[] { 0, 1, 1, 0 }
            };
            Replace(path, w =>
            {
                w.Write((byte)'}'); CheckpointFixture.Text(w, "indices");
                CheckpointFixture.Symbol(w, "torch._utils", "_rebuild_tensor"); w.Write((byte)'('); w.Write((byte)'(');
                CheckpointFixture.Text(w, "storage"); CheckpointFixture.Symbol(w, "torch", dtype);
                CheckpointFixture.Text(w, "0"); CheckpointFixture.Text(w, "cpu"); CheckpointFixture.Integer(w, 4);
                w.Write((byte)'t'); w.Write((byte)'Q'); CheckpointFixture.Integer(w, 0);
                CheckpointFixture.Tuple(w, new[] { 2, 2 }); CheckpointFixture.Tuple(w, new[] { 1, 2 });
                w.Write((byte)'t'); w.Write((byte)'R'); w.Write((byte)'s');
            });
            using (var archive = ZipFile.Open(path, ZipArchiveMode.Update))
            {
                archive.GetEntry("fixture/data/0").Delete();
                using var writer = new BinaryWriter(archive.CreateEntry("fixture/data/0").Open());
                foreach (var value in values)
                {
                    if (dtype == "UInt32Storage") writer.Write(checked((uint)value));
                    else if (dtype == "LongStorage") writer.Write(value);
                    else if (dtype == "IntStorage") writer.Write(checked((int)value));
                    else writer.Write(checked((byte)value));
                }
            }
            using var parsed = new TorchCheckpoint(path);
            var expected = new[] { values[0], values[2], values[1], values[3] };
            MotionTests.Check(parsed.ReadInt64("indices").SequenceEqual(expected), "Integer tensor lost precision or stride: " + dtype);
            MotionTests.Check(parsed.ReadFloat("indices").SequenceEqual(expected.Select(v => (float)v)), "Integer float conversion changed: " + dtype);
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            MotionTests.Throws<OperationCanceledException>(() => parsed.ReadInt64("indices", cancelled.Token));
        }
    }

    static void Reject(string path, Action<BinaryWriter> write)
    {
        Replace(path, write);
        MotionTests.Throws<InvalidDataException>(() => TorchCheckpoint.OpenScriptData(path, "__torch__.MHRDemo"));
    }

    static void Object(BinaryWriter writer, string module, string name)
    { CheckpointFixture.Symbol(writer, module, name); writer.Write((byte)')'); writer.Write((byte)0x81); }

    static void Replace(string path, Action<BinaryWriter> write)
    {
        using var archive = ZipFile.Open(path, ZipArchiveMode.Update);
        archive.GetEntry("fixture/data.pkl").Delete();
        using var writer = new BinaryWriter(archive.CreateEntry("fixture/data.pkl").Open(), Encoding.UTF8);
        writer.Write((byte)0x80); writer.Write((byte)2); write(writer); writer.Write((byte)'.');
    }
}
