using System.IO.Compression;
using System.Text;
using HumanoidMocapX.Inference;

/// <summary>Writes protocol-2 PyTorch ZIP fixtures without a Python runtime.</summary>
static class CheckpointFixture
{
    internal static void Write(string path, IReadOnlyDictionary<string, Tensor> tensors,
        string? component = "model_state_dict", string prefix = "motion_transformer.")
    {
        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);
        using (var byteorder = archive.CreateEntry("fixture/byteorder").Open()) byteorder.Write("little"u8);
        using var metadata = new MemoryStream(); using var writer = new BinaryWriter(metadata, Encoding.UTF8, true);
        writer.Write((byte)0x80); writer.Write((byte)2); writer.Write((byte)'}');
        if (component is not null) { Text(writer, component); writer.Write((byte)'}'); }
        writer.Write((byte)'(');
        var storageId = 0;
        foreach (var pair in tensors)
        {
            Text(writer, prefix + pair.Key);
            Symbol(writer, "torch._utils", "_rebuild_tensor_v2"); writer.Write((byte)'(');
            writer.Write((byte)'('); Text(writer, "storage"); Symbol(writer, "torch", "FloatStorage");
            Text(writer, storageId.ToString(System.Globalization.CultureInfo.InvariantCulture)); Text(writer, "cpu");
            Integer(writer, pair.Value.Data.Length); writer.Write((byte)'t'); writer.Write((byte)'Q'); Integer(writer, 0);
            var shape = pair.Value.Shape.Reverse().ToArray();
            Tuple(writer, shape);
            var strides = new int[shape.Length]; var stride = 1;
            for (var dim = shape.Length - 1; dim >= 0; dim--) { strides[dim] = stride; stride *= shape[dim]; }
            Tuple(writer, strides); writer.Write((byte)0x89);
            Symbol(writer, "collections", "OrderedDict"); writer.Write((byte)')'); writer.Write((byte)'R');
            writer.Write((byte)'t'); writer.Write((byte)'R');
            using var storage = new BinaryWriter(archive.CreateEntry("fixture/data/" + storageId, CompressionLevel.NoCompression).Open());
            foreach (var value in pair.Value.Data) storage.Write(value);
            storageId++;
        }
        writer.Write((byte)'u');
        if (component is not null) writer.Write((byte)'s');
        writer.Write((byte)'.'); writer.Flush();
        using var entry = archive.CreateEntry("fixture/data.pkl").Open(); metadata.Position = 0; metadata.CopyTo(entry);
    }

    internal static void Text(BinaryWriter writer, string text)
    { var bytes = Encoding.UTF8.GetBytes(text); writer.Write((byte)'X'); writer.Write(bytes.Length); writer.Write(bytes); }
    internal static void Symbol(BinaryWriter writer, string module, string name)
    { writer.Write((byte)'c'); writer.Write(Encoding.ASCII.GetBytes(module + "\n" + name + "\n")); }
    internal static void Integer(BinaryWriter writer, int value) { writer.Write((byte)'J'); writer.Write(value); }
    internal static void Tuple(BinaryWriter writer, int[] values)
    { writer.Write((byte)'('); foreach (var value in values) Integer(writer, value); writer.Write((byte)'t'); }
}
