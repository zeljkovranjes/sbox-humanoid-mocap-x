#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace HumanoidMocapX.EditorTools.Inference;

public sealed partial class TorchCheckpoint
{
    readonly Dictionary<string,IReadOnlyList<string>> stringLists=new(StringComparer.Ordinal);

    /// <summary>Named string lists from a script archive's data, such as joint names.
    /// Keys use dot-separated object fields; ordinary checkpoints leave this empty.</summary>
    public IReadOnlyDictionary<string,IReadOnlyList<string>> StringLists=>stringLists;

    /// <summary>Reads only stored tensors and string lists from a TorchScript archive.
    /// rootType must match its qualified root class, e.g. "__torch__.MHRDemo".
    /// Code entries and methods are never loaded or executed. Rejects executable
    /// pickle calls, invalid tensor storage, cycles and excessive object nesting.
    /// The caller owns the returned reader and must dispose it.</summary>
    public static TorchCheckpoint OpenScriptData(string path,string rootType)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(rootType);
        return new TorchCheckpoint(ZipFile.OpenRead(path),rootType);
    }

    TorchCheckpoint(ZipArchive source,string rootType)
    {
        archive=source;
        try
        {
            if(ReadMetadata(archive,out prefix,true) is not ScriptObject root||root.Type.Module+"."+root.Type.Name!=rootType)
                throw new InvalidDataException("TorchScript data root does not match "+rootType+".");
            var fields=0;Flatten(root,"",new HashSet<ScriptObject>(),0,ref fields);
            if(tensors.Count==0)throw new InvalidDataException("No tensor weights in script data.");
        }
        catch{archive.Dispose();throw;}
    }

    void Flatten(ScriptObject instance,string path,HashSet<ScriptObject> ancestors,int depth,ref int fields)
    {
        if(depth>=64||!ancestors.Add(instance))throw new InvalidDataException("TorchScript data contains a cycle or exceeds 64 nested objects.");
        if(instance.Fields is null)throw new InvalidDataException("TorchScript object has no stored fields: "+path);
        foreach(var pair in instance.Fields)
        {
            if(++fields>100_000)throw new InvalidDataException("TorchScript data exceeds the field limit.");
            if(pair.Key is not string member||member.Length==0||member.Length>1024||member.Any(c=>c!='_'&&!char.IsAsciiLetterOrDigit(c)))
                throw new InvalidDataException("Unexpected TorchScript field name.");
            var name=path.Length==0?member:path+"."+member;
            if(name.Length>4096)throw new InvalidDataException("TorchScript field path exceeds the size limit.");
            if(pair.Value is ScriptObject child)Flatten(child,name,ancestors,depth+1,ref fields);
            else if(pair.Value is TensorRef tensor)
            {
                var info=new TensorInfo(name,tensor.Storage.Key,tensor.Storage.Dtype,tensor.Storage.Count,tensor.Offset,tensor.Shape,tensor.Stride);
                Validate(info);tensors.Add(name,info);
            }
            else if(pair.Value is List<object?> list&&list.All(x=>x is string))
                stringLists.Add(name,Array.AsReadOnly(list.Cast<string>().ToArray()));
        }
        ancestors.Remove(instance);
    }
}
