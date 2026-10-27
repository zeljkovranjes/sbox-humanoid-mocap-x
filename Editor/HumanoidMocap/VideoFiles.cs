using System;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using Sandbox;
using HumanoidMocapX.EditorTools.Video;

namespace HumanoidMocapX.EditorTools;

/// <summary>Copies source footage into the editor's local preview filesystem without changing the original.</summary>
public static class VideoFiles
{
    /// <summary>Streams footage into an editor filesystem and returns its relative cache
    /// path. Call on the editor thread; cancellation or IO failure leaves no partial file.</summary>
    public static Task<string> CacheVideoAsync(string absolutePath, BaseFileSystem filesystem, CancellationToken cancellation=default)
    {
        ArgumentNullException.ThrowIfNull(filesystem);
        const string directory="humanoid_mocap/video";
        filesystem.CreateDirectory(directory);
        var folder=filesystem.GetFullPath(directory);
        return Task.Run(async()=>directory+"/"+await VideoPreviewCache.CopyAsync(absolutePath,folder,cancellation).ConfigureAwait(false),cancellation);
    }

    public static string CacheVideo(string absolutePath)
    {
        using var input=File.OpenRead(absolutePath);
        var hash=Convert.ToHexString(SHA256.HashData(input)).ToLowerInvariant();
        var relative="humanoid_mocap/video/"+hash+Path.GetExtension(absolutePath).ToLowerInvariant();
        var fs=global::Editor.FileSystem.ProjectTemporary;
        fs.CreateDirectory("humanoid_mocap/video");
        var destination=fs.GetFullPath(relative);
        if(!File.Exists(destination))File.Copy(absolutePath,destination);
        return relative;
    }
}
