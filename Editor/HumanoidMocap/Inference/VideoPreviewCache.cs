using System;
using System.Buffers;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace HumanoidMocapX.EditorTools.Inference;

/// <summary>Stages source video in a disk cache without changing or transcoding it.</summary>
internal static class VideoPreviewCache
{
    const int BufferSize = 65536;

    internal static async Task<string> CopyAsync(string source, string folder, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        Directory.CreateDirectory(folder);
        var partial = Path.Combine(folder, Guid.NewGuid().ToString("N") + ".partial");
        var buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
        try
        {
            byte[] digest;
            using (var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
            {
                await using var input = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize, true);
                if (input.Length == 0) throw new InvalidDataException("The source video is empty: " + source);
                await using var output = new FileStream(partial, FileMode.CreateNew, FileAccess.Write, FileShare.None, BufferSize, true);
                int count;
                while ((count = await input.ReadAsync(buffer.AsMemory(0, BufferSize), cancellation).ConfigureAwait(false)) > 0)
                {
                    hash.AppendData(buffer, 0, count);
                    await output.WriteAsync(buffer.AsMemory(0, count), cancellation).ConfigureAwait(false);
                }
                await output.FlushAsync(cancellation).ConfigureAwait(false);
                digest = hash.GetHashAndReset();
            }
            var filename = Convert.ToHexString(digest).ToLowerInvariant() + Path.GetExtension(source).ToLowerInvariant();
            var destination = Path.Combine(folder, filename);
            if (File.Exists(destination))
            {
                await using var cached = new FileStream(destination, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize, true);
                var actual = await SHA256.HashDataAsync(cached, cancellation).ConfigureAwait(false);
                if (CryptographicOperations.FixedTimeEquals(digest, actual)) return filename;
            }
            cancellation.ThrowIfCancellationRequested();
            File.Move(partial, destination, true);
            return filename;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
            File.Delete(partial);
        }
    }
}
