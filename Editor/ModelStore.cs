using System;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace HumanoidMocapX.EditorTools;

/// <summary>Downloads a selected upstream checkpoint into a verified local cache.</summary>
public sealed partial class ModelStore
{
    // Pin the upstream revision so a later model update cannot silently change reconstruction.
    public const string Revision = "1eb105e554e0cfc1846ea5feac8c2d516a36d319";
    const string Repository = "https://huggingface.co/fafsaf1/flowhmr-0.46B/";
    const int BufferBytes = 1024 * 1024;
    const long MaximumCheckpointBytes = 4L * 1024 * 1024 * 1024;
    readonly HttpClient http;
    readonly string cache;

    /// <summary>Uses the caller-owned HTTP client and cache directory. Rejects a null
    /// client or blank directory; does not create files until a download is requested.</summary>
    public ModelStore(HttpClient http, string cache)
    {
        ArgumentNullException.ThrowIfNull(http);
        ArgumentException.ThrowIfNullOrWhiteSpace(cache);
        this.http = http;
        this.cache = Path.GetFullPath(cache);
    }

    /// <summary>Explicitly downloads Base or Latest; call only after a user download action.
    /// Verifies SHA-256 before installation or reuse.
    /// Network, integrity and cancellation failures propagate without replacing a cached file.</summary>
    public async Task<string> EnsureAsync(FlowModel model, Action<string> progress, CancellationToken token)
    {
        var name = model switch
        {
            FlowModel.Base => "flowhmr_base",
            FlowModel.Latest => "flowhmr_latest",
            _ => throw new ArgumentOutOfRangeException(nameof(model))
        };
        var relative = $"{name}/{name}.ckpt";
        using var metadata = await http.GetAsync(Repository + $"raw/{Revision}/" + relative,
            HttpCompletionOption.ResponseHeadersRead, token);
        metadata.EnsureSuccessStatusCode();
        await metadata.Content.LoadIntoBufferAsync(4096, token);
        var pointer = await metadata.Content.ReadAsStringAsync(token);
        var digest = Regex.Match(pointer, @"(?m)^oid sha256:([a-f0-9]{64})\r?$");
        var length = Regex.Match(pointer, @"(?m)^size ([0-9]+)\r?$");
        if (!digest.Success || !length.Success || !long.TryParse(length.Groups[1].Value, out var bytes)
            || bytes <= 0 || bytes > MaximumCheckpointBytes)
            throw new InvalidDataException("Invalid FlowHMR checkpoint metadata: " + relative);
        var directory = Path.Combine(cache, Revision); Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, name + ".ckpt");
        var sha256 = digest.Groups[1].Value;
        await EnsureFileAsync(Repository + $"resolve/{Revision}/" + relative, path, bytes, sha256, progress, token);
        return path;
    }

    async Task DownloadAsync(string url, string path, long expectedBytes, string expectedHash,
        Action<string> progress, CancellationToken token)
    {
        var partial = path + "." + Guid.NewGuid().ToString("N") + ".partial";
        try
        {
            using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, token);
            response.EnsureSuccessStatusCode();
            if (response.Content.Headers.ContentLength is long length && length != expectedBytes)
                throw new InvalidDataException("Unexpected download length for " + Path.GetFileName(path));
            await using (var input = await response.Content.ReadAsStreamAsync(token))
            await using (var output = new FileStream(partial, FileMode.CreateNew, FileAccess.Write, FileShare.None, BufferBytes, true))
            using (var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
            {
                var buffer = new byte[BufferBytes]; long total = 0; int previous = -1;
                while (await input.ReadAsync(buffer, token) is var count && count > 0)
                {
                    total += count;
                    if (total > expectedBytes) throw new InvalidDataException("Download exceeds declared length for " + Path.GetFileName(path));
                    await output.WriteAsync(buffer.AsMemory(0, count), token);
                    hash.AppendData(buffer, 0, count);
                    var percent = (int)(total * 100 / expectedBytes);
                    if (percent != previous) { previous = percent; progress?.Invoke($"Downloading weights · {percent}%"); }
                }
                if (total != expectedBytes || !Convert.ToHexString(hash.GetHashAndReset()).Equals(expectedHash, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("Checksum or length mismatch for " + Path.GetFileName(path) + ". Retry the download.");
            }
            token.ThrowIfCancellationRequested();
            File.Move(partial, path, true);
        }
        finally { if (File.Exists(partial)) File.Delete(partial); }
    }

    static async Task<bool> MatchesAsync(string path, long bytes, string sha256, CancellationToken token)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, BufferBytes, true);
        return stream.Length == bytes && Convert.ToHexString(await SHA256.HashDataAsync(stream, token))
            .Equals(sha256, StringComparison.OrdinalIgnoreCase);
    }
}
