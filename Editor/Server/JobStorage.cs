#nullable enable annotations

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;

namespace HumanoidMocapHost;

/// <summary>Job files plus the host's saved settings.</summary>
internal interface IJobStorage
{
    Stream CreateVideo(Guid id);
    Stream OpenVideo(Guid id);
    string VideoPath(Guid id);
    Stream CreateMotion(Guid id);
    Stream OpenMotion(Guid id);
    void DeleteVideo(Guid id);
    void Delete(Guid id);
    HostSettings? LoadSettings();
    void SaveSettings(HostSettings settings);
}

internal sealed class FileJobStorage : IJobStorage, IDisposable
{
    const string LockName = "run.lock", SettingsName = "settings.json";
    static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);
    readonly string directory, root;
    // Held for the service lifetime so a later start can tell live run folders from abandoned ones.
    readonly FileStream runLock;
    internal string CaptureDirectory => Path.Combine(root, "captures");

    /// <summary>Uses a new run folder under <paramref name="directory"/>, after deleting run folders
    /// that no live host holds. Several hosts may share the directory.</summary>
    internal FileJobStorage(string directory)
    {
        this.directory = Path.GetFullPath(directory);
        Directory.CreateDirectory(this.directory);
        RemoveAbandonedRuns();
        root = Path.Combine(this.directory, Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        runLock = new FileStream(Path.Combine(root, LockName), FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None);
    }

    // A stopped or crashed run's jobs can no longer be listed or deleted on schedule, so its files
    // go at the next start; that keeps results from outliving the 24-hour promise.
    void RemoveAbandonedRuns()
    {
        foreach (var run in Directory.EnumerateDirectories(directory))
        {
            if (!Guid.TryParseExact(Path.GetFileName(run), "N", out _)) continue;
            var marker = Path.Combine(run, LockName);
            if (!File.Exists(marker)) continue;
            try
            {
                using (new FileStream(marker, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { }
                Directory.Delete(run, true);
            }
            catch (IOException) { } // Locked: another host is still running there.
        }
    }

    string PathFor(Guid id, string suffix) => Path.Combine(root, id.ToString("N") + suffix);
    static FileStream Create(string path) => new(path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, true);
    static FileStream Open(string path) => new(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete, 65536, true);
    public Stream CreateVideo(Guid id) => Create(PathFor(id, ".video"));
    public Stream OpenVideo(Guid id) => Open(PathFor(id, ".video"));
    public string VideoPath(Guid id) => PathFor(id, ".video");
    public Stream CreateMotion(Guid id) => Create(PathFor(id, ".hmotion"));
    public Stream OpenMotion(Guid id) => Open(PathFor(id, ".hmotion"));
    public void DeleteVideo(Guid id) => File.Delete(PathFor(id, ".video"));
    public void Delete(Guid id)
    {
        DeleteVideo(id);
        File.Delete(PathFor(id, ".hmotion"));
    }

    public HostSettings? LoadSettings()
    {
        var path = Path.Combine(directory, SettingsName);
        if (!File.Exists(path)) return null;
        try { return JsonSerializer.Deserialize<HostSettings>(File.ReadAllText(path), Json) ?? throw new JsonException("empty"); }
        catch (JsonException error) { throw new InvalidDataException($"{path} is not valid host settings. Fix or delete it.", error); }
    }

    public void SaveSettings(HostSettings settings)
    {
        var path = Path.Combine(directory, SettingsName);
        var temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            File.WriteAllText(temporary, JsonSerializer.Serialize(settings, Json));
            File.Move(temporary, path, true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    /// <summary>Releases the run folder's lock; its files are removed by the next host start.</summary>
    public void Dispose() => runLock.Dispose();

    /// <summary>Deletes this run after every reader/writer has drained. Other run folders
    /// and saved settings remain intact; deletion failures propagate to the operator.</summary>
    internal void DeleteRun()
    {
        runLock.Dispose();
        Directory.Delete(root, true);
    }
}
