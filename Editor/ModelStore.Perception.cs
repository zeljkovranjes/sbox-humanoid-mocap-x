using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace HumanoidMocapX.EditorTools;

/// <summary>The body-feature, camera and person-detection packages required by FlowHMR.</summary>
public enum PerceptionModel { Sam3dBody, VggtOmega, YoloXL }

public sealed partial class ModelStore
{
    internal const string SamRevision = "566ff0482ef71b7a11d0874e2c41e5a801851343";
    internal const string OmegaRevision = "a8c3a718e0cf78e9e4c6847229efea793d37f060";
    internal const string YoloRelease = "0.1.1rc0";
    const string YoloRepository = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/" + YoloRelease + "/";
    const string YoloLicense = "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/e1052df71842031413f6030723c3607b839c80ce/LICENSE";
    const string SamRepository = "https://huggingface.co/mingjiexie/sam-3d-body-dinov3/resolve/" + SamRevision + "/";
    const string OmegaRepository = "https://huggingface.co/1kaiser/vggt-omega-jax/resolve/" + OmegaRevision + "/";
    const string OmegaLicense = "https://raw.githubusercontent.com/facebookresearch/vggt-omega/39a0cb8af88554f15ddcb5354cd52bde588fa014/LICENSE";
    const int MaximumTextBytes = 64 * 1024;

    /// <summary>Explicitly downloads a perception package from its release or a pinned public mirror,
    /// including its license and SAM's configuration/MHR assets. Returns its directory.
    /// Large files must match the mirror hashes recorded in docs/model-sources.md.
    /// Network, integrity and cancellation failures propagate; completed files are retained.</summary>
    public async Task<string> EnsurePerceptionAsync(PerceptionModel model, Action<string> progress, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        if (model == PerceptionModel.YoloXL)
        {
            var detectorDirectory = Path.Combine(cache, "yolox-l", YoloRelease);
            Directory.CreateDirectory(detectorDirectory);
            await DownloadTextAsync(YoloLicense, Path.Combine(detectorDirectory, "LICENSE"), token);
            await EnsureFileAsync(YoloRepository + "yolox_l.pth", Path.Combine(detectorDirectory, "yolox_l.pth"), 434357141,
                YoloHash, progress, token);
            return detectorDirectory;
        }
        if (model is not (PerceptionModel.Sam3dBody or PerceptionModel.VggtOmega)) throw new ArgumentOutOfRangeException(nameof(model));
        var sam = model == PerceptionModel.Sam3dBody;
        var directory = Path.Combine(cache, sam ? "sam-3d-body-dinov3" : "vggt-omega", sam ? SamRevision : OmegaRevision);
        Directory.CreateDirectory(directory);
        progress?.Invoke(sam ? "Preparing SAM 3D Body DINOv3 files…" : "Preparing VGGT-Omega 1B-512 files…");
        await DownloadTextAsync(sam ? SamRepository + "LICENSE" : OmegaLicense, Path.Combine(directory, "LICENSE"), token);
        if (sam)
        {
            await DownloadTextAsync(SamRepository + "model_config.yaml", Path.Combine(directory, "model_config.yaml"), token);
            await EnsureFileAsync(SamRepository + "model.ckpt", Path.Combine(directory, "model.ckpt"), 2109129346,
                SamHash, progress, token);
            Directory.CreateDirectory(Path.Combine(directory, "assets"));
            await EnsureFileAsync(SamRepository + "assets/mhr_model.pt", Path.Combine(directory, "assets", "mhr_model.pt"), 696110248,
                MhrHash, progress, token);
        }
        else
            await EnsureFileAsync(OmegaRepository + "vggt_omega_1b_512.pt", Path.Combine(directory, "vggt_omega_1b_512.pt"), 4576706117,
                OmegaHash, progress, token);
        return directory;
    }

    internal async Task EnsureFileAsync(string url, string path, long bytes, string sha256,
        Action<string> progress, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        progress?.Invoke("Checking " + Path.GetFileName(path) + "…");
        if (File.Exists(path) && await MatchesAsync(path, bytes, sha256, token)) return;
        progress?.Invoke("Downloading " + Path.GetFileName(path) + "…");
        await DownloadAsync(url, path, bytes, sha256, progress, token);
    }

    async Task DownloadTextAsync(string url, string path, CancellationToken token)
    {
        using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, token);
        response.EnsureSuccessStatusCode();
        await response.Content.LoadIntoBufferAsync(MaximumTextBytes, token);
        var bytes = await response.Content.ReadAsByteArrayAsync(token);
        if (bytes.Length == 0) throw new InvalidDataException("Empty model support file: " + Path.GetFileName(path));
        var partial = path + "." + Guid.NewGuid().ToString("N") + ".partial";
        try
        {
            await File.WriteAllBytesAsync(partial, bytes, token);
            token.ThrowIfCancellationRequested();
            File.Move(partial, path, true);
        }
        finally { if (File.Exists(partial)) File.Delete(partial); }
    }
}
