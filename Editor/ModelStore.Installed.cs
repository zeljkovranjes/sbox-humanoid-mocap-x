using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace HumanoidMocapX.EditorTools;

public sealed partial class ModelStore
{
    internal const string SamHash = "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf";
    internal const string MhrHash = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc";
    internal const string OmegaHash = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934";
    internal const string YoloHash = "1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917";
    internal const string SmplHash = "a60a7e29d33f09ef1a6352907e2f485e161f471330ca40592764374063e751df";

    /// <summary>Checks installed weights against pinned release hashes, without network
    /// requests or file changes. Missing or corrupt files require a manual download.</summary>
    internal static async Task<CaptureModels> InstalledAsync(string cache, FlowModel model, Action<string> progress, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        new CaptureOptions { Model = model }.Validate();
        var paths = Paths(cache, model);
        // Check every path first so an absent model is reported before hashing gigabytes.
        if (Absent(paths, model) is { } missing)
            throw new FileNotFoundException("Missing " + missing.Name + " (" + Path.GetFileName(missing.Path) + "). Download it in Mocap adjustments → Processing → Setup → Local models.", missing.Path);
        await VerifyInstalledAsync(paths.Flow, model == FlowModel.Base ? 1841700869 : 1841702021, model == FlowModel.Base
            ? "77b903ba545669bafd6f7709c0f1748f5fa030a4f25f67392c8c082a21f5e672"
            : "b7eb06b119feaf5ad7287bafdb2fff38d42dfc22ea9d256f8cdd2ada26dfc53a", progress, token);
        await VerifyInstalledAsync(paths.Sam, 2109129346, SamHash, progress, token);
        await VerifyInstalledAsync(paths.Mhr, 696110248, MhrHash, progress, token);
        await VerifyInstalledAsync(paths.Omega, 4576706117, OmegaHash, progress, token);
        await VerifyInstalledAsync(paths.Detector, 434357141, YoloHash, progress, token);
        await VerifyInstalledAsync(paths.Smpl, 87259106, SmplHash, progress, token);
        return paths;
    }

    /// <summary>The first model capture needs that has no file yet, by name, or null when every
    /// file is present. Presence only and cheap; capture still verifies hashes before use.</summary>
    internal static string Missing(string cache, FlowModel model) => Absent(Paths(cache, model), model)?.Name;

    /// <summary>Where each model capture needs is installed under <paramref name="cache"/>.</summary>
    internal static CaptureModels Paths(string cache, FlowModel model)
    {
        var sam = Path.Combine(cache, "sam-3d-body-dinov3", SamRevision);
        return new CaptureModels(Path.Combine(cache, Revision, model == FlowModel.Base ? "flowhmr_base.ckpt" : "flowhmr_latest.ckpt"),
            Path.Combine(sam, "model.ckpt"), Path.Combine(sam, "assets", "mhr_model.pt"),
            Path.Combine(cache, "vggt-omega", OmegaRevision, "vggt_omega_1b_512.pt"),
            Path.Combine(cache, "yolox-l", YoloRelease, "yolox_l.pth"), Path.Combine(cache, "smplh-neutral", SmplRevision, "model.npz"));
    }

    static (string Name, string Path)? Absent(CaptureModels paths, FlowModel model)
    {
        foreach (var (name, path) in new[] { ("FlowHMR " + model, paths.Flow), ("SAM 3D Body DINOv3", paths.Sam),
            ("SAM's MHR asset", paths.Mhr), ("VGGT-Omega", paths.Omega), ("YOLOX-L", paths.Detector), ("SMPL-H neutral", paths.Smpl) })
            if (!File.Exists(path)) return (name, path);
        return null;
    }

    internal static async Task VerifyInstalledAsync(string path, long bytes, string hash, Action<string> progress, CancellationToken token)
    {
        token.ThrowIfCancellationRequested(); progress?.Invoke("Verifying installed " + Path.GetFileName(path) + "…");
        if (!File.Exists(path) || !await MatchesAsync(path, bytes, hash, token))
            throw new InvalidDataException("Missing or damaged model: " + path + ". Repair it with its button in Mocap adjustments → Processing → Setup → Local models.");
    }
}
