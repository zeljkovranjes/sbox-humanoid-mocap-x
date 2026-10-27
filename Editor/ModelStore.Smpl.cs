using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace HumanoidMocapX.EditorTools;

public sealed partial class ModelStore
{
    internal const string SmplRevision = "363972831b553e4f5b62d11d509e98830d930c97";
    const string SmplRepository = "https://huggingface.co/lithiumice/models_hub/resolve/" + SmplRevision + "/4_SMPLhub/smplh/";

    /// <summary>Explicitly downloads the neutral SMPL-H skeleton asset and its license
    /// from a pinned mirror. Call only after a user download action. Returns the NPZ path;
    /// network, integrity and cancellation errors propagate without replacing valid files.</summary>
    public async Task<string> EnsureSmplAsync(Action<string> progress, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        var directory = Path.Combine(cache, "smplh-neutral", SmplRevision);
        Directory.CreateDirectory(directory);
        await DownloadTextAsync(SmplRepository + "LICENSE.txt", Path.Combine(directory, "LICENSE.txt"), token);
        var path = Path.Combine(directory, "model.npz");
        await EnsureFileAsync(SmplRepository + "neutral/model.npz", path, 87259106,
            SmplHash, progress, token);
        return path;
    }
}
