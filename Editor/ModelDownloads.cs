using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>Editor controls for explicit model downloads. Closing the widget cancels
/// any transfer and releases the HTTP client after its file operations have stopped.</summary>
internal sealed class ModelDownloads : Widget
{
    readonly CaptureOptions options;
    readonly Action changed;
    readonly HttpClient http = new() { Timeout = Timeout.InfiniteTimeSpan };
    readonly ModelStore store;
    readonly string cache;
    readonly (Label Name, IconButton Button, Func<CaptureModels, bool> Installed)[] rows;
    readonly Button cancel;
    readonly Label status;
    CancellationTokenSource transfer;
    string pendingMessage;
    bool destroyed;

    /// <summary><paramref name="changed"/> runs on the main thread after each download ends, so
    /// whatever waits for the models can recheck them.</summary>
    internal ModelDownloads(Widget parent, CaptureOptions options, Action changed) : base(parent)
    {
        this.options = options; this.changed = changed;
        cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HumanoidMocapX", "models");
        store = new ModelStore(http, cache);
        Layout = Layout.Column(); Layout.Spacing = 6;
        rows = new[]
        {
            Model("FlowHMR · selected model", "", () => _ = DownloadAsync(null), paths => File.Exists(paths.Flow)),
            Model("SMPL-H neutral skeleton", "87.3 MB", () => _ = DownloadAsync(null, smpl: true), paths => File.Exists(paths.Smpl)),
            Model("SAM 3D Body DINOv3", "2.81 GB", () => _ = DownloadAsync(PerceptionModel.Sam3dBody), paths => File.Exists(paths.Sam) && File.Exists(paths.Mhr)),
            Model("VGGT-Omega 1B-512", "4.58 GB", () => _ = DownloadAsync(PerceptionModel.VggtOmega), paths => File.Exists(paths.Omega)),
            Model("YOLOX-L detector", "434 MB", () => _ = DownloadAsync(PerceptionModel.YoloXL), paths => File.Exists(paths.Detector))
        };
        var footer = Layout.AddRow(); footer.Spacing = 8;
        status = footer.Add(SettingsText.Note(this, "Downloads start only when clicked. Capture checks installed files without downloading."), 1);
        cancel = footer.Add(Ui.Secondary("Cancel download", "cancel", () => transfer?.Cancel())); cancel.Visible = false;
        footer.Add(Ui.Icon(this, "folder_open", () => { Directory.CreateDirectory(cache); EditorUtility.OpenFolder(cache); }, "Open the models folder: " + cache, Ui.RowHeight));
        Layout.Add(SettingsText.Note(this, "SAM, VGGT and SMPL-H come from public mirrors; their licenses are saved beside the files. SMPL-H allows noncommercial research, education and artistic use only."));
        Refresh();
    }

    /// <summary>Grays out models whose files are installed and shows a check instead of the download
    /// arrow. Clicking it only verifies the file: the download is skipped when the hash matches and
    /// happens only to repair a damaged file. Presence only here; capture verifies hashes too.</summary>
    internal void Refresh()
    {
        var paths = ModelStore.Paths(cache, options.Model);
        foreach (var (name, button, installed) in rows)
        {
            var done = installed(paths);
            button.Enabled = transfer is null;
            button.Icon = done ? "check" : "download";
            button.Foreground = done ? Theme.Green : Theme.Text;
            button.ToolTip = done ? name.Text + " is downloaded. Click to verify it; only a damaged file is downloaded again." : "Download " + name.Text;
            name.SetStyles($"color: {(done ? Theme.TextDisabled : Theme.Text).Hex};");
        }
    }

    (Label, IconButton, Func<CaptureModels, bool>) Model(string name, string size, Action download, Func<CaptureModels, bool> installed)
    {
        var row = Layout.AddRow(); row.Spacing = 8;
        var label = row.Add(new Label(name, this), 1);
        row.Add(SettingsText.Note(this, size));
        return (label, row.Add(Ui.Icon(this, "download", download, "Download " + name, Ui.RowHeight)), installed);
    }

    async Task DownloadAsync(PerceptionModel? perception, bool smpl = false)
    {
        if (destroyed || transfer is not null) return;
        var selected = options.Model;
        var name = smpl ? "SMPL-H neutral" : perception switch
        {
            PerceptionModel.Sam3dBody => "SAM 3D Body DINOv3",
            PerceptionModel.VggtOmega => "VGGT-Omega 1B-512",
            PerceptionModel.YoloXL => "YOLOX-L",
            _ => "FlowHMR " + selected
        };
        using var cancellation = new CancellationTokenSource(); transfer = cancellation;
        foreach (var row in rows) row.Button.Enabled = false;
        cancel.Visible = true; status.Text = "Preparing " + name + "…";
        string result;
        try
        {
            void Progress(string message) => Report(name + ": " + message);
            await Task.Run(() => smpl ? store.EnsureSmplAsync(Progress, cancellation.Token) : perception is { } model
                ? store.EnsurePerceptionAsync(model, Progress, cancellation.Token)
                : store.EnsureAsync(selected, Progress, cancellation.Token), cancellation.Token);
            result = name + " files saved and verified.";
        }
        catch (OperationCanceledException) { result = name + " download cancelled. Completed files are kept; the unfinished file restarts next time."; }
        catch (Exception error) { result = name + " download failed: " + error.Message; }
        await EditorPipeline.SwitchToMainThread();
        transfer = null; Interlocked.Exchange(ref pendingMessage, null);
        if (destroyed) { http.Dispose(); return; }
        Refresh();
        cancel.Visible = false; status.Text = result;
        changed();
    }

    void Report(string message) => Interlocked.Exchange(ref pendingMessage, message);

    [EditorEvent.Frame]
    void UpdateStatus()
    {
        if (!destroyed && this.IsValid() && Interlocked.Exchange(ref pendingMessage, null) is { } message)
            status.Text = message;
    }

    public override void OnDestroyed()
    {
        destroyed = true;
        if (transfer is null) http.Dispose(); else transfer.Cancel();
        base.OnDestroyed();
    }
}
