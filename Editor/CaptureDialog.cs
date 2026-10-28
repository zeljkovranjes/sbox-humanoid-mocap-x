using System;
using System.IO;
using System.Threading.Tasks;
using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>The X window: an account bar across the top (provider account and queue, the processing
/// machine, this computer, or a pointer to settings) above the shared workspace. Until a processing
/// choice is saved the drop area is locked and the settings button shows a red dot; the choice is made
/// in settings › Processing › Setup, after which the window rebuilds and reopens that page.</summary>
sealed class CaptureDialog : Dialog
{
    const string SetupPage = "Setup";
    readonly CaptureOptions options;
    readonly HostingSettings hosting;
    readonly HostedCapture hosted;
    readonly SshSettings machine;
    readonly SshCapture remote;
    readonly LocalCapture local;
    readonly string models;
    QueueTracker queue;
    RetargetWindow workspace;
    FlowCapture capture;

    internal CaptureDialog(CaptureOptions options) : base(null)
    {
        this.options = options;
        hosting = new HostingSettings
        {
            Enabled = EditorCookie.Get(ProcessingSetup.HostedCookie, false),
            ServiceUrl = EditorCookie.Get(ProcessingSetup.AddressCookie, ""),
            Dispatch = MainThread.Queue
        };
        var cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HumanoidMocapX");
        hosted = new HostedCapture(hosting, options, Path.Combine(cache, "hosted"));
        models = Path.Combine(cache, "models");
        machine = ProcessingSetup.SavedMachine();
        remote = new SshCapture(machine, options, Path.Combine(cache, "captures"));
        local = new LocalCapture(options, models, Path.Combine(cache, "captures"));
        Window.Title = MocapWindow.Title; Window.SetWindowIcon("videocam");
        Layout = Layout.Column();
        ShowWorkspace();
    }

    // Deferred: the button that asked belongs to the view being replaced.
    void Later(Action show) => MainThread.Queue(() => { if (this.IsValid()) show(); });

    void ShowWorkspace()
    {
        Layout.Clear(true);
        queue?.Dispose(); queue = null;
        if (hosting.Enabled)
            queue = new QueueTracker(hosting, hosted, path => workspace.IsValid() ? workspace.OpenMotionAsync(path) : Task.CompletedTask);
        var configured = ProcessingSetup.Configured;
        Layout.Add(new AccountBar(this, hosting, machine, queue, configured, OpenSetup));
        capture = new FlowCapture(options, hosting, hosted, machine, remote, local);
        workspace = Layout.Add(new RetargetWindow(this, MocapWindow.Title, capture,
            host => MocapWindow.Settings(host, options, hosting, machine, () => Later(Applied), RefreshGate)), 1);
        var account = workspace.AddSettingsPage("Processing", "account_circle", "Account");
        account.Layout.Add(new AccountPage(account, hosting, machine, configured, OpenSetup));
        RefreshGate();
        Fit(new Vector2(760, 520), new Vector2(1040, 700));
        if (hosting.Enabled && hosting.SignedIn) _ = RefreshProfileAsync();
    }

    void OpenSetup() { if (workspace.IsValid()) workspace.ShowAdjustments(SetupPage); }

    // Capture stays locked (grayed drop area, red dot on settings) until a processing choice is saved
    // and, on this computer, every model file exists. Rechecked after downloads and model changes.
    void RefreshGate()
    {
        if (!workspace.IsValid()) return;
        var reason = !ProcessingSetup.Configured ? ProcessingSetup.SetupMessage
            : !hosting.Enabled && !machine.Enabled && ModelStore.Missing(models, options.Model) is { } missing ? ProcessingSetup.ModelsMessage(missing) : null;
        capture.Blocked = reason;
        workspace.SetSetupRequired(reason);
    }

    // A saved processing choice changes the account bar, queue and capture backend, so the window
    // rebuilds around it; settings reopen where the choice was made.
    void Applied() { ShowWorkspace(); OpenSetup(); }

    // Picks up whitelist and role changes made since sign-in. An unreachable provider already
    // shows in the queue, so a failed refresh keeps the current profile.
    async Task RefreshProfileAsync()
    {
        using var client = new HostSignIn();
        try { await client.RefreshAsync(hosting, default); }
        catch (Exception failure) when (failure is System.Net.Http.HttpRequestException or InvalidDataException
            or System.Text.Json.JsonException or OperationCanceledException or ArgumentException)
        { Log.Info($"Humanoid Mocap X: profile refresh failed ({failure.Message})."); }
    }

    void Fit(Vector2 minimum, Vector2 preferred)
    {
        Window.MinimumSize = minimum;
        if (Window.Size.x < preferred.x || Window.Size.y < preferred.y) Window.Size = Vector2.Max(Window.Size, preferred);
    }

    public override void OnDestroyed()
    {
        queue?.Dispose(); hosted.Dispose();
        base.OnDestroyed();
    }
}
