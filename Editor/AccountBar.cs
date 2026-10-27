using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Editor;
using HumanoidMocap.EditorTools;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>The strip across the top of the X window. With a provider: avatar, name, status pills,
/// account ID and limits, then icon buttons for the queue, adding videos and sign-out. On this
/// computer: the GPU. Before setup: a pointer to settings. Missing names and pictures show as not
/// provided; account details and setup live in Mocap adjustments › Processing.</summary>
sealed class AccountBar : Widget
{
    readonly HostingSettings settings;
    readonly QueueTracker queue;
    readonly AccountAvatar avatar;
    readonly Label name, details, problem;
    readonly Layout pills;
    readonly Button signIn;
    readonly BadgeButton queueButton;
    readonly IconButton add, signOut;
    CancellationTokenSource pending;

    /// <summary>Provider mode when <paramref name="queue"/> is set, otherwise this computer, or a
    /// pointer to settings while processing is not <paramref name="configured"/>.</summary>
    internal AccountBar(Widget parent, HostingSettings settings, QueueTracker queue, bool configured, Action openSetup) : base(parent)
    {
        this.settings = settings; this.queue = queue;
        Layout = Layout.Row(); Layout.Margin = new Sandbox.UI.Margin(12, 10, 12, 10); Layout.Spacing = 10;
        avatar = Layout.Add(new AccountAvatar(this, 34));
        var text = Layout.AddColumn(1); text.Spacing = 2;
        var top = text.AddRow(); top.Spacing = 8;
        name = top.Add(Ui.Bold(new Label("", this), 14));
        pills = top.AddRow(); pills.Spacing = 6;
        top.AddStretchCell();
        details = text.Add(Ui.Muted(new Label("", this), small: true));
        problem = Layout.Add(Ui.Error(this));
        if (!configured)
        {
            avatar.ShowIcon("settings");
            name.Text = "Not set up";
            details.Text = "Choose where videos are processed: settings (top right) › Processing › Setup.";
            return;
        }
        if (queue is null)
        {
            avatar.ShowIcon("memory");
            name.Text = "This computer";
            var gpu = Sandbox.Engine.SystemInfo.Gpu is { Length: > 0 } gpuName && gpuName != "unset" ? gpuName : "This GPU";
            details.Text = $"{gpu} · up to {FlowVideoProcessor.LaneCapacity()} frames at once · videos never leave this computer";
                return;
        }
        signIn = Layout.Add(Ui.Primary("Sign in", "login", () => _ = SignInAsync(openSetup)));
        queueButton = Layout.Add(new BadgeButton(this, "format_list_bulleted", OpenQueue, "Queue: your jobs on the provider"));
        add = Layout.Add(Ui.Icon(this, "playlist_add", () => QueuePopup.AddVideos(queue), "Add videos to the queue"));
        signOut = Layout.Add(Ui.Icon(this, "logout", () => _ = SignOutAsync(), "Sign out"));
        settings.Changed += Refresh; queue.Changed += Refresh;
        Refresh();
    }

    void OpenQueue()
    {
        var popup = new QueuePopup(this, queue);
        popup.OpenAt(queueButton.ScreenRect.BottomLeft + new Vector2(queueButton.Width - popup.FixedWidth, 6));
    }

    void Refresh()
    {
        if (!this.IsValid()) return;
        var profile = settings.SignedIn ? settings.Profile : null;
        var label = EditorCookie.Get(ProcessingSetup.ProviderLabelCookie, "");
        signIn.Visible = profile is null; signIn.Text = label.Length > 0 ? "Sign in with " + label : "Sign in";
        queueButton.Visible = add.Visible = signOut.Visible = profile is not null;
        queueButton.Count = queue.Active;
        pills.Clear(true);
        if (profile is null)
        {
            name.Text = "Signed out"; details.Text = Host(); avatar.ShowIcon("person");
            return;
        }
        var user = profile.User;
        name.Text = Clip(user.Name ?? user.Username, 32) ?? "Name not provided";
        var (status, color) = profile switch
        {
            { Admin: true } => ("Admin", Ui.Accent),
            { CanProcess: true } => ("Whitelisted", Ui.Accent),
            { InGuild: false } => ("Join the Discord server", Theme.Yellow),
            _ => ("Not whitelisted", Theme.Red)
        };
        pills.Add(new Pill(this, status, color));
        if (profile.Limits is { Priority: true }) pills.Add(new Pill(this, "Priority", Theme.Yellow));
        details.Text = $"{Clip(user.Id, 40)} · {Host()}" + (profile.Limits is { } limit ? " · " + Describe(limit) : "");
        details.ToolTip = user.Id + ": your account ID on this provider. Give it to the provider to be whitelisted. Click to copy.";
        avatar.ShowAccount(user.Avatar, user.Name ?? user.Username);
    }

    protected override void OnMouseClick(MouseEvent e)
    {
        base.OnMouseClick(e);
        // The account ID line copies the ID, so users can send it to the provider.
        if (e.LeftMouseButton && details.IsUnderMouse && settings.Profile?.User.Id is { } id) EditorUtility.Clipboard.Copy(id);
    }

    static string Describe(HostLimits limit)
        => (limit.MaximumClipSeconds is { } clip ? $"clips up to {HostedCapture.Duration((int)clip)}" : "clips up to the provider's limit")
            + $" · uploads up to {limit.MaximumUploadBytes / (1024 * 1024)} MiB";

    string Host()
    {
        try { return settings.Endpoint().Host; } catch (ArgumentException) { return "No provider address"; }
    }

    static string Clip(string text, int length) => text is { } value && value.Length > length ? value[..(length - 1)] + "…" : text;

    async Task SignInAsync(Action openSetup)
    {
        if (pending is not null) return;
        if (ProcessingSetup.SavedProvider() is not { } provider) { openSetup(); return; }
        using var cancellation = new CancellationTokenSource(); pending = cancellation; var token = cancellation.Token;
        Ui.Show(problem, null); signIn.Enabled = false; signIn.Text = "Finish in your browser…";
        try { await ProcessingSetup.SignInAsync(settings, provider, token); }
        catch (Exception failure) when (ProcessingSetup.IsSignInFailure(failure))
        {
            await EditorPipeline.SwitchToMainThread();
            if (!token.IsCancellationRequested) Ui.Show(problem, failure is OperationCanceledException ? "Sign-in timed out. Try again." : failure.Message);
        }
        finally
        {
            await EditorPipeline.SwitchToMainThread();
            pending = null;
            if (signIn.IsValid()) { signIn.Enabled = true; Refresh(); }
        }
    }

    async Task SignOutAsync()
    {
        using var client = new HostSignIn();
        await client.SignOutAsync(settings);
    }

    public override void OnDestroyed()
    {
        settings.Changed -= Refresh; pending?.Cancel();
        if (queue is not null) queue.Changed -= Refresh;
        base.OnDestroyed();
    }

    protected override void OnPaint()
    {
        Paint.SetPen(Ui.CardEdge, 1);
        Paint.DrawLine(new Vector2(0, Height - .5f), new Vector2(Width, Height - .5f));
    }
}
