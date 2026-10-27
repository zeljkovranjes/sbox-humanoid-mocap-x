using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Editor;
using HumanoidMocap.EditorTools;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>Mocap adjustments › Processing › Account: who is signed in on the provider, what the
/// account may do (whitelist, Discord server, priority, clip and upload limits), and signing out or
/// switching accounts. On this computer it says no account is needed; before setup, how to start.
/// The provider shares no email address, so none is shown.</summary>
sealed class AccountPage : Widget
{
    const float LabelWidth = 150;
    readonly HostingSettings settings;
    readonly bool configured;
    readonly Action openSetup;
    readonly AccountAvatar avatar;
    readonly Label name, handle, problem;
    readonly Layout pills, actions;
    readonly Widget facts;
    CancellationTokenSource pending;

    internal AccountPage(Widget parent, HostingSettings settings, bool configured, Action openSetup) : base(parent)
    {
        this.settings = settings; this.configured = configured; this.openSetup = openSetup;
        Layout = Layout.Column(); Layout.Spacing = 12;
        // Identity: a plain row on the page's gray, like the facts below it.
        var identity = Layout.Add(new Widget(this) { Layout = Layout.Row(), FixedHeight = 44 });
        var row = identity.Layout; row.Spacing = 10;
        avatar = row.Add(new AccountAvatar(identity, 40));
        var who = row.AddColumn(1); who.Spacing = 4; who.AddStretchCell();
        var top = who.AddRow(); top.Spacing = 8;
        name = top.Add(Ui.Bold(new Label("", identity), 13));
        pills = top.AddRow(); pills.Spacing = 6;
        top.AddStretchCell();
        handle = who.Add(Ui.Muted(new Label("", identity)));
        who.AddStretchCell();
        actions = row.AddRow(); actions.Spacing = 8;
        problem = Layout.Add(Ui.Error(this));
        facts = Layout.Add(new Widget(this) { Layout = Layout.Column() });
        facts.Layout.Spacing = 12;
        // Spare height goes below the facts, so the identity panel keeps its own height.
        Layout.AddStretchCell();
        settings.Changed += Refresh;
        Refresh();
    }

    void Refresh()
    {
        if (!this.IsValid()) return;
        pills.Clear(true); actions.Clear(true); facts.Layout.Clear(true);
        if (!configured) { ShowNotSetUp(); return; }
        if (!settings.Enabled) { ShowThisComputer(); return; }
        var profile = settings.SignedIn ? settings.Profile : null;
        var provider = ProcessingSetup.SavedProvider();
        if (profile is null)
        {
            avatar.ShowIcon("person"); name.Text = "Signed out";
            handle.Text = $"Sign in to {Host()} to see your account and send videos.";
            actions.Add(Ui.Primary(provider is null ? "Sign in" : "Sign in with " + provider.Label, "login", () => _ = SignInAsync(false)));
            return;
        }
        var user = profile.User;
        avatar.ShowAccount(user.Avatar, user.Name ?? user.Username);
        name.Text = user.Name ?? user.Username ?? "Name not provided";
        handle.Text = string.Join(" · ", new[] { user.Username is { } username && username != user.Name ? "@" + username : null,
            "Signed in with " + (provider?.Label ?? user.Provider) }.Where(part => part is not null));
        var (status, color) = profile switch
        {
            { Admin: true } => ("Admin", Ui.Accent),
            { CanProcess: true } => ("Whitelisted", Ui.Accent),
            { InGuild: false } => ("Join the Discord server", Theme.Yellow),
            _ => ("Not whitelisted", Theme.Red)
        };
        pills.Add(new Pill(this, status, color));
        if (profile.Limits is { Priority: true }) pills.Add(new Pill(this, "Priority", Theme.Yellow));
        actions.Add(Ui.Secondary("Switch account", "switch_account", () => _ = SignInAsync(true))).ToolTip =
            "Signs out here and opens sign-in in your browser. Choose or log in to the other account there.";
        actions.Add(Ui.Secondary("Sign out", "logout", () => _ = SignOutAsync()));

        var access = Group("Access");
        Fact(access, "Whitelist", profile.Whitelisted ? "Whitelisted" : "Not whitelisted", profile.Whitelisted ? Ui.Accent : Theme.Red);
        if (profile.InGuild is { } inGuild)
            Fact(access, "Discord server", inGuild ? profile.GuildOwner ? "Member · server owner" : "Member" : "Not a member",
                inGuild ? Ui.Accent : Theme.Yellow);
        Fact(access, "Can send videos", profile.CanProcess ? "Yes" : "Not yet: ask your provider to whitelist your account ID",
            profile.CanProcess ? Ui.Accent : Theme.Yellow);
        if (profile.Admin) Fact(access, "Role", "Admin · manages this provider", Ui.Accent);

        var limits = Group("Limits");
        if (profile.Limits is { } limit)
        {
            Fact(limits, "Queue", limit.Priority ? "Priority · starts ahead of standard jobs" : "Standard", limit.Priority ? Theme.Yellow : null);
            Fact(limits, "Longest clip", limit.MaximumClipSeconds is { } clip ? HostedCapture.Duration((int)clip) : "The provider's limit");
            Fact(limits, "Largest upload", Bytes(limit.MaximumUploadBytes));
        }
        else Fact(limits, "Limits", "Not reported by this provider");

        var details = Group("Details");
        Fact(details, "Account ID", user.Id, copy: user.Id);
        Fact(details, "Username", user.Username ?? "Not provided");
        Fact(details, "Provider", Host());
    }

    void ShowNotSetUp()
    {
        avatar.ShowIcon("settings"); name.Text = "Not set up";
        handle.Text = "Choose whether videos are processed on this computer or by a provider.";
        actions.Add(Ui.Primary("Set up", "tune", openSetup));
    }

    void ShowThisComputer()
    {
        avatar.ShowIcon("memory"); name.Text = "This computer";
        handle.Text = "No account needed: videos are processed here and never leave this computer.";
        pills.Add(new Pill(this, "Local", Ui.Accent));
        var device = Group("Device");
        Fact(device, "GPU", Sandbox.Engine.SystemInfo.Gpu is { Length: > 0 } gpu && gpu != "unset" ? gpu : "This GPU");
        Fact(device, "Frames at once", $"Up to {FlowVideoProcessor.LaneCapacity()} on this GPU");
    }

    Widget Group(string title)
    {
        var group = facts.Layout.Add(new Widget(facts) { Layout = Layout.Column() });
        group.Layout.Spacing = 2;
        group.Layout.Add(new SectionLine(group, title));
        return group;
    }

    static void Fact(Widget group, string label, string value, Color? color = null, string copy = null)
    {
        var row = group.Layout.Add(new FactRow(group)).Layout;
        var caption = row.Add(Ui.Muted(new Label(label, group))); caption.FixedWidth = LabelWidth;
        var text = row.Add(new Label(value, group) { WordWrap = true }, 1);
        if (color is { } tint) Ui.Colored(text, tint);
        if (copy is not null)
            row.Add(Ui.Icon(group, "content_copy", () => EditorUtility.Clipboard.Copy(copy), "Copy: give this to your provider to be whitelisted", 24));
    }

    static string Bytes(long bytes) => bytes >= 1L << 30 ? $"{bytes / (double)(1L << 30):0.#} GB" : $"{bytes / (1024 * 1024)} MB";

    string Host()
    {
        try { return settings.Endpoint().Host; } catch (ArgumentException) { return "your provider"; }
    }

    async Task SignInAsync(bool switchAccount)
    {
        if (pending is not null) return;
        if (ProcessingSetup.SavedProvider() is not { } provider) { openSetup(); return; }
        using var cancellation = new CancellationTokenSource(); pending = cancellation; var token = cancellation.Token;
        Ui.Show(problem, null);
        try
        {
            if (switchAccount) { using var client = new HostSignIn(); await client.SignOutAsync(settings); }
            await ProcessingSetup.SignInAsync(settings, provider, token);
        }
        catch (Exception failure) when (ProcessingSetup.IsSignInFailure(failure))
        {
            await EditorPipeline.SwitchToMainThread();
            if (!token.IsCancellationRequested && problem.IsValid())
                Ui.Show(problem, failure is OperationCanceledException ? "Sign-in timed out. Try again." : failure.Message);
        }
        finally { pending = null; }
    }

    async Task SignOutAsync()
    {
        using var client = new HostSignIn();
        await client.SignOutAsync(settings);
    }

    public override void OnDestroyed()
    {
        settings.Changed -= Refresh; pending?.Cancel();
        base.OnDestroyed();
    }

    /// <summary>One label and value, with a hairline below so rows scan like a table.</summary>
    sealed class FactRow : Widget
    {
        internal FactRow(Widget parent) : base(parent) { Layout = Layout.Row(); Layout.Margin = new Sandbox.UI.Margin(0, 6, 0, 6); Layout.Spacing = 8; MinimumHeight = 30; }

        protected override void OnPaint()
        {
            Paint.SetPen(MocapButton.Edge.WithAlpha(.6f), 1);
            Paint.DrawLine(new Vector2(0, Height - .5f), new Vector2(Width, Height - .5f));
        }
    }
}
