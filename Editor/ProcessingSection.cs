using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Editor;
using HumanoidMocapX.EditorTools;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>Settings › Processing › Setup › Processing: where captures run. This computer is saved at
/// once. A provider needs its address, then a browser sign-in with an option its site offers; until
/// that succeeds the current choice and address stay in use. <c>applied</c> runs after a choice is
/// saved, so the window can rebuild around it.</summary>
sealed class ProcessingSection : Widget
{
    readonly HostingSettings settings;
    readonly Action applied;
    readonly float labelWidth;
    readonly Widget provider;
    readonly LineEdit address;
    readonly Button connect;
    readonly Widget signIns;
    readonly Label status, problem;
    CancellationTokenSource pending;
    bool building = true;

    internal ProcessingSection(Widget parent, HostingSettings settings, Action applied, float labelWidth) : base(parent)
    {
        this.settings = settings; this.applied = applied; this.labelWidth = labelWidth;
        Layout = Layout.Column(); Layout.Spacing = 6;
        var configured = ProcessingSetup.Configured; var hosted = configured && settings.Enabled;
        var row = Layout.AddRow(); row.Spacing = 8;
        row.Add(new Label("Runs on", this) { FixedWidth = labelWidth });
        var mode = row.Add(Ui.Framed(new ComboBox(this)), 1);
        if (!configured) mode.AddItem("Not set up · choose one", "help_outline", () => ShowProvider(false), selected: true);
        mode.AddItem("This computer", "memory", UseThisComputer,
            LocalCapture.Availability ?? "Your GPU processes videos; nothing leaves this computer.", configured && !hosted);
        mode.AddItem("A provider", "cloud", () => ShowProvider(true),
            "A provider processes videos for you after you sign in.", hosted);

        provider = Layout.Add(new Widget(this) { Layout = Layout.Column(), Visible = hosted });
        provider.Layout.Spacing = 6;
        var line = provider.Layout.AddRow(); line.Spacing = 8;
        line.Add(new Label("Address", provider) { FixedWidth = labelWidth });
        address = line.Add(Ui.Framed(new LineEdit(provider) { Text = settings.ServiceUrl, PlaceholderText = "mocap.example.com", MaxLength = 2048 }), 1);
        connect = line.Add(Ui.Secondary(hosted ? "Reconnect" : "Connect", "link", () => _ = ConnectAsync()));
        address.ReturnPressed += () => _ = ConnectAsync();
        signIns = provider.Layout.Add(new Widget(provider) { Layout = Layout.Row(), Visible = false });
        signIns.Layout.Spacing = 8;
        status = provider.Layout.Add(Ui.Muted(new Label("", provider) { WordWrap = true, Visible = false }));
        problem = Layout.Add(Ui.Error(this));
        Layout.Add(SettingsText.Note(this, !configured ? "Choose this computer or a provider before capturing videos."
            : hosted ? "Your provider receives the videos you capture and your account ID. Results are deleted 24 hours after processing."
            : LocalCapture.Availability ?? "Videos are processed on this computer with the installed models (Local models below)."));
        building = false;
    }

    // The address and sign-in rows belong to the provider choice only.
    void ShowProvider(bool shown)
    {
        if (building) return;
        provider.Visible = shown;
        if (!shown) { signIns.Visible = false; Ui.Show(problem, null); Ui.Show(status, null); }
    }

    void UseThisComputer()
    {
        ShowProvider(false);
        if (building || ProcessingSetup.Configured && !settings.Enabled) return;
        pending?.Cancel();
        ProcessingSetup.Choose(settings, false);
        applied();
    }

    // Reaches the typed site without touching the address in use, then offers its sign-in options.
    async Task ConnectAsync()
    {
        Ui.Show(problem, null); signIns.Visible = false;
        var typed = address.Text.Trim();
        if (typed.Length == 0) { Ui.Show(problem, "Enter your provider's address first."); return; }
        connect.Enabled = false;
        try
        {
            using var client = new HostSignIn();
            var offered = await client.ProvidersAsync(new HostingSettings { ServiceUrl = typed }.Endpoint(), default);
            await EditorPipeline.SwitchToMainThread();
            if (this.IsValid()) ShowSignIns(typed, offered);
        }
        catch (Exception failure) when (failure is ArgumentException or System.Net.Http.HttpRequestException or System.IO.InvalidDataException
            or System.Text.Json.JsonException or OperationCanceledException)
        {
            await EditorPipeline.SwitchToMainThread();
            if (problem.IsValid()) Ui.Show(problem, failure is OperationCanceledException ? "The provider did not answer in time." : failure.Message);
        }
        finally { if (connect.IsValid()) connect.Enabled = true; }
    }

    void ShowSignIns(string typed, IReadOnlyList<HostProvider> offered)
    {
        signIns.Layout.Clear(true);
        signIns.Layout.AddSpacingCell(labelWidth + 8);
        foreach (var option in offered)
            signIns.Layout.Add(new MocapButton("Sign in with " + option.Label, "login") { FixedHeight = Ui.RowHeight, Clicked = () => _ = SignInAsync(typed, option) });
        signIns.Layout.AddStretchCell();
        signIns.Visible = true;
    }

    // One sign-in at a time: the buttons stay disabled until it ends, so two attempts never
    // swap the address under each other.
    async Task SignInAsync(string typed, HostProvider option)
    {
        if (pending is not null) return;
        Ui.Show(problem, null); Ui.Show(status, "Finish signing in in your browser. This window continues by itself.");
        using var cancellation = new CancellationTokenSource(); pending = cancellation; var token = cancellation.Token;
        signIns.Enabled = connect.Enabled = false;
        var previous = settings.ServiceUrl;
        settings.ServiceUrl = typed;
        try
        {
            await ProcessingSetup.SignInAsync(settings, option, token);
            await EditorPipeline.SwitchToMainThread();
            if (token.IsCancellationRequested) { settings.ServiceUrl = previous; return; }
            ProcessingSetup.Choose(settings, true);
            applied();
        }
        catch (Exception failure) when (ProcessingSetup.IsSignInFailure(failure))
        {
            await EditorPipeline.SwitchToMainThread();
            settings.ServiceUrl = previous;
            if (!token.IsCancellationRequested && problem.IsValid())
                Ui.Show(problem, failure is OperationCanceledException ? "Sign-in timed out. Try again." : failure.Message);
        }
        finally
        {
            pending = null;
            if (status.IsValid()) { Ui.Show(status, null); signIns.Enabled = connect.Enabled = true; }
        }
    }

    public override void OnDestroyed()
    {
        pending?.Cancel();
        base.OnDestroyed();
    }
}
