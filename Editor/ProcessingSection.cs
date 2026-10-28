using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>Settings › Processing › Setup › Processing: where captures run. This computer is saved at
/// once. A provider needs its address, then a browser sign-in with an option its site offers. A
/// processing machine needs its SSH address and worker path, then a successful Check. Until then the
/// current choice and addresses stay in use. <c>applied</c> runs after a choice is saved, so the
/// window can rebuild around it.</summary>
sealed class ProcessingSection : Widget
{
    static readonly TimeSpan CheckTimeout = TimeSpan.FromSeconds(30);
    readonly HostingSettings settings;
    readonly SshSettings machine;
    readonly Action applied;
    readonly float labelWidth;
    readonly Widget provider, remote;
    readonly LineEdit address, machineAddress, worker;
    readonly Button connect, check;
    readonly Widget signIns;
    readonly Label status, problem;
    CancellationTokenSource pending;
    bool building = true;

    internal ProcessingSection(Widget parent, HostingSettings settings, SshSettings machine, Action applied, float labelWidth) : base(parent)
    {
        this.settings = settings; this.machine = machine; this.applied = applied; this.labelWidth = labelWidth;
        Layout = Layout.Column(); Layout.Spacing = 6;
        var configured = ProcessingSetup.Configured; var hosted = configured && settings.Enabled; var remoteChosen = configured && !hosted && machine.Enabled;
        var row = Layout.AddRow(); row.Spacing = 8;
        row.Add(new Label("Runs on", this) { FixedWidth = labelWidth });
        var mode = row.Add(Ui.Framed(new ComboBox(this)), 1);
        if (!configured) mode.AddItem("Not set up · choose one", "help_outline", () => Show(null), selected: true);
        mode.AddItem("This computer", "memory", UseThisComputer,
            LocalCapture.Availability ?? "Your GPU processes videos; nothing leaves this computer.", configured && !hosted && !remoteChosen);
        mode.AddItem("A provider", "cloud", () => Show(provider),
            "A provider processes videos for you after you sign in.", hosted);
        mode.AddItem("Processing machine (SSH)", "dns", () => Show(remote),
            "Your own Linux machine processes videos with humanoid-mocap-worker over SSH.", remoteChosen);

        provider = Layout.Add(new Widget(this) { Layout = Layout.Column(), Visible = hosted });
        provider.Layout.Spacing = 6;
        var line = provider.Layout.AddRow(); line.Spacing = 8;
        line.Add(new Label("Address", provider) { FixedWidth = labelWidth });
        address = line.Add(Ui.Framed(new LineEdit(provider) { Text = settings.ServiceUrl, PlaceholderText = "mocap.example.com", MaxLength = 2048 }), 1);
        connect = line.Add(Ui.Secondary(hosted ? "Reconnect" : "Connect", "link", () => _ = ConnectAsync()));
        address.ReturnPressed += () => _ = ConnectAsync();
        signIns = provider.Layout.Add(new Widget(provider) { Layout = Layout.Row(), Visible = false });
        signIns.Layout.Spacing = 8;

        remote = Layout.Add(new Widget(this) { Layout = Layout.Column(), Visible = remoteChosen });
        remote.Layout.Spacing = 6;
        var host = remote.Layout.AddRow(); host.Spacing = 8;
        host.Add(new Label("Address", remote) { FixedWidth = labelWidth });
        machineAddress = host.Add(Ui.Framed(new LineEdit(remote) { Text = machine.Address, PlaceholderText = "user@192.168.1.20", MaxLength = 512 }), 1);
        check = host.Add(Ui.Secondary("Check", "network_check", () => _ = CheckAsync()));
        var path = remote.Layout.AddRow(); path.Spacing = 8;
        path.Add(new Label("Worker", remote) { FixedWidth = labelWidth });
        worker = path.Add(Ui.Framed(new LineEdit(remote) { Text = machine.Worker, PlaceholderText = SshSettings.DefaultWorker, MaxLength = 1024 }), 1);
        machineAddress.ReturnPressed += () => _ = CheckAsync(); worker.ReturnPressed += () => _ = CheckAsync();

        status = Layout.Add(Ui.Muted(new Label("", this) { WordWrap = true, Visible = false }));
        problem = Layout.Add(Ui.Error(this));
        Layout.Add(SettingsText.Note(this, !configured ? "Choose this computer, a provider or a processing machine before capturing videos."
            : hosted ? "Your provider receives the videos you capture and your account ID. Results are deleted 24 hours after processing."
            : remoteChosen ? $"Videos are copied to {machine.Address} over SSH and processed there; each job's files are removed afterwards. Sign-in must work without a password prompt (an SSH key)."
            : LocalCapture.Availability ?? "Videos are processed on this computer with the installed models (Local models below)."));
        building = false;
    }

    // The address rows belong to the provider and processing machine choices; null shows neither.
    void Show(Widget panel)
    {
        if (building) return;
        provider.Visible = panel == provider; remote.Visible = panel == remote;
        signIns.Visible = false; Ui.Show(problem, null); Ui.Show(status, null);
    }

    void UseThisComputer()
    {
        Show(null);
        if (building || ProcessingSetup.Configured && !settings.Enabled && !machine.Enabled) return;
        pending?.Cancel();
        ProcessingSetup.Choose(settings, machine, false);
        applied();
    }

    // Checks the typed machine without touching the one in use; a ready machine becomes the choice.
    async Task CheckAsync()
    {
        if (pending is not null) return;
        Ui.Show(problem, null);
        var typed = new SshSettings { Address = machineAddress.Text.Trim(), Worker = worker.Text.Trim() };
        if (typed.Problem is { } invalid) { Ui.Show(problem, invalid); return; }
        using var cancellation = new CancellationTokenSource(CheckTimeout); pending = cancellation; var token = cancellation.Token;
        check.Enabled = false; Ui.Show(status, "Checking " + typed.Address + "…");
        try
        {
            await SshCapture.CheckAsync(typed, token);
            await EditorPipeline.SwitchToMainThread();
            if (!this.IsValid()) return;
            Ui.Show(status, typed.Address + " is ready: the worker starts and FFmpeg is installed.");
            if (ProcessingSetup.Configured && machine.Enabled && machine.Address == typed.Address && machine.Worker == typed.Worker) return;
            machine.Address = typed.Address; machine.Worker = typed.Worker;
            ProcessingSetup.Choose(settings, machine, false, remote: true);
            applied();
        }
        catch (Exception failure) when (failure is InvalidOperationException or ArgumentException or OperationCanceledException or System.IO.IOException)
        {
            await EditorPipeline.SwitchToMainThread();
            Ui.Show(status, null);
            Ui.Show(problem, failure is OperationCanceledException ? "The processing machine did not answer in time." : failure.Message);
        }
        finally
        {
            pending = null;
            if (check.IsValid()) check.Enabled = true;
        }
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
            ProcessingSetup.Choose(settings, machine, true);
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
