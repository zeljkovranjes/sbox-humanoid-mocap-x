using System;
using System.Threading;
using System.Threading.Tasks;
using Editor;

namespace HumanoidMocapX.EditorTools;

/// <summary>The saved processing choice (this computer or a provider) and browser sign-in, shared by
/// the settings' Processing section, the account bar and the Account page.</summary>
static class ProcessingSetup
{
    internal const string SetupCookie = "HumanoidMocapX.SetupDone", HostedCookie = "HumanoidMocapX.Hosted",
        AddressCookie = "HumanoidMocapX.HostUrl", ProviderCookie = "HumanoidMocapX.Provider", ProviderLabelCookie = "HumanoidMocapX.ProviderLabel";

    internal const string SetupMessage = "Finish setup first: open settings (top right) and choose where videos are processed.";

    /// <summary>Why capture on this computer waits: <paramref name="missing"/> names the first absent model.</summary>
    internal static string ModelsMessage(string missing)
        => $"Download the models first (missing: {missing}): open settings (top right) › Processing › Setup › Local models.";

    /// <summary>True once this computer or a signed-in provider has been chosen.</summary>
    internal static bool Configured => EditorCookie.Get(SetupCookie, false);

    /// <summary>Saves the choice; the window then rebuilds around it.</summary>
    internal static void Choose(HostingSettings settings, bool hosted)
    {
        settings.Enabled = hosted;
        EditorCookie.Set(HostedCookie, hosted); EditorCookie.Set(SetupCookie, true);
        if (hosted) EditorCookie.Set(AddressCookie, settings.ServiceUrl);
    }

    /// <summary>Signs in through the browser and remembers the provider for one-click sign-in
    /// next time. Throws the sign-in's own errors.</summary>
    internal static async Task SignInAsync(HostingSettings settings, HostProvider provider, CancellationToken token)
    {
        using var client = new HostSignIn();
        await client.SignInAsync(settings, provider.Name, Ui.OpenBrowser, token);
        await EditorPipeline.SwitchToMainThread();
        EditorCookie.Set(ProviderCookie, provider.Name); EditorCookie.Set(ProviderLabelCookie, provider.Label);
    }

    /// <summary>The provider last signed in with, or null before the first sign-in.</summary>
    internal static HostProvider SavedProvider()
        => EditorCookie.Get(ProviderCookie, "") is { Length: > 0 } name ? new HostProvider(name, EditorCookie.Get(ProviderLabelCookie, name)) : null;

    /// <summary>Failures a browser sign-in reports to the user rather than throws: bad address or
    /// reply, network errors, timeouts and cancellation.</summary>
    internal static bool IsSignInFailure(Exception failure) => failure is ArgumentException or System.Net.Http.HttpRequestException
        or System.IO.InvalidDataException or System.Text.Json.JsonException or OperationCanceledException or System.Net.Sockets.SocketException;
}
