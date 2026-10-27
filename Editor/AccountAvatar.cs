using System;
using System.Net.Http;
using System.Threading.Tasks;
using Editor;
using HumanoidMocapX.EditorTools;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>A round account picture, the name's initial, or an icon when neither is known. Pictures
/// load only over HTTPS from the image CDNs providers use, without redirects, up to 1 MiB.</summary>
sealed class AccountAvatar : Widget
{
    const long MaximumBytes = 1024 * 1024;
    // Avatars are tiny public images; one client for the editor session avoids socket churn.
    static readonly HttpClient Client = new(new HttpClientHandler { AllowAutoRedirect = false }) { Timeout = TimeSpan.FromSeconds(10) };
    // Only the image CDNs providers use; the website drops other avatar URLs too.
    static readonly System.Text.RegularExpressions.Regex Hosts = new(@"^(cdn\.discordapp\.com|[a-z0-9-]+\.googleusercontent\.com)$",
        System.Text.RegularExpressions.RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100));
    readonly float diameter;
    Pixmap image;
    string initial, icon, shown;

    internal AccountAvatar(Widget parent, float diameter) : base(parent) { this.diameter = diameter; FixedSize = diameter; }

    /// <summary>Shows an icon instead of an account.</summary>
    internal void ShowIcon(string fallbackIcon) { shown = null; image = null; initial = null; icon = fallbackIcon; Update(); }

    /// <summary>Shows the name's initial now and the picture at <paramref name="url"/> once it loads.</summary>
    internal void ShowAccount(string url, string name)
    {
        initial = name is { Length: > 0 } ? name[..1].ToUpperInvariant() : null; icon = "person";
        if (url == shown && (image is not null || url is null)) { Update(); return; }
        shown = url; image = null; Update();
        _ = LoadAsync(url);
    }

    async Task LoadAsync(string url)
    {
        Pixmap picture = null;
        if (Uri.TryCreate(url, UriKind.Absolute, out var uri) && uri.Scheme == Uri.UriSchemeHttps && uri.IsDefaultPort && Hosts.IsMatch(uri.Host))
        {
            try
            {
                using var response = await Client.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead);
                if (response.IsSuccessStatusCode && response.Content.Headers.ContentLength is null or <= MaximumBytes)
                {
                    await response.Content.LoadIntoBufferAsync(MaximumBytes);
                    var bytes = await response.Content.ReadAsByteArrayAsync();
                    await EditorPipeline.SwitchToMainThread();
                    picture = Bitmap.CreateFromBytes(bytes) is { } bitmap ? Pixmap.FromBitmap(bitmap) : null;
                }
            }
            // A missing picture only keeps the initial; the profile itself is unaffected.
            catch (Exception failure) when (failure is HttpRequestException or TaskCanceledException or InvalidOperationException or ArgumentException)
            { Log.Info($"Humanoid Mocap X: profile picture unavailable ({failure.GetType().Name})."); }
        }
        await EditorPipeline.SwitchToMainThread();
        if (this.IsValid() && shown == url) { image = picture; Update(); }
    }

    protected override void OnPaint()
    {
        Paint.Antialiasing = true;
        var bounds = new Rect(0, 0, diameter, diameter);
        if (image is not null) { Paint.Draw(bounds, image, 1, diameter / 2); return; }
        Paint.ClearPen(); Paint.SetBrush(Ui.Accent.WithAlpha(.16f)); Paint.DrawRect(bounds, diameter / 2);
        Paint.SetPen(Ui.Accent);
        if (initial is null) Paint.DrawIcon(bounds, icon ?? "person", diameter * .58f);
        else { Paint.SetDefaultFont(diameter * .32f, 700); Paint.DrawText(bounds, initial, TextFlag.Center); }
    }
}
