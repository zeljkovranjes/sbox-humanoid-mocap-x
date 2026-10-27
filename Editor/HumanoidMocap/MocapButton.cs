using System;
using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>The secondary button: a fill a step lighter than the card, a hairline edge, light
/// text and a lighter hover, in the weapon importer's style. A drop-in <see cref="Button"/>,
/// so callers keep its Text, Icon, Clicked and layout behaviour.</summary>
public class MocapButton : Button
{
    const float Radius = 4, IconSize = 16, IconGap = 6;

    public MocapButton(string title, Widget parent = null) : base(title, parent) { }

    public MocapButton(string title, string icon, Widget parent = null) : base(title, icon, parent) { }

    /// <summary>The fill under the secondary button and icon buttons.</summary>
    public static Color Fill => Color.Lerp(Theme.ControlBackground.WithAlpha(1), Color.White, .07f);

    /// <summary>The secondary button's hairline edge.</summary>
    public static Color Edge => Color.Lerp(Theme.ControlBackground.WithAlpha(1), Color.White, .15f);

    protected override void OnPaint()
    {
        Paint.Antialiasing = true;
        var hover = Enabled && Paint.HasMouseOver;
        Paint.SetPen(Edge, 1);
        Paint.SetBrush(Enabled && Paint.HasPressed ? Fill.Lighten(.15f) : hover ? Fill.Lighten(.3f) : Fill);
        Paint.DrawRect(LocalRect.Shrink(.5f), Radius);
        Paint.SetDefaultFont();
        var text = Paint.MeasureText(Text);
        var hasIcon = !string.IsNullOrWhiteSpace(Icon);
        var width = text.x + (hasIcon ? IconSize + IconGap : 0);
        var x = LocalRect.Center.x - width / 2;
        Paint.SetPen(Enabled ? Theme.Text : Theme.TextDisabled);
        if (hasIcon)
        {
            Paint.DrawIcon(new Rect(x, LocalRect.Center.y - IconSize / 2, IconSize, IconSize), Icon, IconSize);
            x += IconSize + IconGap;
        }
        Paint.DrawText(new Rect(x, LocalRect.Top, text.x, LocalRect.Height), Text, TextFlag.Center);
    }
}

/// <summary>A square icon button with the secondary button's fill and a green active state.</summary>
public class MocapIconButton : IconButton
{
    const float BadgeSize = 9;
    bool badge;

    public MocapIconButton(string icon, Action onClick = null, Widget parent = null) : base(icon, onClick, parent)
    {
        IconSize = 16; Background = MocapButton.Fill; Foreground = Theme.Text;
        BackgroundActive = Theme.Green.WithAlpha(.2f); ForegroundActive = Theme.Green;
    }

    /// <summary>A small red dot on the top-right corner: something behind this button needs attention.</summary>
    public bool Badge { get => badge; set { if (badge == value) return; badge = value; Update(); } }

    protected override void OnPaint()
    {
        base.OnPaint();
        if (!badge) return;
        Paint.Antialiasing = true;
        Paint.SetPen(Theme.WidgetBackground, 2); Paint.SetBrush(Theme.Red);
        Paint.DrawRect(new Rect(Width - BadgeSize - 1, 1, BadgeSize, BadgeSize), BadgeSize / 2);
    }
}
