using System;
using Editor;
using HumanoidMocapX.EditorTools;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>The X window's look, shared with the weapon importer and Collaborator: rounded dark
/// cards on the window gray, a green accent, framed inputs, 30 px header controls and 26 px row
/// controls.</summary>
static class Ui
{
    internal const float ControlHeight = 30, RowHeight = 26, Gap = 10, Padding = 16, Radius = 4;

    internal static Color Accent => Theme.Green;
    // Text on the accent fill: near-black with a green cast, so labels stay readable.
    internal static Color OnAccent => new(.06f, .09f, .03f);
    internal static Color CardFill => Theme.ControlBackground;
    internal static Color CardEdge => Theme.ControlBackground.Lighten(.25f);

    /// <summary>A text field with the buttons' fill and hairline edge, so inputs and buttons match.</summary>
    internal static T Framed<T>(T input, float height = ControlHeight) where T : Widget
    {
        input.SetStyles($"background-color: {MocapButton.Fill.Hex}; border: 1px solid {MocapButton.Edge.Hex}; border-radius: {Radius}px; padding-left: 6px;");
        input.FixedHeight = height;
        return input;
    }

    internal static Label Muted(Label label, bool small = false)
    {
        label.SetStyles($"color: {Theme.TextLight.Hex};" + (small ? " font-size: 11px;" : ""));
        return label;
    }

    internal static Label Title(Label label, float size = 15)
    {
        label.SetStyles($"font-weight: 700; font-size: {size}px;");
        return label;
    }

    internal static Label Bold(Label label, float size = 0)
    {
        label.SetStyles(size > 0 ? $"font-weight: 600; font-size: {size}px;" : "font-weight: 600;");
        return label;
    }

    internal static Label Error(Widget parent) => Colored(new Label("", parent) { WordWrap = true, Visible = false }, Theme.Red);

    internal static Label Colored(Label label, Color color, bool bold = false)
    {
        label.SetStyles($"color: {color.Hex};" + (bold ? " font-weight: 600;" : ""));
        return label;
    }

    internal static void Show(Label label, string text)
    {
        if (!label.IsValid()) return;
        label.Text = text ?? ""; label.Visible = !string.IsNullOrEmpty(text);
    }

    internal static Button Primary(string text, string icon, Action clicked)
        => new Button.Primary(text) { Icon = icon, Tint = Accent, FixedHeight = ControlHeight, Clicked = clicked };

    internal static Button Secondary(string text, string icon, Action clicked)
        => new MocapButton(text, icon) { FixedHeight = RowHeight, Clicked = clicked };

    /// <summary>A square icon button in the secondary style; 30 px in headers, 26 px in rows.</summary>
    internal static IconButton Icon(Widget parent, string icon, Action clicked, string tooltip, float size = ControlHeight)
        => new MocapIconButton(icon, clicked, parent) { FixedSize = size, ToolTip = tooltip };

    internal static void OpenBrowser(string url)
    {
        // The OS default browser; editor code may start the shell handler for http(s) URLs.
        System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(url) { UseShellExecute = true });
    }
}
