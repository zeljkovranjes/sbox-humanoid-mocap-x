using Editor;

namespace HumanoidMocapX.EditorTools;

/// <summary>Secondary notes for the X settings card, in the editor's default colors.</summary>
static class SettingsText
{
    /// <summary>Secondary wrapped text, muted like the shared Capture details card.</summary>
    internal static Label Note(Widget parent, string text)
    {
        var label = new Label(text, parent) { WordWrap = true };
        label.SetStyles($"color: {Theme.TextLight.Hex}; font-size: 11px;");
        return label;
    }
}
