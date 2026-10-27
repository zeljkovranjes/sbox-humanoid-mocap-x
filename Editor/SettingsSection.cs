using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>A heading that shows or hides its section, keeping the adjustments page
/// short. Click it, or Tab to it and press Space or Enter. Each section remembers
/// whether it was open.</summary>
sealed class SettingsSection : Widget
{
    const float HeaderHeight = 24, IconSize = 16;
    readonly string title, cookie;
    readonly Widget body;

    SettingsSection(Widget parent, string title, Widget body, bool openByDefault) : base(parent)
    {
        this.title = title; this.body = body; cookie = "HumanoidMocapX.Section." + title;
        FixedHeight = HeaderHeight; Cursor = CursorShape.Finger; FocusMode = FocusMode.TabOrClick;
        body.Visible = EditorCookie.Get(cookie, openByDefault);
    }

    /// <summary>Adds a section to host and returns its body; add the section's controls
    /// to the body's column layout.</summary>
    internal static Widget Add(Widget host, string title, bool openByDefault)
    {
        var body = new Widget(host) { Layout = Layout.Column() };
        body.Layout.Spacing = 6;
        host.Layout.Add(new SettingsSection(host, title, body, openByDefault));
        host.Layout.Add(body);
        return body;
    }

    protected override void OnMouseClick(MouseEvent e)
    {
        base.OnMouseClick(e);
        if (e.LeftMouseButton) Toggle();
    }

    protected override void OnKeyPress(KeyEvent e)
    {
        if (e.Key is KeyCode.Space or KeyCode.Return or KeyCode.Enter && !e.IsAutoRepeat) Toggle();
        else base.OnKeyPress(e);
    }

    void Toggle()
    {
        body.Visible = !body.Visible; EditorCookie.Set(cookie, body.Visible); Update();
    }

    protected override void OnPaint()
    {
        if (IsFocused) { Paint.SetPen(Theme.Primary, 1); Paint.ClearBrush(); Paint.DrawRect(LocalRect.Shrink(1), 3); }
        Paint.SetPen(Paint.HasMouseOver ? Theme.Text : Theme.TextLight);
        Paint.DrawIcon(new Rect(0, (HeaderHeight - IconSize) / 2, IconSize, IconSize), body.Visible ? "expand_more" : "chevron_right", IconSize);
        Paint.SetPen(Theme.Text); Paint.SetDefaultFont(weight: 600);
        Paint.DrawText(new Rect(IconSize + 4, 0, Width - IconSize - 4, HeaderHeight), title, TextFlag.LeftCenter);
    }
}
