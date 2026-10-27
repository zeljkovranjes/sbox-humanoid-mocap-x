using System;
using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>A small uppercase rounded label tinted by its color: WHITELISTED, PRIORITY, QUEUED.</summary>
sealed class Pill : Widget
{
    readonly string text;
    readonly Color color;

    internal Pill(Widget parent, string text, Color color) : base(parent)
    {
        this.text = text.ToUpperInvariant(); this.color = color;
        FixedHeight = 18; FixedWidth = MathF.Ceiling(6.2f * this.text.Length + 16);
    }

    protected override void OnPaint()
    {
        Paint.Antialiasing = true; Paint.ClearPen();
        Paint.SetBrush(color.WithAlpha(.16f)); Paint.DrawRect(LocalRect, Height / 2);
        Paint.SetPen(color); Paint.SetDefaultFont(7, 600);
        Paint.DrawText(LocalRect, text, TextFlag.Center);
    }
}

/// <summary>A slim progress bar; Value is 0 to 1.</summary>
sealed class UiProgress : Widget
{
    float value;

    internal UiProgress(Widget parent) : base(parent) { FixedHeight = 3; }

    internal float Value { get => value; set { this.value = Math.Clamp(value, 0, 1); Update(); } }

    protected override void OnPaint()
    {
        Paint.Antialiasing = true; Paint.ClearPen();
        Paint.SetBrush(Color.White.WithAlpha(.08f)); Paint.DrawRect(LocalRect, 1.5f);
        if (value <= 0) return;
        Paint.SetBrush(Ui.Accent); Paint.DrawRect(new Rect(0, 0, Width * value, Height), 1.5f);
    }
}

/// <summary>Small uppercase caption with a hairline to the right, separating groups.</summary>
sealed class SectionLine : Widget
{
    readonly string text;
    string count;

    internal SectionLine(Widget parent, string text) : base(parent) { this.text = text.ToUpperInvariant(); FixedHeight = 18; }

    internal string Count { set { count = value; Update(); } }

    protected override void OnPaint()
    {
        Paint.SetDefaultFont(7, 600); Paint.SetPen(Theme.TextLight);
        var x = Paint.MeasureText(text).x + 8;
        Paint.DrawText(new Rect(0, 0, x, Height), text, TextFlag.LeftCenter);
        if (!string.IsNullOrEmpty(count))
        {
            Paint.SetPen(Theme.TextLight.WithAlpha(.7f));
            var size = Paint.MeasureText(count).x;
            Paint.DrawText(new Rect(x - 2, 0, size + 2, Height), count, TextFlag.LeftCenter);
            x += size + 6;
        }
        Paint.SetPen(Ui.CardEdge, 1);
        Paint.DrawLine(new Vector2(x, Height / 2), new Vector2(Width, Height / 2));
    }
}

/// <summary>An icon button with a small count badge, for the queue.</summary>
sealed class BadgeButton : Widget
{
    readonly string icon;
    int count;
    internal Action Clicked;

    internal BadgeButton(Widget parent, string icon, Action clicked, string tooltip) : base(parent)
    {
        this.icon = icon; Clicked = clicked; ToolTip = tooltip;
        FixedSize = Ui.ControlHeight; Cursor = CursorShape.Finger; MouseTracking = true;
    }

    internal int Count { set { if (count == value) return; count = value; Update(); } }

    protected override void OnMouseClick(MouseEvent e)
    {
        base.OnMouseClick(e);
        if (e.LeftMouseButton) Clicked?.Invoke();
    }

    protected override void OnMouseEnter() { base.OnMouseEnter(); Update(); }
    protected override void OnMouseLeave() { base.OnMouseLeave(); Update(); }

    protected override void OnPaint()
    {
        Paint.Antialiasing = true; Paint.ClearPen();
        Paint.SetBrush(Paint.HasMouseOver ? Ui.Accent.WithAlpha(.2f) : MocapButton.Fill);
        Paint.DrawRect(LocalRect, Ui.Radius);
        Paint.SetPen(Paint.HasMouseOver ? Ui.Accent : Theme.Text);
        Paint.DrawIcon(LocalRect, icon, 16);
        if (count <= 0) return;
        var label = count > 9 ? "9+" : count.ToString();
        var badge = new Rect(Width - 14, -1, 15, 15);
        Paint.SetPen(Theme.WindowBackground, 2); Paint.SetBrush(Ui.Accent); Paint.DrawRect(badge, 7.5f);
        Paint.SetPen(Ui.OnAccent); Paint.SetDefaultFont(7, 700); Paint.DrawText(badge, label, TextFlag.Center);
    }
}
