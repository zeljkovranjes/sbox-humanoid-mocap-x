using System;
using System.IO;
using System.Linq;
using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>The queue popup: each job with its place in line, progress and time estimates, cancel
/// for waiting jobs and open for finished ones, and an add button for several videos at once.</summary>
sealed class QueuePopup : PopupWidget
{
    const string VideoFilter = "Video files (*.mp4 *.mov *.m4v *.avi *.mkv *.webm)";
    readonly QueueTracker tracker;
    readonly SectionLine title;
    readonly Widget list;
    readonly Label problem;

    internal QueuePopup(Widget parent, QueueTracker tracker) : base(parent)
    {
        this.tracker = tracker;
        FixedWidth = 380; MaximumHeight = 460;
        Layout = Layout.Column(); Layout.Margin = 12; Layout.Spacing = 8;
        var header = Layout.AddRow(); header.Spacing = 6;
        title = header.Add(new SectionLine(this, "Queue"), 1);
        header.Add(Ui.Icon(this, "playlist_add", () => AddVideos(tracker), "Add videos to the queue", Ui.RowHeight));
        problem = Layout.Add(Ui.Error(this));
        var scroll = Layout.Add(new ScrollArea(this), 1);
        list = new Widget(scroll) { Layout = Layout.Column() };
        list.Layout.Spacing = 2;
        scroll.Canvas = list;
        tracker.Changed += Rebuild;
        Rebuild();
    }

    /// <summary>Opens the file picker and queues every selected video.</summary>
    internal static void AddVideos(QueueTracker tracker)
    {
        var dialog = new FileDialog(null) { Title = "Add videos to the queue" };
        dialog.SetFindExistingFiles(); dialog.SetModeOpen(); dialog.SetNameFilter(VideoFilter);
        if (dialog.Execute()) tracker.Add(dialog.SelectedFiles);
    }

    void Rebuild()
    {
        if (!this.IsValid()) return;
        title.Count = tracker.Active > 0 ? tracker.Active.ToString() : null;
        Ui.Show(problem, tracker.Problem);
        list.Layout.Clear(true);
        if (!tracker.Ready) { list.Layout.Add(Empty("lock", "Sign in to see your queue.")); return; }
        if (tracker.Uploading is { } uploading) list.Layout.Add(new JobRow(list, Path.GetFileName(uploading), "Uploading…", Ui.Accent));
        foreach (var job in tracker.Jobs.Reverse()) list.Layout.Add(Row(job));
        var slots = tracker.Limits?.MaximumActiveJobs ?? 1;
        foreach (var video in tracker.Waiting)
            list.Layout.Add(new JobRow(list, Path.GetFileName(video), $"Waits for a free slot · {slots} at a time", Theme.TextLight))
                .Action("close", "Remove", () => tracker.Remove(video));
        if (tracker.Jobs.Length == 0 && tracker.Waiting.Count == 0 && tracker.Uploading is null)
            list.Layout.Add(Empty("movie", "Nothing queued. Drop a video in the workspace, or add several here."));
        else if (tracker.Limits?.MaximumClipSeconds is { } clip)
            list.Layout.Add(Ui.Muted(new Label($"Added videos are cut to the first {HostedCapture.Duration((int)clip)}.", list), small: true));
        AdjustSize();
    }

    JobRow Row(HostedJob job)
    {
        var (color, line) = job.State switch
        {
            "queued" => (Theme.Yellow, Place(job)),
            "processing" => (Ui.Accent, job.Progress is { } done ? $"{Math.Round(done * 100)}%" + Left(job) : "Starting" + Left(job)),
            "completed" => (Ui.Accent, job.ExpiresAt is { } expires ? "Ready · deleted in " + HostedCapture.Duration((int)(expires - DateTimeOffset.UtcNow).TotalSeconds) : "Ready"),
            "failed" => (Theme.Red, job.Message),
            "cancelled" => (Theme.TextLight, "Cancelled"),
            _ => (Theme.TextLight, job.Message)
        };
        var row = new JobRow(list, job.Name ?? "Untitled video", line, color, job.Priority);
        if (job.State == "processing") row.Progress((float)(job.Progress ?? 0), job.Message);
        if (job.State == "completed") row.Action("file_open", "Open in the workspace", () => _ = tracker.OpenAsync(job));
        if (job.State != "processing")
            row.Action(QueueTracker.Finished(job) ? "delete" : "close", QueueTracker.Finished(job) ? "Remove" : "Cancel", () => _ = tracker.CancelAsync(job));
        else row.ToolTip = "A job that is processing cannot be cancelled. Its result is deleted 24 hours after it finishes.";
        return row;
    }

    static string Place(HostedJob job)
        => (job.QueuePosition is { } position ? $"#{position} in line" : "Waiting")
            + (job.SecondsUntilStart is { } start ? " · starts in " + HostedCapture.Duration(start) : "") + Left(job);

    static string Left(HostedJob job) => job.SecondsRemaining is { } left ? " · about " + HostedCapture.Duration(left) + " left" : "";

    Widget Empty(string icon, string text)
    {
        var empty = new Widget(list) { Layout = Layout.Column() };
        empty.Layout.Margin = new Sandbox.UI.Margin(0, 16, 0, 16); empty.Layout.Spacing = 6;
        var iconRow = empty.Layout.AddRow(); iconRow.AddStretchCell();
        iconRow.Add(new IconCell(empty, icon, Theme.TextLight.WithAlpha(.5f), 28)); iconRow.AddStretchCell();
        empty.Layout.Add(Ui.Muted(new Label(text, empty) { WordWrap = true, Alignment = TextFlag.Center }));
        return empty;
    }

    public override void OnDestroyed()
    {
        tracker.Changed -= Rebuild;
        base.OnDestroyed();
    }

    /// <summary>One job: a status dot, its name and pills, a status line, and icon actions that
    /// appear on the right.</summary>
    sealed class JobRow : Widget
    {
        readonly Color color;
        readonly Layout text, actions;

        internal JobRow(Widget parent, string name, string status, Color color, bool priority = false) : base(parent)
        {
            this.color = color; MouseTracking = true;
            Layout = Layout.Row(); Layout.Margin = new Sandbox.UI.Margin(20, 6, 4, 6); Layout.Spacing = 6;
            text = Layout.AddColumn(1); text.Spacing = 2;
            var top = text.AddRow(); top.Spacing = 6;
            top.Add(Ui.Bold(new Label(name.Length > 34 ? name[..33] + "…" : name, this) { ToolTip = name }));
            if (priority) top.Add(new Pill(this, "Priority", Theme.Yellow));
            top.AddStretchCell();
            text.Add(Ui.Muted(new Label(status, this) { WordWrap = true }, small: true));
            actions = Layout.AddRow(); actions.Spacing = 4;
        }

        internal JobRow Action(string icon, string tooltip, Action clicked)
        {
            actions.Add(Ui.Icon(this, icon, clicked, tooltip, Ui.RowHeight));
            return this;
        }

        internal void Progress(float value, string message)
        {
            text.Add(new UiProgress(this) { Value = value });
            text.Add(Ui.Muted(new Label(message, this) { WordWrap = true }, small: true));
        }

        protected override void OnMouseEnter() { base.OnMouseEnter(); Update(); }
        protected override void OnMouseLeave() { base.OnMouseLeave(); Update(); }

        protected override void OnPaint()
        {
            Paint.Antialiasing = true; Paint.ClearPen();
            if (Paint.HasMouseOver) { Paint.SetBrush(Color.White.WithAlpha(.04f)); Paint.DrawRect(LocalRect, Ui.Radius); }
            Paint.SetBrush(color); Paint.DrawRect(new Rect(7, 11, 7, 7), 3.5f);
        }
    }
}

/// <summary>A tinted material icon in a square cell.</summary>
sealed class IconCell : Widget
{
    readonly string icon;
    readonly Color color;

    internal IconCell(Widget parent, string icon, Color color, float size = 20) : base(parent)
    {
        this.icon = icon; this.color = color; FixedSize = size;
    }

    protected override void OnPaint()
    {
        Paint.SetPen(color); Paint.DrawIcon(LocalRect, icon, Width - 4);
    }
}
