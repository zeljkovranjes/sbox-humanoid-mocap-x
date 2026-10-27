using System;
using System.IO;
using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

// First-load surface, styled after Humanoid Rigger's ModelDropArea: a dark-gray
// rounded box on the gray window, centered prompt and a green choose button.
sealed class VideoDropArea : Widget
{
    readonly Action<string> import;
    readonly VideoIcon icon;
    readonly Label title, help, or;
    readonly Button choose;
    readonly string prompt;
    int hover;
    public VideoDropArea(Widget parent, Action<string> import, Action chooseVideo) : base(parent)
    {
        this.import=import;AcceptDrops=true;Layout=Layout.Column();Layout.Margin=12;Layout.Spacing=8;
        Layout.AddStretchCell();
        var row=Layout.AddRow();row.AddStretchCell();var center=row.AddColumn();center.Spacing=12;
        icon=center.Add(new VideoIcon(this));
        title=center.Add(new Label.Subtitle("Video to motion"){Alignment=TextFlag.Center});
        prompt="Drag and drop a video here (.mp4, .mov, .m4v, .avi, .mkv, .webm)";
        help=center.Add(new Label(prompt,this){Alignment=TextFlag.Center});
        help.SetStyles($"color: {Theme.TextLight.Hex};");
        or=center.Add(new Label("or",this){Alignment=TextFlag.Center});
        var choice=center.AddRow();choice.Spacing=8;choice.AddStretchCell();
        choose=choice.Add(new Button.Primary("Choose Video"){Icon="video_file",Tint=Theme.Green,MinimumWidth=140,FixedHeight=32,Clicked=chooseVideo});
        choice.AddStretchCell();
        row.AddStretchCell();Layout.AddStretchCell();
    }
    /// <summary>Grays the area out and refuses drops: a blocked icon, "Currently disabled" and
    /// <paramref name="reason"/> replace the prompt and buttons, until called again with null.</summary>
    public void SetLocked(string reason)
    {
        var locked=reason is not null;
        Enabled=!locked;AcceptDrops=!locked;hover=0;
        icon.Icon=locked?"block":"video_file";
        title.Text=locked?"Currently disabled":"Video to motion";
        help.Text=reason??prompt;help.SetStyles($"color: {(locked?Theme.Yellow:Theme.TextLight).Hex};");
        or.Visible=choose.Visible=!locked;
        Update();
    }
    bool Supported(string path)=>Path.GetExtension(path).ToLowerInvariant() switch {
        ".mp4" or ".mov" or ".m4v" or ".avi" or ".mkv" or ".webm" => true,
        _ => false
    };
    // Copy is the one action every file manager offers; an X11 drag source (Dolphin through
    // XWayland) can refuse a drop that asks for Link, so the file never arrives.
    public override void OnDragHover(DragEvent e)
    {bool valid=DroppedFile(e.Data) is { } path&&Supported(path);hover=valid?1:-1;if(valid)e.Action=DropAction.Copy;Update();}
    public override void OnDragDrop(DragEvent e)
    {hover=0;if(DroppedFile(e.Data) is { } path&&Supported(path)){e.Action=DropAction.Copy;import(path);}Update();}
    /// <summary>The first dropped local file, read from its file URL. s&amp;box's text fallback
    /// strips the leading '/' from Linux paths. Null when the drag carries no local file.</summary>
    static string DroppedFile(DragData data)
    {
        if(data.Url is { IsFile: true } url)return url.LocalPath;
        // A '#' in a file name would otherwise start a URI fragment.
        var first=data.Text?.Split('\n')[0].Trim().Replace("#","%23");
        return Uri.TryCreate(first,UriKind.Absolute,out var uri)&&uri.IsFile?uri.LocalPath:null;
    }
    public override void OnDragLeave(){hover=0;Update();}
    protected override void OnPaint()
    {
        Paint.SetPen(hover==1?Theme.Green:hover<0?Theme.Red:Theme.ControlBackground.Lighten(.2f),1);
        Paint.SetBrush(hover==1?Theme.Green.WithAlpha(.06f):Enabled&&Paint.HasMouseOver?Theme.ControlBackground.Lighten(.3f):Theme.ControlBackground);
        Paint.DrawRect(LocalRect.Shrink(1),4);
    }
    sealed class VideoIcon : Widget
    {
        string icon="video_file";
        public VideoIcon(Widget parent):base(parent){FixedHeight=48;}
        public string Icon{get=>icon;set{icon=value;Update();}}
        protected override void OnPaint(){Paint.SetPen(Theme.TextLight);Paint.DrawIcon(new Rect((Width-40)*.5f,4,40,40),icon,40);}
    }
}
