using System;
using Editor;
using HumanoidMocap.EditorTools;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>Third-person FlowHMR capture using Humanoid Mocap's preview, rigs and exporter.</summary>
public static partial class MocapWindow
{
    public const string Title = "Humanoid Mocap X";
    const float LabelWidth = 130;
    static Dialog window;

    [Event("tools.editorwindow.createview")]
    static void Register(Menu menu) => EditorWindow.DockManager.RegisterDockType(new DockManager.DockInfo
    {
        Title = Title,
        Icon = "videocam",
        CreateAction = () => { Open(); return null; }
    });

    [Event("tools.editorwindow.postcreateview")]
    static void Configure(Menu menu)
    {
        var option = menu.GetOption(Title);
        if (option is null) return;
        option.Toggled = null; option.Checkable = false; option.Triggered = Open;
    }

    /// <summary>Opens or raises the X workspace. Human is selected initially.</summary>
    public static void Open()
    {
        if (window.IsValid()) { window.Show(); window.Window.Raise(); return; }
        var options = new CaptureOptions
        {
            Model = EditorCookie.Get("HumanoidMocapX.Model", FlowModel.Latest),
            CameraInterval = EditorCookie.Get("HumanoidMocapX.CameraInterval", 10),
            Lanes = EditorCookie.Get("HumanoidMocapX.Lanes", CaptureLanes.Auto)
        };
        if (options.Lanes is < CaptureLanes.Auto or > CaptureLanes.Maximum) options.Lanes = CaptureLanes.Auto;
        if (options.CameraInterval is not (1 or 10 or 30)) options.CameraInterval = 10;
        if (options.Model is not (FlowModel.Base or FlowModel.Latest)) options.Model = FlowModel.Latest;
        window = new CaptureDialog(options);
        window.Show();
    }

    /// <summary>The X part of Mocap adjustments: where videos are processed, the motion model and,
    /// for this computer, frames at once and model downloads.</summary>
    internal static void Settings(Widget host, CaptureOptions options, HostingSettings hosting, Action applied, Action modelsChanged)
    {
        // Where captures run, and the only place to set or change it.
        var where = SettingsSection.Add(host, "Processing", openByDefault: true);
        where.Layout.Add(new ProcessingSection(where, hosting, applied, LabelWidth));
        // Built in Local models below; Base or Latest decides which FlowHMR file it checks.
        ModelDownloads list = null;
        var section = SettingsSection.Add(host, "Motion model", openByDefault: true);
        var models = Row(section, "FlowHMR model", Ui.Framed(new ComboBox(section)));
        foreach (var model in new[] { FlowModel.Latest, FlowModel.Base })
            models.AddItem(model.ToString(), onSelected: () =>
            {
                options.Model = model; EditorCookie.Set("HumanoidMocapX.Model", model); modelsChanged(); list?.Refresh();
            }, selected: model == options.Model);
        var sampling = Row(section, "Camera sampling", Ui.Framed(new ComboBox(section)
        {
            ToolTip = "Sparser camera sampling uses less GPU memory but can lose accuracy with fast camera motion. Animation stays at 30 fps."
        }));
        foreach (var (interval, label) in new[] { (1, "Every frame"), (10, "Every 10th frame"), (30, "Every 30th frame") })
            sampling.AddItem(label, onSelected: () =>
            {
                options.CameraInterval = interval; EditorCookie.Set("HumanoidMocapX.CameraInterval", interval);
            }, selected: interval == options.CameraInterval);
        if (hosting.Enabled) return;
        var capacity = FlowVideoProcessor.LaneCapacity();
        var lanes = Row(section, "Parallel frames", Ui.Framed(new ComboBox(section)
        {
            ToolTip = "Frames processed at once during local capture. More is faster and uses more GPU memory; results are identical. Any choice is capped to what this GPU's free memory allows when capture starts."
        }));
        for (var count = CaptureLanes.Auto; count <= CaptureLanes.Maximum; count++)
        {
            var choice = count;
            var label = choice == CaptureLanes.Auto ? $"Auto · up to {capacity} on this GPU"
                : choice > capacity ? $"{choice} · capped to {capacity} on this GPU" : choice.ToString();
            lanes.AddItem(label, onSelected: () =>
            {
                options.Lanes = choice; EditorCookie.Set("HumanoidMocapX.Lanes", choice);
            }, selected: choice == options.Lanes);
        }
        var downloads = SettingsSection.Add(host, "Local models", openByDefault: false);
        list = downloads.Layout.Add(new ModelDownloads(downloads, options, modelsChanged));
    }

    /// <summary>Adds a labelled row whose control takes the remaining width.</summary>
    static T Row<T>(Widget host, string label, T control) where T : Widget
    {
        var row = host.Layout.AddRow(); row.Spacing = 8;
        row.Add(new Label(label, host) { FixedWidth = LabelWidth });
        return row.Add(control, 1);
    }
}
