using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Editor;
using HumanoidMocapX.Core.Formats.Fbx;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Core.Skeleton;
using HumanoidMocapX.Core.Target;
using Sandbox;
using HumanoidMocapX.Core;

namespace HumanoidMocapX.EditorTools;

public sealed partial class RetargetWindow
{
    bool _exportingMotion;

    async void PickMotionExport(bool retarget)
    {
        var dialog = new FileDialog(null) { Title = "Export armature and animation", DefaultSuffix = ".fbx" };
        dialog.SelectFile("capture.fbx"); dialog.SetFindFile(); dialog.SetModeSave(); dialog.SetNameFilter("FBX animation (*.fbx)");
        string path;
        try { if (!dialog.Execute()) return; path = dialog.SelectedFile; }
        finally { dialog.Destroy(); }
        if (!Path.GetExtension(path).Equals(".fbx", StringComparison.OrdinalIgnoreCase)) path += ".fbx";
        try
        {
            await ExportMotionFbxAsync(path, retarget);
            await EditorPipeline.SwitchToMainThread();
            if (this.IsValid()) _captureStatus.Text = $"Exported {Path.GetFileName(path)} · armature and bone animation.";
        }
        catch (Exception e) { await EditorPipeline.SwitchToMainThread(); if (this.IsValid()) _captureStatus.Text = e.Message; }
    }

    internal async Task ExportMotionFbxAsync(string path, bool retarget)
    {
        if (_exportingMotion || _processing is not null) throw new InvalidOperationException("Wait for the current operation to finish.");
        if (_editedMotion is null) throw new InvalidOperationException("Reconstruct or open motion before exporting.");
        if (retarget && _target is null) throw new InvalidOperationException("Select a target rig to retarget the animation.");
        var destination = Path.GetFullPath(path);
        if (_target?.ModelFilePath is { } modelPath && string.Equals(destination, Path.GetFullPath(modelPath), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Choose a new animation filename to preserve the original target model.");
        _exportingMotion = true;
        // A slowed-down clip (see MocapClipSpeed) exports at that speed: the same frames, further apart.
        var speed=_animationSpeed;
        UpdateExportAvailability();
        try
        {
            while(retarget)
            {
                await EditorPipeline.SwitchToMainThread();
                if(!_previewPending)break;
                await _mocapPreviewTask;
            }
            await EditorPipeline.SwitchToMainThread();
            if (!this.IsValid()) return;
            var motion = _editedMotion;var preview=_bakedPreview;
            if(retarget&&preview is null)throw new InvalidOperationException("Wait for a valid animation preview before exporting.");
            await Write(destination,motion,preview);
        }
        finally { await EditorPipeline.SwitchToMainThread(); _exportingMotion = false;if(this.IsValid())UpdateExportAvailability(); }

        async Task Write(string destination,MotionDocument motion,BakedPreview preview)
        {
            var bytes = await Task.Run(() =>
            {
                if (!retarget) return MotionFbxExport.WriteForModelDoc(ClipSpeed.Retimed(motion,speed));
                // Export the exact baked transforms shown in the viewport. Unapplied
                // edits in Advanced must not silently change the exported animation.
                var clip=preview.Clip;var target=preview.Target;
                var fps=clip.Fps*speed;
                return ModelDocFbxExport.Write(target.Rig.Skeleton, clip.ClipName, clip.SolvedFrames,
                    Enumerable.Range(0, clip.SolvedFrames.Count).Select(i => i / (double)fps).ToArray(), fps,
                    target.UpAxis,
                    preview.Space + "; retargeted");
            });
            // Stage next to the destination. A failed write leaves any existing export intact.
            var temporary = destination + "." + Guid.NewGuid().ToString("N") + ".tmp";
            try
            {
                await File.WriteAllBytesAsync(temporary, bytes);
                File.Move(temporary, destination, true);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }
    }
}
