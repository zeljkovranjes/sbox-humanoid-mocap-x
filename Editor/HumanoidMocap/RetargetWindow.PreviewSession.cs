using System.Threading.Tasks;
using Sandbox;
using HumanoidMocapX.Core;

namespace HumanoidMocapX.EditorTools;

public sealed partial class RetargetWindow
{
    sealed record BakedPreview(ClipResult Clip,RetargetTargetSpec Target,string Space,int Revision);
    BakedPreview _bakedPreview;
    Task _mocapPreviewTask=Task.CompletedTask;
    int _previewRevision,_motionLoadRevision;
    bool _previewPending;

    Task RefreshMocapPreviewAsync()
        =>_mocapPreviewTask=BuildMocapPreviewAsync(++_previewRevision);

    void InvalidateMocapPreview()
    {
        ++_previewRevision;_bakedPreview=null;_previewPending=false;
        UpdateExportAvailability();
    }
    void UpdateExportAvailability()
    {
        if(_exportMotionButton.IsValid())
            _exportMotionButton.Enabled=_editedMotion is not null&&_processing is null&&!_exportingMotion
                &&(_exportCaptured||(!_previewPending&&_bakedPreview is not null));
    }
}
