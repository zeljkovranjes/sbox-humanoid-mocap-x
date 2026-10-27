using System;
using System.Globalization;
using HumanoidMocapX.Core.Cleanup;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.EditorTools;

public sealed partial class RetargetWindow
{
    MocapAdjustmentStore.Session _editSession;
    CleanupSettings _appliedCleanup;
    string _adjustmentSaveError;

    void ResetAdjustmentFields()
    {
        var cleanup=_appliedCleanup??new CleanupSettings();
        _rootSmooth.Text=cleanup.Root.ToString(CultureInfo.InvariantCulture);
        _armSmooth.Text=cleanup.Arms.ToString(CultureInfo.InvariantCulture);
        _fingerSmooth.Text=cleanup.Fingers.ToString(CultureInfo.InvariantCulture);
        _smoothing.Text=cleanup.Smoothing.ToString(CultureInfo.InvariantCulture);
        FitMocapPlacementToTarget();
    }

    void RestoreTargetAdjustments()
    {
        if(_editSession is null||_target is null)return;
        if(!_editSession.State.Targets.TryGetValue(MocapAdjustmentStore.TargetKey(_target.Spec,false),out var edit))return;
        string F(float v)=>v.ToString("R",CultureInfo.InvariantCulture);
        var c=edit.Corrections;
        _ground.Text=F(c.GroundOffset);_facing.Text=F(c.FacingDegrees);
        _stabilizeFeetControl.Value=c.StabilizeFeet;
        _rootMotion=edit.RootMotion;_inPlaceControl.Value=_rootMotion==RootMotionMode.InPlace;
    }

    void SaveAppliedAdjustments(MocapAdjustmentStore.Session session,string key,MocapAdjustmentStore.TargetEdit edit,
        CleanupSettings cleanup,MotionDocument motion)
    {
        if(session is null||session!=_editSession)return;
        session.State.Cleanup=cleanup;session.State.Contacts=motion.Contacts;session.State.Targets[key]=edit;
        try{MocapAdjustmentStore.Save(session);_adjustmentSaveError=null;}
        catch(Exception error){_captureStatus.Text=_adjustmentSaveError="Preview ready, but adjustments could not be saved: "+error.Message;}
    }
}
