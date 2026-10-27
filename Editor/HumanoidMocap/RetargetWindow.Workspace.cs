using System;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Editor;
using Sandbox;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Core;

namespace HumanoidMocapX.EditorTools;

public sealed partial class RetargetWindow
{
    Widget _videoHost, _targetHost;
    Label _captureStatus, _clock;
    LineEdit _sourcePath, _rangeStart, _rangeEnd, _rootSmooth, _armSmooth, _fingerSmooth, _smoothing, _ground, _facing;
    MocapVideoWidget _video;
    PreviewWidget _mocapPreview;
    FloatSlider _timeline;
    MotionDocument _rawMotion, _editedMotion;
    string _motionPath;
    Checkbox _inPlaceControl;
    Checkbox _stabilizeFeetControl;
    bool _showTargetBones=true;
    CancellationTokenSource _processing;
    Widget _advancedPanel;
    Label _motionDetails;
    Button _cancelCaptureButton, _retryCaptureButton;
    IconButton _exportMotionButton, _uploadVideoButton;
    MocapIconButton _settingsButton;
    IconButton _resetViewButton;
    bool _exportCaptured;
    Widget _welcome, _uploadBar, _previewArea, _transportBar;
    ProcessingIndicator _loader;string _loaderStatus;StatusDot _statusDot;
    Label _videoName;
    Dialog _adjustments;

    // Direct reuse of retargeter Group/Theme controls and sbox-public's SegmentedControl.
    void BuildMocapUi()
    {
        Layout.Margin=12;Layout.Spacing=10;
        // Header: settings and export on the right.
        var header=Layout.AddRow();header.Spacing=8;
        header.AddStretchCell();
        _settingsButton=header.Add(new MocapIconButton("tune",ShowAdjustments,this){FixedSize=30,ToolTip="Mocap adjustments"});
        _exportMotionButton=header.Add(new MocapIconButton("file_download",()=>PickMotionExport(!_exportCaptured),this)
            {FixedSize=30,Enabled=false,ToolTip="Export the armature and animated bones as FBX."});

        _sourcePath=new LineEdit(this){Visible=false,ReadOnly=true};
        _welcome=Layout.Add(new VideoDropArea(this,ImportVideoAndProcess,ChooseVideo),1);
        _previewArea=Layout.Add(new Widget(this){Visible=false},1);
        _previewArea.Layout=Layout.Row();_previewArea.Layout.Spacing=10;
        // Source card: the video, with its file name and the way to replace it in the header.
        var sourcePanel=_previewArea.Layout.Add(new MocapCard(this),1);
        var sourceHeader=sourcePanel.Header("movie","Source");
        _uploadBar=sourceHeader.Add(new Widget(sourcePanel),1);_uploadBar.Layout=Layout.Row();
        var upload=_uploadBar.Layout;upload.Spacing=4;
        _videoName=upload.Add(new Label("",sourcePanel){MinimumWidth=20},1);
        _videoName.SetStyles($"color: {Theme.TextLight.Hex};");
        // IconButton centers its glyph; an empty-label Button still reserves room for text.
        _uploadVideoButton=upload.Add(new MocapIconButton("video_file",ChooseVideo,sourcePanel){FixedSize=26,IconSize=16,ToolTip="Replace the video: choose another file"});
        _videoHost=sourcePanel.Layout.Add(new Widget(sourcePanel),1);_videoHost.Layout=Layout.Column();
        // Animation card: the character playing the capture, with its character choice in the header.
        var animationPanel=_previewArea.Layout.Add(new MocapCard(this),1);
        var viewBar=animationPanel.Header("accessibility_new","Animation");
        viewBar.AddStretchCell();
        _previewTargetPicker=Framed(viewBar.Add(new ComboBox(animationPanel){FixedWidth=112,FixedHeight=26,ToolTip="Preview and export character. Citizen is Terry (citizen.vmdl). Changing character reuses the captured motion."}));
        _previewTargetPicker.AddItem("Human",onSelected:()=>SelectBuiltinPreviewTarget(false),selected:true);
        _previewTargetPicker.AddItem("Citizen",onSelected:()=>SelectBuiltinPreviewTarget(true),description:"Terry · models/citizen/citizen.vmdl");
        _previewTargetPicker.AddItem("Custom",enabled:false);
        // Native IconButton centers its glyph. Button reserves a trailing text gap
        // even with an empty label, shifting this icon two pixels to the left.
        // Centred on the framed character dropdown beside it, which its border and padding make 32 px tall
        // (measured in the editor: the holder sits 3 px above the dropdown, so the 26 px button needs 6 px above it).
        var reset=viewBar.Add(new Widget(animationPanel){FixedWidth=26,FixedHeight=32});reset.Layout=Layout.Column();
        reset.Layout.Margin=new Sandbox.UI.Margin(0,6,0,0);
        _resetViewButton=reset.Layout.Add(new MocapIconButton("center_focus_strong",()=>_mocapPreview?.ResetView(),reset)
            {FixedSize=26,IconSize=16,ToolTip="Reset preview camera"});
        _targetHost=animationPanel.Layout.Add(new Widget(animationPanel),1);_targetHost.Layout=Layout.Column();
        _loader=_targetHost.Layout.Add(new ProcessingIndicator(_targetHost){Busy=false},1);_loader.SetMessage("Choose a video to begin.");

        // Playback card: play, the timeline, time and the bone overlay.
        var transportCard=new MocapCard(this,true){Visible=false};_transportBar=Layout.Add(transportCard);
        transportCard.Layout.Margin=new Sandbox.UI.Margin(8,6,10,6);
        var transport=transportCard.Layout;transport.Spacing=10;
        // IconButton centers the play/pause glyph; a Button keeps a text gap even with no label.
        _playButton=transport.Add(new MocapIconButton("play_arrow",TogglePlayback,this){FixedSize=30,IconSize=18,ToolTip="Play / pause"});
        var tracks=transport.Add(new Widget(this),1);tracks.Layout=Layout.Column();tracks.Layout.Spacing=2;
        _timeline=tracks.Layout.Add(new FloatSlider(tracks));_timeline.Minimum=0;_timeline.Maximum=1;
        _timeline.OnValueEdited=()=>SeekPlaybackFraction(_timeline.Value);
        _clock=transport.Add(new Label("0.00 s",this){MinimumWidth=70});
        _clock.SetStyles($"color: {Theme.TextLight.Hex};");
        var bones=transport.Add(new Checkbox("Bones"){Value=true});
        bones.Clicked=()=>{_showTargetBones=bones.Value;if(_mocapPreview.IsValid())_mocapPreview.ShowTargetBones=bones.Value;};

        // Status line: a light (working, ready) and what is happening, with Cancel and Retry beside it.
        var status=Layout.AddRow();status.Spacing=8;
        _statusDot=status.Add(new StatusDot(this));
        _captureStatus=status.Add(new Label(_bodyCapture.UnavailableReason??"Upload a video or open a saved body motion.",this){WordWrap=true,MinimumHeight=24},1);
        _cancelCaptureButton=status.Add(new MocapButton("Cancel","cancel"){Visible=false,Clicked=CancelCapture});
        _retryCaptureButton=status.Add(new MocapButton("Retry","refresh"){Visible=false,Clicked=()=>_=ProcessImportedVideoAsync()});

        _advancedPanel=new Widget(this){Visible=false};_advancedPanel.Layout=Layout.Column();_advancedPanel.Layout.Spacing=10;
        // Mocap adjustments pages, grouped in the window's tree. The host's own settings
        // come first under Processing, before Capture / Character / Motion.
        if(_captureSettings is not null)
        {
            var setupCard=AddAdjustmentPage("Processing","tune","Setup").Layout.Add(new MocapCard(this){Flat=true});
            setupCard.Header("tune","Setup");_captureSettings(setupCard);
        }
        // Capture › Video: the processed range, reprocessing and saved motions.
        var videoCard=AddAdjustmentPage("Capture","movie","Video").Layout.Add(new MocapCard(this){Flat=true});
        videoCard.Header("movie","Video");
        var range=videoCard.Layout.AddRow();range.Spacing=8;
        _rangeStart=Field(range,"Start (s)","0");_rangeEnd=Field(range,"End (s)","");
        var videoActions=videoCard.Layout.AddRow();videoActions.Spacing=8;
        videoActions.Add(new MocapButton("Process again","refresh"){Clicked=()=>_=ProcessImportedVideoAsync()});
        videoActions.Add(new MocapButton("Open motion…","folder_open"){ToolTip="Open a saved full-body .hmotion capture.",Clicked=()=>{
            var file=EditorUtility.OpenFileDialog("Open body motion","Humanoid Motion (*.hmotion)",null);
            if(!string.IsNullOrEmpty(file))_=OpenCaptureFileAsync(file);
        }});
        videoActions.AddStretchCell();
        // Character › Target: which rig the capture drives, or the captured skeleton itself.
        var targetCard=AddAdjustmentPage("Character","person","Target").Layout.Add(new MocapCard(this){Flat=true});
        targetCard.Header("person","Target");
        var targetRow=targetCard.Layout.AddRow();targetRow.Spacing=8;
        targetRow.Add(new Label("Character:",this));
        var target=_advancedTargetPicker=Framed(targetRow.Add(new ComboBox(this)));
        target.AddItem("s&box Human","person",()=>SelectBuiltinPreviewTarget(false),selected:true);
        target.AddItem("s&box Citizen","person",()=>SelectBuiltinPreviewTarget(true));
        target.AddItem("Custom VMDL…","folder_open",()=>{if(!_updatingTargetPickers){PickCustomModelTarget();RefreshTargetPickers();}});
        target.AddItem("Custom FBX / GLB…","folder_open",()=>{if(!_updatingTargetPickers){PickCustomFbxTarget();RefreshTargetPickers();}});
        var captured=targetRow.Add(new Checkbox("Export captured skeleton"));
        captured.ToolTip="Export the source skeleton without target retargeting or target foot locking.";
        captured.Clicked=()=>{_exportCaptured=captured.Value;UpdateExportAvailability();};
        targetRow.AddStretchCell();
        // Character › Placement: ground, facing and foot locking on the target.
        var placementPage=AddAdjustmentPage("Character","accessibility_new","Placement");
        var groundCard=placementPage.Layout.Add(new MocapCard(this){Flat=true});groundCard.Layout.Spacing=10;
        groundCard.Header("directions_walk","Ground and facing");
        var groundOptions=groundCard.Layout.AddRow();groundOptions.Spacing=24;
        _ground=Field(groundOptions,"Ground offset (m)","0");_facing=Field(groundOptions,"Facing (degrees)","0");
        var inPlace=_inPlaceControl=groundOptions.Add(new Checkbox("In place"));
        inPlace.ToolTip="Remove horizontal travel from the baked animation. Camera-relative capture remains camera-relative.";
        inPlace.Clicked=()=>{_rootMotion=inPlace.Value?HumanoidMocapX.Core.Cleanup.RootMotionMode.InPlace:HumanoidMocapX.Core.Cleanup.RootMotionMode.Off;_=RefreshMocapPreviewAsync();};
        var contactOptions=groundCard.Layout.AddRow();contactOptions.Spacing=12;
        _stabilizeFeetControl=contactOptions.Add(new Checkbox("Foot locking"){Value=true,Enabled=false,
            ToolTip="Keep predicted stationary ankles and toes anchored on the final target rig. Applies to world-relative captures with backend stationary-joint probabilities. Preserves toe-only pivots and bone lengths; predictions still need review."});
        _stabilizeFeetControl.Clicked=()=>_=RefreshMocapPreviewAsync();

        var cleanupCard=AddAdjustmentPage("Motion","auto_fix_high","Cleanup").Layout.Add(new MocapCard(this){Flat=true});
        cleanupCard.Header("auto_fix_high","Cleanup");
        // Two rows so the card fits the dialog's width instead of clipping the apply button.
        var cleanup=cleanupCard.Layout.AddRow();cleanup.Spacing=8;
        _rootSmooth=Field(cleanup,"Root cleanup","0.25");_armSmooth=Field(cleanup,"Arms","0.10");_fingerSmooth=Field(cleanup,"Fingers","0.025");
        cleanup=cleanupCard.Layout.AddRow();cleanup.Spacing=8;
        _speedField=Field(cleanup,"Animation speed (%)","100");
        _speedField.ToolTip="How fast the animation plays and exports, in percent of the video's speed.";
        _speedField.EditingFinished+=AnimationSpeedEdited;
        _smoothing=Field(cleanup,"Smoothing","7");_smoothing.ToolTip="Zero-phase smoothing strength, 0.5 to 10 (higher is smoother); 0 turns it off. 7 matches Rokoko's default.";
        cleanup.AddStretchCell();
        cleanup.Add(new MocapButton("Apply adjustments","check"){Clicked=()=>_=ProcessMotionAsync()});
        // What the capture did and assumed, for reference: kept out of the way in small, muted text.
        var detailsCard=AddAdjustmentPage("Capture","info","Details").Layout.Add(new MocapCard(this){Flat=true});
        detailsCard.Header("info","Details");
        _motionDetails=detailsCard.Layout.Add(new Label(this){WordWrap=true});
        _motionDetails.SetStyles($"color: {Theme.TextLight.Hex}; font-size: 11px;");
    }

    /// <summary>Inputs take the buttons' fill and hairline edge, so they read as controls on the gray window.</summary>
    static T Framed<T>(T input) where T:Widget
    {
        input.SetStyles($"background-color: {MocapButton.Fill.Hex}; border: 1px solid {MocapButton.Edge.Hex}; border-radius: 4px; padding-left: 4px;");
        return input;
    }
    LineEdit Field(Layout layout,string title,string value)
    {
        var row=layout.AddRow();row.Spacing=6;row.Add(new Label(title,this));
        return Framed(row.Add(new LineEdit(this){Text=value,MinimumWidth=65,FixedHeight=24},1));
    }
    /// <summary>The Smoothing field: 0 (off) or 0.5 to 10, clamped; blank or invalid keeps the default.</summary>
    float SmoothingStrength()
    {
        var value=Number(_smoothing,7);if(!float.IsFinite(value))return 7;
        return value<=0?0:Math.Clamp(value,.5f,10);
    }
    void ChooseVideo()
    {
        var path=EditorUtility.OpenFileDialog("Upload video","Video (*.mp4 *.mov *.m4v *.avi *.mkv *.webm)",null);
        if(!string.IsNullOrEmpty(path))ImportVideoAndProcess(path);
    }
    /// <summary>Loads the source preview without starting capture. An idle preview never shows a processing spinner.</summary>
    public void LoadVideo(string path)
    {
        _welcome.Visible=false;_uploadBar.Visible=true;_previewArea.Visible=true;_transportBar.Visible=true;
        Update();
        _videoName.Text=Path.GetFileName(path);_videoName.ToolTip=path;
        _sourcePath.Text=path;_videoHost.Layout.Clear(true);
        // The animation side waits at the preview's size while the video is processed.
        _targetHost.Layout.Clear(true);_mocapPreview=null;
        _loader=_targetHost.Layout.Add(new ProcessingIndicator(_targetHost){Busy=_processing is not null},1);
        _loader.SetMessage(_processing is null?"Video preview. Capture has not started.":"Preparing…");_loaderStatus=null;
        _video=_videoHost.Layout.Add(new MocapVideoWidget(_videoHost,path,useEngineVideo:true),1);
        _video.TogglePlayback=TogglePlayback;
        _video.Show();
        ResetPlayback();
        _captureStatus.Text=_processing is null&&_bodyCapture.UnavailableReason is { } reason?reason:
            "Inspect source visibility and lens distortion. No camera calibration is assumed.";
    }
    public async Task LoadMotionAsync(string path)
    {
        var revision=++_motionLoadRevision;
        try
        {
            var loaded=await Task.Run(()=>{var session=MocapAdjustmentStore.Load(path,null,null);return(session,quality:MotionDiagnostics.Analyze(session.Raw));});
            var doc=loaded.session.Edited;
            await EditorPipeline.SwitchToMainThread();if(!this.IsValid()||revision!=_motionLoadRevision)return;
            if(!doc.Bones.Any(b=>b.Role==HumanoidMocapX.Core.Mapping.BoneRole.Hips))
                throw new InvalidOperationException("This workspace accepts full-body motion only.");
            InvalidateMocapPreview();
            _editSession=loaded.session;_rawMotion=loaded.session.Raw;_editedMotion=doc;_motionPath=path;
            // Foot locking needs world-relative motion with the backend's stationary-joint probabilities.
            _stabilizeFeetControl.Enabled=_rawMotion is {Space:MotionSpace.WorldRelative,StationaryJoints.Count:>0};
            _appliedCleanup=loaded.session.State.Cleanup;ResetAdjustmentFields();
            if(File.Exists(doc.SourceVideo))
            {
                // The same video reloaded with its capture keeps the original file's name.
                var same=!string.IsNullOrEmpty(_sourcePath.Text)&&string.Equals(Path.GetFullPath(doc.SourceVideo),Path.GetFullPath(_sourcePath.Text),StringComparison.OrdinalIgnoreCase);
                var name=_videoName.Text;var tip=_videoName.ToolTip;
                LoadVideo(doc.SourceVideo);
                if(same){_videoName.Text=name;_videoName.ToolTip=tip;}
            }
            else
            {
                _videoHost.Layout.Clear(true);_video=null;_videoName.Text="Source video unavailable";_videoName.ToolTip=doc.SourceVideo;
                _sourcePath.Text=doc.SourceVideo;_videoHost.Layout.Add(new Label("The source video could not be found.\nAnimation playback is still available.",_videoHost){Alignment=TextFlag.Center,WordWrap=true},1);
                ResetPlayback();
            }
            _captureStatus.Text=$"Ready · {doc.Frames.Count} frames. Review the animation, then export.";
            _motionDetails.Text=$"{doc.Backend} · {doc.Space}. "+loaded.quality.HandSummary+" "+string.Join(" ",doc.Diagnostics);
            if(loaded.session.Notice is { } notice)_captureStatus.Text+=" "+notice;
            await RefreshMocapPreviewAsync();
        }
        catch(Exception e){await EditorPipeline.SwitchToMainThread();if(this.IsValid()&&revision==_motionLoadRevision)_captureStatus.Text=e.Message;}
    }

    async Task BuildMocapPreviewAsync(int revision)
    {
        if(_editedMotion is null || _target is null)return;
        _previewPending=true;_bakedPreview=null;UpdateExportAvailability();
        if(_loader.IsValid()){_loader.Busy=true;_loader.SetMessage("Building the preview…");}
        try
        {
            if(_fbxPreviewTask is not null)await _fbxPreviewTask;
            await EditorPipeline.SwitchToMainThread();if(!this.IsValid()||revision!=_previewRevision)return;
            var target=_target;var spec=target.Spec;var motion=_editedMotion;
            var bytes=Encoding.UTF8.GetBytes(motion.ToJson());var corrections=CaptureTargetCorrections();var rootMotion=_rootMotion;
            var session=_editSession;var cleanup=_appliedCleanup;var targetKey=MocapAdjustmentStore.TargetKey(spec,false);
            var edit=new MocapAdjustmentStore.TargetEdit{Corrections=corrections,RootMotion=rootMotion};
            var result=await Task.Run(()=>Retargeter.Convert(new RetargetRequest{SourceData=bytes,SourceFileName="capture.hmotion",
                FootPlantCleanup=corrections.StabilizeFeet,ArmEffectorIk=false,MocapCorrections=corrections,RootMotion=rootMotion},spec));
            await EditorPipeline.SwitchToMainThread();if(!this.IsValid()||revision!=_previewRevision)return;
            var clip=result.Clips.FirstOrDefault(c=>c.Success)??
                throw new InvalidOperationException("No convertible motion: "+(result.Clips.Select(c=>c.Error).FirstOrDefault(e=>!string.IsNullOrWhiteSpace(e))??string.Join(" ",result.Errors)));
            _targetHost.Layout.Clear(true);
            _mocapPreview=_targetHost.Layout.Add(new PreviewWidget(_targetHost,spec.Rig,target.PreviewModelPath,target.PreviewPositionScale,spec.UpAxis),1);
            _mocapPreview.Playing=false;_mocapPreview.ShowTargetBones=_showTargetBones;
            _welcome.Visible=false;_uploadBar.Visible=true;_previewArea.Visible=true;_transportBar.Visible=true;
            Update();
            _previewFps=clip.Fps;_mocapPreview.SetClip(clip);_mocapPreview.ResetView();
            _mocapPreview.Show();
            _targetHost.Update();
            _bakedPreview=new BakedPreview(clip,spec,motion.Space.ToString(),revision);
            SynchronizePreview();
            SaveAppliedAdjustments(session,targetKey,edit,cleanup,motion);
        }
        catch(Exception e){await EditorPipeline.SwitchToMainThread();if(this.IsValid()&&revision==_previewRevision)_captureStatus.Text=e.Message;}
        finally
        {
            await EditorPipeline.SwitchToMainThread();
            if(this.IsValid()&&revision==_previewRevision){_previewPending=false;UpdateExportAvailability();}
        }
    }
    async Task ProcessMotionAsync()
    {
        if(_rawMotion is null || _processing is not null)return;
        _processing=new CancellationTokenSource();var token=_processing.Token;
        SetCaptureBusy(true);
        try
        {
            var settings=new CleanupSettings{Root=Number(_rootSmooth,.25f),Arms=Number(_armSmooth,.1f),Fingers=Number(_fingerSmooth,.025f)};
            settings.Smoothing=settings.PositionSmoothing=SmoothingStrength();
            var raw=_rawMotion;var contacts=_editedMotion.Copy().Contacts;var session=_editSession;
            var doc=await Task.Run(()=>{token.ThrowIfCancellationRequested();var copy=raw.Copy();copy.Contacts=contacts;var edited=MotionCleanup.Apply(copy,settings);token.ThrowIfCancellationRequested();return edited;},token);
            await EditorPipeline.SwitchToMainThread();if(!this.IsValid())return;token.ThrowIfCancellationRequested();
            if(session!=_editSession)return;
            _editedMotion=doc;_appliedCleanup=settings;
            _captureStatus.Text="Adjustments applied. Original observations are preserved.";
            await RefreshMocapPreviewAsync();
        }
        catch(OperationCanceledException){await EditorPipeline.SwitchToMainThread();if(this.IsValid())_captureStatus.Text="Cancelled. Original motion preserved.";}
        catch(Exception e){await EditorPipeline.SwitchToMainThread();if(this.IsValid())_captureStatus.Text=e.Message;}
        finally
        {
            await EditorPipeline.SwitchToMainThread();_processing.Dispose();_processing=null;
            if(this.IsValid())
            {
                SetCaptureBusy(false);
                StartNextQueuedVideo();
            }
        }
    }
    static float Number(LineEdit field,float fallback)=>float.TryParse(field.Text,NumberStyles.Float,CultureInfo.InvariantCulture,out var n)&&float.IsFinite(n)?n:fallback;
    void FitMocapPlacementToTarget()
    {
        if(_target is null||_ground is null)return;
        _ground.Text="0";_facing.Text="0";
        _rootMotion=HumanoidMocapX.Core.Cleanup.RootMotionMode.Off;_inPlaceControl.Value=false;
        _stabilizeFeetControl.Value=true;
        RestoreTargetAdjustments();
    }

    // Body captures read only ground, facing and foot locking; the arm and camera fields
    // keep their defaults because only hand captures use them.
    TargetCorrectionSettings CaptureTargetCorrections() => new()
    {
        ContactTargetKey=_target is null?"":MocapAdjustmentStore.TargetKey(_target.Spec,false),
        GroundOffset=Number(_ground,0),FacingDegrees=Number(_facing,0),
        StabilizeFeet=_stabilizeFeetControl.Value
    };
    /// <summary>Backend progress lines that are bookkeeping ("temporal-inference", "complete"), not something to show.</summary>
    static bool IsStepToken(string message)=>message.Length>0&&message.All(c=>c is >='a' and <='z' or '-');
    [EditorEvent.Frame]
    public void MocapTick()
    {
        if(!this.IsValid())return;
        _video?.Present();
        if(Interlocked.Exchange(ref _workerMessage,null) is { } message&&!IsStepToken(message))
        {
            _captureStatus.Text=message;
            if(_loader.IsValid()&&_loader.Busy)_loader.SetMessage(message);
        }
        if(_statusDot.IsValid())_statusDot.Color=_processing is not null?Theme.Yellow:_captureStatus.Text.StartsWith("Ready",StringComparison.Ordinal)?Theme.Green:Theme.TextLight;
        if(_loader.IsValid())
        {
            // Once a capture stops without a preview, the loader shows how it ended instead of spinning.
            if(!_loader.Busy&&_loaderStatus!=_captureStatus.Text){_loaderStatus=_captureStatus.Text;_loader.SetMessage(_loaderStatus);}
            _loader.Tick();
        }
        TickPlayback();
    }
}
