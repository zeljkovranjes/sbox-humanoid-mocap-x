if (args is ["--body-clearance"]) { BodyClearanceTests.Run(); return; }
if (args is ["--video-layout"]) { VideoLayoutTests.Run(); return; }
CcdIkTests.Run();
BodyClearanceTests.Run();
VideoLayoutTests.Run();
MotionTests.Run();
FingerTests.Run();
HandTransferTests.Run();
HandContactTests.Run();
FingerTransferTests.Run();
HeadPitchTests.Run();
RootAnchorTests.Run();
CaptureSpinTests.Run();
HandPlantTests.Run();
CameraTests.Run();
InferenceTests.Run();
RefinerTests.Run();
DenoiserTests.Run();
CheckpointTests.Run();
ScriptCheckpointTests.Run();
DecoderTests.Run();
YoloTests.Run();
OmegaTests.Run();
OmegaVisionTests.Run();
SamBackboneTests.Run();
SamDecoderTests.Run();
SamConditioningTests.Run();
SamCompactPoseTests.Run();
SamHeadTests.Run();
SamKeypointTests.Run();
SamProjectionTests.Run();
SamFeedbackTests.Run();
SamInitialTests.Run();
ImageSamplingTests.Run();
SamCropTests.Run();
SamHandBoxTests.Run();
CaptureInputTests.Run();
CaptureLaneTests.Run();
MhrSkeletonTests.Run();
MhrSkinningTests.Run();
MhrBodyTests.Run();
if (args is ["--mhr-checkpoint", var mhrPath])
{
    MhrSkeletonTests.Released(mhrPath);
    MhrSkinningTests.Released(mhrPath);
    MhrBodyTests.Released(mhrPath);
}
if (args is ["--sam-checkpoint", var samPath, "--mhr-checkpoint", var samMhrPath]) SamHeadTests.Released(samPath, samMhrPath);
if (args is ["--flow-checkpoint", var flowPath, "--smpl-model", var smplPath]) DecoderTests.Released(flowPath, smplPath);
await PreviewTests.Run();
await VideoFramesTests.Run();
await FfmpegVideoTests.Run();
await InstalledTests.Run();
await DownloadTests.Run();
await HostedTests.Run();
await FlowCaptureTests.Run();
await SshCaptureTests.Run();
