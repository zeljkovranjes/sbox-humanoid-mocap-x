Development
===========

Keep the modified `humanoid-mocap` checkout beside `humanoid-mocap-x`. The
editor workspace depends on its new `IBodyCapture` API and third-person
constructor. No shared solver or UI source is copied into X.

The projects in `dev/` are independent build checks. s&box generates its own
IDE projects under `Code/` and `Editor/`; do not edit those generated files.

## Checks

Use .NET SDK 10.0.403 and set `FACEPUNCH_ENGINE` to the s&box directory
containing `bin/managed`. Run from the X repository:

```sh
dotnet build dev/Editor.csproj -c Release -m:1
dotnet build dev/GpuChecks.csproj -c Release -m:1
dotnet run --project tests/MocapX.Tests.csproj -c Release
dotnet build dev/Whitelist.csproj -c Release -m:1
dotnet dev/bin/Whitelist/Release/net10.0/Whitelist.dll \
  dev/bin/Tensor/Release/net10.0/package.notpointless.chomnr_humanoid_mocap_x.tensor.dll \
  dev/bin/Core/Release/net10.0/package.notpointless.chomnr_humanoid_mocap_x.dll
```

`Whitelist` calls the installed engine's `AccessControl.VerifyAssembly` on
both halves of the runtime assembly. `dev/Tensor.csproj` builds `Code/Tensor`
against the engine, so `dev/Core.csproj` and the tests stay engine-free. It
does not change the engine's rules. Editor code has separate compiler
permissions in s&box; network and disk access live in `Editor/`. The runtime
`FlowMotion` DTO validates its own format without shared-library references;
`Editor/FlowMotionAdapter.cs` converts it into the shared motion document.

The library manifest explicitly selects Release compilation, including in
consumer editors. Build diagnostic projects with `-c Release`
when measuring performance. Debug and Release captures produced identical
saved output for the five-second fixture, but substantially different timings.

Tests exercise metre scale, timestamps, all 52 SMPL-H bone roles, Human and
Citizen retargeting, FBX round trips, planted feet, airborne release, input
validation, streamed downloads, checksum failure, cancellation and cache
replacement. Inference checks cover the twenty-step Euler sampler, attention
masks, time embeddings, the two-block video refiner and all eighteen
denoiser blocks. The complete prediction path and its integration with the
sampler are compared against independent double-precision equations using
small synthetic weights. Repeated predictions allocate no managed memory
on the calling thread. Checkpoint fixtures also exercise the shared C#
reader, tensor layout, rejected executable pickle data and missing or
malformed weights. These checks are not real video inference or parity
tests against the released checkpoint.

MHR fixtures cover inert TorchScript data, exact integer tensor reads,
forbidden calls, cyclic objects and metadata limits. The skeleton and
skinning checks compare parameter mapping, Euler rotations, prerotations,
parent transforms, joint scales and weighted vertices with independent
double-precision matrix calculations. Repeated execution allocates no
managed memory. To also validate a locally downloaded MHR asset:

```sh
dotnet run --project tests/MocapX.Tests.csproj -- \
  --mhr-checkpoint /path/to/assets/mhr_model.pt
```

The released-data checks passed for all 127 joints, 249 model parameters,
18,439 vertices and 51,337 skinning influences. Neutral skeleton transforms
compose with the saved inverse bind states to identity, and neutral skinning
reproduces the saved mesh. Posed skinning's maximum absolute difference from
the matrix reference was 0.0000340171 model units. This validates the C#
skeleton and skinning operations, not upstream SAM inference parity.

By default, `MhrBlendShapes`, `MhrCorrectives` and `MhrBody` evaluate the complete
body-model path: identity and expression deformation, sparse rotation
features excluding the first two root joints, ReLU, dense pose corrections,
and skinning. Identity coefficients deform vertices; their skeleton
parameter slots stay zero, matching the archive. Synthetic tests cover
independent blend and rotation equations, duplicate sparse entries, input
validation and allocation-free reuse. Released data reproduces the neutral
base mesh, gives finite deformed output and repeats deterministically.
Sampled dense corrective outputs differed from double-precision dot products
by at most 0.00000000599074 model units. The dense projection retains all
166 million weights; no corrective layer or blendshape is omitted.

The editor capture loader opts into `MhrBody.Restrict` using the union of
vertices referenced by the body and hand keypoint mappings. It computes the
same corrective dot products for those vertices and clears corrections for
the rest. Joint states and mapped keypoints retain the full evaluation's
values; unreferenced mesh vertices are not fully pose-corrected on this path.
Consumers needing the complete posed mesh must use an unrestricted body.
The five-second human capture produced eight byte-identical artifacts after
this change; see [video validation](video-validation.md).

`dev/mhr_checks.csx` also runs the complete released MHR body model
inside the editor through `run_csharp`. Put the absolute checkpoint path in
the session exchange directory's `mhr-model-path.txt`. The live Linux
editor check passed for all vertices, including identical repeated posed
output; its maximum neutral-mesh error was 0.0000457764 model units.
Editor and GPU diagnostic builds pass with zero
warnings or errors, and the live editor reports zero compile errors.

SAM's body and hand pose heads now bind the released projection weights,
learned initial predictions, scale components, dedicated finger parameters,
wrist transform and landmark mapping. `SamDecoderFeedback` connects the
MHR body model to the five intermediate decoder updates and final
prediction. `SamInitialTokens` binds the learned pose/camera initialization,
CLIFF crop condition, invalid point prompt, hand box and keypoint embeddings.
Its camera projection preserves hand scale, crop coordinates
and signed depths. Image feedback uses bilinear sampling with zero padding;
invalid points suppress positional embeddings and sampled features while
retaining the image projection's learned bias. The 3D embeddings use the
midpoint of the two hip landmarks as their origin.

Synthetic tests cover independent double-precision rotation/projection
equations, gimbal locks, parameter ordering, image-boundary sampling,
behind-camera masking, input rejection and allocation-free repeated calls.
The following optional check also runs both released six-layer decoders,
their full body/hand heads, all five intermediate updates and the final
prediction twice:

```sh
dotnet run --project tests/MocapX.Tests.csproj -- \
  --sam-checkpoint /path/to/model.ckpt \
  --mhr-checkpoint /path/to/assets/mhr_model.pt
```

Both released paths passed with finite, exactly repeatable output. This
integration fixture uses learned initial tokens with synthetic image
features on a six-patch grid. It does not cover real image preparation,
full feature extraction or video accuracy.

`SamCrop` retains the upstream body/hand padding, float32 affine source
points, OpenCV coordinate quantization and byte-rounded RGB interpolation.
It reuses `LinearRgbImage` from Humanoid Mocap; the original hand crop uses
the same sampler with its existing coordinate conventions. Tests cover
black borders, fractional rounding, alpha removal, mirrored left-hand
sampling and allocation-free reuse. `SamHandBoxes` compares all three MLP
layers and original-image box conversion with independent double arithmetic.

`SamFeatures.LoadAsync` loads a body/hand inference pipeline for each capture
lane. Immutable GPU weights and the MHR corrective projection are shared;
each lane retains its own mutable work buffers. `SamGpuFeatures.RunAsync`
accepts original RGBA8 pixels, a person box and original-pixel intrinsics,
and writes 3072 features in body/left/right order. It takes the first pose
token from each initial decoder pass, exactly the selection used by
FlowHMR's runtime feature hooks. SAM's subsequent hand-pose merging and
prompted body refinement do not feed those selected tokens and are not run
by this feature extractor. This does not replace FlowHMR's final motion
prediction or its finger output.

`dev/sam_feature_checks.csx` exercises that adapter in the editor. Supply the
same three model/assembly path files as the released checks, plus
`detector-model-path.txt` and `sam-feature-frames.json` containing an array
of objects with absolute `RgbaPath`, `Width` and `Height` fields. It writes
body/hand landmarks, hand boxes and all feature channels for visual review.
The check also tests invalid cameras, cancellation, concurrent-call rejection
and exact repeatability. See [GPU validation] for actual run results;
the existence of a diagnostic does not mean it has passed.

[GPU validation]: gpu-validation.md

Finger fixtures articulate all 30 SMPL-H finger joints. Source FBX preserves
their rotations; retargeting and target FBX preserve motion for all 30 Human
finger joints and Citizen's 24 supported joints. Classic Citizen lacks the
six pinky joints. The real five-second captures also retained all supported
finger tracks through FBX export. Finger-pose accuracy still needs a
ground-truth comparison; see [video validation].

Decoder checks cover clip-averaged body shape, all 52 joint rotations,
root displacement at 30 fps, absolute pelvis height and contact scores.
Known axis rotations include the identity, half-turns and near-zero angles;
the conversion reuses Humanoid Mocap's six-dimensional rotation decoder.
Checkpoint fixtures cover the normalization and shaped-joint buffers too.
Camera checks cover 1/10/30-frame sampling with both endpoints, original-pixel
intrinsics, focal-length interpolation, relative rotations, changes of world
axes, quaternion sign/scale, invalid inputs and allocation-free repeated
decoding. These use analytic predictions, not outputs from a VGGT checkpoint.
Preview tests cover pixel channel order, aspect ratio, bounded storage,
allocation-free repeated frame copies and playback decisions. Disk-cache
checks cover exact source bytes, corrupted-cache replacement, cancellation
before copying, empty/missing input and removal of partial files.

CI is configured for these engine-free tests on Windows and Linux.
Editor compilation and whitelist verification require an installed engine
and run locally.

## Implementation and acceptance

The Node example implements Discord OAuth through `oauth4webapi`, expiring
editor credentials and the provider's queued capture API. Its automated
tests and a live connection to the C# host pass. A real Discord consent flow
still needs application credentials. See `hosted-oauth.md` and
`../examples/hosted-web/README.md`. The s&box library remains C#/HLSL.

The five-second video fixtures exercise both target rigs, finger export and
foot locking. Broad motion has been inspected in preview, but anatomical
finger accuracy and skin-surface contact still need visual acceptance.
The [body-contact report](body-contact.md) describes the solver's tested
bone-derived collision proxies and their limits.

After capture, preview and export acceptance passes, profile the complete
pipeline and optimize its measured bottlenecks. Compare before/after timings
and peak memory using the same clips, checkpoints, initial noise, settings
and target rigs. Preserve image resolution, sampling steps, camera interval
and tracking detail. Check numerical outputs, foot contacts, finger motion,
root motion and exported animation for regressions. Isolated stage timings
are not evidence of improved complete-capture throughput. Performance work
and quality-preservation validation are required before completion.

The original five-second model-stage baseline was about 19 minutes. The
3–5-minute milestone was reached in a fixed-lane run of 272.033 seconds;
the subsequent Auto run took 299.756 seconds. These use predecoded frames,
exclude production video decoding, and are affected by competing GPU load.
The latest complete Release capture took 148.057 seconds elapsed with all
eight saved artifacts byte-identical to the preceding baseline. It includes
up to six detection/feature lanes, staged camera-model loading and shared MHR
corrective weights, and passed the diagnostic memory guards. The controlled
feature-only comparison took 127.81 seconds with four lanes and 120.23 with
six. These are individual samples; repeated-job memory acceptance remains open.
Current measurements are in [video validation]. The next target is
15 processing seconds per video second: 75 seconds for this fixture. It
remains unmet; other GPUs need their own measurements.
Continue testing reductions while preserving
output quality. Keep only measured improvements; do not lower model or image
settings to meet the timing target.
Exhaust concrete implementation options identified by profiling before
starting deep algorithm research. If research becomes necessary, compare
established and newer approaches and share specific findings through the
agent communication files before changing implementation ownership.

`LocalCapture` connects timestamped Windows decoding, installed-model
verification and the complete C#/HLSL [upstream pipeline] to the workspace.
There is no inference worker, Python runtime or process bridge in X.
`VideoFrames` streams original RGBA pixels to a temporary random-access
30 fps timeline, selecting the displayed source frame at each sample time.
It bounds dimensions, length and cache size, rejects changing dimensions or
non-increasing timestamps and deletes temporary pixels when disposed.
Tests cover variable/high frame rates, trims, exact pixels, malformed input
and cancellation cleanup without the engine. The shared Windows decoder
reads [presentation duration] from Media Foundation; native execution needs
a Windows acceptance run. Linux preview is retained, but its callbacks are
unsuitable for deterministic inference (see `video-validation.md`).

[presentation duration]: https://learn.microsoft.com/en-us/windows/win32/medfound/processing-media-data-with-the-source-reader#getting-the-file-duration

The checkpoint reader expects `model_state_dict.motion_transformer` and
304 model tensors. Both Base and Latest have completed all stages on
five-second predecoded videos.

The shared reader streams complete contiguous FP32 storage directly into its
output array on little-endian machines. Partial views, strided tensors and
other data types retain their existing conversion path. Regression checks cover
exact float bits, views, multi-chunk reads, cancellation before allocation and
the absence of a duplicate full-storage buffer.

For one 16 MiB released Omega tensor, seven warm reads per version allocated
33,554,616 bytes before this change and 16,777,376 bytes after it. All tensor
SHA-256 hashes were identical. Median read time was 2.1435 versus 1.6003 ms;
this isolated result does not establish complete-capture throughput or resolve
repeated-editor memory growth. Evidence is in the local
`_scratch/checkpoint-read-probe/` directory. The full engine-independent suite
passed all 39 groups, and the editor build had zero warnings or errors.
The earlier reader-only guarded capture completed tracking, camera and
features, then cancelled during motion generation when system swap grew
beyond 2 GiB.
Generation and both rig exports subsequently completed from the saved features
in a fresh editor; all eight artifacts matched the preceding baseline exactly.
This validates output preservation across those separate runs, not a successful
uninterrupted capture or a throughput improvement. A diagnostic collection
after generation reclaimed about 1.79 GiB of managed data; it does not identify
the earlier perception-stage memory cause. Later shared-weight and six-lane
captures passed the guards; an intermediate instrumented attempt cancelled
at its swap guard. See [video validation] for these results, measurements and
remaining acceptance.
Released-weight upstream parity and complete motion quality remain unverified.

Hosted capture is connected through the same `IBodyCapture` entry point.
The client streams video, polls owned jobs and validates the downloaded
motion before import. The opt-in `EditorHost` runs the authenticated API,
queue and local GPU capture inside s&box; see [the operator guide](editor-host.md).
It decodes with Media Foundation on Windows or FFmpeg on Linux and needs
manually installed, verified model files.
The separate `services/humanoid-mocap-host` starter links the same queue and
transport source for custom providers and engine-independent tests; its
default processor explicitly reports unavailable. See `../DOCUMENT.MD` for
the transport contract. The Node example in
`examples/hosted-web` adds Discord OAuth, expiring host keys and a browser
client while reusing the C# host's queue. Its tests cover OAuth validation
and live HTTP proxy behavior; real Discord consent requires credentials.

`FlowCheckpoint.LoadDecoder` reads saved normalization statistics and a
separate SMPL-H NPZ asset: the released checkpoints omit `body_model.*`.
`SmplModel` reuses Humanoid Mocap's numeric NPY reader, validates the
52-joint hierarchy and array shapes, and passes vertex data to `SmplJoints`.
That pure regression computes `J_regressor @ v_template` and joint shape
directions from all sixteen vertex shape directions. Tests use separate
checkpoint/NPZ fixtures, weighted regression, missing arrays, incorrect
hierarchies, shapes and non-finite data. Model files remain outside the repo.
`FlowDecoder` averages shape over the clip, integrates XZ velocity, retains
absolute pelvis Y and applies upstream's export-facing Y half-turn. Contact
scores are clamped to the shared motion format's probability range. It
leaves ground placement and foot locking to the shared retargeter; it does
not reproduce upstream's mesh-based floor fitting. The released checkpoints
and their rotation convention still need a direct numerical comparison.

The denoiser uses the published 491-channel, 1024-wide architecture, six
dual-stream blocks, twelve single-stream blocks and two context-refiner
blocks. It accepts unpadded clips of up to 5000 frames, matching the
backbone's position table. Rotary layout uses adjacent channel pairs and
base-10000 frequencies. Static comparison with the available [FlowHMR rotary
implementation] and its backbone confirms that formula and the separate
position indexing of motion/video branches. Released-checkpoint numerical
parity remains unverified.

[FlowHMR rotary implementation]: https://github.com/flowhmr/flowhmr/blob/89a78b821194897139b6df91a82f3128635ec6e8/flowhmr/network/mmdit/positional_encoding.py

HLSL compute shaders are permitted inside s&box libraries. The built-in
Tensor Inference code in `Code/Tensor` and `Assets/shaders/tensor-inference`
contains C# dispatch and float32 matrix multiplication, normalization,
attention, activation, modulation, packing, rotary-position and reduction
kernels. It uses standard s&box GPU buffers and no external process. The
attention kernel uses online softmax instead of allocating a dense score
matrix. It accepts the same key ranges produced by
`FlowAttention.CopyKeyRanges`.

The GPU library runs every model stage through `FlowVideoProcessor`. `TensorBatch`
records operations in an isolated 8×8 engine view, providing the graphics
context required by asynchronous GPU readback. It waits for readback before
reporting cancellation. Its C# assembly passes the engine whitelist, and
isolated HLSL compilation and SPIR-V validation passed for the original
shader set. Later accepted kernel changes also pass the full-clip artifact
comparison recorded in [video validation].
The Tensor Inference library's `dev/gpu_checks.csx` provides numerical checks
for a live editor, including
dependent dispatches, position-table resets and cancellation after
submission. It now passes on the AMD RX 9070 in the live editor; results and
shader-loading limitations are in [GPU validation]. Full predecoded model-stage
timings are recorded in `video-validation.md`; isolated batch overhead has
not been measured.

[GPU validation]: gpu-validation.md

`Editor/Gpu/FlowGpuSampler.cs` connects weight uploads, a shared workspace,
the video refiner and all eighteen model blocks through twenty sampling
steps. Weights and working tensors remain on the GPU; each step fences with
a one-float readback, followed by a complete final output readback. This
bounds queued work and waits for submitted work before reporting cancellation.
The live synthetic numerical comparison passes on AMD. Released-model
execution and capture timings are recorded in [video validation]; direct
upstream numerical parity remains unverified.

`dev/GpuChecks.csproj` builds a diagnostic assembly comparing six complete
GPU captures with the CPU reference, using synthetic 64-channel attention
heads and clips crossing the attention windows. It also checks repeated
captures with changed conditioning, malformed output and pre-cancellation.
CPU fixtures and the live AMD GPU checks pass.
With both library projects open in an editor, build the diagnostic project,
then invoke its public `HumanoidMocapX.Checks.FlowGpuChecks.Run` through
trusted `sbox-cli` automation. For example, replace the path in this
`run_csharp` method body with the absolute build output path:

```csharp
var assembly = System.Reflection.Assembly.LoadFrom(
    "/absolute/path/dev/bin/GpuChecks/Debug/net10.0/HumanoidMocapX.GpuChecks.dll");
var run = assembly.GetType("HumanoidMocapX.Checks.FlowGpuChecks", true)
    .GetMethod("Run");
return await (Task<object>)run.Invoke(null, new object[] { cancellationToken });
```

Keep the assembly's dependencies beside it. This diagnostic runs inside the
editor and loads only the test assembly's public entry point. It is not a
capture worker. Restart the editor before loading a rebuilt diagnostic DLL.

`FlowCamera` decodes VGGT-Omega's translation/XYZW/FoV output into the
original-pixel intrinsics SAM needs and the first-camera-relative rotations
FlowHMR needs. It interpolates rotations with shortest-arc SLERP and focal
lengths linearly between sampled frames. The shared world-axis conversion
cancels in the relative rotations. Translation is not a FlowHMR conditioning
input. The released camera model runs over the sampled frames of the
five-second video fixtures.
Temporal accuracy and upstream parity remain unverified.

The camera transformer's CPU and GPU implementations now include affine
layer normalization, masked QKV bias, optional per-head Q/K normalization,
layer scaling, max-normalized axial positions and error-function GELU.
`OmegaCameraHead` and `OmegaGpuCameraHead` implement all four prediction
blocks, camera-token selection and the nine-value output. They consume final
aggregator tokens from `DinoImageEncoder` and `OmegaAggregator`. These
implement all 24 DINO image blocks and 24 frame/inter-frame block pairs,
including the five register-only stages and separate first-frame tokens.
`OmegaGpuModel` runs the connected path, loading one RGB image at a time
and retaining image and camera features on the GPU. Its camera head consumes
the final GPU buffer directly. Working tensors respect the engine's
4 GiB per-buffer limit; this is not a GPU-memory budget or a measured peak.

`OmegaImage` center-crops extreme aspect ratios and resizes the long side
to 512 pixels with patch-aligned dimensions. Its bicubic resampler uses
antialiasing, byte rounding after each pass and a white alpha background.
Tests cover crop placement, interpolation values, transparent inputs,
both pass orders and allocation-free reuse. Pixel parity with a running
Pillow installation remains unverified.

`VisionCheckpoint.LoadOmega` reads image, aggregator and camera tensors through
the shared restricted reader, excluding unused depth and text heads.
`LoadOmegaCameraHead` retains the smaller head-only path for diagnostics.
Fixtures check tensor orientation, shape rejection, non-finite values and
cancellation. CPU tests compare the complete image encoder, aggregator and
camera head with independent double-precision equations. GELU is
checked against numerical integration of the Gaussian density. The
`OmegaGpuChecks.Run` diagnostic covers 64/128-channel heads, Q/K normalization,
split-half positions and the complete image-to-camera path against the CPU
implementation, including repeated runs and incomplete image readers.
It passes on the AMD GPU with synthetic weights. A real frame also completes
the released-weight camera path; details and limits are in [GPU validation].

`DinoImageEncoder` and `DinoGpuEncoder` also implement SAM's 32-layer
DINOv3 ViT-H/16+ backbone, with 1280 channels, 20 attention heads and
SwiGLU feed-forward gates. The shared transformer supports both this
configuration and Omega's 24-layer GELU encoder. SAM normalizes rotary
coordinates separately along each axis. The GPU adapter retains its patch
features for later GPU stages without requiring a CPU readback.

`VisionCheckpoint.LoadSamBackbone` selects backbone tensors from either a
root state dictionary or `state_dict`, excluding decoder and body-model
data. CPU tests compare all 32 layers with double-precision equations, and
cover checkpoint layout, malformed values, cancellation and allocation-free
reuse. `SamGpuChecks.Run` compares the encoder against the CPU reference in
a live editor and passes on AMD. The released weights are also bound in
the full video fixtures. Numerical parity against upstream bfloat16
execution remains unverified.

`SamDecoder` and `SamGpuDecoder` implement the six body/hand transformer
layers: self-attention, asymmetric image cross-attention, layer-normalized
positional additions, first-layer position handling, GELU feed-forward
layers and final normalization. `VisionCheckpoint.LoadSamDecoder` reads
either transformer's weights. Intermediate pose feedback runs after the
first five layers and is a required callback. Final pose prediction belongs
to the caller and uses the returned normalized tokens. The GPU adapter
keeps image features and weights resident, reads back tokens after each
layer, and invokes pose feedback on a worker thread before uploading the
updated keypoint tokens and positions.

CPU tests compare both decoders and their intermediate updates with
independent double-precision equations and check allocation-free reuse.
The GPU diagnostic also compares both decoders, changed image context,
cancellation and recovery after invalid feedback. Its small fixtures use
synthetic feedback; the released-head integration is covered separately
above. `MhrCheckpoint` reads the MHR
asset's numeric skeleton and skinning data without executing TorchScript.
`MhrSkeleton` maps parameters and calculates global transforms;
`MhrSkinning` applies inverse bind poses and vertex influences. `MhrBody`
combines these with identity/expression blendshapes and learned pose
correctives. Keypoint mapping and pose-head binding now connect the complete
body model to the SAM decoder.
`SamRays`, `SamImageConditioning` and `SamGpuImageConditioning` implement
camera-ray Fourier channels, learned no-mask embeddings, body/hand image
projection, layer normalization and positional context. Independent CPU
equations and live AMD GPU comparisons pass. `SamGpuFeatures` connects
body/hand crop preparation, hand detection and final 3072-channel feature
extraction. This connected path has run over all 150 frames of the five-second
fixtures; it supplies FlowHMR conditioning, while FlowHMR predicts the final
motion and fingers.

[upstream pipeline]: https://github.com/flowhmr/flowhmr/blob/main/docs/inference.md

`ModelStore` downloads the selected Base or Latest FlowHMR checkpoint from a
pinned Hugging Face revision. `ModelDownloads` exposes explicit editor
buttons for that checkpoint, SAM 3D Body DINOv3, VGGT-Omega 1B-512 and
YOLOX-L. The
perception packages use pinned mirror revisions and fixed checkpoint lengths
and SHA-256 values. SAM's MHR asset, configuration and model licenses are
included. The capture path never triggers downloads. Closing the widget
cancels its transfer and disposes HTTP resources after file operations stop.
Fixture tests cover transport, cache repair, cancellation, mirror selection
and bounded support-file downloads. Actual downloads of all packages passed
integrity checks through the C# downloader. Invoking the YOLOX-L editor
button also completed and displayed its saved-files status; the other
buttons still need live interaction checks. Complete perception inference
has run on a five-second predecoded clip. Mirror provenance and verification limits are in
[model sources]; original asset requirements are in the [installation guide].

[model sources]: model-sources.md

[installation guide]: https://github.com/flowhmr/flowhmr/blob/main/docs/install.md

The tensor library now includes HWC convolution with OIHW weights for vision
layers and patch embedding. It forms 16×16 patch tiles without an expanded
image buffer. Engine-free tests cover shape/dispatch bounds and overflow;
the live GPU diagnostic includes a double-precision convolution comparison.
Shader compilation and live AMD numerical checks pass. Full-stage timings
include these kernels; individual production-shape kernels need profiling.

`YoloGpuDetector` implements the fixed 640-pixel YOLOX-L person-detection
stage: RGBA-to-BGR image preparation, CSPDarknet, the feature pyramid,
eighty-class prediction heads, grid decoding and class-agnostic suppression
before person filtering. Image preparation and suppression run on worker
threads; convolution and feature tensors stay in s&box GPU buffers. It
returns zero people when none are detected, rather than inventing a person
covering the frame. `FlowVideoProcessor` tracks its detections across the clip.

`YoloCheckpoint` reuses the restricted checkpoint reader and folds batch
normalization with the upstream epsilon of 0.001. The fixed network has
54,208,895 trainable parameters and 302 operations. Its static liveness plan
uses 75,366,400 scratch bytes, excluding weights, input/output staging and
the graphics context. This is a calculated allocation size, not a measured
hardware peak. Tests simulate every buffer lifetime, check checkpoint
orientation and normalization, and cover resizing, suppression, finite
values, cancellation and allocation-free repeated CPU preprocessing.

`YoloGpuChecks.Run` in `dev/GpuChecks.csproj` compares all 8400 output rows
against a CPU evaluation using sparse synthetic weights through the full
network. The tensor diagnostic additionally checks focus packing, nearest
upsampling and max-pool padding using negative values. These diagnostics
pass on the AMD GPU. The released YOLOX-L weights also detect the visible
people in Desktop and FlowHMR video frames with repeatable output; timings
are in [GPU validation]. OpenCV resize rounding and upstream-logit parity
remain unverified. YOLOX-L, camera estimation and SAM body/hand features have
run over all 150 frames of a predecoded clip inside the AMD editor.
Production Windows decoding and complete motion-quality validation remain
required.

X's shared video-preview widget selects s&box's built-in `VideoPlayer`.
Source footage is streamed into the editor's preview filesystem with a
content hash, without transcoding or changing the original. One reusable
960×540 pixel buffer supplies the UI. Normal playback tolerates up to 100 ms
of clock drift; scrubbed/slow playback seeks at 30 fps. The engine API gives
a playback clock rather than decoded frame timestamps, so this adapter is
for preview only. The Desktop test clip now displays through this path as
WebM in the live Linux editor, with an 8-second duration and no preview
error. This engine build reports H.264 decoding as Windows-only; H.264/MP4
support is retained and still needs Windows validation. Live seek checks
returned stale or inconsistent frames, so this preview API is not used for
inference. Normal playback has been inspected; exact decoded-frame timing
remains unverified.

The upstream source is Apache-2.0, while its [model weights] are restricted to
non-commercial educational, research and nonprofit use. These terms are
separate from this library's MIT license. Adapted source files and their
license are listed in `THIRD_PARTY_NOTICES.md`; no model weights are bundled.
VGGT-Omega image, camera and transformer adaptations have separate FAIR
Noncommercial Research terms; see `LICENSE-VGGT`. The bicubic resampler's
Pillow portions retain the MIT-CMU terms in `LICENSE-PILLOW`.

[model weights]: https://huggingface.co/fafsaf1/flowhmr-0.46B#license

## Validation limits

The final acceptance tests must use downloaded video clips whose measured
duration is between 5 and 10 seconds. Synthetic tensors and imported motion
files do not satisfy that requirement. Record each source URL, measured
duration, selected model and memory setting, target rig, GPU and export
result. Three existing public videos and an eight-second segment of the
user's Desktop video are prepared; sources, exact durations and hashes are
in [video validation]. Both Base and Latest have completed predecoded
five-second captures and Human/Citizen exports. Production native decoding
and detailed motion-quality acceptance remain separate requirements.

[video validation]: video-validation.md

The current environment can build against the installed engine assemblies.
The editor build completed with zero warnings and errors. Both runtime
assemblies passed the installed engine's whitelist verifier, and the motion,
inference and download tests passed. Source formatting is checked in folder
mode because project-mode formatting requires unavailable local build IPC.
The editor now launches and is reachable through `sbox-cli` MCP. The AMD
GPU and network are accessible, real model downloads have completed, and
all five synthetic GPU diagnostic groups pass. Editor asset indexing needed
manual shader compilation and a restart as described in [GPU validation].
End-to-end UI capture with native Windows decoding, native ModelDoc import,
NVIDIA execution and complete motion-quality acceptance remain unverified.
