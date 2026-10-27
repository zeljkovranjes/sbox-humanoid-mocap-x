GPU validation
==============

The following checks passed inside the live s&box editor on 2026-10-27,
using an AMD Radeon RX 9070, RADV GFX1201, Mesa 26.2.5-arch3.1 and Vulkan
1.4.354. The editor compiled Humanoid Mocap X with zero errors.

| Check | Scope | Largest absolute error |
| --- | --- | ---: |
| Tensor library `dev/gpu_checks.csx` | All nine kernels, dependent dispatch, cancellation/readback and input rejection | 1.907349e-6 |
| `FlowGpuChecks.Run` | Six synthetic captures, eighteen model blocks and twenty sampling steps | 3.278255e-7 |
| `OmegaGpuChecks.Run` | Synthetic image encoder, aggregator and camera head, including saved all-zero bias masks | 6.705523e-7 |
| `SamGpuChecks.Run` | Synthetic DINOv3, body/hand decoders, camera/image conditioning and position tensors | 2.741814e-6 |
| `SamImageChecks.Run` | Reusable image adapter, body/hand conditioners, changed pixels, cancellation and invalid-input recovery | 1.192093e-6 |
| `YoloGpuChecks.Run` | Full detector network with sparse synthetic filters, all 8400 output rows | 9.313226e-10 |

These compare GPU results with CPU calculations; the tensor checks use
double-precision equations. They establish GPU dispatch and numerical
agreement for the covered inputs. They do not establish parity with the
upstream Python models or successful video-to-animation capture.

## Editor setup

The local checkout uses `Libraries/humanoid-mocap` and
`Libraries/tensor-inference` symlinks to the sibling repositories. These
development links are excluded locally and do not copy shared source.

This Linux engine fork initially failed to index the tensor shader assets.
Trusted editor automation compiled the nine `.shader` sources through the
engine's shader compiler. All nine compute programs compiled successfully.
The first dispatches still used missing shader pipelines; restarting the
editor after compilation resolved this. `mat_reloadshaders` alone did not.
The compile outputs are ignored build artifacts, not bundled source files.

Asset registration in this engine build also emitted malformed external
paths and unrelated prefab compilation errors. The successful numerical
checks above were collected after restarting; they do not prove that a
fresh install will automatically compile and mount all assets correctly.
Fresh-install shader discovery remains an acceptance check.

Build `dev/GpuChecks.csproj`, then invoke its diagnostic methods through
`sbox-cli` as described in [development]. Restart the editor before loading
a rebuilt diagnostic DLL. The diagnostic assembly is development tooling,
not an inference worker or part of the shipped capture path.

[development]: development.md

NVIDIA execution and full-model memory peaks have not been measured.
The five-second human fixture improved from 1148.781 to 576.639 seconds
after the register-tiled linear change. Every saved perception/motion
artifact and both Human/Citizen FBX files matched the baseline byte for byte.
The expanded tensor checks include output dimensions around the new 64x64
tile boundary; the largest absolute error remains 1.907349e-6. See
[video validation] for provenance, stage timings and acceptance limits.

A repeat using the same accepted kernels took 475.772 seconds with all eight
artifacts unchanged. This reflects substantial timing variation, not another
optimization. Competing editor load was uncontrolled; small gains require
adjacent baseline comparisons and repeat measurements.

Two concurrent feature extractors subsequently completed the same fixture
in 394.504 seconds (6m35s), again with all eight artifacts byte-identical.
That experiment duplicated model weights and reached a sampled combined GPU
use of 14,470 MiB with another editor open. Shared-weight extraction then
completed in 350.723 seconds with two feature lanes, 311.165 seconds with
three, and **272.033 seconds (4m32s)** with four feature lanes and two detector
lanes. All eight artifacts and the input fixture matched the accepted
baseline in independent SHA-256 comparisons. Shared-weight ownership and
disposal also passed a live GPU check.

The four-lane result meets the 3–5-minute model-stage target for this
five-second input. It excludes production decoding; native Windows H.264,
NVIDIA performance and lower-memory GPU acceptance remain unverified.
The live engine memory probe reported a 12,822,249,472-byte budget,
12,862,488,576-byte GPU memory value and 877,973,332 bytes used; Auto selected
four lanes. A full capture with default Auto then passed in 299.756 seconds
(28.108 detection, 34.678 camera, 224.801 features, 12.169 generation). The
same fixture and all eight artifacts matched `human-capture-register/` by
SHA-256. Logs show four-frame feature batches. Evidence is in
`human-capture-auto/`. This is a repeat acceptance of the default path, not
a speed improvement over the 272.033-second fixed-lane run.

### Rejected larger linear tiles

Four further layouts preserved all 27 checked output hashes: three large
linear shapes, two unchanged attention controls and eleven boundary shapes
with and without bias. Boundary checks included partial input dimensions and
63/64/65 and 127/128/129 output dimensions, with a double-precision reference.
None established a consistent speed improvement, so the accepted 4x4 shader
and dispatch remain in use.

| Layout | 1029x1280x3840 | 1029x1280x5120 | 1029x5120x1280 |
| --- | ---: | ---: | ---: |
| Accepted 4x4, before | 1.537 ms | 1.671 ms | 1.820 ms |
| 4x8 | 1.539 ms | 1.687 ms | 1.891 ms |
| 8x4 | 1.502 ms | 1.685 ms | 1.815 ms |
| 8x8, K-major shared memory | 1.670 ms | 1.702 ms | 1.860 ms |
| 8x4, K-major shared memory | 1.653 ms | 1.621 ms | 1.908 ms |
| Accepted 4x4, restored | 1.571 ms | 1.677 ms | 1.841 ms |

Each value is the best of three 64-dispatch batches after one warm-up batch,
including a one-float readback. All layouts use 256 threads per group. The
K-major layouts pad shared rows by four floats and give each thread contiguous
outputs. These results were collected in one editor session with another
editor open. Small isolated differences are not reliable full-capture gains.
Sources, scripts and raw reports are retained under `codex-tiles/` in the
local validation directory. The restored shader passed the same hash checks.

[video validation]: video-validation.md

For long diagnostic calls, launch `sbox-cli` with `--request-timeout 3600`.
Its default 300-second request timeout closes a CLI-owned editor when
exceeded. Changing only `--idle-timeout` does not extend active requests.

## Released SAM decoders with MHR feedback

`SamReleasedChecks.Run` passed inside the same AMD/Linux editor on
2026-10-27. It loads the released body and hand decoder weights, pose/camera
heads, the complete MHR body model and keypoint feedback projections.
Each decoder runs six layers, five intermediate pose-feedback updates and
the final pose prediction. The CPU and GPU consume identical synthetic
image features on a six-patch grid, with learned initial pose/camera,
CLIFF crop conditioning, invalid point prompt and keypoint tokens.

| Decoder | Maximum token difference | Maximum keypoint difference, metres |
| --- | ---: | ---: |
| Body | 0.00000619888 | 0.000000298023 |
| Hand | 0.00000977516 | 0.000000111759 |

Both GPU runs repeated exactly, including final keypoints. MHR evaluation
and pose feedback run in C# on a worker thread between GPU decoder layers.
These differences measure CPU/GPU agreement on the fixture; they are not
measurements of capture accuracy. Real-frame checks are recorded separately
below; full video capture remains acceptance work.

Build `dev/GpuChecks.csproj` and use `dev/sam_released_checks.csx` through
`run_csharp`. Supply absolute paths in `sam-model-path.txt`,
`mhr-model-path.txt` and `gpu-checks-path.txt` inside the session exchange
directory. The diagnostic DLL must be loaded into a fresh editor after
rebuilding it.

## Real-frame SAM body pass

`SamFrameChecks.Run` completed detection, 512 × 512 cropping, DINOv3 image
encoding, six decoder layers with five complete MHR feedback updates, and
final body prediction on the Desktop clip's frame at 5 s. Image encoding
repeated exactly; its first/repeated execution took 2111/1874 ms. Body
decoding and final prediction took 585 ms. These times exclude checkpoint
loading and construction.

The 70 projected landmarks visually follow the character's head, arms,
hips, legs, boots and hands. Depths were finite and positive, from 2.394 to
3.014 metres in the inferred camera frame. This is a qualitative check on a
rendered game character, not measured pose accuracy or an upstream parity
comparison. Separate hand passes and full video capture are not covered by
this body-only check.

`dev/sam_frame_checks.csx` writes projected landmarks, 3D landmarks, model
parameters and the 1024-channel body feature to `sam-frame-result.json` in
the editor session exchange directory. Supply the model path files above
and `sam-frame.json` with absolute `DetectorPath`, `RgbaPath`, `Width` and
`Height` fields. RGBA frames are developer fixtures; this does not validate
production video decoding.

## Real-frame body and hand features

`SamFeatureChecks.Run` passed on the AMD/Linux editor for two RGBA video
frames. The adapter performs body detection/cropping, predicts hand boxes,
mirrors the left image, runs each hand decoder with complete MHR feedback,
and writes body/left/right features in FlowHMR's required order. All 3072
channels, hand boxes and projected landmarks repeated exactly on each
frame. Cancellation before execution, concurrent-call rejection, invalid
camera rejection and recovery passed.

| Input frame | Selected person | First / repeated extraction |
| --- | --- | ---: |
| Desktop clip, 5 s | Rendered character | 6894 / 6828 ms |
| FlowHMR sample, 2 s | Woman on the left | 6813 / 6793 ms |

These times include three image encodings, crop preparation and three
decoders with MHR feedback, but exclude detection and model loading. Shared
model loading took 12,769 ms. They are isolated observations, not video
throughput or an optimization result. The Desktop body feature matched the
separate body-only diagnostic exactly across all 1024 channels.

The engine logged GPU fence waits exceeding 250 ms during image encoding.
This remains a responsiveness concern for capture integration; asynchronous
readback alone does not prevent a long command batch from delaying rendering.

Visual overlays place body and hand landmarks on the selected subjects.
The human frame includes motion blur and overlapping hands; this does not
establish individual-finger accuracy. No upstream numerical comparison or
video-to-animation/export acceptance has passed yet.

The first integration run rejected hand feedback because its constructor
was accidentally given the body selector. Passing the selected branch to
`SamKeypointFeedback` fixed the binding; the table records the successful
rerun. `dev/sam_feature_checks.csx` saves all features, boxes and landmarks
to `sam-features-result.json` for inspection and future comparisons.

## Checkpoint bias masks

The initial released Omega check failed during weight binding because the
port required a K-only bias mask in every layer. Inspection of the installed
checkpoint found all-zero masks in the first image, frame and inter-frame
blocks, while the camera head retained its K-only mask. Upstream's
[linear layer] multiplies the saved bias by the saved mask.

[linear layer]: https://github.com/facebookresearch/vggt-omega/blob/main/vggt_omega/models/layers/attention.py

`OmegaBlock.MaskedBias`, shared by CPU and GPU, now uses the saved binary
mask. The regression fixture reproduces the original constructor rejection;
full Omega fixtures now use all-zero image/aggregator masks. Non-binary
masks still fail validation. This changes weight binding, not the shaders.

After the correction, the full CPU suite and live Omega GPU comparison
passed. The released Omega checkpoint then completed single-frame camera
inference on the FlowHMR example at 2 s, prepared through `OmegaImage` at
512 × 288. The nine outputs were finite:

```text
translation: -0.000016095117, 0.000040577725, 0.00015494414
quaternion:   0.000011546072, -0.00024672505, -0.000104473904, -0.9999024
field of view: 0.5181201, 0.8959742
```

GPU execution/readback took 1927 ms, excluding checkpoint loading, model
construction and image preparation. This verifies released-weight execution
for one image; temporal camera accuracy and upstream parity remain untested.

## Released YOLOX-L weights

`ReleasedGpuChecks.Yolo` loaded the installed official checkpoint through
the restricted C# reader and ran person detection twice on each RGBA8 frame.
Results were identical between repeated calls:

| Input frame | People | Confidence | First / repeated call |
| --- | ---: | --- | --- |
| Desktop `videoplayback.mp4`, 5 s, 1080 × 1920 | 1 | 0.9490453 | 287 / 256 ms |
| FlowHMR example, 2 s, 1920 × 1080 | 2 | 0.9267242, 0.9184292 | 304 / 293 ms |

Times include CPU image preparation, GPU execution/readback and suppression;
they exclude loading the weights and constructing the detector. They are two
observations per input, not a throughput benchmark. Frames were extracted
with FFmpeg for this isolated diagnostic; production video decoding is not
covered. The detected boxes agree with the visible character/dancers, but
there is no independent upstream-logit comparison yet.

## Released FlowHMR weights

Both installed checkpoints loaded all denoiser weights, bound the full
1024-channel GPU sampler and completed twenty steps for three frames:

| Model | Output minimum / maximum | Sampling time |
| --- | --- | ---: |
| Latest | -2.528044 / 2.8310182 | 915 ms |
| Base | -3.0360627 / 2.8809352 | 874 ms |

Conditioning was synthetic: zero body features, identity camera rotations
and sine-wave initial values. All 1473 output values were finite. This checks
released-weight loading, binding and execution; it does not establish motion
quality, upstream numerical parity, or video reconstruction. Times exclude
loading and construction and are not capture benchmarks.
