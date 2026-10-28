Video validation
================

The following source videos were downloaded and prepared locally. `ffprobe`
measured each final clip between 5 and 10 seconds. Downloading and preparing
these inputs does not establish successful motion capture.

Files are in `/mnt/projects/_scratch/humanoid-mocap-x-validation/clips/`.
They are validation inputs, not assets bundled with the library.

| Clip | Duration | Size | Frame rate | Preparation |
| --- | --- | --- | --- | --- |
| `flowhmr-example.mp4` | 5.041667 s | 1920 × 1080 | 24 fps | Original file |
| `gvhmr-tennis.mp4` | 8.000000 s | 812 × 720 | 30 fps | Eight seconds starting at 1 s |
| `opencv-pedestrians.mp4` | 8.000000 s | 768 × 576 | 10 fps | Eight seconds starting at 2 s |
| `desktop-videoplayback.mp4` | 8.000000 s | 1080 × 1920 | 60 fps | Eight seconds starting at 2 s |
| `desktop-five-seconds.mp4` | 5.000000 s | 1080 × 1920 | 30 fps | First five seconds of the prepared Desktop trim |

Sources and SHA-256 hashes of the **prepared clips**:

- [FlowHMR example]:
  `094cc009fb7bee43146e35fbb41a3e5f036d6fdfd04528329fc53e5cd7a010a9`
- [GVHMR tennis]:
  `75673af4acd859e744bebd97d7d4f41459ddf2131988c172da0140230c5777c9`
- [OpenCV pedestrians]:
  `cfe7e35ac9fa8748ffa5788e84c21deb263cda4ac4ae1c82856fca659720546d`
- User-provided `/home/chomnr/Desktop/videoplayback.mp4`:
  `6b85f3ff30a2a804277b195e2c882f7096e7b93b4cd0e51afab52ff073c524d5`

[FlowHMR example]: https://raw.githubusercontent.com/flowhmr/flowhmr/main/assets/example/example.mp4
[GVHMR tennis]: https://raw.githubusercontent.com/zju3dv/GVHMR/main/docs/example_video/tennis.mp4
[OpenCV pedestrians]: https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi

The trimmed videos use H.264, CRF 18, the fast preset and fast-start MP4,
without audio. FFmpeg was used to prepare test inputs only; the library
does not launch it or require a separate inference process.

The FlowHMR example contains two dancers. Acceptance must identify the
selected person. Its visible hand motion includes blur, so it cannot by
itself establish accurate individual-finger capture.

The Desktop source is 19.966667 seconds and shows a full-body animated
character. Its original SHA-256 is
`f37ebdce7df605d84a08305d086885baa7d3309864abd2762e46b41f666aa616`.
The original file is unchanged. Its stylized appearance makes it a useful
additional case alongside the human footage.

The five-second, 150-frame Desktop test input has SHA-256
`22b24ecbb84993f61c32b2fe39869866a8559c33d176721a0ef62ec91d625504`.
Its separate five-second VP9 preview copy has SHA-256
`a8cf1b7cd4bb7f97faaef14ecb24a99d7e4f573bb3ab1fec10349d13b0a761cc`.
Both retain 1080 × 1920 dimensions. The MP4 remains the input reference.

## Editor decoding

The live Linux editor opened the third-person workspace with Human selected.
Its built-in decoder logged `H.264 decoding is only supported on Windows`
for the prepared MP4. H.264/MP4 input support is retained; the original and
prepared MP4 files were not replaced.

A separate `desktop-videoplayback.webm` test copy uses VP9, 1080 × 1920,
60 fps and exactly 8.000000 seconds. Its SHA-256 is
`830bb074a9cb33d32af83f6590c8504ee54a0452b33f0d294bd69f03d072d19b`.
The editor's built-in player displayed the animated character, reported an
8-second duration and delivered frames without an error. At one observation,
the preview clock was 3.530114 seconds. This establishes live WebM preview,
not deterministic inference frame extraction or completed capture.

The WebM copy was prepared for validation using FFmpeg/libvpx-vp9, CRF 30,
zero target bitrate, realtime deadline and CPU setting 6. The library has no
automatic conversion step or external decoder process. Windows H.264 preview
still needs a live check on that platform.

The first preview-only load incorrectly left the animation pane spinning
with "Preparing…" despite no active capture. `RetargetWindow.LoadVideo` now
sets its busy state from the actual capture task and retains an unavailable
provider's explanation. The live `dev/preview_idle_checks.csx` regression
check passes after restarting the editor: `Busy = false`, with the local
capture-unavailable message visible. The editor build has zero warnings and
errors. This fixes status reporting; it does not enable local capture.

## Frame extraction checks

SAM now extracts body/left/right features from individual RGBA frames of the
Desktop and FlowHMR clips inside the AMD/Linux editor. See [GPU validation]
for timings and scope. These frames were prepared with FFmpeg as diagnostic
inputs; this is not production video decoding.

[GPU validation]: gpu-validation.md

A separate five-second, 256 × 128, 30 fps lossless VP9 fixture exposed a
problem with using the engine preview API for deterministic capture.
Pausing in `OnLoaded` before the first pixel callback prevented delivery of
the first frame in this test. Allowing the first frame through before
pausing resolved that initial wait.

Subsequent paused `Seek` calls changed `PlaybackTime` and triggered pixel
callbacks, but the pixels did not match the requested times. Requests at
0.0333, 0.2, 1.5 and 4.9 seconds returned the initial keyframe; later seeks
also returned stale keyframes. Comparing all 150 reference frames identified
frames 0 or 120 where frames 1, 6, 45 or 147 were required. A changed pixel
callback and the reported playback clock therefore cannot establish a
correct inference frame. This path is not enabled for capture. Linux
preview remains available; deterministic frame extraction is unfinished.

Normal playback was also checked against all 150 reference frames with
`dev/video_playback_checks.csx`. It delivered 150 callbacks, but the first
contained invalid pixels; callbacks 1–149 matched source frames 0–148.
Continuing to call `Present` for 250 ms after `OnFinished` gave the same
result. Callback counts and playback timestamps alone therefore cannot
validate a complete inference timeline. The final source frame was missing.
These findings do not affect the independently predecoded GPU fixture.

## Acceptance record

The five-second Desktop fixture completed all model stages and Human/Citizen
FBX export on 2026-10-27. This is predecoded-frame acceptance, not a completed
editor video-decoding path. The result was opened in the actual X workspace
with Human selected and the supported WebM source preview. A visual sample
showed broad pose alignment; it does not establish complete finger accuracy
or foot-lock quality across the clip.

| Stage | Seconds |
| --- | ---: |
| Person tracking | 29.575 |
| Camera reconstruction, interval 10 | 40.025 |
| Body/left/right features | 1092.219 |
| Latest sampling and motion decoding | 11.868 |
| Total model-stage time | 1173.688 |

The AMD RX 9070-family GPU ran all stages inside s&box through C#/HLSL.
The shared retargeter produced 150 finite frames for both Human (94 target
bones, 1400636-byte FBX) and Citizen (95 target bones, 1415404-byte FBX).
Both FBX files passed shared importer frame-count checks. The source motion
contains 52 joints and four foot-contact tracks. Evidence is retained under
`desktop-capture-third/` in the local validation directory, including
`result.json`, `timings.json`, `capture.hmotion`, both FBX files and a native
editor screenshot. Body/hand extraction dominates this measured baseline.

Base also generated 150 frames from the same saved Desktop body/hand and
camera features, with the same initial-noise seed. Both target exports pass
finite-pose and FBX frame-count checks. `desktop-capture-base/` retains the
separate model-tagged motion and exports. This rerun reuses the perception
results and does not measure a second complete capture.

`dev/MotionChecks.csproj` checks every mapped finger on every exported frame
and compares target foot movement with stabilization enabled/disabled:

```sh
dotnet run --project dev/MotionChecks.csproj -- \
  /path/to/capture.hmotion \
  Assets/data/humanoid_mocap /path/to/quality.json
```

Both Desktop variants preserve all 30 Human finger joints and all 24 mapped
Citizen finger joints. Classic Citizen lacks the six separate pinky joints.
Every exported finger rotation differs from the solved rotation by less than
0.1 degree. Latest's Human finger excursions from the first pose are only
0.85–2.90 degrees on this clip; Base's are 24.83–89.23 degrees. Thus export
preserves the available finger motion, but this does not establish accurate
finger reconstruction, especially for Latest on this animated source.

For Latest/Human, average ankle speeds during adjacent samples with predicted
contact probability at least 0.8 fell from 0.1213 to 0.0032 m/s (left) and
0.1214 to 0.0087 m/s (right). Both toe tracks, both Citizen ankles/toes and
all Base target contact tracks also moved less with locking enabled. The
correction is active on actual generated motion. These predictions select
the samples; they are not measured contact ground truth. Reports include
the sample counts and retargeter correction notes in each `quality.json`.

The actual-human five-second FlowHMR sample also completed all stages on
2026-10-27. The detector tracked the man on the right across all 150 output
frames. The H.264 trim is exactly 5.000 seconds at 1920×1080 and 24 fps,
SHA-256 `d198f57e0615ac81996e4e2fdfaabc07beb45a385cd4df5d9b6954453e87c524`.
Developer-only FFmpeg decoding used `fps=30:round=up` to hold displayed source
frames on the 30 fps inference timeline. No frame-resolution reduction was
used. This still excludes production decoding.

| Stage | Seconds |
| --- | ---: |
| Person tracking | 35.074 |
| Camera reconstruction, interval 10 | 45.138 |
| Body/left/right features | 1059.517 |
| Latest sampling and motion decoding | 9.052 |
| Total model-stage time | 1148.781 |

Evidence is under `human-capture-first/`; Base generation from the same
features is under `human-capture-base/`. Both produced 150 finite frames and
FBX files for Human (1400620 bytes) and Citizen (1415388 bytes). Native X
preview screenshots were inspected for both variants, showing broadly
corresponding body poses with Human selected. Every mapped finger frame
passed the export-rotation check, and locking reduced movement on all
predicted ankle/toe contact tracks for both variants and rigs.

Latest's Human finger excursions were 1.04–5.14 degrees; Base's were
11.53–75.62 degrees on this human clip. The checkpoint hand-normalization
means and deviations are identical. The difference is present in generated
motion, rather than introduced by FBX export. Finger accuracy remains an
open quality question; these blurred hands provide no fine-grained ground
truth. Do not interpret small nonzero articulation as a full finger-accuracy
acceptance result.

Full editor capture, NVIDIA, Windows H.264 and native ModelDoc import remain
unverified.
For each completed run, record model revision, memory option, selected person,
target rig,
hardware, processing time, foot contact behavior, finger articulation,
scale, preview result and exported-animation result. Test both FlowHMR
variants and the supported Human/Citizen armatures. Numerical model tests
and synthetic motion exports are separate checks.

## Linear optimization

The same five-second human fixture was rerun on the same AMD GPU after
changing the dense linear shader to compute a 64x64 output tile with
16x16 threads. Each thread accumulates 4x4 outputs, reusing operands while
retaining each dot product's summation order. The matching C# dispatch
change is required; shader and dispatch must be updated together.

| Stage | Baseline seconds | Register-tiled linear seconds |
| --- | ---: | ---: |
| Person tracking | 35.074 | 47.991 |
| Camera reconstruction, interval 10 | 45.138 | 32.453 |
| Body/left/right features | 1059.517 | 487.215 |
| Latest sampling and motion decoding | 9.052 | 8.980 |
| Total model-stage time | 1148.781 | 576.639 |

This run took 9m37s versus 19m9s, approximately twice as fast. The input
frames, weights, model choice, initial noise, camera interval, image
resolution and both target rigs were unchanged. All eight saved artifacts
have identical SHA-256 hashes: person boxes, camera rotations, intrinsics,
3072-channel features, generated motion JSON, `.hmotion`, Human FBX and
Citizen FBX. This includes every saved finger, contact and root-motion
value; it establishes unchanged output for this fixture, not new accuracy
ground truth or NVIDIA performance.

Evidence is in `human-capture-register/`, including `result.json`,
`timings.json` and `comparison.json`. Repeated editor swap-chain/fence wait
messages occurred during the run. Unrelated UI edits failed to compile
mid-run, so the editor retained the prior assembly; the capture completed
and produced identical artifacts. Detection itself was unchanged and slower
in this run. These are individual complete model-stage measurements, not
averages across repeated captures, and exclude production decoding.

This linear-only run did not meet the 3–5-minute target. Later concurrent
extraction reached it, as recorded below. Further improvements must retain
the same capture settings and pass output comparison.

### Timing repeatability

After reverting the dense-attention experiment below, the accepted kernels
were rerun with temporary stage timers. The same fixture took 475.772 seconds
(7m56s): tracking 32.639, camera 31.496, features 405.198 and generation 6.439.
All eight saved artifacts again matched `human-capture-register/` byte for
byte. Evidence is in `human-capture-revert-profile/result.json` and
`claude/compare-revert.json`.

This is a repeat measurement, not a further optimization. The roughly 20%
timing variation means small differences between single captures are not
reliable speed gains. Another editor was open on the same GPU, so competing
load was not controlled. Compare candidates with adjacent baseline runs,
record total stage times and output differences, and repeat small apparent
gains under comparable load before accepting them. These times still exclude
production video decoding and do not meet the five-minute target.

## MHR vertex restriction

The capture loader now requests pose corrections only for vertices used by
the body or hand keypoint mapping. Each retained row uses the same
dot product as full evaluation. Skinning does not mix different vertices,
and unreferenced vertices have zero landmark weights, preserving the
keypoints consumed by later stages.

`human-capture-mhr-restrict/` completed all 150 frames and produced eight
artifacts with SHA-256 hashes identical to `human-capture-register/`,
including both rig exports. Total stage time was 519.628 seconds (8m40s):
tracking 50.951, camera 34.623, features 426.712 and generation 7.341.
This establishes unchanged saved output for the fixture. Concurrent load
and scheduling variation prevent attributing the total timing difference
to this change. This run did not meet the five-minute target.

## Concurrent feature extraction

Two independent SAM extractors processed adjacent frames concurrently,
overlapping CPU feedback with GPU work while preserving per-frame math.
`human-capture-lanes/` used exactly the same fixture and settings as
`human-capture-register/`. All eight artifacts again had identical SHA-256
hashes, including both rig exports.

| Stage | Seconds |
| --- | ---: |
| Person tracking | 43.388 |
| Camera reconstruction | 39.014 |
| Body/left/right features | 302.249 |
| Latest sampling and motion decoding | 9.853 |
| Total model-stage time | 394.504 |

This observed run took 6m35s; it still misses the five-minute target and
excludes production decoding. It used two full weight copies. A sampled GPU
memory reading reached 14,470 MiB of 16,304 MiB with another editor open;
that is combined device use, not isolated library memory.

Subsequent runs share immutable CPU and GPU weights across extractors while
keeping per-frame buffers separate. Each run used the same fixture and
produced the same eight SHA-256 hashes as `human-capture-register/`:

| Feature lanes / detector lanes | Detection | Camera | Features | Generation | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 / 1, shared weights | 36.837 | 27.951 | 279.504 | 6.431 | 350.723 |
| 3 / 1, shared weights | 42.726 | 26.458 | 234.349 | 7.631 | 311.165 |
| 4 / 2, shared weights | 27.759 | 27.968 | 208.105 | 8.202 | 272.033 |

The last run took **4m32s**, meeting the 3–5-minute model-stage target and
improving on the original 19m9s measurement by about 4.2 times. Evidence is
in `human-capture-lanes-shared/`, `human-capture-lanes3/` and
`human-capture-lanes4-det2/`. Both agents independently compared their
fixture and output hashes. Shared-weight lifetime checks also passed in the
live editor: disposing one extractor frees its private buffers without
freeing another extractor's shared weights; disposing the cache frees them.

These are individual runs with competing editor load, not controlled timing
averages. They exclude production H.264 decoding and do not establish native
Windows or NVIDIA performance. Auto lane selection now uses engine-reported
memory budget/use and caps manual selections; its live default-path check
is separate from these fixed-lane measurements. The subsequent default-Auto
run completed in 299.756 seconds with four-frame feature batches and all
eight artifacts unchanged (`human-capture-auto/`, including `comparison.json`).
Lower-memory GPUs may use fewer lanes and take longer. Further gains must
preserve output quality. The next requested target is 15 processing seconds
per video second: 75 seconds for this five-second fixture. It is not met.

## Body-contact solver validation

The shared library now adapts DigitalRune CCD for final torso contact correction.
The [body-contact report](body-contact.md) records provenance, Human/Citizen
continuity and scale tests, and the limits of bone-derived collision proxies.
The existing five-second capture was retargeted again on CPU: both rigs kept all
150 frames and passed finger/FBX checks, with no reported unresolved proxy
contacts. Evidence is in `body-clearance/actual-five-second-report.json`.
This intentionally changes corrected arm motion and FBX; the earlier performance
hash comparisons describe the solver version used for those runs.

Video headers now reject either dimension above 8192 pixels, more than
33,177,600 pixels per frame, and row strides outside the decoded width plus
4096 bytes of alignment. Checks run before managed RGBA allocation, including
format changes. H.264 decoding and Linux preview support are retained. Synthetic
buffer tests cover channel order, rotations, bottom-up rows and rejected headers;
native Windows decode remains a separate acceptance step.

## Native ModelDoc import attempt

The accepted Release Human and Citizen FBX files were tested through the live
Linux editor's asset system. The shared `VmdlWriter` generated separate models
that inherit the corresponding shipped model and reference the exported
animation. Their scale remained 1 because `ModelDocFbxExport` already bakes the
conversion into the FBX. Existing models and the user's scene were unchanged.

Both `asset_compile` requests returned success. An initial load reported the
imported animation and 96 Human / 95 Citizen native bones. However, the asset
system continued reporting invalid dependency information and did not recognize
the generated compiled files. Subsequent loads returned error models with zero
bones. The all-frame sequence sampling check therefore never ran. These results
do not pass native import acceptance.

Both compiled files existed on disk, so the editor's separate warning that the
ModelDoc window is unavailable on non-Windows builds does not establish that
resource compilation itself is unsupported. The cause of the reload failure
remains unconfirmed. Durable import and playback still require validation,
including on Windows.

Evidence is under `modeldoc-release/`: fixture paths, initial model metadata,
compiler/console output, the failed `acceptance.json`, probe scripts and the
preserved fixture sources/compiled files. All ten temporary fixture files were
copied and byte-checked before their unique directory was removed from the
library. No production source or engine code changed for this check.

## Rejected dense attention experiment

A separate dense-attention kernel shared keys/values across 64 queries.
It improved an isolated kernel benchmark, but the full five-second capture
took 576.378 seconds versus the accepted 576.639 seconds. That difference
does not establish a useful complete-pipeline improvement.

The output also failed the conservative preservation thresholds: feature
maximum absolute difference was 0.083147 (RMS 0.001883), source rotations
changed by up to 0.04833 degrees, and exported local rotations changed by
up to 0.09152 degrees. Maximum target world-position changes were 1.443 mm
for Human and 0.968 mm for Citizen, exceeding the 0.1 mm threshold. Contact
decisions were unchanged. Small isolated-kernel rounding differences thus
did not establish sufficiently preserved full-pipeline output.

The experiment was rejected and its shader, dispatch API and call-site
changes were removed. Evidence remains in `human-capture-dense/` and
`claude/compare-dense.json`; the accepted register-tiled linear version is
unchanged. The C# comparer uses the shared FBX importer, whose translations
are already normalized to centimetres, then converts them to metres.

## Exact per-query attention

The DINO encoder now uses a separate per-query attention kernel for uniform
key ranges. Its isolated comparison produced zero mismatches across 1.32 million
values and reduced that tested call from 8.42 ms to 1.11 ms. The shader preserves
the original product rounding; an earlier version that allowed fused operations
failed the exactness check and was not accepted.

The complete five-second capture was then compared against a new baseline with
the current body-contact solver. All eight artifacts were byte-identical:
boxes, rotations, intrinsics, features, flow motion, capture motion, and the
Human and Citizen FBX exports. Evidence is in `claude/baseline-current/` and
`claude/attention-query/`.

| Stage | Baseline seconds | Per-query seconds |
| --- | ---: | ---: |
| Person tracking | 28.405 | 28.979 |
| Camera reconstruction | 31.309 | 30.625 |
| Body/left/right features | 321.430 | 259.080 |
| Latest sampling and motion decoding | 7.131 | 7.160 |
| Total model-stage time | 388.274 | 325.844 |

This adjacent comparison reduced model-stage time by 62.430 seconds, about
16 percent. Reported elapsed capture times, including work outside these stages,
were 389.4 and 326.9 seconds. These runs used the predecoded fixture and do not
include production video decoding. Other applications were using the GPU, so
the older 272.033-second run is not a controlled comparison against this run.
The 75-second target remains unmet; NVIDIA and native Windows decoding still
require separate validation.

The subsequent `claude/attention-omega/` run also uses the per-query kernel for
Omega's uniform global/register attention ranges. Its eight artifacts remain
byte-identical to `claude/baseline-current/`. Model-stage times were 27.015 seconds
for detection, 22.697 for camera, 251.581 for features and 9.289 for generation:
310.582 seconds total (reported elapsed capture: 311.6 seconds). This is about
20 percent below the adjacent session baseline. The runtime reported capacity
for four feature lanes; competing applications remain a timing confounder.
This run still misses the 75-second target.

## Exact linear prefetch

The linear kernel now prefetches the next matrix tile into registers while
preserving each output's multiply-add order. Interleaved kernel comparisons
found zero output mismatches and improvements of 8.9–14.4 percent across the
tested production shapes. The full `claude/linear-prefetch/` capture again
produced eight byte-identical artifacts against `claude/baseline-current/`.

| Stage | Seconds |
| --- | ---: |
| Person tracking | 27.042 |
| Camera reconstruction | 21.966 |
| Body/left/right features | 245.053 |
| Latest sampling and motion decoding | 6.073 |
| Total model-stage time | 300.136 |

Reported elapsed capture time was 301.2 seconds, about 23 percent below this
session's 389.4-second baseline. This is still a predecoded-fixture measurement
under competing GPU load, and the 75-second target remains unmet.

A subsequent scheduler let each feature lane take the next frame immediately
instead of waiting at each group boundary. Its eight artifacts also matched,
but feature time changed only from 245.1 to 244.6 seconds. That single difference
does not establish a useful speedup; the scheduling change was reverted.
Evidence remains in `claude/lanes-steal/`. The accepted source retains grouped
feature lanes, the per-query attention kernels and linear register prefetch.

## Release compilation

The diagnostic builds above used Debug compilation. Rebuilding unchanged
sources in Release reduced the same five-second capture from 301.2 to
175.294 seconds elapsed. Both agents independently compared all eight saved
artifacts with `claude/baseline-current/`; every file was byte-identical.
Evidence is in `claude/release-unchanged/`, including the capture, timings,
result and sampled GPU utilization.

| Stage | Debug seconds | Release seconds |
| --- | ---: | ---: |
| Person tracking | 27.042 | 24.699 |
| Camera reconstruction | 21.966 | 17.122 |
| Body/left/right features | 245.053 | 126.213 |
| Latest sampling and motion decoding | 6.073 | 6.344 |
| Total model-stage time | 300.136 | 174.378 |

The source-published libraries also inherited Debug compilation in consumer
editors. All three `.sbproj` files now set
`Metadata.Compiler.ReleaseMode` to `Release`. After an editor restart, its
native compilation succeeded with zero errors, and the loaded runtime/editor
assemblies reported JIT optimization enabled. Benchmark diagnostics must also
be built with `-c Release`; compare candidates in the same configuration.

This is a roughly 42 percent reduction from the preceding Debug capture,
with unchanged saved output for this fixture. It meets the original 3–5-minute
milestone and corresponds to about 35 processing seconds per video second.
It still exceeds the requested 75 seconds per five-second clip. The test uses
predecoded frames on AMD under competing GPU load; native Windows H.264 input,
NVIDIA performance and detailed motion accuracy remain separate acceptance
checks. Release compilation does not establish those results.

## Exact tiled convolution

The convolution kernel now computes a 64×64 output tile, keeping 4×4 outputs
per thread. It retains the original image-patch indexing, ascending order of
products and final bias addition. The C# dispatch covers the larger tile with
ceiling division by four on both axes.

Nine production-shaped kernel comparisons produced no bit mismatches. The
primitive GPU checks also passed against their CPU reference, with largest
absolute error 0.000001907349. They include optional bias, tiny channel counts,
odd reduction sizes, padding, stride and a new case crossing both 64-element
tile boundaries: 90 output pixels, 70 channels and 45 kernel elements.

The full `claude/conv-tiled/` capture produced eight byte-identical artifacts
against `claude/baseline-current/`, independently checked by both agents.
Compared with the preceding Release capture:

| Stage | Previous Release seconds | Tiled convolution seconds |
| --- | ---: | ---: |
| Person tracking | 24.699 | 12.766 |
| Camera reconstruction | 17.122 | 21.380 |
| Body/left/right features | 126.213 | 129.294 |
| Latest sampling and motion decoding | 6.344 | 8.390 |
| Total model-stage time | 174.378 | 171.830 |
| Elapsed capture time | 175.294 | 172.862 |

Detection fell by 11.933 seconds, about 48 percent. Other stages increased
by 9.386 seconds on unchanged code. Competing GPU use and scheduling remain
timing confounders; these individual runs do not establish a repeatable
whole-pipeline percentage improvement. Production decoding is excluded and
the requested 75-second target remains unmet.

The 128×128 output-tile and 32-element reduction-tile linear experiments
retained exact output but were slower and were removed. The accepted linear
kernel is unchanged.

## Rejected checkpoint prefetch

Two candidates read the next stage's checkpoints into the operating system's
file cache during GPU work. Each uses one reusable 1 MiB buffer and discards
the bytes. The second candidate adds explicit warning and cancellation handling,
waits for its reader on disposal and uses ordinary file-cache policy. Both
captures started with the checkpoint files evicted from the file cache and
preserved all eight artifacts byte for byte, independently checked by both
agents.

The first candidate took 160.474 seconds; the revised candidate took 184.294
seconds, versus 172.862 seconds for the preceding convolution capture. Repeating
the revised candidate took 201.628 seconds with all eight artifacts still
identical. This does not establish a repeatable total-time improvement. The
prefetch implementation and its tests were removed, and the processor was
restored byte for byte to the accepted convolution version already in the
provider snapshot.

Whole-machine memory samples during the revised capture reached 26,537 MiB used
and 4,412 MiB minimum available. Swap grew from 5,076 MiB to a sampled peak of
9,873 MiB. The first candidate's memory trace starts 34.38 seconds after capture
began, so it cannot establish that run's full peak. The convolution baseline has
no matching memory trace; swap growth cannot be attributed solely to prefetch.
Reclaimable file-cache pages alone do not establish a negligible memory cost.
The repeated capture also increased memory pressure. Restarting the editor
released substantial memory, but that does not distinguish retained live
objects, uncollected managed allocations and native or driver caches. Repeated
jobs in one editor still require memory-lifetime acceptance for hosted use.

Evidence is in `claude/prefetch/`, `claude/prefetch-v2/`,
`claude/prefetch-v2-repeat/` and
`prefetch-review.json` under the local validation directory. These are AMD
predecoded-fixture checks; Windows cache behavior remains unmeasured.

## Direct checkpoint reads and guarded capture

The shared reader now reads complete contiguous FP32 storage directly into its
output array on little-endian hosts. Other data types and tensor views keep
the preceding conversion path. Engine-independent tests preserve exact bits,
including NaNs and signed zero, and cover multi-chunk reads, cancellation and
the absence of the duplicate storage allocation. All 39 test groups pass; the
Release editor and GPU diagnostic builds have zero warnings or errors.

Seven warm reads of one 16 MiB Omega tensor allocated 33,554,616 bytes before
and 16,777,376 bytes after the change. Tensor hashes were identical. Median
read time was 2.1435 versus 1.6003 ms; this isolated measurement does not
establish whole-capture throughput.

A fresh-editor five-second capture used the unchanged 150-frame human fixture
and model settings. Its diagnostic monitor sampled memory once per second
and cancelled if available system RAM fell below 4 GiB or system swap grew
by more than 2 GiB. These are test guards, not production capture limits.
Tracking, camera and body/hand features completed in 13.436, 19.809 and
141.685 seconds. The guard cancelled during final motion generation; normal
cancellation drained the work after 183.417 seconds total. No process kill
or out-of-memory exception ended the capture.

At cancellation, swap had grown by 2.069 GiB with 6.242 GiB RAM available.
Across the complete trace including cancellation drain, swap growth reached
2.618 GiB and available RAM reached a minimum of 5.386 GiB. The final managed
estimate was 9.794 GiB and process RSS was 12.491 GiB. A concurrent UI
hotload and another project's editor affected this run; it is neither a
clean performance comparison nor a completed uninterrupted capture.

The four completed perception outputs were byte-identical to
`claude/baseline-current/`. In a separate fresh editor, generation/export
from those saved features completed in 4.067 seconds. Both rigs have 150
finite frames and pass FBX round-trip checks. All eight final artifacts are
byte-identical to the baseline. This separate completion preserves output
evidence but does not turn the cancelled run into a full-capture pass. The
generation timing also overlapped another agent's CPU build.

After generation, one diagnostic full collection took 0.460 seconds. The
managed estimate fell from 3,018,879,856 to 1,100,283,344 bytes; committed GC
memory fell from 3,533,152,256 to 3,356,307,456 bytes, and process RSS from
6,124,068,864 to 5,948,264,448 bytes. This shows that substantial data from
that generation phase was collectible while much allocated heap capacity
remained committed. It does not prove a leak or explain the full perception
run's memory growth. No production collection policy was added.

Evidence is retained in `codex-direct-read/` and
`codex-direct-read-generation/` under the local validation directory, and
`_scratch/checkpoint-read-probe/` contains the reader measurements, source
fingerprints and diagnostic scripts. The original failed capture report is
preserved alongside the separate generation result and artifact comparison.
Repeated-capture memory acceptance and the 75-second target remain unmet.

## Detection lane experiment

A separate candidate removed detection's two-lane cap, retaining the user's
parallel-frame choice, four-lane maximum and existing VRAM capacity check.
Its fresh-editor run used four detection lanes and the unchanged five-second
fixture after evicting model files from the file cache. Detection took
11.756 seconds, and person boxes matched the baseline byte for byte. This
is one sample, about one second below the preceding tiled-convolution run;
it does not establish repeatable whole-capture improvement.

The memory guard cancelled the run after 25.150 seconds during the camera
stage. The cancellation stack is in the first GPU camera batch, after model
upload. Available system RAM fell from 11.311 GiB to 3.940 GiB, below the
4 GiB guard; maximum swap growth was 1.090 GiB. Cancellation drained normally,
and the owning editor was closed afterward. Camera, feature and final export
comparisons were not reached. This candidate was held out of the provider
archive pending the complete capture recorded below.

Evidence is in `claude/detect-lanes/` and the independent
`detection-lanes-review.json` under the local validation directory. This
run demonstrates the need to reduce memory pressure before another complete
performance comparison; it does not establish a memory leak or its cause.

## Staged Omega loading

A subsequent candidate loads the camera model in four parts: image encoder and
tokens, frame blocks, inter-frame blocks, and camera head. Each part is read
off the editor thread and uploaded on the main thread. The GPU operations
and tensors remain the same. Releasing each part's references limits live
CPU weights; it does not itself bound GC heap capacity or native staging.

The preceding Release assembly's generated camera state machine retains its
full CPU weight field throughout the awaited camera run. A metadata-only
inspection confirmed that field is not cleared after GPU construction. The
installed engine defines its upload call as synchronous. This supports
shortening CPU-weight lifetime without changing the numerical operations.
The inspection and assembly hash are in
`_scratch/mocap-camera-il-review/baseline.json`.

With staged loading, a fresh-editor run completed detection in 11.816 seconds
and camera reconstruction in 16.579 seconds. Person boxes, camera rotations
and intrinsics were byte-identical to the baseline; fixture settings were
unchanged. The run also included the then-unaccepted detection-lane candidate.

Camera-stage samples peaked at 2.924 GiB estimated managed memory, versus
about 5.2 GiB in the preceding failed camera attempt. Available system RAM
stayed above 9.29 GiB in that interval. Stage boundaries are approximated from
completed-stage durations and the memory timestamps; machine background load
and initial available RAM differed between runs. This is not an isolated
whole-capture performance comparison.

The guard later cancelled while loading MHR corrective weights for body/hand
features. Cancellation drained after 41.150 seconds. Across the complete
trace, minimum available RAM was 3.335 GiB, maximum estimated managed memory
6.559 GiB and maximum swap growth 1.555 GiB. Feature and final export
comparisons were not reached. No production forced collection was added.

Evidence is in `claude/omega-staged/` and the independent
`omega-staged-review.json` under the local validation directory. The earlier
null-after-upload camera probe was abandoned without a run. Staged loading
and detection-lane changes were held out of the provider archive pending
the complete capture recorded below.

## Shared MHR projection and guarded full capture

Capture lanes now share the immutable MHR corrective projection and sparse
weights. Each lane retains separate features, hidden values and restricted
vertex selections. Four lanes therefore use one 663,804,000-byte projection
instead of four: a calculated reduction of 1,991,412,000 bytes, with unchanged
dot products. Tests clone an already restricted instance, verify the clone
is unrestricted and compare concurrent, different-pose runs with independent
sequential references. The Omega part tests also reject tensors belonging
to other parts. All 40 engine-independent test groups pass.

A fresh-editor full capture combined this sharing with staged Omega loading
and four detection lanes. The unchanged five-second, 150-frame fixture
completed in 172.642 seconds without either memory guard firing:

| Stage | Seconds |
| --- | ---: |
| Detection | 11.718 |
| Camera | 15.064 |
| Body/hand features | 139.348 |
| Motion generation | 5.606 |

All eight artifacts matched `claude/baseline-current/` byte for byte: boxes,
camera rotations, intrinsics, features, Flow motion, shared motion, Human FBX
and Citizen FBX. Both target exports contain 150 finite frames, retain their
94/95-bone rigs and pass FBX round-trip checks. This preserves the reference
output; it does not establish anatomical accuracy beyond that reference.

The one-second memory trace contains 174 samples. Available system RAM
started at 11.161 GiB and stayed above 6.579 GiB. Peak estimated managed memory
was 7.248 GiB, peak process RSS 10.917 GiB and maximum system swap growth
0.0526 GiB (about 54 MiB). No forced collection policy was added. Another
project's editor was open, and a short CPU test-project build overlapped the
capture. The total is close to the preceding 172.862-second accepted run;
these samples do not establish repeatable total-time improvement.

Evidence is in `claude/mhr-shared/` and the independent
`mhr-shared-review.json` under the local validation directory.

A subsequent capture in a separate instrumented assembly, intended to profile
feature phases, cancelled at the same swap-growth guard after 42.139 seconds.
At cancellation the monitor reported 5.800 GiB available RAM and 2.033 GiB
of system swap growth. It stopped inside the SAM image encoder and produced
no completed feature/export comparison. Its result and memory trace remain
in `claude/sam-profile/`; it is not a successful performance sample. The
guard measures system-wide pressure, so this result alone cannot identify
the cause. One successful fresh-editor run does not establish repeated-job
memory acceptance. That requirement and the 75-second target remain open.

## Six concurrent frames

The maximum local capture lane count is now six. Auto and manual requests
remain capped by the GPU memory budget; model resolution, sampling steps,
camera interval and numerical operations are unchanged. Tracking still
consumes completed detection frames in order.

A controlled feature-only comparison used the same uninstrumented assembly
and script, explicit requested/resolved lane counts, a fresh editor for each
run and evicted SAM model files:

| Requested/resolved lanes | Feature seconds | Mean GPU busy | Minimum available RAM | Maximum swap growth |
| --- | ---: | ---: | ---: | ---: |
| 4 / 4 | 127.81 | 84.5% | 5.45 GiB | 0.40 GiB |
| 6 / 6 | 120.23 | 87.2% | 5.83 GiB | 0.36 GiB |

Both feature outputs matched the baseline exactly. These single-run stage
samples exclude detection, camera and final generation. The earlier
instrumented phase measurements include logging and overlapping lane waits;
they are not the four-lane timing control.

The subsequent full Auto capture completed the unchanged five-second,
150-frame fixture in 148.057 seconds without a memory-guard cancellation:

| Stage | Seconds |
| --- | ---: |
| Detection | 7.694 |
| Camera | 15.725 |
| Body/hand features | 118.953 |
| Motion generation | 4.699 |

All eight saved artifacts are independently byte-identical to
`claude/baseline-current/`. Both Human and Citizen exports retain 150 finite
frames and pass their FBX round trips. The 149-sample memory trace starts at
11.739 GiB available system RAM and reaches a minimum of 5.097 GiB. Peak
estimated managed memory is 7.705 GiB, peak RSS 10.670 GiB and maximum system
swap growth 0.601 GiB. The editor was closed after the run.

Evidence is in `claude/six-lanes/` and `six-lanes-review.json`. The preceding
172.642-second full run overlapped a CPU build, so its entire time difference
cannot be attributed to the lane change. These AMD predecoded-fixture runs
do not establish other-GPU throughput or repeated jobs in one editor. The
75-second target remains unmet.

Host Auto reserves the maximum lane count before starting a job and passes
that same bound to capture. The host tests cover maximum configuration values,
rejected over-limit requests and a short clip whose six-lane feature reserve
holds the queue when four lanes would fit. The operator can explicitly retain
four lanes with `MOCAP_LANES=4`; see [the host guide](editor-host.md).

## Observed detail

SAM 3D Body's per-frame keypoints now refine FlowHMR's motion. Fingers follow
the hand passes, head pitch follows the face keypoints, and, when the camera
turns less than 2° over the clip, the root follows SAM's pelvis position.
Both validation clips ran on an RTX 2080 Ti with the Linux worker. The
measurements compare each capture before and after refinement against SAM's
own observations:

| Clip | Root offset at the end | Largest lateral offset | Median head-pitch disagreement | Finger curl range |
| --- | ---: | ---: | ---: | ---: |
| Breakin, 272 frames | 0.30 → 0.02 m | 0.36 → 0.14 m | 11.6° → 4.1° | 1.6° → 18.9° |
| Five-second clip, 150 frames | 0.04 → 0.03 m | 0.04 → 0.02 m | 20.3° → 0.8° | 1.9° → 19.5° |

Offsets are horizontal, in camera axes, relative to the first ten frames.
Finger curl is the mean proximal-joint bend of the eight non-thumb fingers; the
range runs from its 5th to 95th percentile over the clip. FlowHMR held nearly one
hand pose throughout. The head-pitch constant (3.15° between the head's forward
axis and the ear-to-nose line) comes from the SMPL template's COCO face vertices.

The breakin fixture's boxes, camera rotations, intrinsics, features and
FlowHMR motion stay byte-identical to the run before this change; only
`capture.hmotion` and both FBX exports differ. Hand-pass frames are used in
proportion to how well each hand pass agrees with the body pass in the image:
63% of hand frames on the breakin clip, whose gloved hands are about 30 pixels
across, and 91% on the five-second clip. On the breakin clip's idle frames SAM
sees the hands less curled than FlowHMR's default pose (proximal bend 22–38°
against 50–55°). A camera that moves without turning is not detected, and its
motion would be read as the person's.
