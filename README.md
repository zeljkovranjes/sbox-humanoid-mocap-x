Humanoid Mocap X
================

[<img alt="docs" src="https://img.shields.io/badge/docs-development-66c2a5?style=for-the-badge" height="20">][development]

[development]: docs/development.md

This library provides [`MocapWindow`], a third-person workspace that reuses
Humanoid Mocap's rigs, preview, foot locking and FBX exporter, with optional
hosted processing and local C#/HLSL FlowHMR inference.

[`MocapWindow`]: Editor/MocapWindow.cs

Use these local projects in s&box until the shared changes are published:

```text
sbox/libraries/humanoid-mocap/humanoid-mocap.sbproj
sbox/shaders/compute/tensor-inference/tensor-inference.sbproj
sbox/libraries/humanoid-mocap-x/humanoid-mocap-x.sbproj
```

Open **View > Humanoid Mocap X**, choose where videos are processed, then drop
in a video. **Advanced** opens Mocap adjustments, where **Open motion…** loads
an existing full-body `.hmotion` capture. The published Humanoid Mocap package
does not yet include the constructor this workspace requires.

<br>

## Details

- Human is the initial target. Classic Citizen and custom VMDL, FBX and GLB
  targets use the shared retargeter. Exported FBX uses ModelDoc import scale 1.
- Foot locking is enabled by default for target retargeting. Source-skeleton
  export preserves the capture without target corrections.
- Settings > **Processing > Setup** chooses between this computer (download
  the models) and a provider: enter its address, sign in through the browser,
  and the editor shows your profile, limits and queue. Capture stays disabled
  until a choice is saved. Providers send the selected video,
  Base / Latest choice and camera interval to their host.
- Providers process videos with the [in-editor host]: the library serves the
  capture API from an s&box editor on Windows and runs the same GPU pipeline as
  local capture. The [provider guide] covers the website, sign-in, roles and
  HTTPS. The standalone C# service in `services/humanoid-mocap-host` is only a
  starter for custom processors.

[in-editor host]: docs/editor-host.md
[provider guide]: docs/provider-guide.md
- `FlowMotionAdapter` converts validated numeric SMPL-H motion into the shared
  metre-based format. It does not reconstruct motion from video.
- **Mocap adjustments > Processing > Setup > Local models** has download
  buttons for the selected FlowHMR checkpoint, SAM 3D Body DINOv3,
  VGGT-Omega 1B-512 and YOLOX-L person detection, plus the separate SMPL-H
  neutral skeleton.
  Transfers show progress and can be cancelled. Opening the workspace or
  starting capture does not download weights.
- Local capture uses Windows Media Foundation, including H.264 decoding.
  Its production integration still needs a Windows acceptance run. Linux
  supports WebM preview, saved-motion export and hosted capture; its engine
  preview decoder cannot yet supply a reliable inference timeline.

Build commands and remaining work are in [development].

The [web example] adds Discord OAuth to a processing host. The [video checks]
record the first five-second predecoded GPU capture and Human/Citizen exports;
Windows capture and motion-quality acceptance remain unfinished.

[web example]: examples/hosted-web/README.md
[video checks]: docs/video-validation.md
Host setup and the HTTP contract are in [DOCUMENT.MD].

[DOCUMENT.MD]: DOCUMENT.MD

<br>

#### License

<sup>
Original code is licensed under the <a href="LICENSE-MIT">MIT license</a>.
FlowHMR adaptations retain the
<a href="THIRD_PARTY_NOTICES.md">upstream Apache-2.0 terms</a>.
VGGT-Omega adaptations retain the
<a href="LICENSE-VGGT">FAIR Noncommercial Research License</a>.
DINOv3 adaptations retain the <a href="LICENSE-DINOV3">DINOv3 License</a>.
SAM decoder adaptations retain the <a href="LICENSE-SAM">SAM License</a>.
Pillow resampling adaptations retain the
<a href="LICENSE-PILLOW">MIT-CMU license</a>.
FlowHMR weights have separate
<a href="https://huggingface.co/fafsaf1/flowhmr-0.46B#license">non-commercial
terms</a>. No model weights are bundled.
</sup>
