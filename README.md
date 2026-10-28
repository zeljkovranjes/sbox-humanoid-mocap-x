Humanoid Mocap X
================

[<img alt="github" src="https://img.shields.io/badge/github-sbox--humanoid--mocap--x-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">][github]
[<img alt="docs" src="https://img.shields.io/badge/docs-development-66c2a5?style=for-the-badge" height="20">][development]
[<img alt="build status" src="https://img.shields.io/github/actions/workflow/status/zeljkovranjes/sbox-humanoid-mocap-x/ci.yml?branch=main&style=for-the-badge" height="20">][actions]

[github]: https://github.com/zeljkovranjes/sbox-humanoid-mocap-x
[development]: docs/development.md
[actions]: https://github.com/zeljkovranjes/sbox-humanoid-mocap-x/actions?query=branch%3Amain

This library provides [`MocapWindow`], a third-person workspace with built-in
Humanoid Mocap rigs, preview, foot locking and FBX export, with optional hosted
processing and local C#/HLSL FlowHMR inference.

[`MocapWindow`]: Editor/MocapWindow.cs

Download the latest [release] and put it in your project's `Libraries`
folder. Humanoid Mocap and Tensor Inference are built in and do not need to be
installed.

```text
Libraries/humanoid-mocap-x/     this repository
```

[release]: https://github.com/zeljkovranjes/sbox-humanoid-mocap-x/releases/latest

On Linux, capturing on this computer also needs FFmpeg (`ffmpeg` and `ffprobe`
on `PATH`). Open **View > Humanoid Mocap X**, choose where videos are processed,
then drop in a video. **Advanced** opens Mocap adjustments, where **Open motion…** loads
an existing full-body `.hmotion` capture.

To keep capture off this computer's GPU, choose **Processing machine (SSH)**
and enter a Linux machine's `user@host` and the path of `humanoid-mocap-worker`
there. Sign-in must work with an SSH key; **Check** confirms the worker starts
before the choice is saved. See [processing machine].

[processing machine]: docs/processing-machine.md

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
Code copied from Humanoid Mocap keeps its
<a href="THIRD_PARTY_NOTICES.md#humanoid-mocap">notices</a>.
FlowHMR weights have separate
<a href="https://huggingface.co/fafsaf1/flowhmr-0.46B#license">non-commercial
terms</a>. No model weights are bundled.
</sup>
