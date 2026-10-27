Model sources
=============

The editor downloads models only after a button click. No weights are bundled
and the capture path does not initiate downloads. On 2026-10-27, the C#
downloader installed and verified both FlowHMR variants, SAM including its
MHR asset, VGGT-Omega and YOLOX-L in the local model cache.
The separate neutral SMPL-H asset was downloaded and hash-verified for the
decoder on the same date; it has its own manual download button.

## Selected sources

| Package | Download source | Revision |
| --- | --- | --- |
| FlowHMR Base / Latest | [FlowHMR] | `1eb105e554e0cfc1846ea5feac8c2d516a36d319` |
| SAM 3D Body DINOv3 | [SAM mirror] | `566ff0482ef71b7a11d0874e2c41e5a801851343` |
| VGGT-Omega 1B-512 | [Omega mirror] | `a8c3a718e0cf78e9e4c6847229efea793d37f060` |
| YOLOX-L detector | [YOLOX release] | `0.1.1rc0`, asset `42724954` |
| SMPL-H neutral | [SMPL-H mirror] | `363972831b553e4f5b62d11d509e98830d930c97` |

[FlowHMR]: https://huggingface.co/fafsaf1/flowhmr-0.46B
[SAM mirror]: https://huggingface.co/mingjiexie/sam-3d-body-dinov3/commit/566ff0482ef71b7a11d0874e2c41e5a801851343
[Omega mirror]: https://huggingface.co/1kaiser/vggt-omega-jax/commit/a8c3a718e0cf78e9e4c6847229efea793d37f060
[YOLOX release]: https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0
[SMPL-H mirror]: https://huggingface.co/lithiumice/models_hub/tree/363972831b553e4f5b62d11d509e98830d930c97/4_SMPLhub/smplh

FlowHMR downloads use the LFS digest and length from the pinned upstream
revision. Offline capture verification uses the same recorded values.
The installed files use these fixed values:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `flowhmr_base.ckpt` | 1841700869 | `77b903ba545669bafd6f7709c0f1748f5fa030a4f25f67392c8c082a21f5e672` |
| `flowhmr_latest.ckpt` | 1841702021 | `b7eb06b119feaf5ad7287bafdb2fff38d42dfc22ea9d256f8cdd2ada26dfc53a` |
| SAM `model.ckpt` | 2109129346 | `b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf` |
| SAM `assets/mhr_model.pt` | 696110248 | `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` |
| `vggt_omega_1b_512.pt` | 4576706117 | `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` |
| `yolox_l.pth` | 434357141 | `1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917` |
| SMPL-H `neutral/model.npz` | 87259106 | `a60a7e29d33f09ef1a6352907e2f485e161f471330ca40592764374063e751df` |

SAM values come from the mirror commit's LFS pointers. The Omega digest is
published on its [file page]; its byte count and digest also appear in the
[Meridian mirror's inventory]. These identify the selected mirror artifacts.
The official gated file pages did not expose hashes to this environment, so
identity with the original publisher has not been independently verified.
The SMPL-H digest is published on its [mirror file page]. Its provenance
likewise has not been independently verified against an original download.

[mirror file page]: https://huggingface.co/lithiumice/models_hub/blob/363972831b553e4f5b62d11d509e98830d930c97/4_SMPLhub/smplh/neutral/model.npz

YOLOX-L comes directly from its publisher's GitHub release. GitHub reports
the byte count but no asset digest; the fixed hash above was calculated
from the downloaded release asset. Its Apache-2.0 license is downloaded
from release commit `e1052df71842031413f6030723c3607b839c80ce`.

[file page]: https://huggingface.co/1kaiser/vggt-omega-jax/blob/main/vggt_omega_1b_512.pt
[Meridian mirror's inventory]: https://huggingface.co/t8star/Meridian-Comfy

The SAM button also downloads `model_config.yaml` and `LICENSE` from its
pinned mirror revision. The Omega license comes from the [pinned original
source]. These support files are limited to 64 KiB and replaced atomically;
they are not executable and do not share the weight digest checks. Transfers
stream to temporary files. Checkpoint length and hash must match before a
file is installed or reused. Cancellation deletes only the unfinished file.

[pinned original source]: https://raw.githubusercontent.com/facebookresearch/vggt-omega/39a0cb8af88554f15ddcb5354cd52bde588fa014/LICENSE

## Requirements and remaining validation

Original sources are [Meta SAM] and [Meta Omega]. Original licenses continue
to apply to mirrored files. FlowHMR requires SAM's DINOv3 variant and the
512-resolution Omega checkpoint; the downloader does not substitute ViT-H
SAM or 256/416 Omega weights.

[Meta SAM]: https://huggingface.co/facebook/sam-3d-body-dinov3
[Meta Omega]: https://huggingface.co/facebook/VGGT-Omega

FlowHMR's released checkpoints omit the body-model buffers. Its decoder
requires the separate 52-joint, 6890-vertex, sixteen-shape SMPL-H neutral
model. The C# loader reads only its numeric joint regressor, template,
shape directions and hierarchy; it does not execute pickle objects.
The button downloads `LICENSE.txt` before the asset. The [SMPL-H license]
permits noncommercial scientific research, education and artistic projects,
and restricts commercial use and redistribution. It is separate from this
library's MIT license. No SMPL-H data is included in the repository.
Hosted operators must review this license for their intended service.

[SMPL-H license]: https://huggingface.co/lithiumice/models_hub/blob/363972831b553e4f5b62d11d509e98830d930c97/4_SMPLhub/smplh/LICENSE.txt

The installed checkpoints passed download integrity checks. Metadata parsing
read 1127 SAM tensors, 1411 Omega tensors and 318 FlowHMR Latest tensors.
These counts do not establish complete inference or numerical parity. The
shared restricted checkpoint reader reads numeric tensors without executing
pickle code. Its explicit `OpenScriptData` path reads SAM's MHR archive as
inert objects: 59 tensors, 127 joint names and 249 parameter names. The
default reader continues to reject script objects; neither path executes
archive code. MHR parameter mapping, skeleton transforms, mesh skinning,
identity/expression blending and learned pose correctives pass analytic and
released-data checks. The complete body model produces a neutral base mesh
and finite, repeatable posed output. Both SAM pose heads now produce keypoints
through the complete MHR model. Released CPU decoder checks pass all six
layers, five real pose-feedback updates and final prediction with exact
repeatability, including the learned initial pose/camera/prompt tokens.
Those decoder comparisons use synthetic image features. The combined
body/left/right feature extractor has separately passed real-frame runs on
the Desktop character and FlowHMR's human sample, including exact repeats.
Upstream numerical parity and full video accuracy remain unverified.
The detector has passed released-weight inference on frames from two
videos; the camera model has produced finite output for one real frame.
Both FlowHMR variants have completed sampling with synthetic conditioning.
Latest has also completed a five-second predecoded GPU capture. See
[GPU validation]. Windows production decoding and full motion-quality
acceptance remain unverified. Downloading weights does not enable local
video capture on Linux.

[GPU validation]: gpu-validation.md

Other candidates considered were [BlendCap SAM] and [Wincent Omega]. Their
publishers describe unmodified upstream copies and include licenses. They
are not configured fallback URLs; the selected mirrors above have identified
revisions and artifact hashes available for the downloader.

[BlendCap SAM]: https://huggingface.co/Blendcap/sam-3d-body
[Wincent Omega]: https://huggingface.co/wincentIsMe/VGGT-Omega
