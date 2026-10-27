Third-party notices
===================

FlowHMR
-------

The Euler sampler, denoiser, attention mask, time conditioning and video
refiner are C# adaptations of the FlowHMR authors' source, distributed under
Apache License 2.0. They use reusable buffers and explicit shape validation
instead of PyTorch.

Source: https://github.com/flowhmr/flowhmr

Adapted files:

- `flowhmr/pipeline/pipeline_v2m_simple.py` → `Code/FlowSampler.cs`
- `flowhmr/pipeline/pipeline_v2m.py`, `v2m_inference.py` and
  `flowhmr/core/bodymodels/smpl_skeleton.py` → `Editor/Inference/FlowDecoder.cs`
  and `SmplJoints.cs` (joint-template and shape-direction regression)
- `flowhmr/network/hymotion_mmdit_for_v2m.py` →
  `Editor/Inference/FlowAttention.cs`, `FlowDenoiser.cs`, `FlowDoubleBlock.cs`
  and `FlowSingleBlock.cs`
- `flowhmr/network/token_refiner.py` → `Editor/Inference/FlowRefiner.cs`
- `flowhmr/network/mmdit/encoders.py` and `modulate_layers.py` →
  `Editor/Inference/FlowConditioning.cs` and `FlowRefiner.cs`
- `flowhmr/network/mmdit/bricks.py` → `Editor/Inference/FlowNormalization.cs`
- The same sampler, backbone and refiner sources → `Editor/Gpu/FlowGpuSampler.cs`,
  `FlowGpuRefiner.cs`, `FlowGpuDoubleBlock.cs` and `FlowGpuSingleBlock.cs`
- `flowhmr/utils/runtime_vggt_camera.py` and
  `flowhmr/pipeline/v2m_inference.py` → sampling, interpolation and relative
  rotation portions of `Code/FlowCamera.cs`

The applicable license is reproduced in [LICENSE-FLOWHMR](LICENSE-FLOWHMR).
Model weights are downloaded separately under their own terms.

SMPL-H neutral model data is downloaded separately from a pinned public
mirror. Its Max Planck MANO/SMPL+H license is saved alongside the asset and
is not replaced by the library's MIT license. See [model sources] for
provenance, integrity checks and license links; no SMPL-H data is bundled.

[model sources]: docs/model-sources.md

`FlowRotary.cs` implements the adjacent-pair rotary embedding formula used
by the [HY-Motion backbone]. Its full-checkpoint parity with FlowHMR remains
unverified.

[HY-Motion backbone]: https://github.com/Tencent-Hunyuan/HY-Motion-1.0/blob/master/hymotion/network/positional_encoding.py

YOLOX
-----

`Editor/Inference/YoloNetwork.cs`, `YoloWeights.cs`, `YoloImage.cs` and
`YoloDetection.cs` adapt the YOLOX-L inference architecture, batch
normalization, image preparation and detection decoding to C#. The GPU
adapter executes the network through s&box compute shaders. Person
filtering follows FlowHMR's `runtime_video_processing.py`.

Copyright (c) Megvii, Inc. and its affiliates. Distributed under Apache
License 2.0; the license text is reproduced in
[LICENSE-FLOWHMR](LICENSE-FLOWHMR).

Sources: https://github.com/Megvii-BaseDetection/YOLOX/tree/main/yolox/models
and https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py

VGGT-Omega
----------

The camera pose decoding portions of `Code/FlowCamera.cs` are adapted from
`vggt_omega/utils/pose_enc.py`. They use System.Numerics for quaternion
normalization, interpolation and conversion, and validate input before use.

`Editor/Inference/OmegaPosition.cs`, `OmegaBlock.cs` and `OmegaCameraHead.cs`
adapt `models/layers/rope_position_encoding.py`, `attention.py`, `block.py`
and `models/heads/camera_head.py`. `Editor/Gpu/OmegaGpuBlock.cs` and
`OmegaGpuCameraHead.cs` execute those operations through s&box compute
shaders. The upstream layer files retain DINOv3 source notices.

`DinoImageEncoder.cs`, `DinoGpuEncoder.cs`, `OmegaAggregator.cs` and
`OmegaGpuModel.cs` adapt
the DINO image-encoding and alternating-attention paths in
`models/layers/vision_transformer.py` and `models/aggregator.py`, retaining
only the final features required by the camera head. The crop and target
dimensions in `OmegaImage.cs` follow `utils/load_fn.py`'s max-size mode.

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
These adaptations are subject to the FAIR Noncommercial Research License,
reproduced in [LICENSE-VGGT](LICENSE-VGGT), including its acceptable use
policy. The project's MIT license does not relicense these portions.

Source: https://github.com/facebookresearch/vggt-omega

DINOv3
------

The SAM configuration of `DinoShape`, `DinoImageEncoder` and
`DinoGpuEncoder` follows `dinov3/hub/backbones.py` and
`dinov3/models/vision_transformer.py`. The shared transformer and position
implementations support its SwiGLU feed-forward layers and separately
normalized axial positions, following `dinov3/layers/ffn_layers.py` and
`rope_position_encoding.py`.

Copyright (c) Meta Platforms, Inc. and affiliates. These adaptations retain
the [DINOv3 License](LICENSE-DINOV3). The project's MIT license does not
relicense them. Model weights are downloaded separately under their own
terms, including SAM's terms for its fine-tuned checkpoint.

Source: https://github.com/facebookresearch/dinov3

SAM 3D Body
-----------

`SamDecoder.cs`, `SamDecoderBlock.cs` and their GPU adapters implement the
body/hand transformer paths in `models/decoders/promptable_decoder.py` and
`models/modules/transformer.py`, including intermediate pose-feedback
updates. Tensor layout, validation and storage use the existing C# inference
and s&box compute libraries.

`SamRays.cs`, `SamImageConditioning.cs` and `SamGpuImageConditioning.cs`
adapt SAM's camera-ray Fourier encoding, no-mask image conditioning,
body/hand projection and positional context. Ray downsampling evaluates
the affine ray map at antialiased sample coordinates instead of materializing
the full-resolution map.

`SamCompactPose.cs`, `SamMhrParameters.cs`, `SamPredictionHead.cs`,
`SamPoseHead.cs` and `SamKeypoints.cs` adapt `models/modules/mhr_utils.py`
and the MHR/camera head implementations. `SamRotation.cs` adapts SAM's
continuous-rotation conversion and uses the RoMa conventions noted below.
`SamProjection.cs` and `SamKeypointFeedback.cs` adapt the camera projection
and keypoint update paths in `models/heads/camera_head.py` and
`models/meta_arch/sam3d_body.py`.
`SamInitialTokens.cs` adapts the initial body/hand pass and invalid point
prompt embedding in `models/decoders/prompt_encoder.py`.
`SamCrop.cs` adapts `data/transforms/common.py` and `bbox_utils.py`, retaining
the body/hand padding, aspect expansion and affine source-point conventions.
`SamGpuImageEncoder.cs` connects the existing DINO and camera-conditioning
implementations without an external inference runtime.
`SamHandBoxes.cs` adapts the hand-box MLP and square crop selection, including
left-image reflection. `SamGpuPoseDecoder.cs` and `SamGpuFeatures.cs` connect
the body/left/right passes. Feature selection follows FlowHMR's
`flowhmr/utils/runtime_sam_features.py`: the first pose token from each pass,
before SAM's later pose merging and prompted body refinement.

Copyright (c) Meta Platforms, Inc. and affiliates. These adaptations retain
the [SAM License](LICENSE-SAM); the project's MIT license does not relicense
them. Checkpoint files retain their separate upstream terms.

Source: https://github.com/facebookresearch/sam-3d-body

OpenCV
------

The affine coordinate quantization in `SamCrop.cs` adapts the CPU
`warpAffine` path in `modules/imgproc/src/imgwarp.cpp`. Pixel interpolation
reuses `LinearRgbImage` from the MIT Humanoid Mocap library. The applicable
OpenCV source-file BSD notice is reproduced in [LICENSE-OPENCV](LICENSE-OPENCV).

Source: https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/imgwarp.cpp

RoMa and SciPy
-------------

`SamRotation.cs` specializes RoMa's matrix-to-quaternion and
quaternion-to-Euler conversions for SAM's XYZ/ZYX conventions, including
its singularity handling. RoMa's Euler implementation derives from SciPy.
These portions retain the BSD terms in [LICENSE-ROMA](LICENSE-ROMA) and
[LICENSE-SCIPY](LICENSE-SCIPY).

Copyright (c) 2020 NAVER Corp.
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.

Sources: https://github.com/naver/roma/blob/master/roma/euler.py and
https://github.com/naver/roma/blob/master/roma/mappings.py

Momentum
--------

`MhrSkeleton.cs`, `MhrState.cs`, `MhrSkinning.cs`, `MhrBlendShapes.cs`,
`MhrCorrectives.cs` and `MhrBody.cs` adapt parameter mapping, Euler rotations,
global skeleton states, mesh skinning, blendshapes and pose corrections from
Momentum's `pymomentum/torch/character.py`, `quaternion.py`, `skel_state.py`,
`backend/skel_state_backend.py`, `mhr/mhr.py` and `mhr/utils.py`, as
serialized in SAM's MHR asset. Archive code is inspected as source; it is
never executed by the library.

Copyright (c) Meta Platforms, Inc. and affiliates. Distributed under the
MIT license, reproduced in [LICENSE-MOMENTUM](LICENSE-MOMENTUM).
The separately downloaded MHR model retains its own terms.

Source: https://github.com/facebookresearch/momentum

Pillow
------

The bicubic filter, fixed-point coefficient rounding and separable pass
ordering in `Editor/Inference/OmegaImage.cs` adapt Pillow's `Resample.c`.
They run as C# code inside the library; Pillow is not a runtime dependency.
Copyright and MIT-CMU terms are reproduced in [LICENSE-PILLOW](LICENSE-PILLOW).

Source: https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c

Humanoid Mocap
--------------

`Code/HumanoidMocap` and `Editor/HumanoidMocap` hold the parts of Humanoid Mocap
this workspace uses, copied at `0540875` with its unpublished working-tree
changes, so this library does not depend on that package. The root namespace is
renamed to `HumanoidMocapX`; its first-person, hand, phone, MotionBricks and
native-worker features are not included. Humanoid
Mocap's retargeting, cleanup and export code was adapted from the author's
`humanoid-retargeter` at `69ccfcd76f6aecca443f055be912ac3831a64003`, and its
window layout follows `sbox-humanoid-rigger` at
`6e7304bf1d5941037f50496eccf0a7e03ddf9e10`. Editor widgets follow exposed
s&box APIs and conventions in Facepunch/sbox-public at
`95eb0e4c1409d684de8bec636732c0548b7733fe`.

Source: https://github.com/zeljkovranjes/sbox-humanoid-mocap

`Code/HumanoidMocap/Core/Motion/MocapSmooth.cs` ports the zero-phase
Butterworth and Gaussian smoothing of Dylanyz/MocapSmooth
(`Tools/smooth_core.py`), Apache License 2.0.

Source: https://github.com/Dylanyz/MocapSmooth

`Code/HumanoidMocap/Core/Cleanup/CcdIk.cs` adapts the CCD iteration, bone gain
and per-joint constraint callback of DigitalRune `CcdIKSolver.cs` at
`a907452c8922a3360761e0569aa33110b0201b91`. Copyright (c) 2007-2016,
DigitalRune GmbH. The BSD 3-Clause license is retained in
[DigitalRune.LICENSE](Code/HumanoidMocap/Core/Cleanup/DigitalRune.LICENSE).

Source: https://github.com/DigitalRune/DigitalRune

Windows Media Foundation signatures and GUIDs in the copied video decoder
follow Windows SDK 10.0.26100.0; the decoder calls the OS directly.
