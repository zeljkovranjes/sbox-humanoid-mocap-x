Provider motion output
======================

A processing provider returns one `.hmotion` JSON document from
`GET /v1/jobs/{id}/result` only after the job reaches `completed`. The C# host
validates it before publishing that state; Humanoid Mocap X validates it
again before import. Limits are 128 MiB, 1024 bones and 108000 frames in the
shared document format. The local FlowHMR pipeline currently limits each
inference request to 5000 frames at 30 fps.

## Coordinate and animation conventions

- `schema` is `1`, `units` is `"metres"`, and `axes` is
  `"right-handed-x-right-y-up"`. Do not convert positions into s&box inches
  or FBX centimetres; the existing retarget/export adapters do that once.
- Bones have unique names and a parent index smaller than their own index;
  roots use `-1`. Rest positions and rotations are relative to the parent.
- Each frame has one position, rotation and evidence value per bone, in the
  same order as `bones`. Transforms are local to the parent, with root
  transforms in the document's declared space. Rotations are normalized
  quaternions in `[x,y,z,w]` order, not Euler angles or axis-angle vectors.
- `space` is `"WorldRelative"` for reconstructed world motion or
  `"CameraRelative"` when world placement is unknown. `sourceFps` is finite
  and positive; timestamps are finite, strictly increasing seconds.
  FlowHMR output uses 30 fps. Keep the selected clip's start offset when
  synchronizing with the source video.
- Explicit bone roles drive Human/Citizen/custom retargeting. Include `Hips`
  and the full body. Preserve all available finger roles and rotations;
  filling them with identity rotations is not finger reconstruction.
- `metricScaleCalibrated` remains false for model-estimated scale. Only set
  it true with actual calibration evidence. Reconstruction evidence is
  `"Reconstructed"`; do not label estimates as measured ground truth.

For FlowHMR's 52-joint SMPL-H output, reuse [`FlowMotion.ToDocument`] rather
than duplicating role assignment, quaternion conversion or contact mapping.
The adapter preserves all 30 source finger joints. Human and Citizen targets
keep their supported finger chains through the existing retargeter.

[`FlowMotion.ToDocument`]: ../Editor/FlowMotionAdapter.cs

## Document fields

| Field | Meaning |
| --- | --- |
| `name` | Human-readable clip name |
| `backend`, `modelVersion` | Actual processor and model revision/variant |
| `sourceFps`, `space`, `axes`, `units` | Timing and coordinate conventions above |
| `bones` | Name, parent, role, group, rest position and rest rotation |
| `frames` | Time and arrays of local positions, rotations and evidence |
| `stationaryJoints` | Predicted stationary/contact probability tracks |
| `diagnostics` | Bounded explanatory messages about limitations/calibration |

The shared [MotionDocument] type is the authoritative schema. A production
result is a complete animated body. This one-bone snippet illustrates array
shapes only; it is not a useful capture or an acceptance-test result:

[MotionDocument]: ../../humanoid-mocap/Code/HumanoidMocap/Core/Motion/MotionDocument.cs

```json
{
  "schema": 1,
  "name": "capture",
  "backend": "FlowHMR",
  "modelVersion": "flowhmr_latest",
  "sourceFps": 30,
  "space": "WorldRelative",
  "axes": "right-handed-x-right-y-up",
  "units": "metres",
  "metricScaleCalibrated": false,
  "bones": [{
    "name": "Hips", "parent": -1, "role": "Hips", "group": "body",
    "restPosition": [0, 0.9, 0], "restRotation": [0, 0, 0, 1]
  }],
  "frames": [{
    "time": 0,
    "positions": [[0, 0.9, 0]], "rotations": [[0, 0, 0, 1]],
    "evidence": ["Reconstructed"]
  }]
}
```

## Feet, fingers and metadata

Each `stationaryJoints` entry names an existing bone, identifies its `source`
and has a `probability` array matching the frame count. Values are finite
numbers in `[0,1]`. FlowHMR supplies left ankle, left toe, right ankle and
right toe tracks. These are contact predictions, not confidence in overall
3D accuracy. Preserve them for the existing foot-locking solver; do not
convert them into authored contact constraints without review.

Set the correct finger roles, including side and segment. The source order
and all roles are in the adapter linked above. A service may report blurred
or occluded hands in `diagnostics`; it must not claim accurate fingers from
successful serialization alone.

Do not include server filesystem paths, credentials or internal URLs in the
result. The editor overwrites `sourceVideo` with the local input and clears
`originalReconstruction` before saving. Return motion data, not code,
scripts, remote asset references or executable archives.

## Processor lifecycle

`IClipProcessor.ProcessAsync` receives validated range/model settings,
extension, readable video stream, writable result stream, a short progress
callback and cancellation. The host owns both streams. Decode and validate
the actual container; an allowed filename extension is not codec validation.
Do not dispose caller-owned streams. Finish writing a complete document
before returning. Throw on failure and honor cancellation throughout work.

The host moves `queued → processing → completed` only after result validation.
Failure or cancellation removes incomplete results. The default processor
is intentionally unavailable. Replace that registration with a real
processor and expose `IsAvailable = true` only when its models and execution
backend are ready. Review each model's license for the hosted deployment.
