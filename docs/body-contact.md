Body contact
============

Humanoid Mocap X uses a C# adaptation of DigitalRune's cyclic-coordinate descent
(CCD) solver in the shared Humanoid Mocap library. The port uses the existing
domain skeleton and System.Numerics, without another engine or native worker.

[Upstream source]: https://github.com/DigitalRune/DigitalRune/blob/a907452c8922a3360761e0569aa33110b0201b91/Source/DigitalRune.Animation/Character/IK/CcdIKSolver.cs

The [upstream source] is pinned at
`a907452c8922a3360761e0569aa33110b0201b91`. Its BSD 3-Clause copyright and license
are retained beside `Core/Cleanup/CcdIk.cs` in `DigitalRune.LICENSE`, and in the
shared library's third-party notices. The adaptation retains the tip-to-root
iteration, correction gain, and optional joint-constraint callback. It adds
bounded iteration, input validation, reusable storage, and reach residuals.

The contact adapter checks the entire forearm segment and mapped finger-bone
segments against an elliptical torso proxy with rounded ends. IK helper bones
are excluded. All contacts on one hand share its outward direction, preventing
the correction from switching between the front and back of the body when a
different finger becomes the deepest contact.

Corrections ease across neighboring frames, then the solved pose is checked
again. The pass runs after target adjustments and optional arm IK. It preserves
local bone translations, captured finger articulation, and wrist world
orientation. Existing foot locking, scale conversion, and helper-bone baking
remain in the shared retargeting pipeline.

Unreachable contacts keep the best measured pose and add a residual note to the
conversion report. Bones are never stretched to conceal an unreachable target.

Validation
----------

The default test suite covers:

- Reachable and unreachable CCD goals, joint callbacks, tip offsets, rejected
  malformed inputs, and zero allocations during repeated solves.
- Both hands on Human and Citizen, including 0.01x and 10x rig scales.
- Smooth free/contact/free motion with animated fingers. Independent samples
  along the forearm and finger segments verify proxy clearance; free endpoint
  poses stay unchanged. Maximum joint steps in these 30 fps fixtures were
  2.59 degrees on Human and 2.96 degrees on Citizen.
- An impossible short-arm rig, which must report residual contact without
  stretching or changing wrist attitude.
- The existing five-second capture: 150 frames on each rig, 30 Human and 24
  Citizen finger tracks checked through FBX export. Neither rig reported an
  unresolved proxy contact in this capture.

The recorded capture check reuses verified inference output; it does not rerun
the neural networks. Contact correction intentionally changes arm motion and
exported FBX, so old performance-run FBX hashes are historical baselines. The
inference artifacts and numerical model calculations are unchanged.

Limits
------

The torso is inferred from bones, not the rendered skin or clothing. Proxy
clearance is not proof that an arbitrary mesh cannot intersect. This pass does
not solve cloth, object contacts, hand-to-hand collision, or every limb pair.
The synthetic continuity tests and export checks are not visual or ground-truth
acceptance for every gesture. Captured pose errors can still require review.
