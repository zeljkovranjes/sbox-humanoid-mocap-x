using System;
using System.Numerics;
using HumanoidMocapX.Core.Maths;

namespace HumanoidMocapX.Core.Motion;
using Vector3 = System.Numerics.Vector3;

public readonly record struct ArmSolution(Vector3 Shoulder,Vector3 Elbow,Vector3 Wrist,
    Quaternion WristRotation,float ReachError,JointEvidence ShoulderEvidence,JointEvidence ElbowEvidence);
