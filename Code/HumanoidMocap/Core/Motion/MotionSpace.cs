using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Skeleton;

namespace HumanoidMocapX.Core.Motion;
using Vector3 = System.Numerics.Vector3;

public enum MotionSpace { CameraRelative, WorldRelative }
