using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;

namespace HumanoidMocapX.Core.Motion;
using Vector3=System.Numerics.Vector3;

public sealed record MissingObservationInterval(double Start,double End);
