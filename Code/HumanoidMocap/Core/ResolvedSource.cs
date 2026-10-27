#nullable enable annotations

using System;
using System.Collections.Generic;
using System.Linq;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Target;
using HumanoidMocapX.Core.Skeleton;

namespace HumanoidMocapX.Core;

/// <summary>Imported source after the public mapping and bind-rest preparation cascade.</summary>
public sealed record ResolvedSource( SourceScene Scene, MappingResult Mapping, MappingReportInfo Report );
