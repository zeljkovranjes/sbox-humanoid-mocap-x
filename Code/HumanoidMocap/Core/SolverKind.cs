#nullable enable annotations

using System;
using System.Collections.Generic;
using HumanoidMocapX.Core.Cleanup;
using HumanoidMocapX.Core.Formats;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Solve;
using HumanoidMocapX.Core.Target;

namespace HumanoidMocapX.Core;

/// <summary>Which solver retargets a request's clips (design §10).</summary>
public enum SolverKind
{
    /// <summary>The deterministic <see cref="Solve.GeometricSolver"/> (default; better
    /// wherever a role mapping exists).</summary>
    Geometric,

    /// <summary>The experimental skeleton-agnostic deep-learning solver
    /// (<see cref="Dl.DlSolver"/>, SAME pretrained checkpoint) — the no-profile fallback.
    /// Requires <see cref="RetargetTargetSpec.DlWeights"/>; ignores per-role mapping
    /// (only hips/alignment heuristics consult it) and leaves fingers at rest.</summary>
    DeepLearning,
}
