using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Regresses the neutral SMPL-H template and sixteen shape directions
/// into the 52 joints used by FlowHMR. Inputs are row-major vertex arrays.</summary>
internal sealed record SmplJoints(float[] Template, float[] Directions)
{
    internal const int Vertices = 6890, Joints = 52, Shapes = 16;

    internal static SmplJoints Regress(ReadOnlySpan<float> regressor, ReadOnlySpan<float> vertices,
        ReadOnlySpan<float> directions, CancellationToken cancellation)
    {
        if (regressor.Length != Joints * Vertices || vertices.Length != Vertices * 3 || directions.Length != Vertices * 3 * Shapes)
            throw new ArgumentException("SMPL-H regression requires 6890 vertices, 52 joints and sixteen shape directions.");
        var template = new float[Joints * 3]; var shaped = new float[Joints * 3 * Shapes];
        Span<double> sums = stackalloc double[3 + 3 * Shapes];
        for (var joint = 0; joint < Joints; joint++)
        {
            cancellation.ThrowIfCancellationRequested(); sums.Clear();
            for (var vertex = 0; vertex < Vertices; vertex++)
            {
                var weight = regressor[joint * Vertices + vertex];
                if (weight == 0) continue;
                for (var axis = 0; axis < 3; axis++)
                {
                    sums[axis] += (double)weight * vertices[vertex * 3 + axis];
                    for (var shape = 0; shape < Shapes; shape++)
                        sums[3 + axis * Shapes + shape] += (double)weight * directions[(vertex * 3 + axis) * Shapes + shape];
                }
            }
            for (var axis = 0; axis < 3; axis++) template[joint * 3 + axis] = (float)sums[axis];
            for (var component = 0; component < 3 * Shapes; component++) shaped[joint * 3 * Shapes + component] = (float)sums[3 + component];
        }
        return new(template, shaped);
    }
}
