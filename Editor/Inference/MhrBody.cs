using System;
using System.Threading;
using HumanoidMocapX.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Evaluates the MHR mesh and global joint states with identity,
/// expression and learned pose deformations. Owns one inference session's buffers.</summary>
internal sealed class MhrBody
{
    readonly MhrSkeleton skeleton;
    readonly MhrSkinning skinning;
    readonly MhrBlendShapes blendShapes;
    readonly MhrCorrectives correctives;
    readonly float[] parameters, jointParameters, unposed, corrections;
    internal int ModelParameters => skeleton.Parameters - blendShapes.Identities;
    internal int Identities => blendShapes.Identities;
    internal int Expressions => blendShapes.Expressions;
    internal int Joints => skeleton.Joints;
    internal int Vertices => skinning.Vertices;

    internal MhrBody(MhrSkeleton skeleton, MhrSkinning skinning, MhrBlendShapes blendShapes, MhrCorrectives correctives)
    {
        ArgumentNullException.ThrowIfNull(skeleton); ArgumentNullException.ThrowIfNull(skinning);
        ArgumentNullException.ThrowIfNull(blendShapes); ArgumentNullException.ThrowIfNull(correctives);
        if (skeleton.Joints != skinning.Joints || skeleton.Joints != correctives.Joints
            || skinning.Vertices != blendShapes.Vertices || skinning.Vertices * 3 != correctives.Coordinates || skeleton.Parameters <= blendShapes.Identities)
            throw new ArgumentException("MHR body components have incompatible dimensions.");
        this.skeleton = skeleton; this.skinning = skinning; this.blendShapes = blendShapes; this.correctives = correctives;
        parameters = new float[skeleton.Parameters]; jointParameters = new float[skeleton.Joints * MhrSkeleton.JointParameters];
        unposed = new float[skinning.Vertices * 3]; corrections = new float[unposed.Length];
    }

    /// <summary>Applies pose corrections only to the given vertices; the rest keep their
    /// blend-shape positions. Use only when consumers read just these vertices.</summary>
    internal void Restrict(ReadOnlySpan<int> vertices) => correctives.Restrict(vertices);

    /// <summary>The pose correctives, for sharing their projection with another body's (MhrCheckpoint.LoadBody).</summary>
    internal MhrCorrectives Correctives => correctives;

    /// <summary>Writes XYZ vertices and eight values per global joint state, in
    /// checkpoint units. Identity coefficients affect the mesh; their slots in the
    /// skeleton parameter vector stay zero. Rejects bad dimensions and aliasing.</summary>
    internal void Run(ReadOnlySpan<float> identity, ReadOnlySpan<float> model, ReadOnlySpan<float> expression,
        Span<float> vertices, Span<float> globalStates, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (identity.Length != Identities || model.Length != ModelParameters || expression.Length != Expressions
            || vertices.Length != unposed.Length || globalStates.Length != Joints * MhrState.Elements)
            throw new ArgumentException("MHR body input/output dimensions do not match the model.");
        if (vertices.Overlaps(globalStates) || identity.Overlaps(vertices) || model.Overlaps(vertices) || expression.Overlaps(vertices)
            || identity.Overlaps(globalStates) || model.Overlaps(globalStates) || expression.Overlaps(globalStates))
            throw new ArgumentException("MHR body outputs must not overlap inputs or each other.");
        model.CopyTo(parameters);
        skeleton.Run(parameters, jointParameters, globalStates, cancellation);
        blendShapes.Run(identity, expression, unposed, cancellation);
        correctives.Run(jointParameters, corrections, cancellation);
        NeuralOps.AddInPlace(unposed, corrections);
        skinning.Run(globalStates, unposed, vertices, cancellation);
    }
}
