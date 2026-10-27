using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Applies MHR identity and facial blendshapes to the base mesh, in model
/// units. Owns reusable working storage; one instance per inference session.</summary>
internal sealed class MhrBlendShapes
{
    readonly Tensor identity, expression, basis;
    readonly float[] face;
    internal int Vertices { get; }
    internal int Identities => identity.Cols;
    internal int Expressions => expression.Cols;

    internal MhrBlendShapes(FlowWeights weights, int vertices, int identities, int expressions)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (vertices < 1 || identities < 1 || expressions < 1) throw new ArgumentException("MHR blendshape dimensions must be positive.");
        Vertices = vertices;
        var coordinates = checked(vertices * 3);
        var source = weights.Get(MhrSkeleton.Prefix + "blend_shape.base_shape", 3, vertices);
        basis = new(source.Name, source.Data, new[] { coordinates });
        identity = Transposed(weights.Get(MhrSkeleton.Prefix + "blend_shape.shape_vectors", 3, vertices, identities), identities, coordinates);
        expression = Transposed(weights.Get("face_expressions_model.shape_vectors", 3, vertices, expressions), expressions, coordinates);
        face = new float[coordinates];
    }

    /// <summary>Writes the full unposed mesh. Rejects invalid dimensions, non-finite
    /// coefficients and overlapping input/output. Cancellation may leave partial output.</summary>
    internal void Run(ReadOnlySpan<float> identityCoefficients, ReadOnlySpan<float> expressionCoefficients,
        Span<float> output, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (identityCoefficients.Length != Identities || expressionCoefficients.Length != Expressions || output.Length != face.Length)
            throw new ArgumentException("MHR blendshape input/output dimensions do not match the model.");
        if (identityCoefficients.Overlaps(output) || expressionCoefficients.Overlaps(output) || output.Overlaps(basis.Data))
            throw new ArgumentException("MHR blendshape output must not overlap inputs or model weights.");
        foreach (var value in identityCoefficients)
            if (!float.IsFinite(value)) throw new ArgumentException("MHR identity coefficients must be finite.", nameof(identityCoefficients));
        foreach (var value in expressionCoefficients)
            if (!float.IsFinite(value)) throw new ArgumentException("MHR expression coefficients must be finite.", nameof(expressionCoefficients));
        FlowLinear.Run(identityCoefficients, 1, identity, basis, output, cancellation);
        FlowLinear.Run(expressionCoefficients, 1, expression, null, face, cancellation);
        NeuralOps.AddInPlace(output, face);
        foreach (var value in output)
            if (!float.IsFinite(value)) throw new ArithmeticException("MHR blendshapes produced a non-finite vertex.");
    }

    static Tensor Transposed(Tensor source, int coefficients, int coordinates)
        => new(source.Name, NeuralOps.Transpose(source.Data, coefficients, coordinates), new[] { coefficients, coordinates });
}
