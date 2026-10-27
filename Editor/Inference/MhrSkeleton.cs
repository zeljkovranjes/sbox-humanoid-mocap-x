using System;
using System.IO;
using System.Threading;
using HumanoidMocapX.EditorTools.MotionBricks;

namespace HumanoidMocapX.Inference;

/// <summary>Maps MHR model parameters to joint parameters and global states in
/// the checkpoint's units. Reuses working storage; one instance per inference session.</summary>
internal sealed class MhrSkeleton
{
    internal const string Prefix = "character_torch.";
    internal const int JointParameters = 7;
    readonly Tensor mapping, offsets, prerotations;
    readonly int[] parents;
    readonly MhrState[] states;
    internal int Joints => parents.Length;
    internal int Parameters => mapping.Cols;

    internal MhrSkeleton(FlowWeights weights, ReadOnlySpan<int> parents, int parameters)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (parents.Length == 0 || parameters < 1) throw new ArgumentException("MHR skeleton dimensions must be positive.");
        mapping = weights.Get(Prefix + "parameter_transform.parameter_transform", parameters, checked(parents.Length * JointParameters));
        offsets = weights.Get(Prefix + "skeleton.joint_translation_offsets", 3, parents.Length);
        prerotations = weights.Get(Prefix + "skeleton.joint_prerotations", 4, parents.Length);
        for (var joint = 0; joint < parents.Length; joint++)
        {
            if (parents[joint] < -1 || parents[joint] >= joint)
                throw new InvalidDataException("MHR parents must precede their children, with -1 for roots.");
            var q = prerotations.Row(joint); double squaredNorm = 0;
            foreach (var value in q) squaredNorm += (double)value * value;
            if (Math.Abs(squaredNorm - 1) > 1e-4) throw new InvalidDataException("MHR prerotation must be a unit quaternion.");
        }
        this.parents = parents.ToArray(); states = new MhrState[parents.Length];
    }

    /// <summary>Writes seven local joint parameters and eight global-state values
    /// (translation, XYZW quaternion, scale) per joint. Rejects non-finite or
    /// overlapping buffers and invalid dimensions; cancellation may leave partial output.</summary>
    internal void Run(ReadOnlySpan<float> modelParameters, Span<float> jointParameters, Span<float> globalStates,
        CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (modelParameters.Length != Parameters || jointParameters.Length != Joints * JointParameters || globalStates.Length != Joints * MhrState.Elements)
            throw new ArgumentException("MHR parameter or skeleton-state dimensions do not match the model.");
        if (modelParameters.Overlaps(jointParameters) || modelParameters.Overlaps(globalStates) || jointParameters.Overlaps(globalStates)
            || AliasesWeights(jointParameters) || AliasesWeights(globalStates))
            throw new ArgumentException("MHR outputs must not overlap inputs, each other or model weights.");
        foreach (var value in modelParameters)
            if (!float.IsFinite(value)) throw new ArgumentException("MHR model parameters must be finite.", nameof(modelParameters));
        FlowLinear.Run(modelParameters, 1, mapping, null, jointParameters, cancellation);
        for (var joint = 0; joint < Joints; joint++)
        {
            cancellation.ThrowIfCancellationRequested();
            var local = MhrState.Local(jointParameters.Slice(joint * JointParameters, JointParameters), offsets.Row(joint), prerotations.Row(joint));
            states[joint] = parents[joint] < 0 ? local : states[parents[joint]].Then(local);
            var output = globalStates.Slice(joint * MhrState.Elements, MhrState.Elements);
            states[joint].Write(output);
            foreach (var value in output)
                if (!float.IsFinite(value)) throw new ArithmeticException("MHR skeleton calculation produced a non-finite state.");
            if (output[7] <= 0) throw new ArithmeticException("MHR skeleton scale underflowed.");
        }
    }

    bool AliasesWeights(Span<float> output)
        => output.Overlaps(mapping.Data) || output.Overlaps(offsets.Data) || output.Overlaps(prerotations.Data);
}
