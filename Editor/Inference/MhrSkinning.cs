using System;
using System.IO;
using System.Numerics;
using System.Threading;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Skins MHR vertices with the saved influence order and inverse bind
/// states. Owns reusable buffers; use one instance per inference session.</summary>
internal sealed class MhrSkinning
{
    readonly int[] joints, vertices;
    readonly float[] weights, bindScales, scales;
    readonly Vector3[] bindTranslations, translations;
    readonly Quaternion[] bindRotations, rotations;
    internal int Joints => scales.Length;
    internal int Vertices { get; }

    internal MhrSkinning(ReadOnlySpan<float> inverseBind, ReadOnlySpan<int> joints,
        ReadOnlySpan<int> vertices, ReadOnlySpan<float> weights, int vertexCount)
    {
        if (inverseBind.Length == 0 || inverseBind.Length % MhrState.Elements != 0 || vertexCount < 1
            || joints.Length == 0 || vertices.Length != joints.Length || weights.Length != joints.Length)
            throw new InvalidDataException("MHR skinning dimensions are invalid.");
        var jointCount = inverseBind.Length / MhrState.Elements;
        for (var i = 0; i < joints.Length; i++)
            if ((uint)joints[i] >= jointCount || (uint)vertices[i] >= vertexCount || !float.IsFinite(weights[i]) || weights[i] < 0)
                throw new InvalidDataException("MHR skinning influence has an invalid index or weight.");
        ValidateStates(inverseBind);
        this.joints = joints.ToArray(); this.vertices = vertices.ToArray(); this.weights = weights.ToArray(); Vertices = vertexCount;
        bindTranslations = new Vector3[jointCount]; bindRotations = new Quaternion[jointCount]; bindScales = new float[jointCount];
        translations = new Vector3[jointCount]; rotations = new Quaternion[jointCount]; scales = new float[jointCount];
        for (var joint = 0; joint < jointCount; joint++)
        {
            var state = inverseBind.Slice(joint * MhrState.Elements, MhrState.Elements);
            bindTranslations[joint] = Point(state); bindRotations[joint] = Rotation(state); bindScales[joint] = state[7];
        }
    }

    /// <summary>Writes posed XYZ vertices in model units. Rejects invalid states,
    /// non-finite vertices and overlapping buffers. Cancellation may leave partial output.</summary>
    internal void Run(ReadOnlySpan<float> globalStates, ReadOnlySpan<float> restVertices, Span<float> output,
        CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (globalStates.Length != Joints * MhrState.Elements || restVertices.Length != (long)Vertices * 3 || output.Length != restVertices.Length)
            throw new ArgumentException("MHR skinning input/output dimensions do not match the model.");
        if (globalStates.Overlaps(output) || restVertices.Overlaps(output)) throw new ArgumentException("MHR skinning output must not overlap inputs.");
        ValidateStates(globalStates);
        foreach (var value in restVertices)
            if (!float.IsFinite(value)) throw new ArgumentException("MHR rest vertices must be finite.", nameof(restVertices));
        for (var joint = 0; joint < Joints; joint++)
        {
            var state = globalStates.Slice(joint * MhrState.Elements, MhrState.Elements); var rotation = Rotation(state);
            translations[joint] = Point(state) + state[7] * Rotate(rotation, bindTranslations[joint]);
            rotations[joint] = Quaternion.Normalize(rotation * bindRotations[joint]);
            scales[joint] = state[7] * bindScales[joint];
        }
        output.Clear();
        for (var i = 0; i < joints.Length; i++)
        {
            if ((i & 255) == 0) cancellation.ThrowIfCancellationRequested();
            var joint = joints[i]; var vertex = vertices[i] * 3;
            var transformed = translations[joint] + Rotate(rotations[joint], scales[joint] * Point(restVertices.Slice(vertex, 3)));
            var weighted = weights[i] * transformed;
            output[vertex] += weighted.X; output[vertex + 1] += weighted.Y; output[vertex + 2] += weighted.Z;
        }
        foreach (var value in output)
            if (!float.IsFinite(value)) throw new ArithmeticException("MHR skinning produced a non-finite vertex.");
    }

    static void ValidateStates(ReadOnlySpan<float> states)
    {
        foreach (var value in states)
            if (!float.IsFinite(value)) throw new ArgumentException("MHR skeleton states must be finite.");
        for (var start = 0; start < states.Length; start += MhrState.Elements)
        {
            double norm = 0;
            for (var axis = 3; axis < 7; axis++) norm += (double)states[start + axis] * states[start + axis];
            if (Math.Abs(norm - 1) > 1e-3 || states[start + 7] <= 0)
                throw new ArgumentException("MHR skeleton states require unit quaternions and positive scales.");
        }
    }

    static Vector3 Point(ReadOnlySpan<float> values) => new(values[0], values[1], values[2]);
    static Quaternion Rotation(ReadOnlySpan<float> state) => Quaternion.Normalize(new(state[3], state[4], state[5], state[6]));
    static Vector3 Rotate(Quaternion q, Vector3 point)
    {
        var axis = new Vector3(q.X, q.Y, q.Z); var cross = Vector3.Cross(axis, point);
        return point + 2 * (q.W * cross + Vector3.Cross(axis, cross));
    }
}
