using System;
using System.IO;
using System.Numerics;
using System.Threading;
using HumanoidMocapX.EditorTools.Inference;

namespace HumanoidMocapX.Inference;

using Vector3 = System.Numerics.Vector3;

/// <summary>Decodes the basic 491-channel representation to metric SMPL-H motion.
/// Adapted from FlowHMR pipeline_v2m.py; see LICENSE-FLOWHMR.</summary>
internal sealed class FlowDecoder
{
    internal const int JointOffset = 3, RotationOffset = 159, ShapeOffset = 471, ContactOffset = 487, ShapeCount = 16;
    const int Coordinates = 3, RotationChannels = 6, ContactCount = 4, Fps = 30;
    readonly float[] mean, deviation, jointTemplate, jointShapeDirections;

    internal FlowDecoder(ReadOnlySpan<float> mean, ReadOnlySpan<float> deviation,
        ReadOnlySpan<float> jointTemplate, ReadOnlySpan<float> jointShapeDirections)
    {
        Validate(mean, FlowSampler.Channels, nameof(mean));
        Validate(deviation, FlowSampler.Channels, nameof(deviation));
        Validate(jointTemplate, FlowMotion.JointCount * Coordinates, nameof(jointTemplate));
        Validate(jointShapeDirections, FlowMotion.JointCount * Coordinates * ShapeCount, nameof(jointShapeDirections));
        foreach (var value in deviation)
            if (value <= 0) throw new InvalidDataException("FlowHMR standard deviations must be positive.");
        this.mean = mean.ToArray(); this.deviation = deviation.ToArray();
        this.jointTemplate = jointTemplate.ToArray(); this.jointShapeDirections = jointShapeDirections.ToArray();
    }

    /// <summary>Decodes an unpadded [frames,491] prediction and applies upstream's
    /// export-facing Y half-turn. Rejects malformed/non-finite data and degenerate
    /// rotations. Ground placement and foot locking remain shared retargeter passes.</summary>
    internal FlowMotion Decode(ReadOnlySpan<float> latent, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        var frames = latent.Length / FlowSampler.Channels;
        if (frames is < 1 or > FlowRotary.MaximumFrames || latent.Length % FlowSampler.Channels != 0)
            throw new ArgumentException("FlowHMR decoding requires 1–5000 complete 491-channel frames.", nameof(latent));
        Validate(latent, latent.Length, nameof(latent));
        var motion = new FlowMotion
        {
            Parents = FlowMotion.JointParents.ToArray(),
            Rest = ShapedRest(latent, frames, cancellation),
            Poses = new float[frames][],
            Translations = new float[frames][],
            Contacts = new float[frames][]
        };
        var travel = Vector3.Zero;
        for (var frame = 0; frame < frames; frame++)
        {
            cancellation.ThrowIfCancellationRequested();
            var values = latent.Slice(frame * FlowSampler.Channels, FlowSampler.Channels);
            // Frame zero has no preceding displacement. Pelvis Y is already absolute.
            if (frame > 0) { travel.X += Denormalize(values, 0) / Fps; travel.Z += Denormalize(values, 1) / Fps; }
            var rest = motion.Rest[0];
            motion.Translations[frame] = new[] {
                Finite(-(Denormalize(values, JointOffset) + travel.X - rest[0])),
                Finite(Denormalize(values, JointOffset + 1) - rest[1]),
                Finite(-(Denormalize(values, JointOffset + 2) + travel.Z - rest[2]))
            };
            motion.Poses[frame] = Rotations(values, frame);
            var contacts = motion.Contacts[frame] = new float[ContactCount];
            // The regressed contact scores are unbounded; shared tracks require probabilities.
            for (var contact = 0; contact < ContactCount; contact++)
                contacts[contact] = Math.Clamp(Denormalize(values, ContactOffset + contact), 0, 1);
        }
        return motion;
    }

    float[][] ShapedRest(ReadOnlySpan<float> latent, int frames, CancellationToken cancellation)
    {
        Span<double> shapes = stackalloc double[ShapeCount]; shapes.Clear();
        for (var frame = 0; frame < frames; frame++)
        {
            cancellation.ThrowIfCancellationRequested();
            var values = latent.Slice(frame * FlowSampler.Channels, FlowSampler.Channels);
            for (var shape = 0; shape < ShapeCount; shape++) shapes[shape] += Denormalize(values, ShapeOffset + shape);
        }
        for (var shape = 0; shape < ShapeCount; shape++) shapes[shape] /= frames;
        var rest = new float[FlowMotion.JointCount][];
        for (var joint = 0; joint < rest.Length; joint++)
        {
            var position = rest[joint] = new float[Coordinates];
            for (var axis = 0; axis < Coordinates; axis++)
            {
                var index = joint * Coordinates + axis; double value = jointTemplate[index];
                for (var shape = 0; shape < ShapeCount; shape++) value += shapes[shape] * jointShapeDirections[index * ShapeCount + shape];
                position[axis] = Finite((float)value);
            }
        }
        return rest;
    }

    float[] Rotations(ReadOnlySpan<float> values, int frame)
    {
        var poses = new float[FlowMotion.JointCount * Coordinates];
        Span<float> rows = stackalloc float[RotationChannels];
        for (var joint = 0; joint < FlowMotion.JointCount; joint++)
        {
            var offset = RotationOffset + joint * RotationChannels;
            // FlowHMR interleaves two columns; the shared decoder consumes two rows.
            for (var axis = 0; axis < Coordinates; axis++)
            {
                rows[axis] = Denormalize(values, offset + axis * 2);
                rows[axis + Coordinates] = Denormalize(values, offset + axis * 2 + 1);
            }
            Quaternion rotation;
            try { rotation = Quaternion.Conjugate(GvhmrDecoder.Rotation6D(rows)); }
            catch (ArgumentException error) { throw new InvalidDataException($"Invalid FlowHMR rotation at frame {frame}, joint {joint}.", error); }
            if (joint == 0) rotation = new Quaternion(0, 1, 0, 0) * rotation;
            if (rotation.W < 0) rotation = -rotation;
            var axisVector = new Vector3(rotation.X, rotation.Y, rotation.Z);
            var sine = axisVector.Length();
            var angleAxis = axisVector * (sine < 1e-8f ? 2 : 2 * MathF.Atan2(sine, rotation.W) / sine);
            poses[joint * Coordinates] = Finite(angleAxis.X);
            poses[joint * Coordinates + 1] = Finite(angleAxis.Y);
            poses[joint * Coordinates + 2] = Finite(angleAxis.Z);
        }
        return poses;
    }

    float Denormalize(ReadOnlySpan<float> values, int channel)
        => Finite(values[channel] * deviation[channel] + mean[channel]);

    static float Finite(float value) => float.IsFinite(value) ? value
        : throw new InvalidDataException("FlowHMR decoded motion exceeds the finite numeric range.");

    static void Validate(ReadOnlySpan<float> values, int length, string name)
    {
        if (values.Length != length) throw new InvalidDataException("Unexpected FlowHMR decoder tensor length: " + name);
        foreach (var value in values)
            if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite FlowHMR decoder tensor: " + name);
    }
}
