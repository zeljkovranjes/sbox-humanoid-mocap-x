// C# adaptation of SAM 3D Body's MHR head. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.IO;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Maps the released SAM head's 519 predictions into MHR's 204 parameters.</summary>
internal sealed class SamMhrParameters
{
    internal const int Predictions = 519, ModelParameters = 204, PoseParameters = 136;
    internal const int ShapeOffset = 266, Shapes = 45, ScaleOffset = 311, Scales = 28,
        HandOffset = 339, FaceOffset = 447, Expressions = 72;
    readonly Tensor scale, scaleMean, hand, handMean, wrist, root, wristRotation;
    readonly int[] leftIndices, rightIndices, nonhandIndices;
    readonly bool isHand;
    readonly float[] bodyPose = new float[SamCompactPose.BodyParameters], handContinuous = new float[SamCompactPose.HandContinuous];
    readonly float[] handAngles = new float[SamCompactPose.HandParameters];

    internal SamMhrParameters(FlowWeights weights, ReadOnlySpan<int> left, ReadOnlySpan<int> right, ReadOnlySpan<int> nonhand, bool isHand = false)
    {
        this.isHand = isHand;
        var name = isHand ? "head_pose_hand." : "head_pose.";
        scale = Transpose(weights.Get(name + "scale_comps", 68, Scales)); scaleMean = weights.Get(name + "scale_mean", 68);
        hand = Transpose(weights.Get(name + "hand_pose_comps", 54, 54)); handMean = weights.Get(name + "hand_pose_mean", 54);
        wrist = weights.Get(name + "right_wrist_coords", 3); root = weights.Get(name + "root_coords", 3);
        wristRotation = weights.Get(name + "local_to_world_wrist", 3, 3);
        if (left.Length != 27 || right.Length != 27 || nonhand.Length != 145)
            throw new InvalidDataException("SAM hand parameter index counts do not match the released model.");
        Span<bool> used = stackalloc bool[ModelParameters]; used.Clear();
        ValidateIndices(left, used, PoseParameters); ValidateIndices(right, used, PoseParameters);
        used.Clear(); ValidateIndices(nonhand, used, ModelParameters);
        leftIndices = left.ToArray(); rightIndices = right.ToArray(); nonhandIndices = nonhand.ToArray();
    }

    /// <summary>Writes the model vector. Hand/jaw slots from the body prediction
    /// are suppressed; dedicated hand predictions replace the saved hand indices.</summary>
    internal void Run(ReadOnlySpan<float> prediction, Span<float> model, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        if (prediction.Length != Predictions || model.Length != ModelParameters || prediction.Overlaps(model))
            throw new ArgumentException("SAM MHR prediction/model dimensions are invalid or overlap.");
        SamDecoder.Finite(prediction); model.Clear();
        SamRotation.GlobalZYX(prediction.Slice(0, 6), model.Slice(3, 3));
        if (isHand) TransformWrist(model);
        SamCompactPose.Body(prediction.Slice(6, SamCompactPose.BodyContinuous), bodyPose);
        bodyPose.AsSpan(62, 54).Clear(); bodyPose.AsSpan(130, 3).Clear();
        bodyPose.AsSpan(0, 130).CopyTo(model.Slice(6, 130));
        FlowLinear.Run(prediction.Slice(ScaleOffset, Scales), 1, scale, scaleMean, model.Slice(PoseParameters), cancellation);
        ReplaceHand(prediction.Slice(HandOffset, 54), leftIndices, model, cancellation);
        ReplaceHand(prediction.Slice(HandOffset + 54, 54), rightIndices, model, cancellation);
        if (isHand) foreach (var index in nonhandIndices) model[index] = 0;
        SamDecoder.Finite(model);
    }

    void ReplaceHand(ReadOnlySpan<float> prediction, ReadOnlySpan<int> indices, Span<float> model, CancellationToken cancellation)
    {
        FlowLinear.Run(prediction, 1, hand, handMean, handContinuous, cancellation);
        SamCompactPose.Hand(handContinuous, handAngles);
        for (var i = 0; i < indices.Length; i++) model[indices[i]] = handAngles[i];
    }

    void TransformWrist(Span<float> model)
    {
        Span<float> rotation = stackalloc float[9]; Span<float> world = stackalloc float[9];
        SamRotation.MatrixXYZ(model.Slice(3, 3), rotation);
        for (var row = 0; row < 3; row++)
            for (var col = 0; col < 3; col++)
                world[row * 3 + col] = rotation[row * 3] * wristRotation.Data[col]
                    + rotation[row * 3 + 1] * wristRotation.Data[3 + col] + rotation[row * 3 + 2] * wristRotation.Data[6 + col];
        SamRotation.Euler(world, model.Slice(3, 3));
        SamRotation.MatrixXYZ(model.Slice(3, 3), rotation);
        for (var row = 0; row < 3; row++)
        {
            var value = root.Data[row];
            for (var col = 0; col < 3; col++) value += rotation[row * 3 + col] * (wrist.Data[col] - root.Data[col]);
            model[row] = -value * 10;
        }
    }

    static Tensor Transpose(Tensor tensor)
        => new(tensor.Name, NeuralOps.Transpose(tensor.Data, tensor.Rows, tensor.Cols), new[] { tensor.Rows, tensor.Cols });

    static void ValidateIndices(ReadOnlySpan<int> indices, Span<bool> used, int limit)
    {
        foreach (var index in indices)
        {
            if (index < 0 || index >= limit || used[index]) throw new InvalidDataException("SAM hand parameter index is out of range or duplicated.");
            used[index] = true;
        }
    }
}
