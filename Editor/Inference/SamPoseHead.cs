// C# adaptation of SAM 3D Body's MHR pose head. See LICENSE-SAM and THIRD_PARTY_NOTICES.md.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System;
using System.Threading;

namespace HumanoidMocapX.Inference;

/// <summary>Runs a body/hand head through the complete MHR body and keypoint model.
/// Owns reusable output storage. A shared body may be used sequentially by both heads.</summary>
internal sealed class SamPoseHead
{
    readonly SamPredictionHead prediction;
    readonly SamMhrParameters parameters;
    readonly SamKeypoints keypoints;
    readonly MhrBody body;
    readonly float[] raw = new float[SamMhrParameters.Predictions], model = new float[SamMhrParameters.ModelParameters];
    readonly float[] vertices = new float[SamKeypoints.Vertices * 3], states = new float[SamKeypoints.Joints * MhrState.Elements];
    readonly float[] points = new float[SamKeypoints.Count * 3], expressions = new float[SamMhrParameters.Expressions];
    internal ReadOnlySpan<float> Prediction => raw;
    internal ReadOnlySpan<float> Model => model;
    internal ReadOnlySpan<float> Keypoints => points;
    /// <summary>Posed vertices in MHR units. When the shared body is restricted (as
    /// <see cref="HumanoidMocapX.EditorTools.SamFeatures"/> does), only <see cref="MappedVertices"/>
    /// carry pose corrections; read other vertices only from an unrestricted body.</summary>
    internal ReadOnlySpan<float> Vertices => vertices;
    internal ReadOnlySpan<float> JointStates => states;

    internal SamPoseHead(SamPredictionHead prediction, SamMhrParameters parameters, SamKeypoints keypoints, MhrBody body)
    {
        ArgumentNullException.ThrowIfNull(prediction); ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(keypoints); ArgumentNullException.ThrowIfNull(body);
        if (prediction.Parameters != raw.Length || body.Vertices != SamKeypoints.Vertices || body.Joints != SamKeypoints.Joints
            || body.ModelParameters != model.Length || body.Identities != SamMhrParameters.Shapes || body.Expressions != expressions.Length)
            throw new ArgumentException("SAM pose head requires the released MHR body dimensions.");
        this.prediction = prediction; this.parameters = parameters; this.keypoints = keypoints; this.body = body;
    }

    /// <summary>Vertices that can affect this head's keypoints; see <see cref="SamKeypoints.MappedVertices"/>.</summary>
    internal int[] MappedVertices() => keypoints.MappedVertices();

    /// <summary>Updates predictions, model parameters and camera-space keypoints.
    /// Vertices/joint states retain MHR units for subsequent hand merging. Discard
    /// outputs after an exception; returned spans expire on the next invocation.</summary>
    internal void Run(ReadOnlySpan<float> token, CancellationToken cancellation = default)
    {
        prediction.Run(token, raw, cancellation); parameters.Run(raw, model, cancellation);
        // The released head suppresses facial predictions before evaluating MHR.
        body.Run(raw.AsSpan(SamMhrParameters.ShapeOffset, SamMhrParameters.Shapes), model, expressions, vertices, states, cancellation);
        keypoints.Run(vertices, states, points, cancellation);
    }
}
