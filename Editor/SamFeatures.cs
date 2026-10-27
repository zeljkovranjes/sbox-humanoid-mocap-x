using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX.Inference;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>Loads installed numeric SAM/MHR assets and constructs the editor's GPU
/// feature extractor. Downloads are a separate, explicit editor action.</summary>
internal static class SamFeatures
{
    /// <summary>Reads checkpoints off the editor thread and builds one extractor that owns
    /// all of its GPU weights. The caller owns the returned extractor.</summary>
    internal static async Task<SamGpuFeatures> LoadAsync(string samPath, string mhrPath, Action<string> progress, CancellationToken cancellation = default)
        => (await LoadAsync(samPath, mhrPath, 1, null, progress, cancellation))[0];

    /// <summary>Reads checkpoints once, off the editor thread, and builds independent
    /// extractors that share DINOv3, decoder and head weights; each keeps its own buffers
    /// and MHR body. GPU weights come from <paramref name="shared"/> when given, which the
    /// caller disposes after every extractor. Reports loading stages and releases GPU
    /// allocations on failure or cancellation. The caller owns the returned extractors.</summary>
    internal static async Task<SamGpuFeatures[]> LoadAsync(string samPath, string mhrPath, int count, FlowGpuWeights shared, Action<string> progress,
        CancellationToken cancellation = default)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        var shape = DinoShape.Sam(); var decoderShape = new SamDecoderShape(SamInitialTokens.Count, shape.Patches);
        var extractors = new SamGpuFeatures[count];
        SamGpuImageEncoder image = null; SamGpuPoseDecoder body = null, hand = null;
        try
        {
            progress?.Invoke("Loading SAM image encoder");
            var backbone = await Task.Run(() => VisionCheckpoint.LoadSamBackbone(samPath, cancellation), cancellation);
            var bodyImage = await Task.Run(() => VisionCheckpoint.LoadSamImageConditioning(samPath, false, cancellation), cancellation);
            var handImage = await Task.Run(() => VisionCheckpoint.LoadSamImageConditioning(samPath, true, cancellation), cancellation);
            progress?.Invoke("Loading SAM decoders");
            var bodyBranch = await Task.Run(() => new Branch(samPath, false, cancellation), cancellation);
            var handBranch = await Task.Run(() => new Branch(samPath, true, cancellation), cancellation);
            var boxWeights = await Task.Run(() => VisionCheckpoint.LoadSamHandBoxes(samPath, cancellation), cancellation);
            // Lanes share the first body's ~664 MB corrective projection; everything mutable stays per lane.
            MhrCorrectives correctives = null;
            for (var lane = 0; lane < count; lane++)
            {
                progress?.Invoke("Loading SAM body model");
                var model = await Task.Run(() => MhrCheckpoint.LoadBody(mhrPath, correctives, cancellation), cancellation);
                correctives ??= model.Correctives;
                var mapped = new SortedSet<int>();
                await GameTask.MainThread(); cancellation.ThrowIfCancellationRequested();
                image = new(backbone, bodyImage, handImage, shape, shared);
                body = await Build(bodyBranch, model, mapped); hand = await Build(handBranch, model, mapped);
                // Both heads read the mesh only through their keypoint mappings, so correcting the
                // mapped vertices gives identical keypoints at a fraction of the projection cost.
                model.Restrict(mapped.ToArray());
                var boxes = new SamHandBoxes(boxWeights);
                await GameTask.MainThread(); cancellation.ThrowIfCancellationRequested();
                extractors[lane] = new(image, body, hand, boxes, shape); image = null; body = hand = null;
            }
            return extractors;
        }
        catch
        {
            await GameTask.MainThread(); hand?.Dispose(); body?.Dispose(); image?.Dispose();
            foreach (var extractor in extractors) extractor?.Dispose();
            throw;
        }

        async Task<SamGpuPoseDecoder> Build(Branch branch, MhrBody model, SortedSet<int> mapped)
        {
            var pose = await Task.Run(() => new SamPoseHead(new(branch.Head, branch.IsHand), SamCheckpoint.LoadParameters(samPath, branch.Head, branch.IsHand, cancellation),
                new(branch.Head, branch.IsHand), model), cancellation);
            mapped.UnionWith(pose.MappedVertices());
            var camera = new SamPredictionHead(branch.Head, branch.IsHand, true);
            var keypoints = new SamKeypointFeedback(branch.Feedback, decoderShape, shape.Width / DinoShape.PatchSize, shape.Height / DinoShape.PatchSize, branch.IsHand);
            var initial = new SamInitialTokens(branch.Initial, branch.IsHand);
            await GameTask.MainThread(); cancellation.ThrowIfCancellationRequested();
            return new(branch.Decoder, decoderShape, initial, new(decoderShape, pose, camera, keypoints), branch.IsHand, shared);
        }
    }

    /// <summary>One body or hand branch's checkpoint tensors, read once and shared by every lane.</summary>
    sealed class Branch(string samPath, bool isHand, CancellationToken cancellation)
    {
        internal readonly bool IsHand = isHand;
        internal readonly FlowWeights Head = VisionCheckpoint.LoadSamHead(samPath, isHand, cancellation);
        internal readonly FlowWeights Feedback = VisionCheckpoint.LoadSamFeedback(samPath, isHand, cancellation);
        internal readonly FlowWeights Initial = VisionCheckpoint.LoadSamInitialTokens(samPath, isHand, cancellation);
        internal readonly FlowWeights Decoder = VisionCheckpoint.LoadSamDecoder(samPath, isHand, cancellation);
    }
}
