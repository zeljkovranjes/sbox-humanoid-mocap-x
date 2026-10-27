using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using HumanoidMocapX.Inference;

namespace HumanoidMocapX.EditorTools;

internal static class VisionCheckpoint
{
    /// <summary>Loads image, aggregator and camera-head tensors from a verified
    /// checkpoint, excluding the unused dense-depth and text-alignment heads.</summary>
    internal static FlowWeights LoadOmega(string path, CancellationToken cancellation = default)
        => Load(path, OmegaElements, cancellation, "camera_head.", "aggregator.");

    /// <summary>Loads only the Omega tensors under <paramref name="prefixes"/>, for building the
    /// model part by part (OmegaGpuModel.LoadAsync). Fails like LoadOmega, and when no tensor matches.</summary>
    internal static FlowWeights LoadOmegaPart(string path, CancellationToken cancellation, params string[] prefixes)
        => Load(path, OmegaElements, cancellation, prefixes);

    const long OmegaElements = 1_200_000_000;

    /// <summary>Loads only camera-head tensors from a verified Omega checkpoint,
    /// without executing pickle code. Run off the editor thread. Shapes are checked
    /// when binding the head; malformed metadata and oversized tensors fail here.</summary>
    internal static FlowWeights LoadOmegaCameraHead(string path, CancellationToken cancellation = default)
        => Load(path, 220_000_000, cancellation, "camera_head.");

    /// <summary>Loads SAM's DINOv3 backbone from its verified checkpoint, using either
    /// a root state dictionary or state_dict. Excludes decoder and MHR tensors.</summary>
    internal static FlowWeights LoadSamBackbone(string path, CancellationToken cancellation = default)
        => Load(path, 900_000_000, cancellation, "backbone.encoder.");

    /// <summary>Loads one SAM pose transformer's weights, excluding image and MHR
    /// tensors. Binding the decoder validates every required layer and its shape.</summary>
    internal static FlowWeights LoadSamDecoder(string path, bool hand, CancellationToken cancellation = default)
        => Load(path, 50_000_000, cancellation, hand ? "decoder_hand." : "decoder.");

    /// <summary>Loads no-mask image conditioning and body/hand camera-position tensors.</summary>
    internal static FlowWeights LoadSamImageConditioning(string path, bool hand, CancellationToken cancellation = default)
        => Load(path, 2_000_000, cancellation, hand ? "ray_cond_emb_hand." : "ray_cond_emb.",
            hand ? "hand_pe_layer." : "prompt_encoder.pe_layer.", "prompt_encoder.no_mask_embed.");

    /// <summary>Loads one body/hand pose and camera head, including continuous-pose
    /// mappings and keypoint regression. Integer index buffers are read separately.</summary>
    internal static FlowWeights LoadSamHead(string path, bool hand, CancellationToken cancellation = default)
    {
        var pose = hand ? "head_pose_hand." : "head_pose.";
        return Load(path, 12_000_000, cancellation, pose + "proj.", pose + "scale_", pose + "hand_pose_",
            pose + "keypoint_mapping", pose + "right_wrist_coords", pose + "root_coords", pose + "local_to_world_wrist",
            hand ? "head_camera_hand.proj." : "head_camera.proj.",
            hand ? "init_pose_hand." : "init_pose.", hand ? "init_camera_hand." : "init_camera.");
    }

    /// <summary>Loads the 2D/3D keypoint position MLPs and sampled-image projection.</summary>
    internal static FlowWeights LoadSamFeedback(string path, bool hand, CancellationToken cancellation = default)
    {
        var suffix = hand ? "_hand." : ".";
        return Load(path, 5_000_000, cancellation, "keypoint_posemb_linear" + suffix,
            "keypoint3d_posemb_linear" + suffix, "keypoint_feat_linear" + suffix);
    }

    /// <summary>Loads initial pose/camera estimates, CLIFF projections, the invalid
    /// point prompt and body/hand keypoint tokens for the first decoder pass.</summary>
    internal static FlowWeights LoadSamInitialTokens(string path, bool hand, CancellationToken cancellation = default)
    {
        var suffix = hand ? "_hand." : ".";
        return Load(path, 4_000_000, cancellation, "init_pose" + suffix, "init_camera" + suffix,
            "init_to_token_mhr" + suffix, "prev_to_token_mhr" + suffix, "prompt_to_token.",
            "prompt_encoder.invalid_point_embed.", "hand_box_embedding.",
            "keypoint_embedding" + suffix, "keypoint3d_embedding" + suffix);
    }

    /// <summary>Loads the three-layer hand-box regressor. The classification logits
    /// are unused by upstream's full inference crop selection.</summary>
    internal static FlowWeights LoadSamHandBoxes(string path, CancellationToken cancellation = default)
        => Load(path, 3_000_000, cancellation, "bbox_embed.");

    static FlowWeights Load(string path, long maximumElements, CancellationToken cancellation, params string[] prefixes)
    {
        cancellation.ThrowIfCancellationRequested();
        using var checkpoint = new TorchCheckpoint(path);
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        long elements = 0;
        foreach (var pair in checkpoint.Tensors)
        {
            if (!Include(pair.Key, prefixes)) continue;
            cancellation.ThrowIfCancellationRequested();
            if (pair.Value.Shape.Length is < 1 or > 4) throw new InvalidDataException("Unexpected vision tensor rank: " + pair.Key);
            long count = 1;
            foreach (var dimension in pair.Value.Shape)
            {
                if (dimension < 1) throw new InvalidDataException("Empty vision tensor: " + pair.Key);
                count = checked(count * dimension);
            }
            elements = checked(elements + count);
            if (elements > maximumElements) throw new InvalidDataException("Vision weights exceed their supported size.");
        }
        if (elements == 0) throw new InvalidDataException("Checkpoint has no tensors for the selected vision model.");
        foreach (var pair in checkpoint.Tensors)
        {
            if (!Include(pair.Key, prefixes)) continue;
            cancellation.ThrowIfCancellationRequested();
            var shape = pair.Value.Shape.AsSpan().ToArray(); Array.Reverse(shape);
            tensors.Add(pair.Key, new Tensor(pair.Key, checkpoint.ReadFloat(pair.Key, cancellation), shape));
        }
        cancellation.ThrowIfCancellationRequested();
        return new(tensors);
    }

    static bool Include(string name, string[] prefixes)
    {
        foreach (var prefix in prefixes) if (name.StartsWith(prefix, StringComparison.Ordinal)) return true;
        return false;
    }
}
