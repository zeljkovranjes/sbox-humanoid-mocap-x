using HumanoidMocapX.Inference;

static class SamFixture
{
    internal static Dictionary<string, Tensor> ImageConditioning(int width, bool hand)
    {
        var tensors = new Dictionary<string, Tensor>(); var name = hand ? "ray_cond_emb_hand" : "ray_cond_emb";
        OmegaFixture.Add(tensors, "prompt_encoder.no_mask_embed.weight", width, 1);
        OmegaFixture.Add(tensors, name + ".conv.weight", 1, 1, width + SamRays.Channels, width);
        OmegaFixture.Norm(tensors, name + ".norm", width);
        OmegaFixture.Add(tensors, (hand ? "hand_pe_layer" : "prompt_encoder.pe_layer") + ".positional_encoding_gaussian_matrix", width / 2, 2);
        if (hand) foreach (var pair in tensors.Where(p => p.Key.StartsWith("ray_cond_emb_hand")))
            for (var i = 0; i < pair.Value.Data.Length; i++) pair.Value.Data[i] *= .6f;
        return tensors;
    }

    internal static Dictionary<string, Tensor> Decoder(SamDecoderShape shape, bool hand)
    {
        var tensors = new Dictionary<string, Tensor>(); var prefix = hand ? "decoder_hand" : "decoder";
        for (var layer = 0; layer < 6; layer++)
        {
            var name = prefix + ".layers." + layer;
            foreach (var suffix in new[] { ".ln1", ".ln2_1", ".ln3", ".ln_pe_1" }) OmegaFixture.Norm(tensors, name + suffix, shape.Width);
            foreach (var suffix in new[] { ".ln2_2", ".ln_pe_2" }) OmegaFixture.Norm(tensors, name + suffix, shape.ContextWidth);
            foreach (var cross in new[] { false, true })
            {
                var attention = name + (cross ? ".cross_attn" : ".self_attn");
                OmegaFixture.Linear(tensors, attention + ".q_proj", shape.Width, shape.AttentionWidth);
                OmegaFixture.Linear(tensors, attention + ".k_proj", cross ? shape.ContextWidth : shape.Width, shape.AttentionWidth);
                OmegaFixture.Linear(tensors, attention + ".v_proj", cross ? shape.ContextWidth : shape.Width, shape.AttentionWidth);
                OmegaFixture.Linear(tensors, attention + ".proj", shape.AttentionWidth, shape.Width);
            }
            OmegaFixture.Linear(tensors, name + ".ffn.layers.0.0", shape.Width, shape.HiddenWidth);
            OmegaFixture.Linear(tensors, name + ".ffn.layers.1", shape.HiddenWidth, shape.Width);
        }
        OmegaFixture.Norm(tensors, prefix + ".norm_final", shape.Width);
        return tensors;
    }

    internal static void Feedback(int width, int layer, ReadOnlySpan<float> normalized, Span<float> tokens, Span<float> positions)
    {
        for (var i = 0; i < width; i++)
        {
            tokens[tokens.Length - width + i] += normalized[i] * (.03f * (layer + 1));
            positions[positions.Length - width + i] = normalized[i] * .04f + layer * .01f;
        }
    }
}
