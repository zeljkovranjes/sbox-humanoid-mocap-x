using HumanoidMocap.EditorTools.MotionBricks;

static class OmegaFixture
{
    internal static Dictionary<string, Tensor> Full(int width, int heads)
    {
        var tensors = Create(width * 2, heads);
        Image(tensors, "aggregator.patch_embed", width, heads, false);
        Add(tensors, "aggregator.camera_token", width, 1, 2, 1); Add(tensors, "aggregator.register_token", width, 16, 2, 1);
        Add(tensors, "aggregator.rope_embed.periods", width / heads / 4);
        for (var i = 0; i < width / heads / 4; i++) tensors["aggregator.rope_embed.periods"].Data[i] = MathF.Pow(100, i / (float)(width / heads / 4));
        for (var i = 0; i < 24; i++)
        {
            Block(tensors, "aggregator.frame_blocks." + i, width, heads, true);
            Block(tensors, "aggregator.inter_frame_blocks." + i, width, heads, true);
            // Released Omega masks all QKV biases in its image and aggregator blocks.
            Array.Clear(tensors["aggregator.patch_embed.blocks." + i + ".attn.qkv.bias_mask"].Data);
            Array.Clear(tensors["aggregator.frame_blocks." + i + ".attn.qkv.bias_mask"].Data);
            Array.Clear(tensors["aggregator.inter_frame_blocks." + i + ".attn.qkv.bias_mask"].Data);
        }
        return tensors;
    }

    internal static Dictionary<string, Tensor> Sam(int width, int heads)
    {
        var tensors = new Dictionary<string, Tensor>();
        Image(tensors, "backbone.encoder", width, heads, true);
        return tensors;
    }

    static void Image(Dictionary<string, Tensor> tensors, string name, int width, int heads, bool sam)
    {
        Add(tensors, name + ".patch_embed.proj.weight", 16, 16, 3, width); Add(tensors, name + ".patch_embed.proj.bias", width);
        Add(tensors, name + ".cls_token", width, 1, 1); Add(tensors, name + ".storage_tokens", width, 4, 1);
        Norm(tensors, name + ".norm", width);
        Add(tensors, name + ".rope_embed.periods", width / heads / 4);
        for (var i = 0; i < width / heads / 4; i++) tensors[name + ".rope_embed.periods"].Data[i] = MathF.Pow(100, i / (float)(width / heads / 4));
        for (var i = 0; i < (sam ? 32 : 24); i++) Block(tensors, name + ".blocks." + i, width, heads, false, sam);
    }

    internal static Dictionary<string, Tensor> Create(int width, int heads)
    {
        var tensors = new Dictionary<string, Tensor>();
        Norm(tensors, "camera_head.token_norm", width); Norm(tensors, "camera_head.trunk_norm", width);
        for (var i = 0; i < 4; i++) Block(tensors, "camera_head.trunk." + i, width, heads, false);
        Linear(tensors, "camera_head.camera_branch.0", width, width / 2);
        Linear(tensors, "camera_head.camera_branch.2", width / 2, 9);
        return tensors;
    }

    internal static void Block(Dictionary<string, Tensor> tensors, string name, int width, int heads, bool qk, bool swiglu = false)
    {
        Norm(tensors, name + ".norm1", width); Norm(tensors, name + ".norm2", width);
        Linear(tensors, name + ".attn.qkv", width, 3 * width); Linear(tensors, name + ".attn.proj", width, width);
        Linear(tensors, name + (swiglu ? ".mlp.w1" : ".mlp.fc1"), width, 4 * width);
        Linear(tensors, name + (swiglu ? ".mlp.w3" : ".mlp.fc2"), 4 * width, width);
        if (swiglu) Linear(tensors, name + ".mlp.w2", width, 4 * width);
        Add(tensors, name + ".ls1.gamma", width); Add(tensors, name + ".ls2.gamma", width);
        var maskName = name + ".attn.qkv.bias_mask"; Add(tensors, maskName, width * 3);
        for (var i = 0; i < width * 3; i++) tensors[maskName].Data[i] = i >= width && i < width * 2 ? 0 : 1;
        if (qk) { Norm(tensors, name + ".attn.q_norm", width / heads); Norm(tensors, name + ".attn.k_norm", width / heads); }
    }

    internal static void Norm(Dictionary<string, Tensor> tensors, string name, int width)
    {
        Add(tensors, name + ".weight", width); Add(tensors, name + ".bias", width);
        for (var i = 0; i < width; i++) tensors[name + ".weight"].Data[i] += 1;
    }
    internal static void Linear(Dictionary<string, Tensor> tensors, string name, int inputs, int outputs)
    { Add(tensors, name + ".weight", inputs, outputs); Add(tensors, name + ".bias", outputs); }
    internal static void Add(Dictionary<string, Tensor> tensors, string name, params int[] shape)
    {
        var count = shape.Aggregate(1, (a, b) => a * b);
        var values = Enumerable.Range(0, count).Select(i => .09f * MathF.Sin(i * .371f + tensors.Count * .17f)).ToArray();
        tensors.Add(name, new(name, values, shape));
    }
}
