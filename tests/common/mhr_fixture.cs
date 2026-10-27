using HumanoidMocap.EditorTools.MotionBricks;
using HumanoidMocapX.Inference;

static class MhrFixture
{
    internal const int Joints = 5, Vertices = 4, Identities = 3, Expressions = 2, ModelParameters = 9, Hidden = 7;
    internal static readonly long[] Indices = { 0, 1, 1, 2, 4, 6, 2, 0, 0, 1, 1, 4, 8, 17, 6, 9 };

    internal static Dictionary<string, Tensor> Tensors()
    {
        var values = new Dictionary<string, Tensor>();
        OmegaFixture.Add(values, MhrSkeleton.Prefix + "blend_shape.base_shape", 3, Vertices);
        OmegaFixture.Add(values, MhrSkeleton.Prefix + "blend_shape.shape_vectors", 3, Vertices, Identities);
        OmegaFixture.Add(values, "face_expressions_model.shape_vectors", 3, Vertices, Expressions);
        OmegaFixture.Add(values, MhrCorrectives.Prefix + "0.sparse_weight", Indices.Length / 2);
        OmegaFixture.Add(values, MhrCorrectives.Prefix + "2.weight", Hidden, Vertices * 3);
        OmegaFixture.Add(values, MhrSkeleton.Prefix + "parameter_transform.parameter_transform", ModelParameters + Identities, Joints * 7);
        OmegaFixture.Add(values, MhrSkeleton.Prefix + "skeleton.joint_translation_offsets", 3, Joints);
        var pre = new float[Joints * 4]; for (var i = 0; i < Joints; i++) pre[i * 4 + 3] = 1;
        var name = MhrSkeleton.Prefix + "skeleton.joint_prerotations"; values.Add(name, new(name, pre, new[] { 4, Joints }));
        return values;
    }

    internal static MhrBody Body(Dictionary<string, Tensor> tensors)
    {
        var weights = new FlowWeights(tensors); var parents = new[] { -1, 0, 1, 1, 3 };
        return new(new(weights, parents, ModelParameters + Identities), Skinning(),
            new(weights, Vertices, Identities, Expressions), new(weights, Indices, Joints, Hidden, Vertices));
    }

    internal static MhrSkinning Skinning()
    {
        var bind = new float[Joints * 8];
        for (var i = 0; i < Joints; i++) { bind[i * 8 + 6] = 1; bind[i * 8 + 7] = 1; }
        return new(bind, new[] { 0, 2, 3, 4 }, new[] { 0, 1, 2, 3 }, new[] { 1f, 1, 1, 1 }, Vertices);
    }
}
