using HumanoidMocapX.EditorTools.MotionBricks;
using HumanoidMocapX.Inference;

static class SamPoseFixture
{
    internal static readonly int[] Triples = [0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 39, 44, 45, 46, 53, 54, 55,
        64, 65, 66, 85, 69, 73, 86, 70, 79, 87, 71, 82, 88, 72, 76, 91, 92, 93, 112, 96, 100,
        113, 97, 106, 114, 98, 109, 115, 99, 103, 130, 131, 132];
    internal static readonly int[] Singles = [1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52,
        56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90,
        94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123];
    internal static readonly int[] HandDofs = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1];
    internal static readonly int[] Left = Enumerable.Range(0, 27).Select(i => 68 + i * 11 % 27).ToArray();
    internal static readonly int[] Right = Enumerable.Range(95, 27).Reverse().ToArray();
    internal static readonly int[] Nonhand = Enumerable.Range(6, 198).Except(Right).Except(Enumerable.Range(136, 26)).ToArray();

    internal static Dictionary<string, Tensor> Head(bool hand, int width = 8)
    {
        var tensors = new Dictionary<string, Tensor>(); var suffix = hand ? "_hand" : "";
        foreach (var camera in new[] { false, true })
        {
            var part = camera ? "camera" : "pose"; var count = camera ? 3 : 519;
            OmegaFixture.Linear(tensors, "head_" + part + suffix + ".proj.layers.0.0", width, width);
            OmegaFixture.Linear(tensors, "head_" + part + suffix + ".proj.layers.1", width, count);
            Add("init_" + part + suffix + ".weight", camera ? new float[] { -1.1f, .02f, -.03f } : Prediction(), count, 1);
        }
        var name = "head_pose" + suffix + ".";
        OmegaFixture.Add(tensors, name + "scale_comps", 68, 28);
        OmegaFixture.Add(tensors, name + "scale_mean", 68);
        Add(name + "hand_pose_mean", Hand(new float[27]), 54);
        var components = new float[54 * 54];
        for (var i = 0; i < 54; i++) { components[i * 54 + i] = .7f; components[i * 54 + (i + 3) % 54] = .13f; }
        Add(name + "hand_pose_comps", components, 54, 54);
        Add(name + "right_wrist_coords", [.7f, .4f, -.2f], 3); Add(name + "root_coords", [.1f, -.3f, .2f], 3);
        var rotation = Matrix(.2, -.3, .4); Add(name + "local_to_world_wrist", rotation.Cast<double>().Select(v => (float)v).ToArray(), 3, 3);
        return tensors;

        void Add(string key, float[] data, params int[] shape) => tensors.Add(key, new(key, data, shape));
    }

    internal static float[] Prediction()
    {
        var output = DenoiserTests.Wave(519, .1f); Columns(.23, -.31, .47).CopyTo(output, 0);
        Body(DenoiserTests.Wave(133, .13f)).CopyTo(output, 6); return output;
    }

    internal static Dictionary<string, Tensor> Feedback(SamDecoderShape shape, bool hand)
    {
        var tensors = new Dictionary<string, Tensor>(); var suffix = hand ? "_hand" : "";
        foreach (var spatial in new[] { false, true })
        {
            var name = (spatial ? "keypoint3d_posemb_linear" : "keypoint_posemb_linear") + suffix + ".layers.";
            OmegaFixture.Linear(tensors, name + "0.0", spatial ? 3 : 2, shape.Width);
            OmegaFixture.Linear(tensors, name + "1", shape.Width, shape.Width);
        }
        OmegaFixture.Linear(tensors, "keypoint_feat_linear" + suffix, shape.ContextWidth, shape.Width);
        return tensors;
    }

    internal static float[] Body(float[] angles)
    {
        var result = new float[260]; var at = 0;
        for (var i = 0; i < Triples.Length; i += 3)
        { Columns(angles[Triples[i]], angles[Triples[i + 1]], angles[Triples[i + 2]]).CopyTo(result, at); at += 6; }
        foreach (var index in Singles) { result[at++] = (float)Math.Sin(angles[index]); result[at++] = (float)Math.Cos(angles[index]); }
        angles.AsSpan(124, 6).CopyTo(result.AsSpan(at)); return result;
    }

    internal static float[] Hand(float[] angles)
    {
        var result = new float[54]; var at = 0;
        foreach (var count in HandDofs)
        {
            if (count == 3) Columns(angles[at], angles[at + 1], angles[at + 2]).CopyTo(result, at * 2);
            else for (var i = at; i < at + count; i++) { result[i * 2] = (float)Math.Sin(angles[i]); result[i * 2 + 1] = (float)Math.Cos(angles[i]); }
            at += count;
        }
        return result;
    }

    internal static float[] Columns(double x, double y, double z)
    {
        var matrix = Matrix(x, y, z); var result = new float[6];
        for (var col = 0; col < 2; col++) for (var row = 0; row < 3; row++) result[col * 3 + row] = (float)matrix[row, col];
        return result;
    }

    internal static double[,] Matrix(double x, double y, double z)
    {
        var a = new double[,] { { 1, 0, 0 }, { 0, Math.Cos(x), -Math.Sin(x) }, { 0, Math.Sin(x), Math.Cos(x) } };
        var b = new double[,] { { Math.Cos(y), 0, Math.Sin(y) }, { 0, 1, 0 }, { -Math.Sin(y), 0, Math.Cos(y) } };
        var c = new double[,] { { Math.Cos(z), -Math.Sin(z), 0 }, { Math.Sin(z), Math.Cos(z), 0 }, { 0, 0, 1 } };
        return Multiply(c, Multiply(b, a));
    }

    internal static double[,] Multiply(double[,] a, double[,] b)
    {
        var result = new double[3, 3];
        for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++) for (var k = 0; k < 3; k++) result[r, c] += a[r, k] * b[k, c];
        return result;
    }
}
