using NumericVector3 = System.Numerics.Vector3;
using System.Text;
using HumanoidMocap.Core;
using HumanoidMocap.Core.Cleanup;
using HumanoidMocap.Core.Formats.Fbx;
using HumanoidMocap.Core.Mapping;
using HumanoidMocap.Core.Maths;
using HumanoidMocap.Core.Motion;
using HumanoidMocap.Core.Target;
using HumanoidMocapX;

static class MotionTests
{
    public static void Run()
    {
        Conversion();
        RejectedInputs();
        Retarget(false);
        Retarget(true);
        FootContacts();
        Console.WriteLine("PASS: motion conversion, input validation, Human/Citizen scale and FBX, planted feet and airborne release");
    }

    static void Conversion()
    {
        var source = Fixture(); source.StartTime = 1.5;
        source.Translations[1][0] = 1;
        source.Poses[1][1] = MathF.PI / 2;
        source.Contacts[0] = new[] { .1f, .2f, .3f, .4f };
        source.Model = FlowModel.Base;
        var motion = source.ToDocument();
        Equal("flowhmr_base", motion.ModelVersion);
        Near(1.5, motion.Frames[0].Time);
        Near(1.5 + 1d / 30, motion.Frames[1].Time);
        Near(1, motion.Frames[1].Positions[0][0]);
        Near(1, motion.Frames[1].Positions[0][1]);
        Near(1, NumericVector3.Transform(NumericVector3.UnitZ, MotionDocument.Q(motion.Frames[1].Rotations[0])).X);
        Equal(new[] { "FootL", "ToeL", "FootR", "ToeR" }, motion.StationaryJoints.Select(t => t.Bone).ToArray());
        for (var i = 0; i < 4; i++) Near((i + 1) / 10f, motion.StationaryJoints[i].Probability[0]);
        Equal(BoneRole.PinkyProxL, motion.Bones[28].Role!.Value);
        Equal(BoneRole.ThumbProxR, motion.Bones[49].Role!.Value);
        var restored = MotionDocument.Parse(Encoding.UTF8.GetBytes(motion.ToJson()));
        Near(1, restored.Frames[1].Positions[0][0]);
        var scene = motion.ToSourceScene();
        Near(100, scene.Clips[0].Frames[1][scene.Skeleton.IndexOf("Hips")].Pos.X, .001);
        var exported = FbxImporter.Import(MotionFbxExport.WriteForModelDoc(motion));
        Near(100, exported.Clips[0].Frames[1][exported.Skeleton.IndexOf("Hips")].Pos.X, .002);
        Near(2.54, exported.UnitScaleCm);
        var restX = motion.Bones[1].RestPosition[0];
        motion.Frames[0].Positions[1][0] += 1;
        Near(restX, motion.Bones[1].RestPosition[0]);
        Near(restX, motion.Frames[1].Positions[1][0]);
    }

    static void RejectedInputs()
    {
        var options = new CaptureOptions(); options.Validate();
        Equal(FlowModel.Latest, options.Model); Equal(10, options.CameraInterval);
        foreach (var interval in new[] { 1, 10, 30 }) { options.CameraInterval = interval; options.Validate(); }
        options.CameraInterval = 0; Throws<ArgumentOutOfRangeException>(options.Validate);
        options.CameraInterval = 10; options.Model = (FlowModel)3; Throws<ArgumentOutOfRangeException>(options.Validate);
        foreach (var corrupt in new Action<FlowMotion>[] {
            m => m.Fps = double.NaN, m => m.Fps = 60, m => m.StartTime = -1,
            m => m.Parents[2] = 1, m => m.Rest[4] = new float[2],
            m => m.Poses[0][3] = float.PositiveInfinity, m => m.Contacts[0][0] = 2,
            m => m.Translations = Array.Empty<float[]>(), m => m.Poses = Array.Empty<float[]>(),
            m => m.Contacts = null, m => m.Model = (FlowModel)9
        }) { var motion = Fixture(); corrupt(motion); Throws<FormatException>(() => motion.ToDocument()); }
        var noContacts = Fixture(); noContacts.Contacts = Array.Empty<float[]>();
        Equal(0, noContacts.ToDocument().StationaryJoints.Count);
    }

    static void Retarget(bool citizen)
    {
        var file = citizen ? "target_rig_sbox_citizen.json" : "target_rig_sbox.json";
        var json = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, file));
        var target = citizen ? RetargetTargetSpec.SboxCitizen(json) : RetargetTargetSpec.SboxDefault(json);
        var motion = Fixture().ToDocument();
        var result = Retargeter.Convert(new RetargetRequest
        {
            SourceData = Encoding.UTF8.GetBytes(motion.ToJson()),
            SourceFileName = "capture.hmotion",
            FootPlantCleanup = false,
            MocapCorrections = new TargetCorrectionSettings { FirstPerson = false, StabilizeFeet = false }
        }, target);
        Check(result.Success, string.Join("; ", result.Errors.Concat(result.Clips.Select(c => c.Error))));
        var clip = result.Clips.Single();
        Check(clip.SolvedFrames.Count > 0, "Retargeting returned no frames");
        Equal(target.Rig.Skeleton.Count, clip.SolvedFrames[0].Length);
        foreach (var frame in clip.SolvedFrames)
            foreach (var pose in frame) Check(float.IsFinite(pose.Pos.Length()) && float.IsFinite(pose.Rot.Length()), "Non-finite target pose");
        var times = Enumerable.Range(0, clip.SolvedFrames.Count).Select(i => i / (double)clip.Fps).ToArray();
        var bytes = ModelDocFbxExport.Write(target.Rig.Skeleton, "capture", clip.SolvedFrames, times,
            clip.Fps, target.UpAxis, "WorldRelative");
        var parsed = FbxImporter.Import(bytes);
        Near(2.54, parsed.UnitScaleCm);
        var hips = target.Rig.BoneForRole(BoneRole.Hips)!.Value;
        var name = target.Rig.Skeleton[hips].Name;
        var importedHips = parsed.Skeleton.IndexOf(name);
        Check(importedHips >= 0, "Export lost pelvis");
        var expected = clip.SolvedFrames[0][hips].Pos;
        if (target.UpAxis == TargetUpAxis.ZUpEngine) expected *= 2.54f;
        Near(0, NumericVector3.Distance(expected, parsed.Clips[0].Frames[0][importedHips].Pos), .003);
    }

    static void FootContacts()
    {
        var fixture = Fixture();
        for (var f = 0; f < fixture.Poses.Length; f++) fixture.Translations[f][0] = f * .0005f;
        var source = fixture.ToDocument().ToSourceScene();
        var rig = TargetRig.FromSkeleton(source.Skeleton, source.AuthoredMapping);
        var frames = source.Clips[0].Frames.Select(f => (XForm[])f.Clone()).ToList();
        var ankle = rig.BoneForRole(BoneRole.FootL)!.Value;
        float Travel(List<XForm[]> poses)
        {
            var first = new HumanoidMocap.Core.Skeleton.Pose(poses[0]).ToWorld(rig.Skeleton);
            var last = new HumanoidMocap.Core.Skeleton.Pose(poses[^1]).ToWorld(rig.Skeleton);
            return Math.Abs(last[ankle].Pos.X - first[ankle].Pos.X);
        }
        var before = Travel(frames);
        var result = CaptureFootLock.Apply(frames, source, source.AuthoredMapping, rig, TargetUpAxis.YUpCm);
        Check(result.ConstrainedSamples > 0, "No foot anchors were applied");
        Check(Travel(frames) < before, "Foot locking did not reduce planted drift");
        for (var f = 0; f < fixture.Poses.Length; f++)
        {
            Array.Clear(fixture.Contacts[f]); fixture.Translations[f][1] = .5f;
        }
        source = fixture.ToDocument().ToSourceScene();
        frames = source.Clips[0].Frames.Select(f => (XForm[])f.Clone()).ToList();
        var airborne = CaptureFootLock.Apply(frames, source, source.AuthoredMapping, rig, TargetUpAxis.YUpCm);
        Equal(0, airborne.ConstrainedSamples);
        for (var f = 0; f < frames.Count; f++)
            for (var j = 0; j < frames[f].Length; j++)
                Near(0, NumericVector3.Distance(frames[f][j].Pos, source.Clips[0].Frames[f][j].Pos));
    }

    internal static FlowMotion Fixture()
    {
        int[] parents = { -1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,
            20,22,23,20,25,26,20,28,29,20,31,32,20,34,35,21,37,38,21,40,41,21,43,44,21,46,47,21,49,50 };
        float[][] rest = {
            new[]{0f,1f,0f}, new[]{.1f,.95f,0f}, new[]{-.1f,.95f,0f}, new[]{0f,1.12f,0f},
            new[]{.1f,.52f,.02f},new[]{-.1f,.52f,.02f},new[]{0f,1.25f,0f},new[]{.1f,.1f,0f},
            new[]{-.1f,.1f,0f},new[]{0f,1.4f,0f},new[]{.1f,.04f,.15f},new[]{-.1f,.04f,.15f},
            new[]{0f,1.55f,0f},new[]{.1f,1.45f,0f},new[]{-.1f,1.45f,0f},new[]{0f,1.7f,0f},
            new[]{.2f,1.45f,0f},new[]{-.2f,1.45f,0f},new[]{.45f,1.45f,0f},new[]{-.45f,1.45f,0f},
            new[]{.7f,1.45f,0f},new[]{-.7f,1.45f,0f}
        };
        Array.Resize(ref rest, 52);
        for (var j = 22; j < 52; j++)
        {
            var parent = rest[parents[j]];
            rest[j] = new[] { parent[0] + (j < 37 ? .03f : -.03f), parent[1], parent[2] + .005f * ((j - 22) / 3 % 5) };
        }
        return new FlowMotion
        {
            Parents = parents,
            Rest = rest,
            Poses = Enumerable.Range(0, 60).Select(_ => new float[156]).ToArray(),
            Translations = Enumerable.Range(0, 60).Select(_ => new float[3]).ToArray(),
            Contacts = Enumerable.Range(0, 60).Select(_ => new[] { 1f, 1f, 1f, 1f }).ToArray()
        };
    }

    internal static void Check(bool condition, string message) { if (!condition) throw new Exception(message); }
    internal static void Near(double expected, double actual, double tolerance = 1e-5)
        => Check(double.IsFinite(actual) && Math.Abs(expected - actual) <= tolerance, $"Expected {expected}, got {actual}");
    internal static void Equal<T>(T expected, T actual) => Check(EqualityComparer<T>.Default.Equals(expected, actual), $"Expected {expected}, got {actual}");
    static void Equal(string[] expected, string[] actual) => Check(expected.SequenceEqual(actual), "Sequences differ");
    internal static void Throws<T>(Action action) where T : Exception
    {
        try { action(); } catch (T) { return; }
        throw new Exception("Expected " + typeof(T).Name);
    }
}
