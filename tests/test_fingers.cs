using System.Numerics;
using System.Text;
using HumanoidMocapX.Core;
using HumanoidMocapX;
using HumanoidMocapX.Core.Formats.Fbx;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Core.Target;

static class FingerTests
{
    internal static void Run()
    {
        var fixture = MotionTests.Fixture();
        for (var frame = 0; frame < fixture.Poses.Length; frame++)
            for (var joint = 22; joint < 52; joint++) fixture.Poses[frame][joint * 3 + 2] = frame / (float)(fixture.Poses.Length - 1) * (.25f + (joint - 22) * .015f);
        var motion = fixture.ToDocument(); var source = FbxImporter.Import(MotionFbxExport.WriteForModelDoc(motion));
        for (var joint = 22; joint < 52; joint++)
        {
            var index = source.Skeleton.IndexOf(motion.Bones[joint].Name);
            MotionTests.Check(index >= 0, "Source FBX lost a finger bone");
            var expected = Similarity(MotionDocument.Q(motion.Frames[0].Rotations[joint]), MotionDocument.Q(motion.Frames[^1].Rotations[joint]));
            MotionTests.Near(expected, Similarity(source.Clips[0].Frames[0][index].Rot, source.Clips[0].Frames[^1][index].Rot), 1e-4);
        }
        Retarget(motion, false); Retarget(motion, true);
        Console.WriteLine("PASS: all 30 source finger rotations survive FBX; Human/Citizen finger articulation survives retargeting and target FBX export");
    }

    static void Retarget(MotionDocument motion, bool citizen)
    {
        var json = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, citizen ? "target_rig_sbox_citizen.json" : "target_rig_sbox.json"));
        var target = citizen ? RetargetTargetSpec.SboxCitizen(json) : RetargetTargetSpec.SboxDefault(json);
        var result = Retargeter.Convert(new RetargetRequest
        {
            SourceData = Encoding.UTF8.GetBytes(motion.ToJson()),
            SourceFileName = "fingers.hmotion",
            FootPlantCleanup = false,
            MocapCorrections = new TargetCorrectionSettings { FirstPerson = false, StabilizeFeet = false }
        }, target);
        MotionTests.Check(result.Success, string.Join("; ", result.Errors.Concat(result.Clips.Select(c => c.Error))));
        var clip = result.Clips.Single(); var times = Enumerable.Range(0, clip.SolvedFrames.Count).Select(i => i / (double)clip.Fps).ToArray();
        var parsed = FbxImporter.Import(ModelDocFbxExport.Write(target.Rig.Skeleton, "fingers", clip.SolvedFrames, times, clip.Fps, target.UpAxis, "WorldRelative"));
        var mapped = 0;
        foreach (var bone in motion.Bones.Skip(22))
        {
            var index = target.Rig.BoneForRole(bone.Role!.Value);
            if (index is null)
            {
                MotionTests.Check(citizen && bone.Role is BoneRole.PinkyProxL or BoneRole.PinkyMidL or BoneRole.PinkyDistL
                    or BoneRole.PinkyProxR or BoneRole.PinkyMidR or BoneRole.PinkyDistR, "Target is missing an expected finger joint");
                continue;
            }
            mapped++;
            var similarity = Similarity(clip.SolvedFrames[0][index.Value].Rot, clip.SolvedFrames[^1][index.Value].Rot);
            MotionTests.Check(similarity < .9999f, "Retargeting froze finger " + bone.Role);
            var exported = parsed.Skeleton.IndexOf(target.Rig.Skeleton[index.Value].Name);
            MotionTests.Check(exported >= 0, "Target FBX lost finger " + bone.Role);
            MotionTests.Near(similarity, Similarity(parsed.Clips[0].Frames[0][exported].Rot, parsed.Clips[0].Frames[^1][exported].Rot), 1e-4);
        }
        MotionTests.Equal(citizen ? 24 : 30, mapped);
    }

    static float Similarity(Quaternion first, Quaternion second) => MathF.Abs(Quaternion.Dot(Quaternion.Normalize(first), Quaternion.Normalize(second)));
}
