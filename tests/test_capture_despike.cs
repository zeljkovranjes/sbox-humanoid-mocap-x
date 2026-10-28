using System.Numerics;
using HumanoidMocapX;
using HumanoidMocapX.Core;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Core.Target;

static class CaptureDespikeTests
{
    internal static void Run()
    {
        // A steady fast spin, 40° a frame, is not a snap.
        var spin = MotionTests.Fixture();
        for (var f = 0; f < spin.Poses.Length; f++) spin.Poses[f][1] = f * 40 * MathF.PI / 180;
        var spinning = spin.ToDocument();
        MotionTests.Equal(0, CaptureDespike.Apply(spinning));

        // One frame where a knee snaps 20° out of line and the body jumps 6 cm.
        var snap = MotionTests.Fixture();
        snap.Poses[30][3 * 4] = 20 * MathF.PI / 180; snap.Translations[30][0] = .06f;
        var snapped = snap.ToDocument();
        var bone = ObservationFixture.Bone(snapped, "LowerLegL");
        MotionTests.Check(CaptureDespike.Apply(snapped) > 0, "The snap was not corrected");
        var q = MotionDocument.Q(snapped.Frames[30].Rotations[bone]);
        var off = 2 * Math.Acos(Math.Min(1, Math.Abs(q.W))) * 180 / Math.PI;
        MotionTests.Check(off < 7, $"The knee still snaps {off:F1} degrees out of line");
        var root = MotionDocument.V(snapped.Frames[30].Positions[0]) - MotionDocument.V(snapped.Frames[29].Positions[0]);
        MotionTests.Check(root.Length() < .02f, $"The body still jumps {root.Length() * 100:F1} cm");
        // On the retargeted Human: a one-frame 25° snap of the right arm is pulled back, while the planted feet and the left
        // hand, held still by earlier cleanup as the body sways, stay exactly where they are.
        var rig = TargetRig.SboxDefault(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "target_rig_sbox.json"))); var skeleton = rig.Skeleton;
        int Bone(BoneRole role) => rig.BoneForRole(role) ?? throw new Exception("missing " + role);
        var frames = Enumerable.Range(0, 40).Select(f =>
        {
            var pose = Enumerable.Range(0, skeleton.Count).Select(b => skeleton[b].RestLocal).ToArray();
            pose[Bone(BoneRole.Spine1)].Rot = Quaternion.Normalize(Quaternion.CreateFromAxisAngle(Vector3.UnitY, .15f * MathF.Sin(f * .2f)) * pose[Bone(BoneRole.Spine1)].Rot);
            return pose;
        }).ToList();
        var held = new[] { BoneRole.HandL, BoneRole.FootL, BoneRole.FootR }.Select(Bone).ToArray();
        var anchors = held.Select(b => World(skeleton, frames[0])[b].Pos).ToArray();
        // A quarter of the way in from the rest pose's straight arm, so the arm is bent and can hold the hand as the body sways.
        anchors[0] = Vector3.Lerp(anchors[0], World(skeleton, frames[0])[Bone(BoneRole.UpperArmL)].Pos, .25f);
        // The swaying body's left hand pinned the way the hand plant pins it.
        HumanoidMocapX.Core.Cleanup.EffectorIk.ApplyGoals(frames, skeleton, new HumanoidMocapX.Core.Cleanup.LimbChain { Upper = Bone(BoneRole.UpperArmL), Lower = Bone(BoneRole.LowerArmL), End = Bone(BoneRole.HandL) },
            Enumerable.Repeat(anchors[0], frames.Count).ToArray(), Vector3.UnitX, soften: 0);
        var shoulder = Bone(BoneRole.UpperArmR); var handR = Bone(BoneRole.HandR);
        frames[20][shoulder].Rot = Quaternion.Normalize(Quaternion.CreateFromAxisAngle(Vector3.UnitX, 25 * MathF.PI / 180) * frames[20][shoulder].Rot);
        var heldBefore = held.Select(b => frames.Select(f => World(skeleton, f)[b].Pos).ToArray()).ToArray();
        var swung = frames.Select(f => World(skeleton, f)[handR].Pos * 2.54f).ToArray();
        var swing = Vector3.Distance(swung[20], (swung[19] + swung[21]) / 2);
        MotionTests.Check(CaptureDespike.Apply(frames, rig, 1 / 2.54f) > 0, "The retargeted snap was not corrected");
        var path = frames.Select(f => World(skeleton, f)[handR].Pos * 2.54f).ToArray();
        var jump = Vector3.Distance(path[20], (path[19] + path[21]) / 2);
        MotionTests.Check(swing > 10 && jump < 2, $"The right hand still snaps {jump:F1} cm out of line (from {swing:F1} cm)");
        for (var h = 0; h < held.Length; h++)
            for (var f = 0; f < frames.Count; f++)
            {
                var moved = Vector3.Distance(World(skeleton, frames[f])[held[h]].Pos, heldBefore[h][f]) * 2.54f;
                MotionTests.Check(moved < .05f, $"{rig.RoleOf(held[h])} moved {moved:F2} cm at frame {f}");
            }
        Console.WriteLine("PASS: capture despike pulls a one-frame snap back onto its neighbours' path, leaves a steady spin alone, and on a retargeted rig keeps held hands and planted feet in place");
    }

    static XForm[] World(HumanoidMocapX.Core.Skeleton.Skeleton skeleton, XForm[] locals)
    {
        var world = new XForm[skeleton.Count];
        for (var i = 0; i < world.Length; i++) world[i] = skeleton[i].ParentIndex < 0 ? locals[i] : XForm.Compose(world[skeleton[i].ParentIndex], locals[i]);
        return world;
    }
}
