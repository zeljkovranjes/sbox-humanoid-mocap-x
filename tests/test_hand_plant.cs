using System.Numerics;
using HumanoidMocapX.Core;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Core.Target;

static class HandPlantTests
{
    internal static void Run()
    {
        var json = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "target_rig_sbox.json"));
        var rig = TargetRig.SboxDefault(json); var skeleton = rig.Skeleton;
        int Bone(BoneRole role) => rig.BoneForRole(role) ?? throw new Exception("missing " + role);
        var up = Vector3.UnitZ; const float Inch = 2.54f;
        // The bind pose upside down on its left hand, the right arm swung off the floor: a one-handed handstand.
        var roll = Quaternion.CreateFromAxisAngle(Vector3.UnitX, -MathF.PI / 2);
        var rest = Enumerable.Range(0, skeleton.Count).Select(b => skeleton[b].RestLocal).ToArray();
        var roots = Enumerable.Range(0, skeleton.Count).Where(b => skeleton[b].ParentIndex < 0).ToArray();
        foreach (var r in roots) rest[r] = new XForm(Vector3.Transform(rest[r].Pos, roll), Quaternion.Normalize(roll * rest[r].Rot));
        var handL = Bone(BoneRole.HandL); var handR = Bone(BoneRole.HandR); var armR = Bone(BoneRole.UpperArmR);
        var world = World(skeleton, rest);
        // Of a few shoulder and elbow turns, keep the one that lifts the right hand highest off the floor.
        var best = rest; var lift = float.MinValue;
        foreach (var joint in new[] { armR, Bone(BoneRole.LowerArmR) })
            foreach (var axis in new[] { Vector3.UnitX, -Vector3.UnitX, Vector3.UnitY, -Vector3.UnitY, Vector3.UnitZ, -Vector3.UnitZ })
                foreach (var angle in new[] { .4f, .6f, .8f })
                {
                    var lifted = (XForm[])rest.Clone(); var before = World(skeleton, lifted);
                    var turned = Quaternion.CreateFromAxisAngle(axis, MathF.PI * angle) * before[joint].Rot;
                    lifted[joint].Rot = Quaternion.Normalize(Quaternion.Inverse(before[skeleton[joint].ParentIndex].Rot) * turned);
                    var after = World(skeleton, lifted);
                    if (after[handR].Pos.Z - after[handL].Pos.Z > lift) { lift = after[handR].Pos.Z - after[handL].Pos.Z; best = lifted; }
                }
        rest = best;
        world = World(skeleton, rest);
        // The hand and its fingers only: rigs hang IK helpers under the hand that sit far from it.
        var handParts = Enumerable.Range(0, skeleton.Count).Where(b => b == handL || IsWithin(skeleton, b, handL) && rig.RoleOf(b)?.ToString() is { } role
            && new[] { "Thumb", "Index", "Middle", "Ring", "Pinky" }.Any(role.StartsWith)).ToArray();
        var lowest = handParts.Min(b => world[b].Pos.Z);
        foreach (var r in roots) rest[r].Pos -= up * (lowest - 1 / Inch);
        world = World(skeleton, rest);
        var feet = new[] { BoneRole.FootL, BoneRole.FootR }.Min(r => world[Bone(r)].Pos.Z) * Inch;
        MotionTests.Check(feet > CaptureHandPlant.SupportFeetCm, $"Fixture feet only {feet:F0} cm up");
        MotionTests.Check(world[handR].Pos.Z * Inch > CaptureHandPlant.LiftCm / 1.5f, $"Fixture right hand only {world[handR].Pos.Z * Inch:F0} cm up");

        // 30 frames sliding 3 cm a frame along the floor: too far and fast for a resting hand, not for one carrying the body.
        var frames = Enumerable.Range(0, 30).Select(f => { var pose = (XForm[])rest.Clone(); foreach (var r in roots) pose[r].Pos += new Vector3(f * 3 / Inch, 0, 0); return pose; }).ToList();
        var held = CaptureHandPlant.Apply(frames, rig, TargetUpAxis.ZUpEngine, 30);
        MotionTests.Check(held >= 25, $"Expected the support hand held, held {held}");
        var start = World(skeleton, frames[0])[handL].Pos; var end = World(skeleton, frames[^1])[handL].Pos;
        MotionTests.Check(Vector3.Distance(Flat(start), Flat(end)) * Inch < 1, $"Support hand slid {Vector3.Distance(Flat(start), Flat(end)) * Inch:F1} cm");

        var last = World(skeleton, frames[^1]);
        var palmLowest = handParts.Min(b => last[b].Pos.Z) * Inch;
        MotionTests.Near(0, palmLowest, 1);
        var index = Bone(BoneRole.IndexProxL); var middle = Bone(BoneRole.MiddleProxL); var pinky = Bone(BoneRole.PinkyProxL);
        var palm = Vector3.Normalize(Vector3.Cross(last[middle].Pos - last[handL].Pos, last[index].Pos - last[pinky].Pos));
        MotionTests.Check(Vector3.Dot(palm, -up) > MathF.Cos(5 * MathF.PI / 180), "The support palm must face the floor");
        foreach (var finger in new[] { "Index", "Middle", "Ring", "Pinky" })
        {
            var chain = new[] { "Prox", "Mid", "Dist" }.Select(k => Bone(Enum.Parse<BoneRole>(finger + k + "L"))).ToArray();
            var a = last[chain[1]].Pos - last[chain[0]].Pos; var b = last[chain[2]].Pos - last[chain[1]].Pos;
            MotionTests.Check(Vector3.Dot(Vector3.Normalize(a), Vector3.Normalize(b)) > MathF.Cos(3 * MathF.PI / 180), finger + " finger of the support hand must lie straight");
        }
        // The body is carried over the support hand, so the raised hand stops sliding with it but stays raised.
        MotionTests.Check(last[handR].Pos.Z * Inch > CaptureHandPlant.LiftCm / 1.5f, $"The raised hand was pulled to {last[handR].Pos.Z * Inch:F0} cm");
        MotionTests.Near(world[handR].Pos.Z, last[handR].Pos.Z, .5);
        Console.WriteLine("PASS: hand plant holds a hand carrying the body however far the capture slides, palm flat and fingers straight, and leaves the raised hand free");
    }

    static Vector3 Flat(Vector3 v) => v with { Z = 0 };
    static bool IsWithin(HumanoidMocapX.Core.Skeleton.Skeleton skeleton, int bone, int ancestor)
    {
        for (var b = bone; b >= 0; b = skeleton[b].ParentIndex) if (b == ancestor) return true;
        return false;
    }
    static XForm[] World(HumanoidMocapX.Core.Skeleton.Skeleton skeleton, XForm[] locals)
    {
        var world = new XForm[skeleton.Count];
        for (var i = 0; i < world.Length; i++) world[i] = skeleton[i].ParentIndex < 0 ? locals[i] : XForm.Compose(world[skeleton[i].ParentIndex], locals[i]);
        return world;
    }
}
