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
        var twoHanded = (XForm[])rest.Clone();
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
        // Stand it on the left wrist with its weight over it, then tip it 15°: a handstand leaning off its hand.
        var balance = World(skeleton, rest); var wrist = balance[handL].Pos;
        var tip = Quaternion.CreateFromAxisAngle(Vector3.UnitX, 15 * MathF.PI / 180) * MathQ.FromTo(Centre(rig, balance) - wrist, up);
        foreach (var r in roots) rest[r] = new XForm(wrist + Vector3.Transform(rest[r].Pos - wrist, tip), Quaternion.Normalize(tip * rest[r].Rot));
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
        // On one hand, the body's weight is over it.
        var centre = Centre(rig, last);
        MotionTests.Check(Vector3.Distance(Flat(centre), Flat(last[handL].Pos)) * Inch < 3, $"Centre of mass {Vector3.Distance(Flat(centre), Flat(last[handL].Pos)) * Inch:F1} cm off the support hand");
        MotionTests.Check(Vector3.Distance(Flat(Centre(rig, World(skeleton, rest))), Flat(world[handL].Pos)) * Inch > 5, "Fixture already balanced; the test proves nothing");
        // The body is carried and balanced over the support hand; the raised hand moves with it but stays raised.
        MotionTests.Check(last[handR].Pos.Z * Inch > CaptureHandPlant.LiftCm / 1.5f, $"The raised hand was pulled to {last[handR].Pos.Z * Inch:F0} cm");

        // The video puts the right hand on the floor too. Out of the arm's reach (here 1.8 m up), that is a mismatch: the
        // hand is left alone and the body stays on the left hand.
        var everyFrame = new Dictionary<BoneRole, bool[]> { [BoneRole.HandR] = Enumerable.Repeat(true, 30).ToArray() };
        var far = Enumerable.Range(0, 30).Select(_ => (XForm[])rest.Clone()).ToList();
        CaptureHandPlant.Apply(far, rig, TargetUpAxis.ZUpEngine, 30, everyFrame);
        var farWorld = World(skeleton, far[^1]);
        MotionTests.Near(0, handParts.Min(b => farWorld[b].Pos.Z) * Inch, 1);
        MotionTests.Check(farWorld[handR].Pos.Z * Inch > 50, "An unreachable observed hand must not be dragged down");

        // A two-handed handstand whose right hand hovers within reach: laid flat on the floor beside the left.
        var hover = Hover(rig, twoHanded, handL, handR, Bone(BoneRole.LowerArmR));
        var near = Enumerable.Range(0, 30).Select(_ => (XForm[])hover.Clone()).ToList();
        var hoverWorld = World(skeleton, hover);
        var partsR = Enumerable.Range(0, skeleton.Count).Where(b => b == handR || IsWithin(skeleton, b, handR) && rig.RoleOf(b)?.ToString() is { } r
            && new[] { "Thumb", "Index", "Middle", "Ring", "Pinky" }.Any(r.StartsWith)).ToArray();
        var hovering = partsR.Min(b => hoverWorld[b].Pos.Z) * Inch;
        MotionTests.Check(hovering > CaptureHandPlant.NearFloorCm, $"Fixture right hand hovers only {hovering:F1} cm");
        CaptureHandPlant.Apply(near, rig, TargetUpAxis.ZUpEngine, 30, everyFrame);
        var down = World(skeleton, near[^1]);
        MotionTests.Near(0, partsR.Min(b => down[b].Pos.Z) * Inch, 1);
        MotionTests.Near(0, handParts.Min(b => down[b].Pos.Z) * Inch, 1);
        var chainR = new[] { "Prox", "Mid", "Dist" }.Select(k => Bone(Enum.Parse<BoneRole>("Middle" + k + "R"))).ToArray();
        MotionTests.Check(Vector3.Dot(Vector3.Normalize(down[chainR[1]].Pos - down[chainR[0]].Pos), Vector3.Normalize(down[chainR[2]].Pos - down[chainR[1]].Pos)) > MathF.Cos(3 * MathF.PI / 180), "The observed hand must lie flat");
        Console.WriteLine("PASS: hand plant holds a hand carrying the body however far the capture slides, balances the body over it, lays palm and fingers flat, leaves a raised hand free, lays down a hovering hand the video shows on the floor and ignores one out of reach");
    }

    static Vector3 Flat(Vector3 v) => v with { Z = 0 };

    /// <summary>The two-handed pose on the floor with the right forearm turned so its hand hovers 10–20 cm up.</summary>
    static XForm[] Hover(TargetRig rig, XForm[] pose, int handL, int handR, int forearm)
    {
        var skeleton = rig.Skeleton; var roots = Enumerable.Range(0, skeleton.Count).Where(b => skeleton[b].ParentIndex < 0).ToArray();
        var parts = Enumerable.Range(0, skeleton.Count).Where(b => b == handL || IsWithin(skeleton, b, handL) && rig.RoleOf(b)?.ToString() is { } r
            && new[] { "Thumb", "Index", "Middle", "Ring", "Pinky" }.Any(r.StartsWith)).ToArray();
        foreach (var axis in new[] { Vector3.UnitX, -Vector3.UnitX, Vector3.UnitY, -Vector3.UnitY })
            for (var angle = .1f; angle < .6f; angle += .05f)
            {
                var turned = (XForm[])pose.Clone(); var before = World(skeleton, turned);
                var rotation = Quaternion.CreateFromAxisAngle(axis, angle) * before[forearm].Rot;
                turned[forearm].Rot = Quaternion.Normalize(Quaternion.Inverse(before[skeleton[forearm].ParentIndex].Rot) * rotation);
                var world = World(skeleton, turned); var floor = parts.Min(b => world[b].Pos.Z);
                foreach (var r in roots) turned[r].Pos -= Vector3.UnitZ * (floor - .2f);
                world = World(skeleton, turned);
                if ((world[handR].Pos.Z - world[handL].Pos.Z) * 2.54f is > 12 and < 22) return turned;
            }
        throw new Exception("No forearm turn lifts the right hand 12–22 cm");
    }

    /// <summary>Centre of mass from de Leva's segment fractions, as the hand plant estimates it.</summary>
    static Vector3 Centre(TargetRig rig, XForm[] world)
    {
        (BoneRole, BoneRole, float)[] segments =
        {
            (BoneRole.Hips, BoneRole.Neck, .43f), (BoneRole.Head, BoneRole.Head, .08f), (BoneRole.UpperArmL, BoneRole.LowerArmL, .027f), (BoneRole.UpperArmR, BoneRole.LowerArmR, .027f),
            (BoneRole.LowerArmL, BoneRole.HandL, .022f), (BoneRole.LowerArmR, BoneRole.HandR, .022f), (BoneRole.UpperLegL, BoneRole.LowerLegL, .14f), (BoneRole.UpperLegR, BoneRole.LowerLegR, .14f),
            (BoneRole.LowerLegL, BoneRole.FootL, .06f), (BoneRole.LowerLegR, BoneRole.FootR, .06f)
        };
        var sum = Vector3.Zero; var mass = 0f;
        foreach (var (a, b, m) in segments) { sum += (world[rig.BoneForRole(a)!.Value].Pos + world[rig.BoneForRole(b)!.Value].Pos) * .5f * m; mass += m; }
        return sum / mass;
    }
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
