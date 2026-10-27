using System.Numerics;
using HumanoidMocapX.Core.Cleanup;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Core.Skeleton;
using HumanoidMocapX.Core.Target;

static class BodyClearanceTests
{
    internal static void Run()
    {
        foreach (var name in new[] { "target_rig_sbox.json", "target_rig_sbox_citizen.json" })
        {
            var rig = TargetRig.Load(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, name)));
            HipContact(rig, true);
            HipContact(rig, false);
            TemporalContact(rig, true);
            TemporalContact(rig, false);
            foreach (var scale in new[] { .01f, 10f })
            {
                var scaled = Scale(rig, scale);
                HipContact(scaled, true);
                HipContact(scaled, false);
            }
        }
        Unreachable();
        Console.WriteLine("PASS: Human/Citizen hip clearance, contact entry/exit, scales, residuals, arm lengths, wrist attitude and finger locals");
    }

    static void HipContact(TargetRig target, bool left)
    {
        var rig = target.Skeleton;
        var frame = rig.Bones.Select(b => b.RestLocal).ToArray();
        var hips = target.BoneForRole(BoneRole.Hips)!.Value;
        var neck = target.BoneForRole(BoneRole.Neck)!.Value;
        var upper = target.BoneForRole(left ? BoneRole.UpperArmL : BoneRole.UpperArmR)!.Value;
        var lower = target.BoneForRole(left ? BoneRole.LowerArmL : BoneRole.LowerArmR)!.Value;
        var hand = target.BoneForRole(left ? BoneRole.HandL : BoneRole.HandR)!.Value;
        var leftShoulder = target.BoneForRole(BoneRole.UpperArmL)!.Value;
        var rightShoulder = target.BoneForRole(BoneRole.UpperArmR)!.Value;
        var world = new Pose(frame).ToWorld(rig);
        var up = Vector3.Normalize(world[neck].Pos - world[hips].Pos);
        var height = Vector3.Distance(world[neck].Pos, world[hips].Pos);
        var shoulderLine = world[leftShoulder].Pos - world[rightShoulder].Pos;
        var halfSpan = shoulderLine.Length() * .5f;
        var across = Vector3.Normalize(shoulderLine - up * Vector3.Dot(shoulderLine, up));
        var forward = Vector3.Cross(across, up);
        var side = left ? 1 : -1;
        var reach = Vector3.Distance(world[upper].Pos, world[lower].Pos) + Vector3.Distance(world[lower].Pos, world[hand].Pos);
        var contactHeight = Math.Max(height * .02f, Vector3.Dot(world[upper].Pos - world[hips].Pos, up) - reach * .9f);
        var goal = world[hips].Pos + up * contactHeight + across * (side * halfSpan * .6f);
        MotionTests.Check(Vector3.Distance(world[upper].Pos, goal) < reach, target.Name + " hip fixture must be reachable");
        var frames = new List<XForm[]> { frame };
        EffectorIk.ApplyGoals(frames, rig, new LimbChain { Upper = upper, Lower = lower, End = hand }, new[] { goal },
            Vector3.Cross(goal - world[upper].Pos, across * side), soften: 0);
        var before = (XForm[])frame.Clone();
        var beforeWorld = new Pose(before).ToWorld(rig);
        var mapping = new MappingResult("body-clearance-fixture", default);
        foreach (var role in Enum.GetValues<BoneRole>())
            if (target.BoneForRole(role) is int bone) mapping.RoleToBone[role] = bone;
        var result = CaptureArmClearance.Apply(frames, new[] { before }, rig, mapping, target, 30);
        var after = new Pose(frame).ToWorld(rig);
        ClearSegments(target, after, left);
        var relative = after[hand].Pos - after[hips].Pos;
        var x = Vector3.Dot(relative, across) / (halfSpan * (CaptureArmClearance.HalfWidth + CaptureArmClearance.ArmRadius));
        var z = Vector3.Dot(relative, forward) / (halfSpan * (CaptureArmClearance.HalfDepth + CaptureArmClearance.ArmRadius));
        var clearance = MathF.Sqrt(x * x + z * z);
        MotionTests.Check(clearance >= .999f, $"{target.Name} {(left ? "left" : "right")} hip hand remains inside torso: {clearance:F6}; moved samples {result.Samples}");
        MotionTests.Near(1, MathF.Abs(Quaternion.Dot(beforeWorld[hand].Rot, after[hand].Rot)), 1e-5);
        Invariants(target, before, frame);
    }

    static TargetRig Scale(TargetRig target, float scale)
    {
        var definitions = target.Skeleton.Bones.Select(b => new BoneDefinition(b.Name,
            b.ParentIndex < 0 ? null : target.Skeleton[b.ParentIndex].Name, new XForm(b.RestLocal.Pos * scale, b.RestLocal.Rot))).ToArray();
        var mapping = new MappingResult(target.Name + "-scaled", default);
        foreach (var role in Enum.GetValues<BoneRole>())
            if (target.BoneForRole(role) is int bone) mapping.RoleToBone[role] = bone;
        return TargetRig.FromSkeleton(Skeleton.Create(definitions), mapping);
    }

    static void Unreachable()
    {
        var definitions = new[]
        {
            new BoneDefinition("Hips", null, XForm.Identity),
            new BoneDefinition("Neck", "Hips", new XForm(Vector3.UnitY, Quaternion.Identity)),
            new BoneDefinition("UpperArmL", "Hips", new XForm(new(.5f,.5f,0), Quaternion.Identity)),
            new BoneDefinition("LowerArmL", "UpperArmL", new XForm(new(0,-.01f,0), Quaternion.Identity)),
            new BoneDefinition("HandL", "LowerArmL", new XForm(new(0,-.01f,0), Quaternion.Identity)),
            new BoneDefinition("UpperArmR", "Hips", new XForm(new(-.5f,.5f,0), Quaternion.Identity)),
            new BoneDefinition("LowerArmR", "UpperArmR", new XForm(new(0,-.01f,0), Quaternion.Identity)),
            new BoneDefinition("HandR", "LowerArmR", new XForm(new(0,-.01f,0), Quaternion.Identity))
        };
        var rig = Skeleton.Create(definitions);
        var mapping = new MappingResult("unreachable-short-arms", default);
        foreach (var bone in rig.Bones) mapping.RoleToBone[Enum.Parse<BoneRole>(bone.Name)] = bone.Index;
        var target = TargetRig.FromSkeleton(rig, mapping);
        var original = rig.Bones.Select(b => b.RestLocal).ToArray();
        var frames = new List<XForm[]> { (XForm[])original.Clone() };
        CaptureArmClearance.Apply(frames, new[] { original }, rig, mapping, target, 30, out var unresolved, out var penetration);
        MotionTests.Check(unresolved == 2 && float.IsFinite(penetration) && penetration > 0,
            "Impossible contacts must report a finite residual, not stretch the arms or claim success");
        Invariants(target, original, frames[0]);
    }

    static void Invariants(TargetRig target, XForm[] before, XForm[] after)
    {
        var mutable = new HashSet<int>();
        foreach (var role in new[] { BoneRole.UpperArmL, BoneRole.LowerArmL, BoneRole.HandL, BoneRole.UpperArmR, BoneRole.LowerArmR, BoneRole.HandR })
            mutable.Add(target.BoneForRole(role)!.Value);
        var beforeWorld = new Pose(before).ToWorld(target.Skeleton);
        var afterWorld = new Pose(after).ToWorld(target.Skeleton);
        foreach (var role in new[] { BoneRole.HandL, BoneRole.HandR })
        {
            var hand = target.BoneForRole(role)!.Value;
            MotionTests.Near(1, MathF.Abs(Quaternion.Dot(beforeWorld[hand].Rot, afterWorld[hand].Rot)), 1e-5);
        }
        for (var bone = 0; bone < after.Length; bone++)
        {
            MotionTests.Near(0, Vector3.Distance(before[bone].Pos, after[bone].Pos), 1e-5);
            if (!mutable.Contains(bone))
                MotionTests.Check(MathF.Abs(Quaternion.Dot(before[bone].Rot, after[bone].Rot)) > .99999f,
                    $"{target.Name} changed non-arm bone {target.Skeleton[bone].Name}");
        }
    }

    static void TemporalContact(TargetRig target, bool left)
    {
        const int count = 91;
        var rig = target.Skeleton;
        var rest = rig.Bones.Select(b => b.RestLocal).ToArray();
        var world = new Pose(rest).ToWorld(rig);
        var hips = target.BoneForRole(BoneRole.Hips)!.Value;
        var neck = target.BoneForRole(BoneRole.Neck)!.Value;
        var upper = target.BoneForRole(left ? BoneRole.UpperArmL : BoneRole.UpperArmR)!.Value;
        var lower = target.BoneForRole(left ? BoneRole.LowerArmL : BoneRole.LowerArmR)!.Value;
        var hand = target.BoneForRole(left ? BoneRole.HandL : BoneRole.HandR)!.Value;
        var up = Vector3.Normalize(world[neck].Pos - world[hips].Pos);
        var shoulders = world[target.BoneForRole(BoneRole.UpperArmL)!.Value].Pos - world[target.BoneForRole(BoneRole.UpperArmR)!.Value].Pos;
        var span = shoulders.Length() * .5f;
        var across = Vector3.Normalize(shoulders - up * Vector3.Dot(shoulders, up));
        var forward = Vector3.Cross(across, up);
        var reach = Vector3.Distance(world[upper].Pos, world[lower].Pos) + Vector3.Distance(world[lower].Pos, world[hand].Pos);
        var height = Math.Max(Vector3.Distance(world[neck].Pos, world[hips].Pos) * .05f,
            Vector3.Dot(world[upper].Pos - world[hips].Pos, up) - reach * .85f);
        var frames = new List<XForm[]>(); var goals = new Vector3[count];
        for (var i = 0; i < count; i++)
        {
            var phase = MathF.PI * i / (count - 1);
            var contact = MathF.Sin(phase) * MathF.Sin(phase);
            var radial = 2.5f - 1.8f * contact;
            goals[i] = world[hips].Pos + up * (height + (1 - contact) * reach * .45f)
                + across * ((left ? 1 : -1) * span * radial);
            MotionTests.Check(Vector3.Distance(world[upper].Pos, goals[i]) < reach, "Contact sweep target must be reachable");
            frames.Add((XForm[])rest.Clone());
        }
        EffectorIk.ApplyGoals(frames, rig, new LimbChain { Upper = upper, Lower = lower, End = hand }, goals,
            Vector3.Cross(goals[count / 2] - world[upper].Pos, across * (left ? 1 : -1)), soften: 0);
        if (target.BoneForRole(left ? BoneRole.IndexProxL : BoneRole.IndexProxR) is int finger)
            for (var i = 0; i < count; i++)
                frames[i][finger].Rot = Quaternion.Normalize(Quaternion.CreateFromAxisAngle(Vector3.UnitZ, .2f * MathF.Sin(MathF.PI * i / (count - 1))) * frames[i][finger].Rot);
        var before = frames.Select(f => (XForm[])f.Clone()).ToArray();
        var mapping = new MappingResult("temporal-contact-fixture", default);
        CaptureArmClearance.Apply(frames, before, rig, mapping, target, 30);
        var maxStep = 0f; var maxTurn = 0f; var last = Vector3.Zero;
        for (var i = 0; i < count; i++)
        {
            Invariants(target, before[i], frames[i]);
            var posed = new Pose(frames[i]).ToWorld(rig);
            ClearSegments(target, posed, left);
            var position = posed[hand].Pos;
            var local = position - posed[hips].Pos;
            var x = Vector3.Dot(local, across) / (span * (CaptureArmClearance.HalfWidth + CaptureArmClearance.ArmRadius));
            var z = Vector3.Dot(local, forward) / (span * (CaptureArmClearance.HalfDepth + CaptureArmClearance.ArmRadius));
            MotionTests.Check(x * x + z * z >= .998f, $"{target.Name} contact frame {i} is inside the hip proxy");
            if (i > 0)
            {
                if (Vector3.Distance(last, position) / span > .15f)
                {
                    var input = new Pose(before[i]).ToWorld(rig);
                    var previousInput = new Pose(before[i - 1]).ToWorld(rig);
                    Console.WriteLine($"JUMP {target.Name} f={i} inputStep={Vector3.Distance(input[hand].Pos, previousInput[hand].Pos) / span} before={previousInput[hand].Pos}->{input[hand].Pos} after={last}->{position}");
                }
                maxStep = Math.Max(maxStep, Vector3.Distance(last, position) / span);
                maxTurn = Math.Max(maxTurn, MathQ.AngleBetween(frames[i - 1][upper].Rot, frames[i][upper].Rot));
                maxTurn = Math.Max(maxTurn, MathQ.AngleBetween(frames[i - 1][lower].Rot, frames[i][lower].Rot));
            }
            last = position;
        }
        Console.WriteLine($"{target.Name} {(left ? "left" : "right")} contact sweep: max wrist step {maxStep:F5} shoulder half-spans; max joint step {maxTurn * 180 / MathF.PI:F3} degrees");
        MotionTests.Check(maxStep < .15f, "Body contact introduced a wrist jump");
        MotionTests.Check(maxTurn < .25f, "Body contact introduced an elbow/shoulder snap");
        foreach (var i in new[] { 0, count - 1 })
            foreach (var bone in new[] { upper, lower, hand })
                MotionTests.Near(0, MathQ.AngleBetween(before[i][bone].Rot, frames[i][bone].Rot), 1e-4);
    }

    static void ClearSegments(TargetRig target, XForm[] world, bool left)
    {
        var rig = target.Skeleton;
        var hips = target.BoneForRole(BoneRole.Hips)!.Value;
        var neck = target.BoneForRole(BoneRole.Neck)!.Value;
        var lower = target.BoneForRole(left ? BoneRole.LowerArmL : BoneRole.LowerArmR)!.Value;
        var hand = target.BoneForRole(left ? BoneRole.HandL : BoneRole.HandR)!.Value;
        var axis = world[neck].Pos - world[hips].Pos;
        var height = axis.Length() * .9f; var up = Vector3.Normalize(axis);
        var shoulders = world[target.BoneForRole(BoneRole.UpperArmL)!.Value].Pos - world[target.BoneForRole(BoneRole.UpperArmR)!.Value].Pos;
        var span = shoulders.Length() * .5f;
        var across = Vector3.Normalize(shoulders - up * Vector3.Dot(shoulders, up));
        var forward = Vector3.Cross(across, up);
        var width = span * (CaptureArmClearance.HalfWidth + CaptureArmClearance.ArmRadius);
        var depth = span * (CaptureArmClearance.HalfDepth + CaptureArmClearance.ArmRadius);
        var cap = span * CaptureArmClearance.ArmRadius;
        for (var bone = hand; bone < rig.Count; bone++)
        {
            if (bone != hand && (target.RoleOf(bone) is not { } role
                || (int)role < (int)BoneRole.ThumbMetaL || (int)role > (int)BoneRole.PinkyDistR)) continue;
            var parent = bone == hand ? lower : rig[bone].ParentIndex;
            var ancestor = parent;
            while (ancestor >= 0 && ancestor != hand) ancestor = rig[ancestor].ParentIndex;
            if (bone != hand && ancestor != hand) continue;
            for (var sample = 0; sample <= 32; sample++)
            {
                var point = Vector3.Lerp(world[parent].Pos, world[bone].Pos, sample / 32f) - world[hips].Pos;
                var x = Vector3.Dot(point, across) / width;
                var z = Vector3.Dot(point, forward) / depth;
                var y = Vector3.Dot(point, up); y = (y - Math.Clamp(y, 0, height)) / cap;
                MotionTests.Check(x * x + y * y + z * z >= .998f, $"{target.Name} {rig[bone].Name} segment penetrates the torso proxy");
            }
        }
    }
}
