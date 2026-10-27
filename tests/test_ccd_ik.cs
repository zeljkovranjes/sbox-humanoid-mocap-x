using System.Numerics;
using HumanoidMocapX.Core.Cleanup;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Skeleton;

static class CcdIkTests
{
    internal static void Run()
    {
        var skeleton = Skeleton.Create(new[]
        {
            new BoneDefinition("root", null, XForm.Identity),
            new BoneDefinition("elbow", "root", new XForm(Vector3.UnitX, Quaternion.Identity)),
            new BoneDefinition("hand", "elbow", new XForm(Vector3.UnitY, Quaternion.Identity)),
            new BoneDefinition("finger", "hand", new XForm(Vector3.UnitY * .2f, Quaternion.CreateFromAxisAngle(Vector3.UnitX, .4f)))
        });
        var rest = skeleton.Bones.Select(b => b.RestLocal).ToArray();
        var solver = new CcdIk(skeleton, new[] { 0, 1 }, 2);
        var pose = (XForm[])rest.Clone();
        var residual = solver.Solve(pose, new Vector3(.7f, 1.3f, .4f), iterations: 128, tolerance: 1e-5f);
        MotionTests.Check(residual < 1e-4, $"CCD must reach a reachable target; residual {residual}");
        Invariants(rest, pose);
        pose = (XForm[])rest.Clone();
        residual = solver.Solve(pose, new Vector3(20, 0, 0), iterations: 128);
        MotionTests.Near(18, residual, .001);
        Invariants(rest, pose);

        var locked = (XForm[])rest.Clone();
        var limits = new LockedJoint(rest);
        solver.Solve(locked, new Vector3(-1, 1, 0), constrain: limits.Apply);
        MotionTests.Near(0, MathQ.AngleBetween(rest[0].Rot, locked[0].Rot), 1e-6);
        MotionTests.Check(limits.Calls > 0, "CCD must enforce the supplied joint constraint");

        var offsetSolver = new CcdIk(skeleton, new[] { 0 }, 2);
        var offset = new Vector3(.1f, .3f, 0);
        var offsetGoal = Vector3.Transform(new Vector3(1, 1, 0) + offset,
            Quaternion.CreateFromAxisAngle(Vector3.UnitZ, .4f));
        pose = (XForm[])rest.Clone();
        MotionTests.Check(offsetSolver.Solve(pose, offsetGoal, tipOffset: offset) < .001f, "CCD tip offset must reach the target");

        Reject(() => new CcdIk(skeleton, new[] { 1, 0 }, 2));
        Reject(() => new CcdIk(skeleton, new[] { -1 }, 2));
        Reject(() => solver.Solve(rest, new Vector3(float.NaN, 0, 0)));
        Reject(() => solver.Solve(rest, Vector3.Zero, iterations: 0));
        Reject(() => solver.Solve(rest, Vector3.Zero, gain: float.NaN));
        var invalid = (XForm[])rest.Clone(); invalid[1].Rot = default;
        Reject(() => solver.Solve(invalid, Vector3.Zero));

        pose = (XForm[])rest.Clone();
        for (var i = 0; i < 10; i++) solver.Solve(pose, new Vector3(.7f, 1.3f, .4f));
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 100; i++) solver.Solve(pose, new Vector3(.7f, 1.3f, .4f));
        MotionTests.Check(GC.GetAllocatedBytesForCurrentThread() == before, "CCD solve must not allocate");
        Console.WriteLine("PASS: DigitalRune CCD adaptation, reach residuals, constraints, offsets, validation and zero-allocation solve");
    }

    static void Invariants(XForm[] before, XForm[] after)
    {
        for (var i = 0; i < before.Length; i++) MotionTests.Near(0, Vector3.Distance(before[i].Pos, after[i].Pos));
        for (var i = 2; i < before.Length; i++) MotionTests.Near(0, MathQ.AngleBetween(before[i].Rot, after[i].Rot));
    }

    static void Reject(Action action)
    {
        try { action(); } catch (ArgumentException) { return; }
        throw new Exception("CCD accepted an invalid chain or input");
    }

    sealed class LockedJoint(XForm[] reference)
    {
        internal int Calls;
        internal void Apply(XForm[] pose, int bone) { Calls++; if (bone == 0) pose[bone].Rot = reference[bone].Rot; }
    }
}
