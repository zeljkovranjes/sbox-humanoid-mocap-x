using System.Numerics;
using System.Text.Json;
using HumanoidMocap.Core;
using HumanoidMocap.Core.Formats.Fbx;
using HumanoidMocap.Core.Mapping;
using HumanoidMocap.Core.Maths;
using HumanoidMocap.Core.Motion;
using HumanoidMocap.Core.Target;

if (args.Length != 3) throw new ArgumentException("Pass capture.hmotion, rig directory and report.json paths.");
var bytes = await File.ReadAllBytesAsync(args[0]);
var motion = MotionDocument.Parse(bytes);
if (motion.Frames.Count is < 150 or > 300) throw new ArgumentException("Motion validation requires 5–10 seconds at 30 fps.");
var targets = new List<object>();
foreach (var citizen in new[] { false, true })
{
    var rig = await File.ReadAllTextAsync(Path.Combine(args[1], citizen ? "target_rig_sbox_citizen.json" : "target_rig_sbox.json"));
    var target = citizen ? RetargetTargetSpec.SboxCitizen(rig) : RetargetTargetSpec.SboxDefault(rig);
    ClipResult Retarget(bool stabilize)
    {
        var result = Retargeter.Convert(new RetargetRequest
        {
            SourceData = bytes,
            SourceFileName = "capture.hmotion",
            FootPlantCleanup = true,
            MocapCorrections = new TargetCorrectionSettings { FirstPerson = false, StabilizeFeet = stabilize }
        }, target);
        if (!result.Success) throw new InvalidDataException(string.Join("; ", result.Errors));
        var clip = result.Clips.Single();
        if (clip.SolvedFrames.Count != motion.Frames.Count) throw new InvalidDataException("Retargeting changed the sample count.");
        return clip;
    }
    var unlocked = Retarget(false); var locked = Retarget(true);
    var frames = locked.SolvedFrames;
    var times = Enumerable.Range(0, frames.Count).Select(i => i / (double)locked.Fps).ToArray();
    var fbx = FbxImporter.Import(ModelDocFbxExport.Write(target.Rig.Skeleton, "motion", frames, times, locked.Fps, target.UpAxis, "WorldRelative"));
    if (fbx.Clips.Count != 1 || fbx.Clips[0].Frames.Count != frames.Count) throw new InvalidDataException("FBX changed the sample count.");
    var fingers = new List<object>();
    foreach (var bone in motion.Bones.Where(b => b.Group == "fingers"))
    {
        if (target.Rig.BoneForRole(bone.Role.Value) is not int index) continue;
        var exported = fbx.Skeleton.IndexOf(target.Rig.Skeleton[index].Name);
        if (exported < 0) throw new InvalidDataException("FBX lost " + bone.Name);
        double range = 0, error = 0;
        for (var frame = 0; frame < frames.Count; frame++)
        {
            range = Math.Max(range, Angle(frames[0][index].Rot, frames[frame][index].Rot));
            error = Math.Max(error, Angle(frames[frame][index].Rot, fbx.Clips[0].Frames[frame][exported].Rot));
        }
        if (error > .1) throw new InvalidDataException("FBX changed finger rotation by more than 0.1 degree: " + bone.Name);
        fingers.Add(new { Bone = bone.Name, MaximumDegreesFromFirstFrame = range, MaximumFbxErrorDegrees = error });
    }
    if (fingers.Count != (citizen ? 24 : 30)) throw new InvalidDataException("Unexpected target finger mapping count.");
    var before = Feet(unlocked.SolvedFrames, target); var after = Feet(frames, target);
    var contacts = new List<object>();
    for (var foot = 0; foot < FootRoles.Length; foot++)
    {
        var role = FootRoles[foot];
        var name = motion.Bones.Single(b => b.Role == role).Name;
        var track = motion.StationaryJoints.Single(t => t.Bone == name).Probability;
        var samples = 0; double unlockedTravel = 0, lockedTravel = 0;
        for (var frame = 1; frame < frames.Count; frame++)
        {
            // Predictions select samples for comparison; they are not measured contact ground truth.
            if (track[frame - 1] < .8 || track[frame] < .8) continue;
            samples++;
            unlockedTravel += Vector3.Distance(before[foot][frame - 1], before[foot][frame]);
            lockedTravel += Vector3.Distance(after[foot][frame - 1], after[foot][frame]);
        }
        var seconds = samples / (double)locked.Fps;
        contacts.Add(new
        {
            Bone = name,
            PredictedContactSteps = samples,
            UnlockedMeanSpeedMetresPerSecond = seconds > 0 ? unlockedTravel / seconds : (double?)null,
            LockedMeanSpeedMetresPerSecond = seconds > 0 ? lockedTravel / seconds : (double?)null
        });
    }
    targets.Add(new { Target = citizen ? "citizen" : "human", Frames = frames.Count, Fingers = fingers, Contacts = contacts, Notes = locked.Mapping.Notes });
}
await File.WriteAllTextAsync(args[2], JsonSerializer.Serialize(new
{
    Source = Path.GetFullPath(args[0]),
    Model = motion.ModelVersion,
    Targets = targets,
    Limitation = "Finger motion/export and predicted-contact diagnostics only; not visual or ground-truth accuracy acceptance."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine("PASS: Human/Citizen finger mapping and every exported finger frame; wrote predicted-contact speed comparison to " + args[2]);

static double Angle(Quaternion a, Quaternion b)
{
    a = Quaternion.Normalize(a); b = Quaternion.Normalize(b);
    var dot = Math.Abs((double)a.X * b.X + (double)a.Y * b.Y + (double)a.Z * b.Z + (double)a.W * b.W);
    // Normalize in double precision so float roundoff is not reported as angular drift.
    var lengthA = Math.Sqrt((double)a.X * a.X + (double)a.Y * a.Y + (double)a.Z * a.Z + (double)a.W * a.W);
    var lengthB = Math.Sqrt((double)b.X * b.X + (double)b.Y * b.Y + (double)b.Z * b.Z + (double)b.W * b.W);
    var angle = 2 * Math.Acos(Math.Clamp(dot / (lengthA * lengthB), 0, 1)) * 180 / Math.PI;
    if (!double.IsFinite(angle)) throw new InvalidDataException("Non-finite finger rotation.");
    return angle;
}

static Vector3[][] Feet(List<XForm[]> frames, RetargetTargetSpec target)
{
    var roles = FootRoles; var indices = roles.Select(role => target.Rig.BoneForRole(role).Value).ToArray();
    var positions = roles.Select(_ => new Vector3[frames.Count]).ToArray();
    var world = new XForm[target.Rig.Skeleton.Count];
    var unitsPerMetre = target.UpAxis == TargetUpAxis.ZUpEngine ? 39.3700787f : 100f;
    for (var frame = 0; frame < frames.Count; frame++)
    {
        for (var bone = 0; bone < world.Length; bone++)
        {
            var parent = target.Rig.Skeleton[bone].ParentIndex;
            world[bone] = parent < 0 ? frames[frame][bone] : XForm.Compose(world[parent], frames[frame][bone]);
            if (!float.IsFinite(world[bone].Pos.Length()) || !float.IsFinite(world[bone].Rot.Length()))
                throw new InvalidDataException("Non-finite target pose.");
        }
        for (var foot = 0; foot < roles.Length; foot++) positions[foot][frame] = world[indices[foot]].Pos / unitsPerMetre;
    }
    return positions;
}

partial class Program
{
    static readonly BoneRole[] FootRoles = [BoneRole.FootL, BoneRole.ToeL, BoneRole.FootR, BoneRole.ToeR];
}
