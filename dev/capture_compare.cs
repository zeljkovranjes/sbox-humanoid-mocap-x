using System.Numerics;
using System.Text.Json;
using HumanoidMocapX.Core.Formats.Fbx;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Motion;

// Numeric acceptance for a non-byte-identical optimization: compares two ClipChecks result
// directories. Exits non-zero when any limit is exceeded or a contact decision changes.
// The optional fourth argument lists stage outputs the experiment must leave exactly unchanged.
if (args.Length is not (3 or 4)) throw new ArgumentException("Pass before and after result directories, report.json and optional unchanged files (comma-separated).");
string before = args[0], after = args[1];
var unchanged = args.Length == 4 ? args[3].Split(',', StringSplitOptions.RemoveEmptyEntries) : [];
var failures = new List<string>();
var report = new Dictionary<string, object>();
foreach (var name in new[] { "boxes.json", "intrinsics.json", "rotations.json", "features.json" })
{
    var (difference, unequal) = Difference(Numbers(Path.Combine(before, name)), Numbers(Path.Combine(after, name)));
    if (unchanged.Contains(name) && unequal > 0) failures.Add($"{name}: {unequal} values changed but must be unchanged");
    report[name] = difference;
}

using (var a = JsonDocument.Parse(await File.ReadAllBytesAsync(Path.Combine(before, "flow-motion.json"))))
using (var b = JsonDocument.Parse(await File.ReadAllBytesAsync(Path.Combine(after, "flow-motion.json"))))
{
    double angle = 0;
    var posesA = Rows(a.RootElement.GetProperty("Poses")); var posesB = Rows(b.RootElement.GetProperty("Poses"));
    SameShape(posesA, posesB, "flow-motion poses");
    for (var frame = 0; frame < posesA.Length; frame++)
        for (var joint = 0; joint < posesA[frame].Length; joint += 3)
            angle = Math.Max(angle, Degrees(AxisAngle(posesA[frame], joint), AxisAngle(posesB[frame], joint)));
    var contacts = Flips(Rows(a.RootElement.GetProperty("Contacts")), Rows(b.RootElement.GetProperty("Contacts")));
    Check(angle < MaximumDegrees, "flow-motion pose angle", angle);
    if (contacts > 0) failures.Add($"flow-motion contacts: {contacts} decisions changed");
    report["flow-motion.json"] = new
    {
        PoseMaximumDegrees = angle, ContactDecisionsChanged = contacts,
        Translations = Difference(Flatten(Rows(a.RootElement.GetProperty("Translations"))), Flatten(Rows(b.RootElement.GetProperty("Translations")))).Report
    };
}

var motionA = MotionDocument.Parse(await File.ReadAllBytesAsync(Path.Combine(before, "capture.hmotion")));
var motionB = MotionDocument.Parse(await File.ReadAllBytesAsync(Path.Combine(after, "capture.hmotion")));
if (motionA.Frames.Count != motionB.Frames.Count || !motionA.Bones.Select(b => b.Name).SequenceEqual(motionB.Bones.Select(b => b.Name))
    || !motionA.StationaryJoints.Select(t => t.Bone).Order().SequenceEqual(motionB.StationaryJoints.Select(t => t.Bone).Order()))
    throw new InvalidDataException("capture.hmotion frames, bones or foot-lock tracks differ in count or identity.");
double motionAngle = 0, motionMetres = 0;
for (var frame = 0; frame < motionA.Frames.Count; frame++)
    for (var bone = 0; bone < motionA.Bones.Count; bone++)
    {
        float[] ra = motionA.Frames[frame].Rotations[bone], rb = motionB.Frames[frame].Rotations[bone];
        motionAngle = Math.Max(motionAngle, Degrees(new Quaternion(ra[0], ra[1], ra[2], ra[3]), new Quaternion(rb[0], rb[1], rb[2], rb[3])));
        float[] pa = motionA.Frames[frame].Positions[bone], pb = motionB.Frames[frame].Positions[bone];
        motionMetres = Math.Max(motionMetres, Vector3.Distance(new Vector3(pa), new Vector3(pb)));
    }
var stationary = motionA.StationaryJoints.Sum(track =>
    Flips([track.Probability], [motionB.StationaryJoints.Single(t => t.Bone == track.Bone).Probability]));
Check(motionAngle < MaximumDegrees, "capture.hmotion rotation", motionAngle);
Check(motionMetres < MaximumMetres, "capture.hmotion position", motionMetres);
if (stationary > 0) failures.Add($"capture.hmotion foot locks: {stationary} decisions changed");
report["capture.hmotion"] = new { RotationMaximumDegrees = motionAngle, PositionMaximumMetres = motionMetres, FootLockDecisionsChanged = stationary };

foreach (var name in new[] { "human.fbx", "citizen.fbx" })
{
    var sceneA = FbxImporter.Import(await File.ReadAllBytesAsync(Path.Combine(before, name)));
    var sceneB = FbxImporter.Import(await File.ReadAllBytesAsync(Path.Combine(after, name)));
    var framesA = sceneA.Clips.Single().Frames; var framesB = sceneB.Clips.Single().Frames;
    if (framesA.Count != framesB.Count || sceneA.UnitScaleCm != sceneB.UnitScaleCm
        || !Enumerable.Range(0, sceneA.Skeleton.Count).Select(i => sceneA.Skeleton[i].Name).SequenceEqual(Enumerable.Range(0, sceneB.Skeleton.Count).Select(i => sceneB.Skeleton[i].Name)))
        throw new InvalidDataException(name + " changed its frames, bones or units.");
    double angle = 0, metres = 0;
    for (var frame = 0; frame < framesA.Count; frame++)
    {
        XForm[] worldA = World(sceneA.Skeleton, framesA[frame]), worldB = World(sceneB.Skeleton, framesB[frame]);
        for (var bone = 0; bone < worldA.Length; bone++)
        {
            angle = Math.Max(angle, Degrees(framesA[frame][bone].Rot, framesB[frame][bone].Rot));
            // FbxImporter's unit policy scales every translation to centimetres.
            metres = Math.Max(metres, Vector3.Distance(worldA[bone].Pos, worldB[bone].Pos) / CentimetresPerMetre);
        }
    }
    Check(angle < MaximumDegrees, name + " local rotation", angle);
    Check(metres < MaximumMetres, name + " world position", metres);
    report[name] = new { LocalRotationMaximumDegrees = angle, WorldPositionMaximumMetres = metres };
}

report["timings"] = new { Before = Numbers(Path.Combine(before, "timings.json")), After = Numbers(Path.Combine(after, "timings.json")) };
report["Failures"] = failures;
await File.WriteAllTextAsync(args[2], JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine(failures.Count == 0 ? "PASS: within 0.01 degree, 0.1 mm and identical contact decisions; wrote " + args[2]
    : "FAIL: " + string.Join("; ", failures));
return failures.Count == 0 ? 0 : 1;

void Check(bool passed, string what, double value) { if (!passed) failures.Add($"{what}: {value:G6}"); }

static double[] Numbers(string path)
{
    using var document = JsonDocument.Parse(File.ReadAllBytes(path));
    var values = new List<double>();
    void Walk(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Number) values.Add(element.GetDouble());
        else if (element.ValueKind == JsonValueKind.Array) foreach (var item in element.EnumerateArray()) Walk(item);
        else if (element.ValueKind == JsonValueKind.Object) foreach (var item in element.EnumerateObject()) Walk(item.Value);
    }
    Walk(document.RootElement);
    return values.ToArray();
}

static (object Report, int Unequal) Difference(double[] a, double[] b)
{
    if (a.Length != b.Length) throw new InvalidDataException($"Value counts differ: {a.Length} and {b.Length}.");
    double maximum = 0, squares = 0; var unequal = 0;
    for (var i = 0; i < a.Length; i++)
    {
        var difference = Math.Abs(a[i] - b[i]);
        if (difference > 0) unequal++;
        maximum = Math.Max(maximum, difference); squares += difference * difference;
    }
    return (new { Count = a.Length, Unequal = unequal, MaximumAbsolute = maximum, Rms = a.Length == 0 ? 0 : Math.Sqrt(squares / a.Length) }, unequal);
}

static float[][] Rows(JsonElement element) => element.EnumerateArray().Select(row => row.EnumerateArray().Select(v => v.GetSingle()).ToArray()).ToArray();

static double[] Flatten(float[][] rows) => rows.SelectMany(row => row.Select(v => (double)v)).ToArray();

// Counts samples whose contact decision (probability >= 0.5, as foot locking uses) differs.
static int Flips(float[][] a, float[][] b)
{
    SameShape(a, b, "contact tracks");
    return a.Zip(b).Sum(rows => rows.First.Zip(rows.Second).Count(p => p.First >= .5f != p.Second >= .5f));
}

static void SameShape(float[][] a, float[][] b, string what)
{
    if (a.Length != b.Length || a.Zip(b).Any(rows => rows.First.Length != rows.Second.Length))
        throw new InvalidDataException(what + " differ in sample count.");
}

static Quaternion AxisAngle(float[] values, int offset)
{
    var axis = new Vector3(values[offset], values[offset + 1], values[offset + 2]);
    var angle = axis.Length();
    return angle < 1e-12f ? Quaternion.Identity : Quaternion.CreateFromAxisAngle(axis / angle, angle);
}

static double Degrees(Quaternion a, Quaternion b)
{
    var dot = Math.Abs((double)a.X * b.X + (double)a.Y * b.Y + (double)a.Z * b.Z + (double)a.W * b.W);
    var lengths = Math.Sqrt(((double)a.X * a.X + (double)a.Y * a.Y + (double)a.Z * a.Z + (double)a.W * a.W)
        * ((double)b.X * b.X + (double)b.Y * b.Y + (double)b.Z * b.Z + (double)b.W * b.W));
    var angle = 2 * Math.Acos(Math.Clamp(dot / lengths, 0, 1)) * 180 / Math.PI;
    if (!double.IsFinite(angle)) throw new InvalidDataException("Non-finite rotation.");
    return angle;
}

static XForm[] World(HumanoidMocapX.Core.Skeleton.Skeleton skeleton, XForm[] local)
{
    var world = new XForm[local.Length];
    for (var bone = 0; bone < local.Length; bone++)
    {
        var parent = skeleton[bone].ParentIndex;
        world[bone] = parent < 0 ? local[bone] : XForm.Compose(world[parent], local[bone]);
    }
    return world;
}

partial class Program
{
    const double MaximumDegrees = 0.01, MaximumMetres = 0.0001, CentimetresPerMetre = 100;
}
