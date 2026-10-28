using System.Diagnostics;
using System.Text;
using System.Text.Json;
using HumanoidMocapX.Core;
using HumanoidMocapX.Core.Formats.Fbx;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.Core.Target;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;
using Sandbox;
using static Sandbox.Internal.GlobalGameNamespace;

namespace HumanoidMocapX.Checks;

/// <summary>Runs the full model sequence on a 5–10-second predecoded video fixture.
/// This developer check excludes production decoding and visual motion acceptance.</summary>
public static class ClipChecks
{
    public sealed record Fixture(string RgbaPath, int Width, int Height, int Frames, string SourceVideo,
        string FlowPath, string SamPath, string MhrPath, string OmegaPath, string DetectorPath, string SmplPath, string RigDirectory,
        int CameraInterval = 10, FlowModel Model = FlowModel.Latest);

    /// <summary>Writes per-stage evidence, motion documents and Human/Citizen FBX
    /// exports. Requires installed released weights and a fresh writable result directory.</summary>
    public static async Task<object> Run(string fixturePath, string directory, CancellationToken cancellation)
    {
        var fixture = JsonSerializer.Deserialize<Fixture>(await File.ReadAllTextAsync(fixturePath, cancellation))
            ?? throw new ArgumentException("Missing clip fixture.");
        if (fixture.Frames is < 150 or > 300) throw new ArgumentException("Clip validation requires 5–10 seconds at 30 fps.");
        if (Directory.Exists(directory)) throw new ArgumentException("Use a fresh validation directory to avoid overwriting evidence.");
        Directory.CreateDirectory(directory);
        await File.WriteAllTextAsync(Path.Combine(directory, "fixture.json"), JsonSerializer.Serialize(fixture), cancellation);
        var processor = Processor(fixture); using var frames = new RawFrames(fixture);
        await Task.Run(() => FlowCheckpoint.LoadDecoder(fixture.FlowPath, fixture.SmplPath, cancellation), cancellation);
        var timer = Stopwatch.StartNew(); var times = new Dictionary<string, double>();
        var person = await processor.DetectAsync(frames, cancellation); times["Detection"] = timer.Elapsed.TotalSeconds;
        if (person.First != 0 || person.Count != frames.Count) throw new InvalidDataException("The validation fixture must show the person in every frame.");
        var boxes = person.Boxes; await Save("boxes.json", boxes); await Save("timings.json", times); timer.Restart();
        var camera = await processor.CameraAsync(frames, cancellation); times["Camera"] = timer.Elapsed.TotalSeconds;
        await Save("rotations.json", camera.Rotations); await Save("intrinsics.json", camera.Intrinsics); await Save("timings.json", times); timer.Restart();
        var (features, observations) = await processor.FeaturesAsync(frames, boxes, camera.Intrinsics, cancellation); times["Features"] = timer.Elapsed.TotalSeconds;
        await Save("features.json", features); await Save("observations.json", observations); await Save("timings.json", times); timer.Restart();
        var motion = await processor.GenerateAsync(features, camera.Rotations, frames.Count, cancellation); times["Generation"] = timer.Elapsed.TotalSeconds;
        var exports = await Export(fixture, motion, new ObservedDetail(observations, camera.Rotations), directory, cancellation); await Save("timings.json", times);
        var result = new { Passed = true, PredecodedFixture = true, Frames = frames.Count, Seconds = frames.Count / 30.0, StageSeconds = times, Exports = exports };
        await Save("result.json", result);
        return result;

        Task Save<T>(string name, T value) => File.WriteAllTextAsync(Path.Combine(directory, name), JsonSerializer.Serialize(value), cancellation);
    }

    /// <summary>Repeats generation/export from this check's saved features. For
    /// developer diagnosis only: the caller must preserve the same fixture and weights.</summary>
    public static async Task<object> Generate(string directory, CancellationToken cancellation)
    {
        var fixture = JsonSerializer.Deserialize<Fixture>(await File.ReadAllTextAsync(Path.Combine(directory, "fixture.json"), cancellation));
        var features = JsonSerializer.Deserialize<float[]>(await File.ReadAllTextAsync(Path.Combine(directory, "features.json"), cancellation));
        var rotations = JsonSerializer.Deserialize<float[]>(await File.ReadAllTextAsync(Path.Combine(directory, "rotations.json"), cancellation));
        var motion = await Processor(fixture).GenerateAsync(features, rotations, fixture.Frames, cancellation);
        var observed = Path.Combine(directory, "observations.json");
        var detail = File.Exists(observed) ? new ObservedDetail(JsonSerializer.Deserialize<float[]>(await File.ReadAllTextAsync(observed, cancellation)), rotations) : null;
        return await Export(fixture, motion, detail, directory, cancellation);
    }

    static FlowVideoProcessor Processor(Fixture fixture) => new(new(fixture.FlowPath, fixture.SamPath, fixture.MhrPath, fixture.OmegaPath, fixture.DetectorPath, fixture.SmplPath),
        new() { CameraInterval = fixture.CameraInterval, Model = fixture.Model }, message => Log.Info(message));

    static Task<object> Export(Fixture fixture, FlowMotion motion, ObservedDetail detail, string directory, CancellationToken cancellation) => Task.Run<object>(async () =>
    {
        motion.Video = fixture.SourceVideo; motion.Name = Path.GetFileNameWithoutExtension(fixture.SourceVideo); var document = motion.ToDocument(); detail?.Apply(document);
        var bytes = Encoding.UTF8.GetBytes(document.ToJson()); await File.WriteAllBytesAsync(Path.Combine(directory, "capture.hmotion"), bytes, cancellation);
        await File.WriteAllTextAsync(Path.Combine(directory, "flow-motion.json"), JsonSerializer.Serialize(motion), cancellation);
        var results = new List<object>();
        foreach (var citizen in new[] { false, true })
        {
            cancellation.ThrowIfCancellationRequested(); var name = citizen ? "citizen" : "human";
            var rig = await File.ReadAllTextAsync(Path.Combine(fixture.RigDirectory, citizen ? "target_rig_sbox_citizen.json" : "target_rig_sbox.json"), cancellation);
            var target = citizen ? RetargetTargetSpec.SboxCitizen(rig) : RetargetTargetSpec.SboxDefault(rig);
            var result = Retargeter.Convert(new RetargetRequest
            {
                SourceData = bytes,
                SourceFileName = "capture.hmotion",
                FootPlantCleanup = true,
                MocapCorrections = new TargetCorrectionSettings { FirstPerson = false, StabilizeFeet = true }
            }, target);
            if (!result.Success) throw new Exception("Clip retarget failed: " + string.Join("; ", result.Errors.Concat(result.Clips.Select(c => c.Error))));
            var clip = result.Clips.Single();
            foreach (var poses in clip.SolvedFrames) foreach (var pose in poses)
                if (!float.IsFinite(pose.Pos.Length()) || !float.IsFinite(pose.Rot.Length())) throw new Exception("Non-finite target animation.");
            var times = Enumerable.Range(0, clip.SolvedFrames.Count).Select(i => i / (double)clip.Fps).ToArray();
            var exported = ModelDocFbxExport.Write(target.Rig.Skeleton, motion.Name, clip.SolvedFrames, times, clip.Fps, target.UpAxis, "WorldRelative");
            var roundTrip = FbxImporter.Import(exported);
            if (roundTrip.Clips.Count != 1 || roundTrip.Clips[0].Frames.Count != clip.SolvedFrames.Count)
                throw new Exception("Target FBX lost animation frames.");
            var path = Path.Combine(directory, name + ".fbx"); await File.WriteAllBytesAsync(path, exported, cancellation);
            results.Add(new { Target = name, Frames = clip.SolvedFrames.Count, Bones = target.Rig.Skeleton.Count, FbxBytes = exported.Length, Path = path });
        }
        return results;
    }, cancellation);

    sealed class RawFrames : ICaptureFrames, IDisposable
    {
        readonly FileStream stream;
        readonly int bytes;
        public int Width { get; }
        public int Height { get; }
        public int Count { get; }

        internal RawFrames(Fixture fixture)
        {
            Width = fixture.Width; Height = fixture.Height; Count = fixture.Frames;
            if (Width < 1 || Height < 1 || (long)Width * Height > 33_177_600) throw new ArgumentException("Invalid RGBA fixture dimensions.");
            bytes = checked(Width * Height * 4);
            stream = new(fixture.RgbaPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.Asynchronous | FileOptions.RandomAccess);
            if (stream.Length != (long)bytes * Count) { stream.Dispose(); throw new ArgumentException("RGBA fixture length does not match its declared frame count."); }
        }

        public async ValueTask ReadAsync(int frame, Memory<byte> rgba, CancellationToken cancellation)
        {
            if ((uint)frame >= (uint)Count || rgba.Length != bytes) throw new ArgumentException("Invalid fixture frame or destination size.");
            cancellation.ThrowIfCancellationRequested(); stream.Position = (long)frame * bytes;
            await stream.ReadExactlyAsync(rgba, cancellation);
        }

        public void Dispose() => stream.Dispose();
    }
}
