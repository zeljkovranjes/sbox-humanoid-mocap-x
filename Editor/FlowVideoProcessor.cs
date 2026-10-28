using System;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX.Inference;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>Connects detection, camera reconstruction, body/hand features and motion
/// sampling. Model stages run sequentially so their GPU weights do not coexist.</summary>
internal sealed class FlowVideoProcessor
{
    readonly CaptureModels models;
    readonly FlowModel model;
    readonly int cameraInterval, requestedLanes;
    readonly Action<string> progress;
    readonly Action<string, double?> reportProgress;

    internal FlowVideoProcessor(CaptureModels models, CaptureOptions options, Action<string> progress,
        Action<string, double?> reportProgress = null)
    {
        ArgumentNullException.ThrowIfNull(models); ArgumentNullException.ThrowIfNull(options); options.Validate();
        this.models = models; model = options.Model; cameraInterval = options.CameraInterval; requestedLanes = options.Lanes; this.progress = progress;
        this.reportProgress = reportProgress;
    }

    /// <summary>Reconstructs one person's motion from a 30 fps timeline, over the frames in
    /// which they are visible; the motion's start time is the first of those, in seconds from
    /// the timeline's start. Requires installed verified models and a live GPU. The caller owns
    /// frames; all failures/cancellation propagate and no partial motion is returned.</summary>
    internal async Task<(FlowMotion Motion, PersonSpan Person)> RunAsync(ICaptureFrames frames, CancellationToken cancellation)
    {
        var person = await DetectAsync(frames, cancellation);
        var visible = person.Visible(frames);
        var camera = await CameraAsync(visible, cancellation);
        var features = await FeaturesAsync(visible, person.Boxes, camera.Intrinsics, cancellation);
        var motion = await GenerateAsync(features, camera.Rotations, visible.Count, cancellation);
        motion.StartTime = person.StartSeconds;
        return (motion, person);
    }

    internal async Task<PersonSpan> DetectAsync(ICaptureFrames frames, CancellationToken cancellation)
    {
        var network = new YoloNetwork(); Report("Loading person detector", 0);
        var weights = await Task.Run(() => YoloCheckpoint.Load(models.Detector, network, cancellation), cancellation);
        await GameTask.MainThread(); cancellation.ThrowIfCancellationRequested();
        // Fast convolutions leave detection waiting on each frame's CPU resize and box decoding,
        // so it uses as many lanes as features do.
        var laneCount = Math.Min(Lanes(), frames.Count);
        var detectors = new YoloGpuDetector[laneCount]; var pixels = new byte[laneCount][];
        var detections = new YoloBox[laneCount][]; var running = new Task<int>[laneCount];
        var track = new PersonTrack(frames.Count);
        try
        {
            for (var lane = 0; lane < laneCount; lane++)
            {
                detectors[lane] = new YoloGpuDetector(network, weights);
                pixels[lane] = Pixels(frames); detections[lane] = new YoloBox[YoloNetwork.Predictions];
            }
            for (var start = 0; start < frames.Count; start += laneCount)
            {
                var count = Math.Min(laneCount, frames.Count - start);
                for (var lane = 0; lane < count; lane++)
                {
                    Report($"Tracking person · {start + lane + 1}/{frames.Count}", 0.10 * start / frames.Count);
                    await frames.ReadAsync(start + lane, pixels[lane], cancellation); await GameTask.MainThread();
                    running[lane] = detectors[lane].DetectAsync(pixels[lane], frames.Width, frames.Height, detections[lane], cancellation);
                }
                // The tracker links people frame to frame, so results are added in frame order.
                for (var lane = 0; lane < count; lane++) track.Add(detections[lane].AsSpan(0, await running[lane]));
                Array.Clear(running);
                reportProgress?.Invoke($"Tracked person · {start + count}/{frames.Count}", 0.10 * (start + count) / frames.Count);
            }
            return track.Complete();
        }
        finally
        {
            foreach (var task in running) if (task is not null) try { await task; } catch { }
            await GameTask.MainThread();
            foreach (var detector in detectors) detector?.Dispose();
        }
    }

    internal async Task<(float[] Rotations, float[] Intrinsics)> CameraAsync(ICaptureFrames frames, CancellationToken cancellation)
    {
        var pixels = Pixels(frames); var count = FlowCamera.SampleCount(frames.Count, cameraInterval);
        var reader = new CameraFrames(frames, pixels, cameraInterval, message => Report(message, 0.10));
        var shape = new OmegaShape(count, reader.Image.Height, reader.Image.Width);
        Report("Loading camera model", 0.10);
        await GameTask.MainThread(); cancellation.ThrowIfCancellationRequested();
        // Release each part's CPU weights after upload so later parts need not keep them live.
        var camera = await OmegaGpuModel.LoadAsync(prefixes => VisionCheckpoint.LoadOmegaPart(models.Omega, cancellation, prefixes), shape, cancellation);
        var poses = new float[count * FlowCamera.PoseChannels];
        try { await camera.RunAsync(reader.ReadAsync, poses, cancellation); }
        finally { await GameTask.MainThread(); camera.Dispose(); }
        var rotations = new float[frames.Count * 9]; var intrinsics = new float[rotations.Length];
        await Task.Run(() => FlowCamera.Decode(poses, frames.Count, cameraInterval, frames.Width, frames.Height, rotations, intrinsics, cancellation), cancellation);
        reportProgress?.Invoke("Camera reconstructed", 0.20);
        return (rotations, intrinsics);
    }

    internal async Task<float[]> FeaturesAsync(ICaptureFrames frames, ReadOnlyMemory<float> boxes, ReadOnlyMemory<float> intrinsics, CancellationToken cancellation)
    {
        if (boxes.Length != frames.Count * 4 || intrinsics.Length != frames.Count * 9)
            throw new ArgumentException("Capture needs one person box and intrinsic matrix per frame.");
        SamDecoder.Finite(boxes.Span); SamDecoder.Finite(intrinsics.Span);
        var features = new float[frames.Count * SamGpuFeatures.Elements];
        var laneCount = Math.Min(Lanes(), frames.Count);
        using var shared = new FlowGpuWeights();
        SamGpuFeatures[] lanes = null; var running = new Task[laneCount];
        try
        {
            lanes = await SamFeatures.LoadAsync(models.Sam, models.Mhr, laneCount, shared, message => Report(message, 0.20), cancellation);
            var pixels = new byte[laneCount][];
            for (var lane = 0; lane < laneCount; lane++) pixels[lane] = Pixels(frames);
            for (var start = 0; start < frames.Count; start += laneCount)
            {
                var count = Math.Min(laneCount, frames.Count - start);
                // Frames are read in order; each lane's GPU work then overlaps the other lane's CPU feedback.
                for (var lane = 0; lane < count; lane++)
                {
                    var i = start + lane;
                    Report($"Reconstructing body and hands · {i + 1}/{frames.Count}", 0.20 + 0.77 * start / frames.Count);
                    await frames.ReadAsync(i, pixels[lane], cancellation); await GameTask.MainThread();
                    running[lane] = lanes[lane].RunAsync(pixels[lane], frames.Width, frames.Height, boxes.Slice(i * 4, 4), intrinsics.Slice(i * 9, 9),
                        features.AsMemory(i * SamGpuFeatures.Elements, SamGpuFeatures.Elements), cancellation);
                }
                await Task.WhenAll(running[..count]);
                Array.Clear(running);
                reportProgress?.Invoke($"Reconstructed body and hands · {start + count}/{frames.Count}", 0.20 + 0.77 * (start + count) / frames.Count);
            }
            return features;
        }
        finally
        {
            // A lane still running after another failed must finish before its GPU buffers are released.
            foreach (var task in running) if (task is not null) try { await task; } catch { }
            await GameTask.MainThread();
            if (lanes is not null) foreach (var lane in lanes) lane.Dispose();
        }
    }

    internal async Task<FlowMotion> GenerateAsync(ReadOnlyMemory<float> features, ReadOnlyMemory<float> rotations, int frames, CancellationToken cancellation)
    {
        if (frames < 1 || frames > CaptureOptions.MaximumFrames || features.Length != frames * SamGpuFeatures.Elements || rotations.Length != frames * 9)
            throw new ArgumentException("Motion sampling requires matching complete body and camera features.");
        SamDecoder.Finite(features.Span); SamDecoder.Finite(rotations.Span); Report("Loading FlowHMR " + model, 0.97);
        var weights = await Task.Run(() => FlowCheckpoint.Load(models.Flow, cancellation), cancellation);
        var decoder = await Task.Run(() => FlowCheckpoint.LoadDecoder(models.Flow, models.Smpl, cancellation), cancellation);
        var noise = new float[frames * FlowSampler.Channels]; var sampled = new float[noise.Length];
        await Task.Run(() => FlowNoise.Fill(noise, 0, cancellation), cancellation);
        await GameTask.MainThread(); cancellation.ThrowIfCancellationRequested(); var sampler = new FlowGpuSampler(weights, frames);
        try { Report("Solving motion · 20 sampling steps", 0.97); await sampler.SampleAsync(features, rotations, noise, sampled, cancellation); }
        finally { await GameTask.MainThread(); sampler.Dispose(); }
        var motion = await Task.Run(() => decoder.Decode(sampled, cancellation), cancellation);
        motion.Model = model; motion.Validate(); return motion;
    }

    void Report(string message, double fraction)
    { progress?.Invoke(message); reportProgress?.Invoke(message, fraction); }

    // Frames processed at once: one frame's CPU work overlaps another's GPU work. Read when
    // each stage starts, so memory freed by the previous stage counts.
    int Lanes() => CaptureLanes.Resolve(requestedLanes, VideoMemoryBudget, Graphics.VideoMemoryUsed);

    /// <summary>Lanes this GPU allows right now, for display in the settings.</summary>
    internal static int LaneCapacity() => CaptureLanes.Capacity(VideoMemoryBudget, Graphics.VideoMemoryUsed);

    // Engine APIs only, so Windows and Linux behave alike: the OS budget when the driver
    // reports one, otherwise the card's total memory.
    static ulong VideoMemoryBudget => Graphics.VideoMemoryBudget is > 0 and var budget ? budget : Sandbox.Engine.SystemInfo.GpuMemory;

    static byte[] Pixels(ICaptureFrames frames)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Width < 1 || frames.Height < 1 || (long)frames.Width * frames.Height > 33_177_600
            || frames.Count < 1 || frames.Count > CaptureOptions.MaximumFrames)
            throw new ArgumentException("Capture frames must have fixed dimensions up to 8K and a length of 1–5000 frames.");
        return new byte[checked(frames.Width * frames.Height * 4)];
    }

    sealed class CameraFrames
    {
        readonly ICaptureFrames frames;
        readonly byte[] pixels;
        readonly int interval;
        readonly Action<string> progress;
        readonly Action prepare;
        Memory<float> output;
        CancellationToken cancellation;
        internal OmegaImage Image { get; }

        internal CameraFrames(ICaptureFrames frames, byte[] pixels, int interval, Action<string> progress)
        {
            this.frames = frames; this.pixels = pixels; this.interval = interval; this.progress = progress;
            Image = new(frames.Width, frames.Height); prepare = Prepare;
        }

        internal async Task ReadAsync(int sample, Memory<float> destination, CancellationToken token)
        {
            progress?.Invoke($"Reconstructing camera · {sample + 1}/{FlowCamera.SampleCount(frames.Count, interval)}");
            await frames.ReadAsync(FlowCamera.SampleFrame(sample, frames.Count, interval), pixels, token);
            output = destination; cancellation = token;
            try { await Task.Run(prepare, token); }
            finally { output = default; cancellation = default; }
        }

        void Prepare() => Image.Prepare(pixels, output.Span, cancellation);
    }
}
