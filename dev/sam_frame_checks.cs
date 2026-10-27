using System.Diagnostics;
using System.Text.Json;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;
using Sandbox;

namespace HumanoidMocapX.Checks;

/// <summary>Runs released YOLOX, SAM DINOv3, body decoder and MHR on a real frame.
/// This is an isolated body-pass diagnostic, not full video or hand refinement.</summary>
public static class SamFrameChecks
{
    /// <summary>Writes crop, keypoint and feature evidence to a JSON file. Requires
    /// a visible person in an RGBA8 frame and installed, verified model files.</summary>
    public static async Task<object> Run(string samPath, string mhrPath, string detectorPath, string rgbaPath,
        int width, int height, string resultPath, CancellationToken cancellation)
    {
        var pixels = await File.ReadAllBytesAsync(rgbaPath, cancellation);
        var box = await Detect(pixels, width, height, detectorPath, cancellation);
        var shape = DinoShape.Sam(); var crop = new SamCrop(box.Left, box.Top, box.Right, box.Bottom, shape);
        var focal = (float)Math.Sqrt((double)width * width + (double)height * height);
        var intrinsics = new float[] { focal, 0, width / 2f, 0, focal, height / 2f, 0, 0, 1 }; var affine = new float[6]; crop.CopyAffine(affine);
        var rgb = new float[shape.ImageElements]; var rayEncoder = new SamRays(shape); var rays = new float[rayEncoder.Elements];
        await Task.Run(() => { crop.Prepare(pixels, width, height, rgb, cancellation); rayEncoder.Prepare(intrinsics, affine, rays, cancellation); }, cancellation);
        var context = new float[shape.Patches * shape.Channels]; var positions = new float[context.Length];
        var imageTimes = await Encode(samPath, shape, rgb, rays, context, positions, cancellation);
        var model = await Task.Run(() => MhrCheckpoint.LoadBody(mhrPath, cancellation), cancellation);
        var pose = await Task.Run(() => SamCheckpoint.LoadPoseHead(samPath, model, false, cancellation), cancellation);
        var decoderShape = new SamDecoderShape(SamInitialTokens.Count, shape.Patches);
        var camera = await Task.Run(() => new SamPredictionHead(VisionCheckpoint.LoadSamHead(samPath, false, cancellation), false, true), cancellation);
        var keypoints = await Task.Run(() => new SamKeypointFeedback(VisionCheckpoint.LoadSamFeedback(samPath, false, cancellation), decoderShape, 32, 32), cancellation);
        var projection = new SamProjection(intrinsics, affine, crop.CenterX, crop.CenterY, crop.Size, shape);
        var feedback = new SamDecoderFeedback(decoderShape, pose, camera, keypoints); feedback.Prepare(projection, context);
        var tokens = new float[decoderShape.TokenElements]; var tokenPositions = new float[tokens.Length]; var decoded = new float[tokens.Length];
        await Task.Run(() => new SamInitialTokens(VisionCheckpoint.LoadSamInitialTokens(samPath, false, cancellation))
            .Run(projection, tokens, tokenPositions, cancellation), cancellation);
        var decoderWeights = await Task.Run(() => VisionCheckpoint.LoadSamDecoder(samPath, false, cancellation), cancellation);
        await GameTask.MainThread(); var decoder = new SamGpuDecoder(decoderWeights, decoderShape); var timer = Stopwatch.StartNew();
        try
        {
            await decoder.RunAsync(tokens, context, tokenPositions, positions, decoded, feedback.Update, cancellation);
            await Task.Run(() => feedback.Finish(decoded, cancellation), cancellation);
        }
        finally { await GameTask.MainThread(); decoder.Dispose(); }
        var decodeMilliseconds = timer.Elapsed.TotalMilliseconds;
        var cropped = new float[140]; var depths = new float[70]; var translation = new float[3]; var imagePoints = new float[140];
        projection.Run(pose.Keypoints, feedback.CameraParameters, cropped, depths, translation);
        for (var i = 0; i < 70; i++)
        {
            imagePoints[i * 2] = ((cropped[i * 2] + .5f) * shape.Width - affine[2]) / affine[0];
            imagePoints[i * 2 + 1] = ((cropped[i * 2 + 1] + .5f) * shape.Height - affine[5]) / affine[4];
        }
        var result = new
        {
            BodyPassOnly = true,
            Width = width,
            Height = height,
            Box = box,
            Crop = new { crop.CenterX, crop.CenterY, crop.Size },
            ImageMilliseconds = imageTimes,
            DecodeMilliseconds = decodeMilliseconds,
            ImageKeypoints = imagePoints,
            KeypointsMetres = pose.Keypoints.ToArray(),
            Depths = depths,
            Translation = translation,
            BodyFeature = decoded.AsSpan(0, decoderShape.Width).ToArray(),
            ModelParameters = pose.Model.ToArray()
        };
        await File.WriteAllTextAsync(resultPath, JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }), cancellation);
        return new
        {
            Passed = true,
            BodyPassOnly = true,
            Box = box,
            Keypoints = 70,
            BodyFeatureChannels = decoderShape.Width,
            ImageMilliseconds = imageTimes,
            DecodeMilliseconds = decodeMilliseconds,
            ResultPath = resultPath
        };
    }

    static async Task<YoloBox> Detect(byte[] pixels, int width, int height, string path, CancellationToken cancellation)
    {
        var network = new YoloNetwork(); var weights = await Task.Run(() => YoloCheckpoint.Load(path, network, cancellation), cancellation);
        await GameTask.MainThread(); var detector = new YoloGpuDetector(network, weights);
        try
        {
            var boxes = new YoloBox[YoloNetwork.Predictions]; var count = await detector.DetectAsync(pixels, width, height, boxes, cancellation);
            if (count < 1) throw new Exception("SAM frame fixture contains no detected person.");
            return boxes[0];
        }
        finally { await GameTask.MainThread(); detector.Dispose(); }
    }

    static async Task<double[]> Encode(string path, DinoShape shape, float[] rgb, float[] rays, float[] context, float[] positions, CancellationToken cancellation)
    {
        var backbone = await Task.Run(() => VisionCheckpoint.LoadSamBackbone(path, cancellation), cancellation);
        var body = await Task.Run(() => VisionCheckpoint.LoadSamImageConditioning(path, false, cancellation), cancellation);
        var hand = await Task.Run(() => VisionCheckpoint.LoadSamImageConditioning(path, true, cancellation), cancellation);
        await GameTask.MainThread(); var encoder = new SamGpuImageEncoder(backbone, body, hand, shape);
        try
        {
            var timer = Stopwatch.StartNew(); await encoder.RunAsync(rgb, rays, context, false, cancellation);
            var first = timer.Elapsed.TotalMilliseconds; var previous = context.ToArray();
            await GameTask.MainThread(); timer.Restart(); await encoder.RunAsync(rgb, rays, context, false, cancellation);
            var repeated = timer.Elapsed.TotalMilliseconds;
            if (!previous.SequenceEqual(context)) throw new Exception("SAM image encoding did not repeat exactly.");
            encoder.CopyPositions(positions); return new[] { first, repeated };
        }
        finally { await GameTask.MainThread(); encoder.Dispose(); }
    }
}
