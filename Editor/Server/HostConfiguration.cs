using System;
using System.Globalization;
using System.IO;
using HumanoidMocapX;

namespace HumanoidMocapHost;

/// <summary>Validated operator settings, read once before opening the listener.</summary>
internal sealed record HostConfiguration(string Models, string Data, int Port, long UploadBytes,
    double ClipSeconds, int ConcurrentJobs, int ActivePerOwner, int Lanes, ApiKeys Keys)
{
    internal static HostConfiguration Read(Func<string, string?> read, string defaultCache)
    {
        var cache = Path.GetFullPath(read("MOCAP_CACHE_DIRECTORY") ?? defaultCache);
        var models = Path.GetFullPath(read("MOCAP_MODELS_DIRECTORY") ?? Path.Combine(cache, "models"));
        var data = Path.GetFullPath(read("MOCAP_DATA_DIRECTORY") ?? Path.Combine(cache, "host"));
        var port = Integer("MOCAP_PORT", 5080, 1, 65535);
        var upload = (long)Integer("MOCAP_MAX_UPLOAD_MB", 512, 1, 65536) * 1024 * 1024;
        var maximumSeconds = CaptureOptions.MaximumFrames / 30d;
        var secondsText = read("MOCAP_MAX_CLIP_SECONDS");
        var seconds = maximumSeconds;
        if (secondsText is not null && (!double.TryParse(secondsText, NumberStyles.Float, CultureInfo.InvariantCulture, out seconds)
            || !double.IsFinite(seconds) || seconds < 1 || seconds > maximumSeconds))
            throw new ArgumentException("MOCAP_MAX_CLIP_SECONDS must be between 1 and 166.66666666666666 seconds.");
        return new(models, data, port, upload, seconds,
            Integer("MOCAP_CONCURRENT_JOBS", 1, 1, CaptureQueue.MaximumConcurrentJobs),
            Integer("MOCAP_MAX_JOBS_PER_USER", 2, 1, 16), Integer("MOCAP_LANES", CaptureLanes.Auto, 0, CaptureLanes.Maximum),
            new ApiKeys(read("MOCAP_API_KEYS"), read("MOCAP_KEY_SECRET")));

        int Integer(string name, int fallback, int minimum, int maximum)
        {
            var text = read(name);
            if (text is null) return fallback;
            if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) || value < minimum || value > maximum)
                throw new ArgumentException($"{name} must be between {minimum} and {maximum}.");
            return value;
        }
    }
}
