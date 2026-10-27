// Diagnostic only: compare every callback's pixels with the known 150-frame fixture.
var fs = Editor.FileSystem.ProjectTemporary;
var input = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "seek-fixture-path.txt"), cancellationToken)).Trim();
var path = await HumanoidMocapX.EditorTools.VideoFiles.CacheVideoAsync(input, fs, cancellationToken);
const int bytesPerFrame = 256 * 128 * 4, maximumCallbacks = 600;
var pixels = new byte[bytesPerFrame * maximumCallbacks]; var times = new float[maximumCallbacks];
var count = 0; var loaded = false; var finished = false; string failure = null;
await GameTask.MainThread();
var player = new VideoPlayer { Repeat = false, Muted = true }; var texture = player.Texture;
player.OnLoaded = () => loaded = true;
player.OnFinished = () => finished = true;
player.OnTextureData = (ReadOnlySpan<byte> source, Vector2 size) =>
{
    if (source.Length != bytesPerFrame || count >= maximumCallbacks) { failure = "Unexpected video callback dimensions or count"; return; }
    source.CopyTo(pixels.AsSpan(count * bytesPerFrame, bytesPerFrame)); times[count] = player.PlaybackTime; count++;
};
try
{
    player.Play(fs, path); var timeout = System.Diagnostics.Stopwatch.StartNew();
    while (!finished)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (failure is not null) throw new Exception(failure);
        if (timeout.Elapsed.TotalSeconds > 20) throw new Exception($"Playback fixture timed out: loaded={loaded}, callbacks={count}, time={player.PlaybackTime}");
        player.Present(); await Task.Delay(5, cancellationToken); await GameTask.MainThread();
    }
}
finally { await GameTask.MainThread(); player.OnLoaded = null; player.OnFinished = null; player.OnTextureData = null; player.Dispose(); texture.Dispose(); }
var output = System.IO.Path.Combine(exchangeDirectory, "playback-frames.rgba");
using (var file = new System.IO.FileStream(output, System.IO.FileMode.CreateNew, System.IO.FileAccess.Write, System.IO.FileShare.None, 65536, true))
    await file.WriteAsync(pixels.AsMemory(0, count * bytesPerFrame), cancellationToken);
var result = new { Loaded = loaded, Finished = finished, Callbacks = count, Times = times.AsSpan(0, count).ToArray(), Frames = output };
await System.IO.File.WriteAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "video-playback-result.json"), System.Text.Json.JsonSerializer.Serialize(result), cancellationToken);
return result;
