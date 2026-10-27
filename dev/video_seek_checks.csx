// Diagnostic only: changed callbacks do not establish frame accuracy. Compare saved pixels against source frames.
var fs = Editor.FileSystem.ProjectTemporary;
var path = await HumanoidMocap.EditorTools.VideoFiles.CacheVideoAsync((await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "seek-fixture-path.txt"), cancellationToken)).Trim(), fs, cancellationToken);
await GameTask.MainThread();
var player = new VideoPlayer { Repeat = false, Muted = true };
var texture = player.Texture;
var sync = new object(); var pixels = new byte[256 * 128 * 4]; var version = 0; var loaded = false;
player.OnLoaded = () => { loaded = true; };
player.OnTextureData = (ReadOnlySpan<byte> source, Vector2 size) => { lock (sync) { if (source.Length != pixels.Length) throw new Exception("Unexpected seek fixture dimensions"); source.CopyTo(pixels); version++; } };
var records = new System.Collections.Generic.List<object>();
try
{
    player.Play(fs, path);
    var timeout = System.Diagnostics.Stopwatch.StartNew();
    while (!loaded || version == 0)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (timeout.Elapsed.TotalSeconds > 30) throw new Exception($"Seek fixture load timeout: loaded={loaded}, version={version}, time={player.PlaybackTime}, size={player.Width}x{player.Height}");
        player.Present(); await Task.Delay(10, cancellationToken); await GameTask.MainThread();
    }
    player.Pause();
    var times = new float[] { 0, 1f / 30, .2f, 1.5f, 4.9f, .2f, 4f, 1f / 30 };
    for (var i = 0; i < times.Length; i++)
    {
        var before = version; player.Seek(times[i]); timeout.Restart(); var lastVersion = before; double lastChanged = 0;
        while (timeout.Elapsed.TotalSeconds < 3)
        {
            cancellationToken.ThrowIfCancellationRequested(); player.Present();
            lock (sync) { if (version != lastVersion) { lastVersion = version; lastChanged = timeout.Elapsed.TotalSeconds; } }
            if (lastVersion != before && timeout.Elapsed.TotalSeconds - lastChanged > .15) break;
            await Task.Delay(10, cancellationToken); await GameTask.MainThread();
        }
        byte[] copy; lock (sync) copy = pixels.ToArray();
        var file = System.IO.Path.Combine(exchangeDirectory, "seek-" + i + ".rgba");
        await System.IO.File.WriteAllBytesAsync(file, copy, cancellationToken); await GameTask.MainThread();
        records.Add(new { Requested = times[i], Playback = player.PlaybackTime, Version = version, Changed = version != before, File = file });
    }
}
finally { await GameTask.MainThread(); player.OnLoaded = null; player.OnTextureData = null; player.Dispose(); texture.Dispose(); }
await System.IO.File.WriteAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "video-seek-result.json"), System.Text.Json.JsonSerializer.Serialize(records), cancellationToken);
return records;
