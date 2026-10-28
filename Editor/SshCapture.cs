using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX.EditorTools;

/// <summary>Captures on a processing machine over SSH with the Linux humanoid-mocap-worker:
/// uploads the video into a new job directory there, runs the worker, forwards its progress lines
/// and downloads the motion into the capture cache. The job directory is removed afterwards.
/// Cancelling ends the worker's session, and the machine then stops the worker.</summary>
// Starting ssh is an approved exception to the library's no-worker-process rule: the user asked
// for captures to run on their own GPU machine over SSH.
internal sealed class SshCapture(SshSettings settings, CaptureOptions options, string cache,
    Action<string, double?> reportProgress = null) : IBodyCapture
{
    internal const string ClientMissing = "Processing on another machine needs an SSH client: install OpenSSH so ssh is on PATH.";
    /// <summary>Each job gets a new directory with this prefix; mkdir refuses an existing one.</summary>
    internal const string JobRoot = "/tmp/humanoid-mocap-x-";
    const string Result = "capture.hmotion";
    const long MaximumResultBytes = 128L * 1024 * 1024;
    // How long a cancelled worker may take to stop before the session is cut. The wrapper kills it after 10 s.
    static readonly TimeSpan StopGrace = TimeSpan.FromSeconds(15);
    static readonly TimeSpan CleanupTimeout = TimeSpan.FromSeconds(20);

    // $1 job directory, $2 file name. A fresh mkdir fails if the path exists, so nothing is written through a planted link.
    const string Upload = """umask 077 && mkdir -- "$1" && cat > "$1/$2" """;

    // $1 job directory, then the worker command. The worker's stdin is /dev/null; the wrapper's own
    // stdin is the session, so its end (cancel, editor exit or a dropped connection) sends TERM, then
    // KILL after 10 s. SIGPIPE is ignored so a dropped session cannot kill the wrapper before that.
    // A failed or stopped job removes its directory; a finished one is left for the download.
    const string Worker = """
        trap "" PIPE
        d=$1; shift
        "$@" </dev/null &
        p=$!
        exec 3<&0
        { while read -r line <&3; do :; done; kill -TERM "$p"; sleep 10; kill -KILL "$p"; } >/dev/null 2>&1 &
        k=$!
        wait "$p"; c=$?
        kill "$k" 2>/dev/null
        [ "$c" -eq 0 ] || rm -rf -- "$d"
        exit "$c"
        """;

    // $1 worker. Without arguments the worker prints its usage, which proves it starts.
    const string Check = """
        command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || { echo "FFmpeg is not installed there: ffmpeg and ffprobe must be on PATH." >&2; exit 1; }
        usage=$("$1" </dev/null 2>&1)
        case $usage in *"humanoid-mocap-worker capture"*) exit 0;; esac
        echo "The worker at $1 did not start: $usage" >&2; exit 1
        """;

    const string Download = """cat -- "$1" """;
    const string Remove = """rm -rf -- "$1" """;

    int active;

    public string UnavailableReason => ClientInstalled ? settings.Problem : ClientMissing;

    static bool ClientInstalled => (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
        .Any(directory => File.Exists(Path.Combine(directory, OperatingSystem.IsWindows() ? "ssh.exe" : "ssh")));

    public async Task<string> CaptureAsync(string video, double start, double? end, Action<string> progress, CancellationToken cancellationToken)
    {
        if (UnavailableReason is { } reason) throw new InvalidOperationException(reason);
        var request = new HostedCaptureOptions(options.Model, options.CameraInterval, start, end); request.Validate();
        var address = settings.Address; var worker = settings.Worker;
        await using var input = new FileStream(video, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, true);
        if (Interlocked.CompareExchange(ref active, 1, 0) != 0)
            throw new InvalidOperationException("Wait for the current capture on the processing machine to finish.");
        var job = JobRoot + Guid.NewGuid().ToString("N");
        void Report(string message, double? fraction) { progress?.Invoke(message); reportProgress?.Invoke(message, fraction); }
        try
        {
            Report("Uploading video to " + address + "…", null);
            var name = VideoName(video);
            await RunAsync(address, Upload, [job, name], cancellationToken, input: input);
            await RunAsync(address, Worker, [job, RemoteWorker(worker), .. WorkerArguments(job, name, request)], cancellationToken,
                output: reader => ForwardAsync(reader, Report), grace: StopGrace);
            Report("Downloading motion from " + address + "…", null);
            return await Task.Run(() => DownloadAsync(address, job, video, cancellationToken), cancellationToken);
        }
        finally
        {
            try { await RemoveAsync(address, job, Report); }
            finally { Volatile.Write(ref active, 0); }
        }
    }

    /// <summary>Checks that the machine answers without a prompt, has FFmpeg and can start the
    /// worker. Throws with the reason otherwise.</summary>
    internal static Task CheckAsync(SshSettings settings, CancellationToken token)
    {
        if (!ClientInstalled) throw new InvalidOperationException(ClientMissing);
        if (settings.Problem is { } problem) throw new ArgumentException(problem);
        return RunAsync(settings.Address, Check, [RemoteWorker(settings.Worker)], token);
    }

    /// <summary>The worker's command line for one job; the video and output live in the job directory.</summary>
    internal static List<string> WorkerArguments(string job, string video, HostedCaptureOptions request)
    {
        var arguments = new List<string> { "capture", job + "/" + video, job + "/" + Result,
            "--start", request.Start.ToString("R", CultureInfo.InvariantCulture), "--model", request.Model == FlowModel.Base ? "base" : "latest",
            "--camera-interval", request.CameraInterval.ToString(CultureInfo.InvariantCulture), "--cache", job };
        if (request.End is { } stop) { arguments.Add("--end"); arguments.Add(stop.ToString("R", CultureInfo.InvariantCulture)); }
        return arguments;
    }

    /// <summary>The remote file name: never the user's, which may hold quotes or spaces; a plain
    /// extension is kept as FFmpeg's format hint.</summary>
    internal static string VideoName(string video)
    {
        var extension = Path.GetExtension(video).ToLowerInvariant();
        return "video" + (extension.Length is > 1 and <= 8 && extension[1..].All(char.IsAsciiLetterOrDigit) ? extension : "");
    }

    // ssh starts commands in the home directory, so ~/ is the same as ./ there.
    internal static string RemoteWorker(string worker) => worker.StartsWith("~/", StringComparison.Ordinal) ? "./" + worker[2..] : worker;

    /// <summary>ssh's arguments: never prompt (the editor cannot answer), give up on an unreachable
    /// machine after 10 s or a silent one after a minute, and no terminal, so stderr stays separate.</summary>
    internal static string[] Arguments(string address, string command)
        => ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-T", address, command];

    /// <summary>The remote command line: sh runs <paramref name="script"/> with each argument quoted.</summary>
    internal static string Command(string script, IEnumerable<string> arguments)
        => "sh -c " + Quote(script) + " humanoid-mocap-x " + string.Join(" ", arguments.Select(Quote));

    /// <summary>Single-quotes <paramref name="text"/> for the login shell. Quotes and backslashes are
    /// escaped outside the quotes, which POSIX shells and fish read the same way.</summary>
    internal static string Quote(string text) => "'" + Regex.Replace(text, @"['\\]", match => "'\\" + match.Value + "'") + "'";

    /// <summary>One worker progress line, <c>{"message":...,"fraction":...}</c>, or null for any other line.</summary>
    internal static (string Message, double? Fraction)? Progress(string line)
    {
        try
        {
            using var json = JsonDocument.Parse(line);
            var root = json.RootElement;
            if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("message", out var message) || message.ValueKind != JsonValueKind.String)
                return null;
            return (message.GetString(), root.TryGetProperty("fraction", out var fraction) && fraction.ValueKind == JsonValueKind.Number ? fraction.GetDouble() : null);
        }
        catch (JsonException) { return null; }
    }

    /// <summary>The last lines of a remote standard error without .NET stack frames, so the
    /// worker's error line is what a failure shows.</summary>
    internal static string Summary(string errors)
    {
        var lines = errors.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(line => !line.StartsWith("at ", StringComparison.Ordinal) && !line.StartsWith("--- ", StringComparison.Ordinal)).TakeLast(4);
        var text = string.Join("\n", lines);
        return text.Length == 0 ? "no error output" : text.Length > 1000 ? "…" + text[^1000..] : text;
    }

    static async Task ForwardAsync(StreamReader reader, Action<string, double?> report)
    {
        while (await reader.ReadLineAsync() is { } line)
            if (Progress(line) is { } update) report(update.Message, update.Fraction);
    }

    async Task<string> DownloadAsync(string address, string job, string video, CancellationToken token)
    {
        Directory.CreateDirectory(cache);
        var path = Path.Combine(cache, Guid.NewGuid().ToString("N") + ".hmotion"); var partial = path + ".partial";
        try
        {
            await using (var output = new FileStream(partial, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, true))
                await RunAsync(address, Download, [job + "/" + Result], token, output: reader => CopyAsync(reader.BaseStream, output, token));
            var motion = MotionDocument.Parse(await File.ReadAllBytesAsync(partial, token));
            if (!motion.Bones.Any(bone => bone.Role == BoneRole.Hips)) throw new InvalidDataException("The processing machine must return full-body motion.");
            // The worker only saw its own copy of the video; point the motion at the local file.
            motion.Name = Path.GetFileNameWithoutExtension(video); motion.SourceVideo = video; motion.OriginalReconstruction = null;
            await using (var output = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, 65536, true))
                await JsonSerializer.SerializeAsync(output, motion, MotionDocument.JsonOptions, token);
            token.ThrowIfCancellationRequested(); File.Move(partial, path);
            return path;
        }
        finally { if (File.Exists(partial)) File.Delete(partial); }
    }

    static async Task CopyAsync(Stream input, Stream output, CancellationToken token)
    {
        var buffer = new byte[65536]; long total = 0;
        while (await input.ReadAsync(buffer, token) is var read && read > 0)
        {
            if ((total += read) > MaximumResultBytes) throw new InvalidDataException("The motion from the processing machine is larger than 128 MiB.");
            await output.WriteAsync(buffer.AsMemory(0, read), token);
        }
    }

    // A failed or stopped job's wrapper has already removed the directory; this covers the rest.
    static async Task RemoveAsync(string address, string job, Action<string, double?> report)
    {
        using var timeout = new CancellationTokenSource(CleanupTimeout);
        try { await RunAsync(address, Remove, [job], timeout.Token); }
        catch (Exception error) when (error is InvalidOperationException or OperationCanceledException or IOException)
        { report("Could not remove " + job + " on the processing machine: " + error.Message, null); }
    }

    /// <summary>Runs <paramref name="script"/> on the machine until ssh exits. <paramref name="input"/>
    /// is streamed to its standard input; otherwise standard input stays open, because the worker
    /// wrapper treats its end as "stop". <paramref name="output"/> reads standard output. Cancelling
    /// closes standard input and cuts the session after <paramref name="grace"/> (at once by default).
    /// Throws with the remote standard error when ssh or the script fails.</summary>
    static async Task RunAsync(string address, string script, IEnumerable<string> arguments, CancellationToken token,
        Stream input = null, Func<StreamReader, Task> output = null, TimeSpan grace = default)
    {
        using var process = Start(address, Command(script, arguments));
        var errors = process.StandardError.ReadToEndAsync(CancellationToken.None);
        using var deadline = new CancellationTokenSource();
        using var kill = deadline.Token.Register(() => Kill(process));
        try
        {
            using (token.Register(() =>
            {
                if (grace <= TimeSpan.Zero) { deadline.Cancel(); return; }
                Close(process.StandardInput); deadline.CancelAfter(grace);
            }))
            {
                var sending = input is null ? Task.FromResult<Exception>(null) : SendAsync(input, process.StandardInput, token);
                await (output?.Invoke(process.StandardOutput) ?? process.StandardOutput.BaseStream.CopyToAsync(Stream.Null));
                await process.WaitForExitAsync(CancellationToken.None);
                var unsent = await sending;
                token.ThrowIfCancellationRequested();
                if (process.ExitCode != 0)
                    throw new InvalidOperationException($"The processing machine failed (exit {process.ExitCode}): {Summary(await errors)}");
                if (unsent is not null) throw new IOException("Could not upload the video: " + unsent.Message, unsent);
            }
        }
        finally { Kill(process); }
    }

    static Process Start(string address, string command)
    {
        // UTF-8 explicitly: Windows would otherwise decode the worker's "·" and "…" with the console code page.
        var info = new ProcessStartInfo("ssh")
        {
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8
        };
        foreach (var argument in Arguments(address, command)) info.ArgumentList.Add(argument);
        try { return Process.Start(info) ?? throw new InvalidOperationException(ClientMissing); }
        catch (Win32Exception error) { throw new InvalidOperationException(ClientMissing, error); }
    }

    // Returns the failure instead of throwing: if ssh failed as well, its error says more.
    static async Task<Exception> SendAsync(Stream input, StreamWriter stdin, CancellationToken token)
    {
        try { await input.CopyToAsync(stdin.BaseStream, token); return null; }
        catch (Exception error) when (error is IOException or OperationCanceledException or ObjectDisposedException) { return error; }
        finally { Close(stdin); }
    }

    static void Close(StreamWriter stdin)
    {
        try { stdin.Close(); } catch (IOException) { }
    }

    static void Kill(Process process)
    {
        try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
        catch (Exception error) when (error is InvalidOperationException or Win32Exception) { }
    }
}
