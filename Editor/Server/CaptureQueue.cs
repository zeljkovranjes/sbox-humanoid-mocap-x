using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Buffers;
using System.Text.Json;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX;

namespace HumanoidMocapHost;

internal sealed class QueueCapacityException(string message) : InvalidOperationException(message);

/// <summary>A request passed a byte or clip-length limit; transports answer 413 with the message.</summary>
internal sealed class LimitExceededException(string message) : Exception(message);

/// <summary>Owns bounded jobs and their lifecycle; storage and inference are injected.
/// <paramref name="maximumActivePerOwner"/> caps one user's unfinished jobs (1 to 16),
/// <paramref name="maximumUploadBytes"/> one video upload (1 byte to 64 GiB),
/// <paramref name="maximumClipSeconds"/> the processed video range (1 s to 24 h) and
/// <paramref name="concurrentJobs"/> how many jobs process at once (1 to 4) until an operator saves
/// another value. <paramref name="canStartJob"/> receives the next job and the running ranges, returning
/// false while the GPU cannot hold one more; waiting jobs then stay queued and it is asked again.
/// Priority jobs wait ahead of others; each group is first come, first served.</summary>
internal sealed class CaptureQueue(IJobStorage storage, IClipProcessor processor, TimeProvider clock, int maximumActivePerOwner = 2,
    ProcessingEstimate? estimate = null, long maximumUploadBytes = CaptureQueue.DefaultMaximumUploadBytes,
    double maximumClipSeconds = CaptureQueue.DefaultMaximumClipSeconds, int concurrentJobs = 1,
    Func<HostedCaptureOptions, IReadOnlyList<HostedCaptureOptions>, bool>? canStartJob = null) : IDisposable
{
    // Memory freed by other applications raises no event, so a refused start is retried on a timer.
    static readonly TimeSpan AdmissionRetry = TimeSpan.FromSeconds(2);
    readonly ProcessingEstimate estimate = estimate ?? new();
    internal const long DefaultMaximumUploadBytes = 512L * 1024 * 1024;
    internal const double DefaultMaximumClipSeconds = 600;
    internal const int MaximumConcurrentJobs = 4;
    readonly long maximumBytes = maximumUploadBytes is >= 1 and <= 64L << 30 ? maximumUploadBytes
        : throw new ArgumentOutOfRangeException(nameof(maximumUploadBytes), "Allow 1 byte to 64 GiB per upload.");
    readonly double maximumSeconds = maximumClipSeconds is >= 1 and <= 86400 ? maximumClipSeconds
        : throw new ArgumentOutOfRangeException(nameof(maximumClipSeconds), "Allow clips of 1 second to 24 hours.");
    internal long MaximumUploadBytes => maximumBytes;
    internal double MaximumClipSeconds => maximumSeconds;

    internal bool IsAvailable => processor.IsAvailable;

    internal const long MaximumResultBytes = 128L * 1024 * 1024;
    // Unfinished jobs across every owner; finished ones count only against their owner's history.
    internal const int MaximumActiveJobs = 64;
    // Jobs one owner keeps listed, finished included; creating more removes their oldest finished job.
    internal const int MaximumJobsPerOwner = 16;
    /// <summary>How long a finished capture's motion stays downloadable before it is deleted.</summary>
    internal static readonly TimeSpan ResultLifetime = TimeSpan.FromHours(24);
    // Failed, cancelled and never-uploaded jobs hold no result, so they go sooner.
    static readonly TimeSpan IdleLifetime = TimeSpan.FromHours(1);
    // Editors measure trims from frame times, so allow a little rounding over the limit.
    const double ClipTolerance = 0.01;
    readonly int maximumActive = maximumActivePerOwner is >= 1 and <= 16 ? maximumActivePerOwner
        : throw new ArgumentOutOfRangeException(nameof(maximumActivePerOwner), "Allow 1 to 16 active jobs per user.");
    internal int MaximumActivePerOwner => maximumActive;
    readonly object gate = new();
    readonly Dictionary<Guid, Job> jobs = new();
    // Released whenever a job becomes ready, finishes or the concurrency changes.
    readonly SemaphoreSlim wake = new(0);
    int concurrent = ValidConcurrency(storage.LoadSettings()?.ConcurrentJobs ?? concurrentJobs);
    long enqueueSequence;

    sealed class Job(string owner, HostedCaptureOptions settings, string? name, JobPolicy policy, long maximumUploadBytes, DateTimeOffset now)
    {
        internal readonly Guid Id = Guid.NewGuid();
        internal readonly string Owner = owner;
        internal HostedCaptureOptions Settings = settings;
        internal readonly string? Name = name;
        internal readonly bool Priority = policy.Priority;
        internal readonly long MaximumUploadBytes = maximumUploadBytes;
        internal readonly DateTimeOffset Created = now;
        internal double? VideoSeconds => Settings.End - Settings.Start;
        internal DateTimeOffset? Started;
        internal double? Progress;
        internal readonly CancellationTokenSource Cancellation = new();
        internal string State = "awaiting-upload", Message = "Waiting for video", Extension = "";
        internal DateTimeOffset Updated = now;
        internal bool InFlight, DeleteRequested;
        internal long Sequence;
        internal bool IsTerminal => State is "completed" or "failed" or "cancelled";
        internal DateTimeOffset Expires => Updated + (State == "completed" ? ResultLifetime : IdleLifetime);
        internal HostedJob View(int queueLength, int? position = null, double? untilStart = null, double? remaining = null)
            => new(Id, State, Message, position, Name, Created, queueLength, State == "processing" ? Progress : null,
                Seconds(untilStart), Seconds(remaining), IsTerminal || State == "awaiting-upload" ? Expires : null, Priority);
        static int? Seconds(double? value) => value is { } seconds ? (int)Math.Ceiling(Math.Min(seconds, int.MaxValue)) : null;
    }

    /// <summary>How many jobs process at once.</summary>
    internal int ConcurrentJobs { get { lock (gate) return concurrent; } }

    /// <summary>Changes and saves how many jobs process at once. Lowering it lets running jobs
    /// finish. Throws <see cref="ArgumentException"/> outside 1 to <see cref="MaximumConcurrentJobs"/>.</summary>
    internal void SetConcurrentJobs(int count)
    {
        lock (gate)
        {
            storage.SaveSettings(new HostSettings(ValidConcurrency(count)));
            concurrent = count;
        }
        wake.Release();
    }

    static int ValidConcurrency(int count) => count is >= 1 and <= MaximumConcurrentJobs ? count
        : throw new ArgumentException($"Allow 1 to {MaximumConcurrentJobs} jobs processing at once.");

    /// <summary>Public host status: limits plus how many jobs wait and run, across all owners.</summary>
    internal HostedCapabilities Capabilities()
    {
        lock (gate)
            return new(1, IsAvailable, MaximumUploadBytes, MaximumActivePerOwner,
                jobs.Values.Count(j => j.State == "queued"), jobs.Values.Count(j => j.State == "processing"), concurrent, MaximumClipSeconds);
    }

    /// <summary>Creates a job waiting for its video. <paramref name="policy"/> sets priority and
    /// tighter limits for the owner. A range without an end is cut at the clip limit; a longer
    /// range throws <see cref="LimitExceededException"/>.</summary>
    internal HostedJob Create(string owner, HostedCaptureOptions options, string? name = null, JobPolicy? policy = null)
    {
        options.Validate();
        policy ??= new(); policy.Validate();
        var clipLimit = Math.Min(policy.MaximumClipSeconds ?? MaximumClipSeconds, MaximumClipSeconds);
        if (options.End is null) options = options with { End = options.Start + clipLimit };
        else if (options.End - options.Start > clipLimit + ClipTolerance)
            throw new LimitExceededException($"Clips are limited to {clipLimit:0.##} seconds for your account. Trim the clip and try again.");
        if (!processor.IsAvailable) throw new InvalidOperationException("The host has not configured its video processor.");
        lock (gate)
        {
            Prune();
            var owned = jobs.Values.Where(j => j.Owner == owner).ToArray();
            if (owned.Count(j => !j.IsTerminal) >= MaximumActivePerOwner)
                throw new QueueCapacityException(MaximumActivePerOwner == 1
                    ? "This host processes one capture per user at a time. Wait for it to finish or cancel it."
                    : $"This host allows {MaximumActivePerOwner} captures per user at a time. Wait for one to finish or cancel it.");
            if (jobs.Values.Count(j => !j.IsTerminal) >= MaximumActiveJobs)
                throw new QueueCapacityException("The processing queue is full. Try again after a capture finishes.");
            foreach (var old in owned.Where(j => j.IsTerminal && !j.InFlight).OrderBy(j => j.Updated).Take(owned.Length - MaximumJobsPerOwner + 1))
                Remove(old);
            var job = new Job(owner, options, name, policy, Math.Min(policy.MaximumUploadBytes ?? MaximumUploadBytes, MaximumUploadBytes),
                clock.GetUtcNow());
            jobs.Add(job.Id, job);
            return View(job);
        }
    }

    internal HostedJob Status(string owner, Guid id)
    {
        lock (gate) return View(Find(owner, id));
    }

    /// <summary>The owner's jobs, oldest first, with queue positions.</summary>
    internal HostedJob[] List(string owner)
    {
        lock (gate) return jobs.Values.Where(j => j.Owner == owner).OrderBy(j => j.Created).Select(View).ToArray();
    }

    Job[] Waiting() => jobs.Values.Where(j => j.State == "queued").OrderByDescending(j => j.Priority).ThenBy(j => j.Sequence).ToArray();

    // Positions and waits count every owner's jobs, since they share the processing slots. Each
    // queued job starts when the earliest running job is expected to finish.
    HostedJob View(Job job)
    {
        var waiting = Waiting();
        if (job.State == "processing") return job.View(waiting.Length, remaining: RemainingOf(job));
        if (job.State != "queued") return job.View(waiting.Length);
        var finishing = new PriorityQueue<double, double>();
        foreach (var running in jobs.Values.Where(j => j.State == "processing")) { var left = RemainingOf(running); finishing.Enqueue(left, left); }
        double start = 0;
        for (var index = 0; ; index++)
        {
            while (finishing.Count >= concurrent) start = Math.Max(start, finishing.Dequeue());
            var end = start + estimate.Expected(waiting[index].VideoSeconds);
            if (waiting[index] == job) return job.View(waiting.Length, index + 1, start, end);
            finishing.Enqueue(end, end);
        }
    }

    double RemainingOf(Job job) => job.Started is { } started
        ? estimate.Remaining(job.VideoSeconds, (clock.GetUtcNow() - started).TotalSeconds, job.Progress)
        : estimate.Expected(job.VideoSeconds);

    internal async Task UploadAsync(string owner, Guid id, string extension, Stream input, CancellationToken cancellation)
    {
        if (extension is not (".mp4" or ".mov" or ".m4v" or ".avi" or ".mkv" or ".webm"))
            throw new ArgumentException("Unsupported video extension.");
        Job job;
        lock (gate)
        {
            job = Find(owner, id);
            if (job.State != "awaiting-upload") throw new InvalidOperationException("This job already received an upload.");
            Update(job, "uploading", "Receiving video"); job.Extension = extension; job.InFlight = true;
        }
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellation, job.Cancellation.Token);
        linked.CancelAfter(TimeSpan.FromMinutes(30));
        try
        {
            await using (var destination = storage.CreateVideo(id))
                await CopyBounded(input, destination, job.MaximumUploadBytes, linked.Token);
            var selected = job.Settings;
            if (processor is IClipInspector inspector)
            {
                selected = await inspector.InspectAsync(selected, storage.VideoPath(id), linked.Token);
                selected.Validate();
                if (selected.Model != job.Settings.Model || selected.CameraInterval != job.Settings.CameraInterval
                    || selected.Start != job.Settings.Start || selected.End is null || selected.End > job.Settings.End)
                    throw new InvalidDataException("Video inspection changed the requested capture settings.");
            }
            lock (gate)
            {
                linked.Token.ThrowIfCancellationRequested();
                job.Settings = selected;
                Update(job, "queued", "Waiting for processing");
                job.Sequence = ++enqueueSequence; job.InFlight = false;
            }
            wake.Release();
        }
        catch
        {
            lock (gate)
            {
                Update(job, linked.IsCancellationRequested ? "cancelled" : "failed", "Video upload did not complete");
                try { storage.Delete(id); }
                finally { job.InFlight = false; }
                if (job.DeleteRequested) Remove(job);
            }
            throw;
        }
    }

    /// <summary>Cancels a waiting job or deletes a finished one. A job that is already processing
    /// cannot be cancelled; its result is deleted <see cref="ResultLifetime"/> after it finishes.</summary>
    internal void Cancel(string owner, Guid id)
    {
        lock (gate)
        {
            var job = Find(owner, id);
            if (job.State == "processing")
                throw new InvalidOperationException("This capture is already processing and cannot be cancelled. Its result is deleted 24 hours after it finishes.");
            job.DeleteRequested = true; job.Cancellation.Cancel();
            // Only uploads are in flight here; the upload's own cleanup removes the job.
            if (job.InFlight) Update(job, "cancelling", "Stopping upload");
            else Remove(job);
        }
    }

    internal Stream OpenResult(string owner, Guid id)
    {
        lock (gate)
        {
            if (Find(owner, id).State != "completed") throw new InvalidOperationException("The motion result is not ready.");
            return storage.OpenMotion(id);
        }
    }

    /// <summary>Starts waiting jobs while fewer than <see cref="ConcurrentJobs"/> run, until
    /// <paramref name="stopping"/> is cancelled; then waits for running jobs to stop.</summary>
    internal async Task RunAsync(Action<Guid, Exception> reportFailure, CancellationToken stopping)
    {
        var running = new List<Task>();
        using var processingStop = CancellationTokenSource.CreateLinkedTokenSource(stopping);
        try
        {
            while (true)
            {
                for (var index = running.Count - 1; index >= 0; index--)
                {
                    var task = running[index];
                    if (!task.IsCompleted) continue;
                    running.RemoveAt(index);
                    await task;
                }
                stopping.ThrowIfCancellationRequested();
                Job? next;
                HostedCaptureOptions[] active;
                lock (gate)
                {
                    next = running.Count < concurrent ? Waiting().FirstOrDefault() : null;
                    active = jobs.Values.Where(job => job.State == "processing").Select(job => job.Settings).ToArray();
                }
                if (next is null)
                {
                    // A worker may signal just before its Task changes to completed.
                    await wake.WaitAsync(running.Count > 0 ? AdmissionRetry : Timeout.InfiniteTimeSpan, stopping);
                    continue;
                }
                // Asked outside the gate: a GPU query may be slow and must not block status requests.
                bool admitted;
                try { admitted = canStartJob?.Invoke(next.Settings, active) ?? true; }
                catch (LimitExceededException error)
                {
                    lock (gate)
                    {
                        if (next.State == "queued" && jobs.ContainsKey(next.Id))
                        { Update(next, "failed", error.Message); storage.DeleteVideo(next.Id); }
                    }
                    continue;
                }
                if (!admitted) { await wake.WaitAsync(AdmissionRetry, stopping); continue; }
                lock (gate)
                {
                    if (running.Count < concurrent && next.State == "queued" && jobs.ContainsKey(next.Id))
                    {
                        next.InFlight = true; next.Started = clock.GetUtcNow();
                        Update(next, "processing", "Reconstructing motion");
                    }
                    else next = null;
                }
                if (next is not null) running.Add(ProcessAsync(next, reportFailure, processingStop.Token));
            }
        }
        finally
        {
            await processingStop.CancelAsync();
            try { await Task.WhenAll(running); }
            catch (OperationCanceledException) when (processingStop.IsCancellationRequested) { }
        }
    }

    // Expected capture failures are recorded; a failing reporter reaches the dispatcher.
    async Task ProcessAsync(Job job, Action<Guid, Exception> reportFailure, CancellationToken stopping)
    {
        await Task.Yield();
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(stopping, job.Cancellation.Token);
        linked.CancelAfter(TimeSpan.FromHours(1));
        try
        {
            await using (var video = storage.OpenVideo(job.Id))
            await using (var motion = storage.CreateMotion(job.Id))
                await processor.ProcessAsync(job.Settings, job.Extension, video, motion, (message, fraction) =>
                {
                    lock (gate)
                    {
                        if (job.State != "processing") return;
                        Update(job, "processing", message.Length > 200 ? message[..200] : message);
                        if (fraction is >= 0 and <= 1) job.Progress = fraction;
                    }
                }, linked.Token);
            await ValidateResult(job.Id, linked.Token);
            lock (gate)
            {
                linked.Token.ThrowIfCancellationRequested();
                Update(job, "completed", "Motion is ready");
                estimate.Completed(job.VideoSeconds, (clock.GetUtcNow() - job.Started!.Value).TotalSeconds);
            }
        }
        catch (OperationCanceledException) when (linked.IsCancellationRequested)
        { lock (gate) Update(job, "cancelled", "Cancelled"); }
        catch (Exception error)
        {
            lock (gate) Update(job, "failed", "Processing failed. Give the job ID to the host for details.");
            reportFailure(job.Id, error);
        }
        finally
        {
            try
            {
                lock (gate)
                {
                    try
                    {
                        storage.DeleteVideo(job.Id);
                        if (job.State != "completed" || job.DeleteRequested) storage.Delete(job.Id);
                        if (job.DeleteRequested) Remove(job);
                    }
                    finally { job.InFlight = false; }
                }
            }
            catch (Exception error)
            {
                lock (gate) Update(job, "failed", "Temporary file cleanup failed. Give the job ID to the host for details.");
                reportFailure(job.Id, error);
            }
            finally { wake.Release(); }
        }
    }

    internal void Prune()
    {
        lock (gate)
        {
            var now = clock.GetUtcNow();
            foreach (var job in jobs.Values.Where(j => !j.InFlight && j.Expires <= now && (j.IsTerminal || j.State == "awaiting-upload")).ToArray())
                Remove(job);
        }
    }

    /// <summary>Releases queue resources after HTTP handlers and RunAsync have stopped.
    /// Throws if an upload or capture still owns job files.</summary>
    public void Dispose()
    {
        lock (gate)
        {
            if (jobs.Values.Any(job => job.InFlight)) throw new InvalidOperationException("Stop uploads and processing before disposing the queue.");
            foreach (var job in jobs.Values) job.Cancellation.Dispose();
            jobs.Clear(); wake.Dispose();
        }
    }

    void Remove(Job job)
    {
        storage.Delete(job.Id); jobs.Remove(job.Id); job.Cancellation.Dispose();
    }

    async Task ValidateResult(Guid id, CancellationToken cancellation)
    {
        await using var stream = storage.OpenMotion(id);
        if (stream.Length is < 1 or > MaximumResultBytes) throw new InvalidDataException("The processor returned an empty or oversized motion.");
        var motion = await JsonSerializer.DeserializeAsync<MotionDocument>(stream, MotionDocument.JsonOptions, cancellation)
            ?? throw new InvalidDataException("The processor returned no motion.");
        motion.Validate();
        if (!motion.Bones.Any(bone => bone.Role == BoneRole.Hips)) throw new InvalidDataException("The processor must return a full-body motion.");
    }

    Job Find(string owner, Guid id) => jobs.TryGetValue(id, out var job) && job.Owner == owner
        ? job : throw new KeyNotFoundException("The capture job was not found.");

    void Update(Job job, string state, string message)
    { job.State = state; job.Message = message; job.Updated = clock.GetUtcNow(); }

    internal static async Task CopyBounded(Stream input, Stream output, long maximum, CancellationToken cancellation)
    {
        var buffer = ArrayPool<byte>.Shared.Rent(65536);
        try
        {
            long total = 0;
            while (await input.ReadAsync(buffer.AsMemory(0, 65536), cancellation) is var read && read > 0)
            {
                total += read;
                if (total > maximum) throw new LimitExceededException(maximum >= 1 << 20
                    ? $"The upload is larger than the {maximum / (double)(1 << 20):0.#} MiB this account may send."
                    : $"The request body is larger than {maximum} bytes.");
                await output.WriteAsync(buffer.AsMemory(0, read), cancellation);
            }
            if (total == 0) throw new InvalidDataException("The video upload is empty.");
        }
        finally { ArrayPool<byte>.Shared.Return(buffer); }
    }
}
