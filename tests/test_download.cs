using System.Net;
using System.Security.Cryptography;
using System.Text;
using HumanoidMocapX;
using HumanoidMocapX.EditorTools;

static class DownloadTests
{
    public static async Task Run()
    {
        var folder = Path.Combine(Path.GetTempPath(), "mocap-x-tests-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var handler = new Checkpoints(); using var http = new HttpClient(handler);
            MotionTests.Throws<ArgumentNullException>(() => new ModelStore(null, folder));
            MotionTests.Throws<ArgumentException>(() => new ModelStore(http, " "));
            var store = new ModelStore(http, folder);
            var path = await store.EnsureAsync(FlowModel.Latest, null, default);
            MotionTests.Check(File.ReadAllBytes(path).SequenceEqual(handler.Bytes), "Downloaded bytes differ");
            await store.EnsureAsync(FlowModel.Latest, null, default);
            MotionTests.Equal(1, handler.Downloads);
            var basePath = await store.EnsureAsync(FlowModel.Base, null, default);
            MotionTests.Check(basePath != path, "Model caches overlap");
            MotionTests.Equal(2, handler.Downloads);
            File.WriteAllText(path, "bad cache");
            handler.Corrupt = true;
            try { await store.EnsureAsync(FlowModel.Latest, null, default); throw new Exception("Corruption was accepted"); }
            catch (InvalidDataException) { }
            MotionTests.Equal("bad cache", File.ReadAllText(path));
            MotionTests.Equal(0, Directory.GetFiles(folder, "*.partial", SearchOption.AllDirectories).Length);
            handler.Corrupt = false;
            using (var interrupted = new CancellationTokenSource())
            {
                try
                {
                    await store.EnsureAsync(FlowModel.Latest,
                        message => { if (message.Contains("100%")) interrupted.Cancel(); }, interrupted.Token);
                    throw new Exception("Interrupted download was installed");
                }
                catch (OperationCanceledException) { }
                MotionTests.Equal("bad cache", File.ReadAllText(path));
                MotionTests.Equal(0, Directory.GetFiles(folder, "*.partial", SearchOption.AllDirectories).Length);
            }
            await store.EnsureAsync(FlowModel.Latest, null, default);
            MotionTests.Check(File.ReadAllBytes(path).SequenceEqual(handler.Bytes), "Cache was not repaired");
            using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
            try { await store.EnsureAsync(FlowModel.Base, null, cancellation.Token); throw new Exception("Cancellation ignored"); }
            catch (OperationCanceledException) { }
            handler.InvalidMetadata = true;
            try { await store.EnsureAsync(FlowModel.Base, null, default); throw new Exception("Invalid metadata was accepted"); }
            catch (InvalidDataException) { }
            Console.WriteLine("PASS: selected-model download, verified cache, corruption, atomic replacement and cancellation");
            await Perception(folder);
        }
        finally { if (Directory.Exists(folder)) Directory.Delete(folder, true); }
    }

    static async Task Perception(string folder)
    {
        using var handler = new PerceptionFiles(); using var http = new HttpClient(handler);
        var store = new ModelStore(http, folder);
        MotionTests.Equal(0, handler.Requests.Count);
        var file = Path.Combine(folder, "perception-fixture.pt");
        var hash = Convert.ToHexString(SHA256.HashData(handler.Bytes));
        const string fixtureUrl = "https://huggingface.co/test/resolve/pinned/file.pt";
        await store.EnsureFileAsync(fixtureUrl, file, handler.Bytes.Length, hash, null, default);
        MotionTests.Check(File.ReadAllBytes(file).SequenceEqual(handler.Bytes), "Perception stream changed bytes");
        var requests = handler.Requests.Count;
        await store.EnsureFileAsync(fixtureUrl, file, handler.Bytes.Length, hash, null, default);
        MotionTests.Equal(requests, handler.Requests.Count);
        File.WriteAllText(file, "existing file");
        try { await store.EnsureFileAsync(fixtureUrl, file, handler.Bytes.Length, new string('0', 64), null, default); throw new Exception("Bad mirror hash accepted"); }
        catch (InvalidDataException) { }
        MotionTests.Equal("existing file", File.ReadAllText(file));
        using (var cancelled = new CancellationTokenSource())
        {
            try
            {
                await store.EnsureFileAsync(fixtureUrl, file, handler.Bytes.Length, hash,
                    message => { if (message.Contains("100%")) cancelled.Cancel(); }, cancelled.Token);
                throw new Exception("Cancelled perception download was installed");
            }
            catch (OperationCanceledException) { }
            MotionTests.Equal("existing file", File.ReadAllText(file));
        }
        foreach (var model in new[] { PerceptionModel.Sam3dBody, PerceptionModel.VggtOmega, PerceptionModel.YoloXL })
        {
            handler.Requests.Clear();
            try { await store.EnsurePerceptionAsync(model, null, default); throw new Exception("Wrong mirror checkpoint accepted"); }
            catch (InvalidDataException) { }
            var sam = model == PerceptionModel.Sam3dBody;
            MotionTests.Equal(sam ? 3 : 2, handler.Requests.Count);
            var prefix = model switch
            {
                PerceptionModel.Sam3dBody => "/mingjiexie/sam-3d-body-dinov3/resolve/" + ModelStore.SamRevision,
                PerceptionModel.VggtOmega => "/1kaiser/vggt-omega-jax/resolve/" + ModelStore.OmegaRevision,
                _ => "/Megvii-BaseDetection/YOLOX/releases/download/" + ModelStore.YoloRelease
            };
            MotionTests.Check(handler.Requests[^1].AbsolutePath.StartsWith(prefix + "/"), "Mirror revision changed");
            MotionTests.Check(handler.Requests[0].AbsolutePath.EndsWith("/LICENSE"), "License was not downloaded first");
            if (sam) MotionTests.Check(handler.Requests[1].AbsolutePath.EndsWith("/model_config.yaml"), "SAM configuration missing");
        }
        handler.Requests.Clear();
        try { await store.EnsureSmplAsync(null, default); throw new Exception("Wrong SMPL-H asset accepted"); }
        catch (InvalidDataException) { }
        MotionTests.Equal(2, handler.Requests.Count);
        MotionTests.Check(handler.Requests[0].AbsolutePath.EndsWith("/smplh/LICENSE.txt"), "SMPL-H license was not downloaded first");
        MotionTests.Equal("/lithiumice/models_hub/resolve/" + ModelStore.SmplRevision + "/4_SMPLhub/smplh/neutral/model.npz", handler.Requests[1].AbsolutePath);
        handler.Requests.Clear(); handler.LargeText = true;
        try { await store.EnsurePerceptionAsync(PerceptionModel.Sam3dBody, null, default); throw new Exception("Oversized support file accepted"); }
        catch (HttpRequestException) { }
        MotionTests.Equal(1, handler.Requests.Count);
        handler.LargeText = false; handler.EmptyText = true;
        try { await store.EnsurePerceptionAsync(PerceptionModel.Sam3dBody, null, default); throw new Exception("Empty license accepted"); }
        catch (InvalidDataException) { }
        try { await store.EnsurePerceptionAsync((PerceptionModel)99, null, default); throw new Exception("Invalid model accepted"); }
        catch (ArgumentOutOfRangeException) { }
        MotionTests.Equal(0, Directory.GetFiles(folder, "*.partial", SearchOption.AllDirectories).Length);
        Console.WriteLine("PASS: explicit body/camera/detector downloads, pinned sources, support files, verified cache reuse and failed/cancelled replacement");
    }

    sealed class PerceptionFiles : HttpMessageHandler
    {
        internal readonly byte[] Bytes = Encoding.UTF8.GetBytes("perception fixture");
        internal readonly List<Uri> Requests = new();
        internal bool LargeText, EmptyText;
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token)
        {
            token.ThrowIfCancellationRequested(); Requests.Add(request.RequestUri);
            var text = request.RequestUri.AbsolutePath.EndsWith("/LICENSE") || request.RequestUri.AbsolutePath.EndsWith("/LICENSE.txt") || request.RequestUri.AbsolutePath.EndsWith(".yaml");
            var bytes = text ? Encoding.UTF8.GetBytes(EmptyText ? "" : LargeText ? new string('x', 65537) : "fixture support file") : Bytes;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) });
        }
    }

    sealed class Checkpoints : HttpMessageHandler
    {
        internal readonly byte[] Bytes = Encoding.UTF8.GetBytes("checkpoint fixture");
        internal int Downloads;
        internal bool Corrupt;
        internal bool InvalidMetadata;
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            var path = request.RequestUri!.AbsolutePath;
            MotionTests.Check(path.Contains(ModelStore.Revision), "Checkpoint revision is not pinned");
            HttpContent content;
            if (path.Contains("/raw/"))
                content = new StringContent(InvalidMetadata ? "not a checkpoint pointer" : $"version https://git-lfs.github.com/spec/v1\noid sha256:{Convert.ToHexString(SHA256.HashData(Bytes)).ToLowerInvariant()}\nsize {Bytes.Length}\n");
            else
            {
                Downloads++;
                content = new ByteArrayContent(Corrupt ? Enumerable.Repeat((byte)0, Bytes.Length).ToArray() : Bytes);
            }
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
        }
    }
}
