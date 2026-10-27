using System.Security.Cryptography;
using HumanoidMocapX;
using HumanoidMocapX.EditorTools;

static class InstalledTests
{
    internal static async Task Run()
    {
        var directory = Path.Combine(Path.GetTempPath(), "mocap-installed-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var path = Path.Combine(directory, "model.bin"); var bytes = new byte[] { 10, 20, 30 };
            var hash = Convert.ToHexString(SHA256.HashData(bytes));
            await VideoFramesTests.Reject<InvalidDataException>(() => ModelStore.VerifyInstalledAsync(path, bytes.Length, hash, null, default));
            await File.WriteAllBytesAsync(path, bytes);
            await ModelStore.VerifyInstalledAsync(path, bytes.Length, hash, null, default);
            await VideoFramesTests.Reject<InvalidDataException>(() => ModelStore.VerifyInstalledAsync(path, 4, hash, null, default));
            await VideoFramesTests.Reject<InvalidDataException>(() => ModelStore.VerifyInstalledAsync(path, 3, new string('0', 64), null, default));
            using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
            await VideoFramesTests.Reject<OperationCanceledException>(() => ModelStore.VerifyInstalledAsync(path, 3, hash, null, cancelled.Token));
            await VideoFramesTests.Reject<FileNotFoundException>(() => ModelStore.InstalledAsync(directory, FlowModel.Latest, null, default));
            MotionTests.Check(File.ReadAllBytes(path).SequenceEqual(bytes), "Installed verification changed a model");
            MotionTests.Equal(1, Directory.GetFiles(directory).Length);
            // Presence check that keeps capture locked: names the first absent model, in capture order.
            MotionTests.Equal("FlowHMR Latest", ModelStore.Missing(directory, FlowModel.Latest));
            var latest = ModelStore.Paths(directory, FlowModel.Latest);
            foreach (var file in new[] { latest.Flow, latest.Sam, latest.Mhr, latest.Omega, latest.Detector })
            { Directory.CreateDirectory(Path.GetDirectoryName(file)!); File.WriteAllBytes(file, []); }
            MotionTests.Equal("SMPL-H neutral", ModelStore.Missing(directory, FlowModel.Latest));
            Directory.CreateDirectory(Path.GetDirectoryName(latest.Smpl)!); File.WriteAllBytes(latest.Smpl, []);
            MotionTests.Equal<string>(null, ModelStore.Missing(directory, FlowModel.Latest));
            MotionTests.Equal("FlowHMR Base", ModelStore.Missing(directory, FlowModel.Base));
            Console.WriteLine("PASS: offline model verification, missing/corrupt weights and cancellation without file replacement");
        }
        finally { Directory.Delete(directory, true); }
    }
}
