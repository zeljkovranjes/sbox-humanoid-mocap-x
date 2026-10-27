// run_csharp method body. Reuse the model/assembly path files from sam_released_checks.csx.
var samPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "sam-model-path.txt"), cancellationToken)).Trim();
var mhrPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "mhr-model-path.txt"), cancellationToken)).Trim();
var assemblyPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "gpu-checks-path.txt"), cancellationToken)).Trim();
using var frame = System.Text.Json.JsonDocument.Parse(await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "sam-frame.json"), cancellationToken));
var detector = frame.RootElement.GetProperty("DetectorPath").GetString();
var rgba = frame.RootElement.GetProperty("RgbaPath").GetString();
var width = frame.RootElement.GetProperty("Width").GetInt32(); var height = frame.RootElement.GetProperty("Height").GetInt32();
if (!new[] { samPath, mhrPath, assemblyPath, detector, rgba }.All(System.IO.Path.IsPathFullyQualified))
    throw new ArgumentException("SAM frame validation requires absolute model, assembly and frame paths.");
var assembly = System.Reflection.Assembly.LoadFrom(assemblyPath);
var method = assembly.GetType("HumanoidMocapX.Checks.SamFrameChecks", true).GetMethod("Run");
return await (Task<object>)method.Invoke(null, new object[] { samPath, mhrPath, detector, rgba, width, height,
    System.IO.Path.Combine(exchangeDirectory, "sam-frame-result.json"), cancellationToken });
