// run_csharp method body. Model/assembly paths use the same files as sam_released_checks.csx.
var samPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "sam-model-path.txt"), cancellationToken)).Trim();
var mhrPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "mhr-model-path.txt"), cancellationToken)).Trim();
var assemblyPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "gpu-checks-path.txt"), cancellationToken)).Trim();
var detectorPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "detector-model-path.txt"), cancellationToken)).Trim();
var assembly = System.Reflection.Assembly.LoadFrom(assemblyPath);
var method = assembly.GetType("HumanoidMocapX.Checks.SamFeatureChecks", true).GetMethod("Run");
return await (Task<object>)method.Invoke(null, new object[] { samPath, mhrPath, detectorPath,
    System.IO.Path.Combine(exchangeDirectory, "sam-feature-frames.json"), System.IO.Path.Combine(exchangeDirectory, "sam-features-result.json"), cancellationToken });
