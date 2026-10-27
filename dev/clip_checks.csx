// Supply absolute paths in gpu-checks-path.txt, clip-fixture-path.txt and clip-results-path.txt.
var assemblyPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "gpu-checks-path.txt"), cancellationToken)).Trim();
var fixturePath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "clip-fixture-path.txt"), cancellationToken)).Trim();
var resultsPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "clip-results-path.txt"), cancellationToken)).Trim();
var assembly = System.Reflection.Assembly.LoadFrom(assemblyPath);
return await (Task<object>)assembly.GetType("HumanoidMocapX.Checks.ClipChecks", true).GetMethod("Run").Invoke(null, new object[] { fixturePath, resultsPath, cancellationToken });
