// run_csharp method body. Supply absolute paths in the three exchange-directory text files.
var samPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "sam-model-path.txt"), cancellationToken)).Trim();
var mhrPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "mhr-model-path.txt"), cancellationToken)).Trim();
var assemblyPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "gpu-checks-path.txt"), cancellationToken)).Trim();
if (!System.IO.Path.IsPathFullyQualified(samPath) || !System.IO.Path.IsPathFullyQualified(mhrPath) || !System.IO.Path.IsPathFullyQualified(assemblyPath))
    throw new ArgumentException("SAM validation requires absolute checkpoint and diagnostic assembly paths.");
var assembly = System.Reflection.Assembly.LoadFrom(assemblyPath);
var method = assembly.GetType("HumanoidMocapX.Checks.SamReleasedChecks", true).GetMethod("Run");
return await (Task<object>)method.Invoke(null, new object[] { samPath, mhrPath, cancellationToken });
