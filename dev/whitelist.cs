using Sandbox;

if (args.Length == 0)
    throw new ArgumentException("Pass runtime assembly paths in dependency order.");

using var access = new AccessControl();
foreach (var path in args)
{
    using var input = File.OpenRead(path);
    var result = access.VerifyAssembly(input, out _);
    Console.WriteLine($"{(result.Success ? "PASS" : "FAIL")}: s&box whitelist · {Path.GetFileName(path)}");
    foreach (var error in result.Errors) Console.Error.WriteLine(error);
    foreach (var error in result.WhitelistErrors)
    {
        Console.Error.WriteLine(error.Name);
        foreach (var location in error.Locations) Console.Error.WriteLine("  " + location.Text);
    }
    if (!result.Success) return 1;
}
return 0;
