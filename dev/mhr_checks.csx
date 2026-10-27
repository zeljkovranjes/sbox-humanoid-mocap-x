// run_csharp method body. Put the installed MHR path in exchangeDirectory/mhr-model-path.txt.
var checkpointPath = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "mhr-model-path.txt"), cancellationToken)).Trim();
if (!System.IO.Path.IsPathFullyQualified(checkpointPath)) throw new ArgumentException("The MHR checkpoint path must be absolute.");
var flags = System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
var checkpointType = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("HumanoidMocapX.EditorTools.MhrCheckpoint")).First(t => t is not null);
return await Task.Run(() =>
{
    var body = checkpointType.GetMethod("LoadBody", flags).Invoke(null, new object[] { checkpointPath, cancellationToken });
    var evaluate = (Action<ReadOnlySpan<float>, ReadOnlySpan<float>, ReadOnlySpan<float>, Span<float>, Span<float>, CancellationToken>)body.GetType().GetMethod("Run", flags)
        .CreateDelegate(typeof(Action<ReadOnlySpan<float>, ReadOnlySpan<float>, ReadOnlySpan<float>, Span<float>, Span<float>, CancellationToken>), body);
    using var checkpoint = HumanoidMocap.EditorTools.Inference.TorchCheckpoint.OpenScriptData(checkpointPath, "__torch__.MHRDemo");
    var rest = checkpoint.ReadFloat("character_torch.blend_shape.base_shape", cancellationToken);
    var parameters = new float[204]; var identity = new float[45]; var expression = new float[72]; var states = new float[127 * 8]; var vertices = new float[rest.Length];
    evaluate(identity, parameters, expression, vertices, states, cancellationToken);
    double maximumError = 0;
    for (var i = 0; i < vertices.Length; i++)
    {
        if (!float.IsFinite(vertices[i])) throw new Exception("MHR skinning produced a non-finite vertex.");
        maximumError = Math.Max(maximumError, Math.Abs(vertices[i] - (double)rest[i]));
    }
    if (maximumError > 1e-4) throw new Exception("Neutral MHR skinning failed to reproduce its saved mesh: " + maximumError);
    for (var i = 0; i < parameters.Length; i++) parameters[i] = .02f * MathF.Sin(i * .37f);
    identity[0] = .2f; expression[0] = .1f;
    evaluate(identity, parameters, expression, vertices, states, cancellationToken);
    if (!vertices.All(float.IsFinite)) throw new Exception("Posed MHR body contains non-finite vertices.");
    var previous = vertices.ToArray();
    evaluate(identity, parameters, expression, vertices, states, cancellationToken);
    if (!previous.SequenceEqual(vertices)) throw new Exception("Repeated MHR body inference changed output.");
    return new { Passed = true, CompleteBodyModel = true, Joints = 127, Vertices = rest.Length / 3, MaximumNeutralMeshError = maximumError, RepeatedPoseMatches = true };
}, cancellationToken);
