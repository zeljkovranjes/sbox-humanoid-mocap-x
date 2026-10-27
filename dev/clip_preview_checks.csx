// Opens a saved full-body capture in the actual X workspace. Use a preview-only
// document copy with a supported source codec when the platform cannot play H.264.
var path = (await System.IO.File.ReadAllTextAsync(System.IO.Path.Combine(exchangeDirectory, "clip-preview-path.txt"), cancellationToken)).Trim();
if (!System.IO.File.Exists(path)) throw new System.IO.FileNotFoundException("Capture document missing", path);
var entry = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("HumanoidMocapX.EditorTools.MocapWindow")).First(t => t is not null);
entry.GetMethod("Open").Invoke(null, null);
var dialog = (Editor.Widget)entry.GetField("window", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static).GetValue(null);
var workspace = dialog.GetDescendants<Editor.Widget>().OfType<HumanoidMocap.EditorTools.RetargetWindow>().Single();
await workspace.LoadMotionAsync(path);
return new { Loaded = path, Workspace = workspace.Name };
