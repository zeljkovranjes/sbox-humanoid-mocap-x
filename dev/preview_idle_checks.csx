// run_csharp method body. Place a supported clip at exchangeDirectory/preview-test.webm.
// Loads source preview in the X window; does not start capture.
var source = System.IO.Path.Combine(exchangeDirectory, "preview-test.webm");
if (!System.IO.File.Exists(source)) throw new System.IO.FileNotFoundException("Copy a preview fixture into the session directory.", source);
var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public;
var entry = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("HumanoidMocapX.EditorTools.MocapWindow")).First(t => t is not null);
entry.GetMethod("Open").Invoke(null, null);
var window = (Editor.Widget)entry.GetField("window", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static).GetValue(null);
var workspace = window.GetDescendants<Editor.Widget>().First(w => w.GetType().Name == "RetargetWindow");
var type = workspace.GetType();
if ((bool)type.GetProperty("CaptureIsRunning", flags).GetValue(workspace)) throw new Exception("Wait for active capture before running the idle-preview check.");
type.GetMethod("LoadVideo").Invoke(workspace, new object[] { source, null });
var indicator = type.GetField("_loader", flags).GetValue(workspace);
if ((bool)indicator.GetType().GetProperty("Busy").GetValue(indicator)) throw new Exception("Idle source preview incorrectly shows a processing spinner.");
var provider = type.GetField("_bodyCapture", flags).GetValue(workspace);
var reason = (string)provider.GetType().GetProperty("UnavailableReason").GetValue(provider);
var message = (string)type.GetProperty("CaptureMessage", flags).GetValue(workspace);
if (reason is not null && reason != message) throw new Exception("Idle preview hid the provider's unavailable message.");
return new { Passed = true, Busy = false, Message = message };
