using System.Linq;

namespace HumanoidMocapX.EditorTools;

/// <summary>The processing machine choice: an SSH address (user@host or a <c>~/.ssh/config</c>
/// alias) and the Linux worker's path there. Key-based sign-in must already work, because ssh is
/// never allowed to prompt.</summary>
internal sealed class SshSettings
{
    internal const string DefaultWorker = "~/humanoid-mocap-worker/humanoid-mocap-worker";
    internal bool Enabled;
    internal string Address = "", Worker = DefaultWorker;

    /// <summary>Why the address or worker path cannot be used, or null. The address may not be
    /// empty, start with '-' (ssh would read it as an option) or contain spaces or control characters.</summary>
    internal string Problem
        => Address.Length == 0 ? "Enter the processing machine's address, for example user@192.168.1.20."
        : Address[0] == '-' || Address.Any(c => char.IsWhiteSpace(c) || char.IsControl(c)) ? "Enter the processing machine's address as user@host, without spaces."
        : Worker.Trim().Length == 0 || Worker.Any(char.IsControl) ? "Enter the worker's path on the processing machine."
        : null;
}
