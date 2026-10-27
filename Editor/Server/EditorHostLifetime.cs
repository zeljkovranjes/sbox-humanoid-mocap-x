using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Editor;
using Sandbox;

namespace HumanoidMocapHost;

/// <summary>Connects a host session to public editor events. Hotload stops the old
/// listener; a replacement must wait for its GPU work to drain before restarting.</summary>
internal sealed class EditorHostLifetime : IHotloadManaged
{
    HostSession session;
    GpuAdmission admission;
    Task shutdown;
    DateTime sampled;
    internal bool IsRunning => shutdown is null && session is not null && !session.IsStopping;

    internal EditorHostLifetime(HostSession session, GpuAdmission admission)
    {
        this.session = session; this.admission = admission;
        EditorEvent.Register(this); SampleMemory();
    }

    [EditorEvent.Frame]
    void SampleMemory()
    {
        if (!IsRunning || DateTime.UtcNow - sampled < TimeSpan.FromSeconds(1)) return;
        sampled = DateTime.UtcNow;
        admission.Update(Graphics.VideoMemoryBudget, Graphics.VideoMemoryUsed, Sandbox.Engine.SystemInfo.GpuMemory);
    }

    internal Task StopAsync()
    {
        if (shutdown is not null) return shutdown;
        EditorEvent.Unregister(this);
        return shutdown = session?.StopAsync() ?? Task.CompletedTask;
    }

    public void Destroyed(Dictionary<string, object> state) => state["shutdown"] = StopAsync();

    public void Created(IReadOnlyDictionary<string, object> state)
    {
        session = null; admission = null;
        shutdown = state.TryGetValue("shutdown", out var task) ? (Task)task : Task.CompletedTask;
    }

    public void Failed() => _ = StopAsync();
}
