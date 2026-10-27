namespace HumanoidMocapX.EditorTools;

/// <summary>Verified installed model paths. Construction does not download files.</summary>
internal sealed record CaptureModels(string Flow, string Sam, string Mhr, string Omega, string Detector, string Smpl);
