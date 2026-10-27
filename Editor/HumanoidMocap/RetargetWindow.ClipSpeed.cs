using System;
using System.Globalization;
using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

/// <summary>Playback and export speed of the animation; Mocap adjustments → Animation speed changes it.</summary>
public sealed partial class RetargetWindow
{
    /// <summary>Playback and export speed of the animation (1 = the video's own).</summary>
    float _animationSpeed=1;
    LineEdit _speedField;
    internal float GateAnimationSpeed=>_animationSpeed;

    /// <summary>A new video starts at its own speed.</summary>
    void ResetClipSpeed()
    {
        _animationSpeed=1;
        if(_speedField.IsValid())_speedField.Text="100";
    }

    void SetAnimationSpeed(float speed)
    {
        _animationSpeed=Math.Clamp(speed,.1f,2f);
        if(_speedField.IsValid())_speedField.Text=(_animationSpeed*100).ToString("0",CultureInfo.InvariantCulture);
    }

    /// <summary>Advanced → Animation speed, in percent.</summary>
    void AnimationSpeedEdited()
    {
        if(float.TryParse(_speedField.Text.TrimEnd('%',' '),NumberStyles.Float,CultureInfo.InvariantCulture,out var percent)&&percent>=10&&percent<=200)
        {
            SetAnimationSpeed(percent/100);
            _captureStatus.Text=Math.Abs(_animationSpeed-1)<1e-3f?"Animation plays at the video's speed.":FormattableString.Invariant($"Animation plays and exports at {_animationSpeed*100:0}% speed.");
        }
        else _speedField.Text=(_animationSpeed*100).ToString("0",CultureInfo.InvariantCulture);
    }
}
