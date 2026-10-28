using System;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;

namespace HumanoidMocapX.Core.Motion;
using Vector3 = System.Numerics.Vector3;

/// <summary>Evens out fast spins. On the breakin clip's one-handed spin the capture turns the body a steady 39°
/// a frame on average, but frame to frame anywhere from 17° to 88°, which jerked the free hand up to 91 cm
/// between frames. Where the heading (the hip line on the floor, defined upright or upside down) turns faster than
/// <see cref="MinDegreesPerSecond"/> for at least <see cref="MinSeconds"/>, it is smoothed over
/// <see cref="SigmaFrames"/> and the body turned about the vertical through the hips to follow; the pose itself
/// is untouched. Slower turns are left as captured.</summary>
public static class CaptureSpin
{
    public const float MinDegreesPerSecond = 300, MinSeconds = .5f, SigmaFrames = 2;
    /// <summary>The hip line must lie at least this share along the floor for its heading to mean anything: through the
    /// breakin clip's windmill it stood near vertical (27–35% along the floor) and its heading jumped 158° in two frames.</summary>
    public const float MinHorizontalShare = .7f;
    /// <summary>No frame is turned further than this.</summary>
    public const float MaxTurnDegrees = 25;

    /// <summary>Turns the root of each spinning frame of a Y-up <paramref name="document"/> whose first bone is
    /// the root. Returns the number of frames turned; 0 when the document is not Y-up, lacks the upper legs or
    /// has fewer than three frames.</summary>
    public static int Apply( MotionDocument document )
    {
        var frames = document.Frames.Count;
        if ( frames < 3 || !document.Axes.Contains( "y-up" ) || document.Bones[0].Parent >= 0 ) return 0;
        int left = document.Bones.FindIndex( b => b.Role == BoneRole.UpperLegL ), right = document.Bones.FindIndex( b => b.Role == BoneRole.UpperLegR );
        if ( left < 0 || right < 0 ) return 0;
        var fps = (frames - 1) / Math.Max( 1e-6, document.Frames[^1].Time - document.Frames[0].Time );
        var heading = new double[frames]; var valid = new bool[frames];
        for ( var f = 0; f < frames; f++ )
        {
            var line = Position( document, document.Frames[f], right ) - Position( document, document.Frames[f], left );
            valid[f] = MathF.Sqrt( line.X * line.X + line.Z * line.Z ) >= MinHorizontalShare * line.Length();
            // Positive about +Y: turning (1, 0, 0) by a about +Y gives (cos a, 0, -sin a).
            heading[f] = Math.Atan2( -line.Z, line.X );
        }
        if ( !Unwrap( heading, valid ) ) return 0;
        var spinning = Spinning( heading, fps );
        var turned = 0;
        var radius = (int)Math.Ceiling( 3 * SigmaFrames );
        for ( var f = 0; f < frames; f++ )
        {
            if ( spinning[f] <= 0 || !valid[f] ) continue;
            double sum = 0, total = 0;
            for ( var j = Math.Max( 0, f - radius ); j <= Math.Min( frames - 1, f + radius ); j++ )
            {
                if ( !valid[j] ) continue;
                var g = Math.Exp( -.5 * (j - f) * (j - f) / (SigmaFrames * SigmaFrames) ); sum += g * heading[j]; total += g;
            }
            var limit = MaxTurnDegrees * Math.PI / 180;
            var turn = (float)(Math.Clamp( sum / total - heading[f], -limit, limit ) * spinning[f]);
            var root = document.Frames[f].Rotations[0];
            var rotation = Quaternion.Normalize( Quaternion.CreateFromAxisAngle( Vector3.UnitY, turn ) * MotionDocument.Q( root ) );
            document.Frames[f].Rotations[0] = new[] { rotation.X, rotation.Y, rotation.Z, rotation.W };
            turned++;
        }
        return turned;
    }

    /// <summary>Unwraps the heading across the frames where it means something and fills the rest by straight lines
    /// between them. Returns false when no frame's heading means anything.</summary>
    static bool Unwrap( double[] heading, bool[] valid )
    {
        var previous = -1;
        for ( var f = 0; f < heading.Length; f++ )
        {
            if ( !valid[f] ) continue;
            if ( previous >= 0 )
            {
                heading[f] = heading[previous] + Math.IEEERemainder( heading[f] - heading[previous], 2 * Math.PI );
                for ( var i = previous + 1; i < f; i++ ) heading[i] = heading[previous] + (heading[f] - heading[previous]) * (i - previous) / (f - previous);
            }
            else for ( var i = 0; i < f; i++ ) heading[i] = heading[f];
            previous = f;
        }
        if ( previous < 0 ) return false;
        for ( var i = previous + 1; i < heading.Length; i++ ) heading[i] = heading[previous];
        return true;
    }

    /// <summary>How much of each frame's smoothing to apply: 1 inside a spin, easing to 0 over a few frames at its
    /// ends so the turn joins the rest of the clip without a step.</summary>
    static double[] Spinning( double[] heading, double fps )
    {
        var frames = heading.Length; var fast = new bool[frames];
        var threshold = MinDegreesPerSecond * Math.PI / 180 / fps;
        // The mean rate over a short window, so one slow frame inside a spin does not split it.
        var window = Math.Max( 1, (int)Math.Round( fps / 10 ) );
        for ( var f = 0; f < frames; f++ )
        {
            int a = Math.Max( 0, f - window ), b = Math.Min( frames - 1, f + window );
            fast[f] = Math.Abs( heading[b] - heading[a] ) / Math.Max( 1, b - a ) > threshold;
        }
        var weight = new double[frames]; var minimum = (int)Math.Ceiling( MinSeconds * fps ); var ease = (int)Math.Ceiling( SigmaFrames * 2 );
        for ( var f = 0; f < frames; )
        {
            if ( !fast[f] ) { f++; continue; }
            var e = f; while ( e + 1 < frames && fast[e + 1] ) e++;
            if ( e - f + 1 >= minimum )
                for ( var i = Math.Max( 0, f - ease ); i <= Math.Min( frames - 1, e + ease ); i++ )
                {
                    var outside = i < f ? f - i : i > e ? i - e : 0;
                    var w = 1 - outside / (double)(ease + 1); w = w * w * (3 - 2 * w);
                    weight[i] = Math.Max( weight[i], w );
                }
            f = e + 1;
        }
        return weight;
    }

    static Vector3 Position( MotionDocument document, MotionFrame frame, int bone )
    {
        // From the bone up to the root: each ancestor turns what lies below it and adds its own offset.
        var position = MotionDocument.V( frame.Positions[bone] );
        for ( var b = document.Bones[bone].Parent; b >= 0; b = document.Bones[b].Parent )
            position = MotionDocument.V( frame.Positions[b] ) + Vector3.Transform( position, MotionDocument.Q( frame.Rotations[b] ) );
        return position;
    }
}
