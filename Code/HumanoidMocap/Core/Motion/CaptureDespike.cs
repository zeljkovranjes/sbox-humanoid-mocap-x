using System;
using System.Numerics;

namespace HumanoidMocapX.Core.Motion;
using Vector3 = System.Numerics.Vector3;

/// <summary>Removes snaps: frames where a joint leaves the smooth path through its neighbours. On an ordinary clip
/// FlowHMR stays within 1.5 cm of that path in 94% of frames; on the breakin clip's upside-down breakdance only 42%
/// do, with jumps of up to 39 cm. Each joint's rotation (and the root's position) is compared with a cubic fitted over
/// <see cref="Window"/> frames around it (Savitzky-Golay); a frame further off than <see cref="StartDegrees"/> (or
/// <see cref="StartCm"/>) is pulled onto the fit, entirely from <see cref="FullDegrees"/> (<see cref="FullCm"/>); the
/// first and last three frames are left as captured.
/// Steady fast motion, a spin included, follows a cubic and is left alone. Contacts are applied later, during
/// retargeting, so planted feet and hands keep their hold.</summary>
public static class CaptureDespike
{
    public const int Window = 7, Passes = 2;
    public const float StartDegrees = 3, FullDegrees = 8, StartCm = 1, FullCm = 3;

    /// <summary>Despikes every bone's rotation and the root's position in place. Returns the number of corrected
    /// samples; 0 for fewer than <see cref="Window"/> frames.</summary>
    public static int Apply( MotionDocument document )
    {
        var frames = document.Frames.Count;
        if ( frames < Window ) return 0;
        var weights = Weights();
        var corrected = 0;
        for ( var pass = 0; pass < Passes; pass++ )
        {
            for ( var bone = 0; bone < document.Bones.Count; bone++ ) corrected += Rotations( document, bone, weights );
            corrected += Root( document, weights );
        }
        return corrected;
    }

    /// <summary>Savitzky-Golay weights for the centre of a cubic fitted over <see cref="Window"/> samples.</summary>
    static double[] Weights()
    {
        // Closed form for a cubic (and quadratic) centre value: (-2, 3, 6, 7, 6, 3, -2) / 21 for seven samples.
        return Window == 7 ? new[] { -2 / 21d, 3 / 21d, 6 / 21d, 7 / 21d, 6 / 21d, 3 / 21d, -2 / 21d } : throw new NotSupportedException();
    }

    static int Rotations( MotionDocument document, int bone, double[] weights )
    {
        var frames = document.Frames.Count; var half = Window / 2;
        // Sign-continuous so neighbouring quaternions average sensibly.
        var track = new Quaternion[frames];
        for ( var f = 0; f < frames; f++ )
        {
            track[f] = MotionDocument.Q( document.Frames[f].Rotations[bone] );
            if ( f > 0 && Quaternion.Dot( track[f], track[f - 1] ) < 0 ) track[f] = Quaternion.Negate( track[f] );
        }
        var count = 0;
        // Only frames with a whole window: repeating an end frame bends steady fast motion off its path.
        for ( var f = half; f < frames - half; f++ )
        {
            var fit = Quaternion.Zero;
            for ( var k = -half; k <= half; k++ )
            {
                var q = track[f + k]; var w = (float)weights[k + half];
                fit = new Quaternion( fit.X + q.X * w, fit.Y + q.Y * w, fit.Z + q.Z * w, fit.W + q.W * w );
            }
            if ( fit.LengthSquared() < 1e-12f ) continue;
            fit = Quaternion.Normalize( fit );
            var off = 2 * MathF.Acos( Math.Clamp( MathF.Abs( Quaternion.Dot( fit, track[f] ) ), 0, 1 ) ) * 180 / MathF.PI;
            var pull = Math.Clamp( (off - StartDegrees) / (FullDegrees - StartDegrees), 0, 1 );
            if ( pull <= 0 ) continue;
            var result = Quaternion.Normalize( Quaternion.Slerp( track[f], fit, pull ) );
            document.Frames[f].Rotations[bone] = new[] { result.X, result.Y, result.Z, result.W };
            count++;
        }
        return count;
    }

    static int Root( MotionDocument document, double[] weights )
    {
        var frames = document.Frames.Count; var half = Window / 2;
        var track = new Vector3[frames];
        for ( var f = 0; f < frames; f++ ) track[f] = MotionDocument.V( document.Frames[f].Positions[0] );
        var count = 0;
        for ( var f = half; f < frames - half; f++ )
        {
            var fit = Vector3.Zero;
            for ( var k = -half; k <= half; k++ ) fit += track[f + k] * (float)weights[k + half];
            var off = Vector3.Distance( fit, track[f] ) * 100;
            var pull = Math.Clamp( (off - StartCm) / (FullCm - StartCm), 0, 1 );
            if ( pull <= 0 ) continue;
            var result = Vector3.Lerp( track[f], fit, pull );
            document.Frames[f].Positions[0] = new[] { result.X, result.Y, result.Z };
            count++;
        }
        return count;
    }
}
