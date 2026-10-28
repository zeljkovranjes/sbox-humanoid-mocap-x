using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using HumanoidMocapX.Core.Cleanup;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Target;

namespace HumanoidMocapX.Core.Motion;
using Vector3 = System.Numerics.Vector3;

/// <summary>Removes snaps: frames where a joint leaves the smooth path through its neighbours. On an ordinary clip
/// FlowHMR stays within 1.5 cm of that path in 94% of frames; on the breakin clip's upside-down breakdance only 42%
/// do, with jumps of up to 39 cm. Each joint's rotation (and the root's position) is compared with a cubic fitted over
/// <see cref="Window"/> frames around it (Savitzky-Golay); a frame further off than <see cref="StartDegrees"/> (or
/// <see cref="StartCm"/>) is pulled onto the fit, entirely from <see cref="FullDegrees"/> (<see cref="FullCm"/>); the
/// first and last three frames are left as captured.
/// Steady fast motion, a spin included, follows a cubic and is left alone. It runs on the capture, before contacts are
/// applied, and again on the retargeted animation, so no cleanup step leaves a snap behind either.</summary>
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
        var corrected = 0; var changed = new bool[frames];
        var rotations = new Quaternion[frames]; var positions = new Vector3[frames];
        for ( var pass = 0; pass < Passes; pass++ )
        {
            for ( var bone = 0; bone < document.Bones.Count; bone++ )
            {
                for ( var f = 0; f < frames; f++ ) rotations[f] = MotionDocument.Q( document.Frames[f].Rotations[bone] );
                corrected += Rotations( rotations, changed );
                for ( var f = 0; f < frames; f++ ) if ( changed[f] ) document.Frames[f].Rotations[bone] = new[] { rotations[f].X, rotations[f].Y, rotations[f].Z, rotations[f].W };
            }
            for ( var f = 0; f < frames; f++ ) positions[f] = MotionDocument.V( document.Frames[f].Positions[0] );
            corrected += Positions( positions, changed, 100 );
            for ( var f = 0; f < frames; f++ ) if ( changed[f] ) document.Frames[f].Positions[0] = new[] { positions[f].X, positions[f].Y, positions[f].Z };
        }
        return corrected;
    }

    /// <summary>A hand or foot moving less than this, in centimetres a frame, is held where it was by
    /// <see cref="Apply(List{XForm[]}, TargetRig, float)"/>; the hold eases in and out over <see cref="EaseFrames"/>.</summary>
    public const float StillCm = .5f;
    public const int EaseFrames = 3;

    /// <summary>Despikes a retargeted animation, then puts every hand and foot that was still before (planted, held or
    /// locked by the cleanup) back where it was: smoothing a spin's arm after its hand was planted slid the hand up to
    /// 9 cm a frame. Returns the number of corrected samples; 0 for fewer than <see cref="Window"/> frames.</summary>
    public static int Apply( List<XForm[]> frames, TargetRig target, float unitsPerCm )
    {
        if ( frames.Count < Window ) return 0;
        var rig = target.Skeleton;
        var chains = new[] { (BoneRole.UpperArmL, BoneRole.LowerArmL, BoneRole.HandL), (BoneRole.UpperArmR, BoneRole.LowerArmR, BoneRole.HandR),
                (BoneRole.UpperLegL, BoneRole.LowerLegL, BoneRole.FootL), (BoneRole.UpperLegR, BoneRole.LowerLegR, BoneRole.FootR) }
            .Select( c => target.BoneForRole( c.Item1 ) is int a && target.BoneForRole( c.Item2 ) is int b && target.BoneForRole( c.Item3 ) is int e ? (A: a, B: b, C: e) : ((int A, int B, int C)?)null )
            .Where( c => c is not null ).Select( c => c!.Value ).ToArray();
        // Where each hand and foot is held (still, as the cleanup planted or locked it), and how firmly: 1 while held,
        // easing out over EaseFrames on either side so a hold ends without a step.
        var world = new XForm[rig.Count];
        var held = chains.Select( _ => new XForm[frames.Count] ).ToArray();
        for ( var f = 0; f < frames.Count; f++ )
        {
            FkUtil.ToWorld( frames[f], rig, world );
            for ( var c = 0; c < chains.Length; c++ ) held[c][f] = world[chains[c].C];
        }
        var still = StillCm * unitsPerCm;
        var holding = held.Select( track => Holding( track.Select( x => x.Pos ).ToArray(), still ) ).ToArray();
        // Toes too: through toe-off the toe stays planted while the ankle lifts, so it is held on its own.
        var toes = new[] { BoneRole.ToeL, BoneRole.ToeR }.Select( target.BoneForRole ).Where( b => b is int t && rig[t].ParentIndex >= 0 ).Select( b => b!.Value ).ToArray();
        var toesHeld = toes.Select( _ => new Vector3[frames.Count] ).ToArray();
        for ( var f = 0; f < frames.Count; f++ )
        {
            FkUtil.ToWorld( frames[f], rig, world );
            for ( var t = 0; t < toes.Length; t++ ) toesHeld[t][f] = world[toes[t]].Pos;
        }
        var toesHolding = toesHeld.Select( track => Holding( track, still ) ).ToArray();
        // Three rounds: each world-space reach can leave a joint above it slightly off its path for the next round to take out.
        var corrected = 0;
        for ( var round = 0; round < 3; round++ ) corrected += Settle( frames, target, unitsPerCm, chains, held, holding, toes, toesHeld, toesHolding );
        return corrected;
    }

    /// <summary>How firmly a joint following <paramref name="path"/> is held per frame: 1 where it moves less than
    /// <paramref name="still"/> to either neighbour, easing out over EaseFrames on either side so a hold ends without a step.</summary>
    static float[] Holding( Vector3[] path, float still )
    {
        var weight = new float[path.Length];
        for ( var f = 1; f + 1 < path.Length; f++ )
        {
            if ( Vector3.Distance( path[f], path[f - 1] ) >= still || Vector3.Distance( path[f + 1], path[f] ) >= still ) continue;
            for ( var j = Math.Max( 0, f - EaseFrames ); j <= Math.Min( path.Length - 1, f + EaseFrames ); j++ )
            {
                var w = 1 - Math.Abs( j - f ) / (float)(EaseFrames + 1); w = w * w * (3 - 2 * w);
                weight[j] = Math.Max( weight[j], w );
            }
        }
        return weight;
    }

    static int Settle( List<XForm[]> frames, TargetRig target, float unitsPerCm, (int A, int B, int C)[] chains, XForm[][] before, float[][] holding,
        int[] toes, Vector3[][] toesHeld, float[][] toesHolding )
    {
        var rig = target.Skeleton; var world = new XForm[rig.Count];
        var corrected = Apply( frames, rig, unitsPerCm );
        if ( corrected == 0 ) return 0;
        // Each held hand or foot is put back where it was held.
        for ( var c = 0; c < chains.Length; c++ )
        {
            var (a, b, e) = chains[c];
            for ( var f = 0; f < frames.Count; f++ )
            {
                var w = holding[c][f]; if ( w <= 0 ) continue;
                FkUtil.ToWorld( frames[f], rig, world );
                var goal = Vector3.Lerp( world[e].Pos, before[c][f].Pos, w );
                var attitude = Quaternion.Normalize( Quaternion.Slerp( world[e].Rot, before[c][f].Rot, w ) );
                var bend = Vector3.Cross( world[b].Pos - world[a].Pos, world[e].Pos - world[b].Pos );
                if ( bend.LengthSquared() < 1e-8f ) bend = Vector3.Transform( Vector3.UnitX, world[a].Rot );
                var ik = TwoBoneIk.Solve( world[a].Pos, world[b].Pos, world[e].Pos, goal, soften: 0, stableBendAxis: bend );
                EffectorIk.ApplyWorldDeltas( frames[f], rig, a, b, e, ik.UpperWorldDelta, ik.LowerWorldDelta, world );
                FkUtil.ToWorld( frames[f], rig, world );
                var parent = rig[e].ParentIndex;
                frames[f][e].Rot = Quaternion.Normalize( (parent < 0 ? Quaternion.Identity : Quaternion.Inverse( world[parent].Rot )) * attitude );
            }
        }
        // Joint turns that each follow a smooth path can still add up to a flick at the hand or foot (a dancer's foot
        // doubled back 5 cm at the top of a swing): each limb's end is despiked in the world too, its limb reaching
        // for the fitted place with the end's attitude kept.
        var path = new Vector3[frames.Count]; var changed = new bool[frames.Count];
        for ( var pass = 0; pass < Passes; pass++ )
            for ( var c = 0; c < chains.Length; c++ )
            {
                var (a, b, e) = chains[c];
                for ( var f = 0; f < frames.Count; f++ ) { FkUtil.ToWorld( frames[f], rig, world ); path[f] = world[e].Pos; }
                corrected += Positions( path, changed, 1 / unitsPerCm );
                for ( var f = 0; f < frames.Count; f++ )
                {
                    // A hand or foot that is held stays where its hold put it.
                    if ( !changed[f] || holding[c][f] >= 1 ) continue;
                    FkUtil.ToWorld( frames[f], rig, world );
                    var attitude = world[e].Rot;
                    var bend = Vector3.Cross( world[b].Pos - world[a].Pos, world[e].Pos - world[b].Pos );
                    if ( bend.LengthSquared() < 1e-8f ) bend = Vector3.Transform( Vector3.UnitX, world[a].Rot );
                    var ik = TwoBoneIk.Solve( world[a].Pos, world[b].Pos, world[e].Pos, path[f], soften: 0, stableBendAxis: bend );
                    EffectorIk.ApplyWorldDeltas( frames[f], rig, a, b, e, ik.UpperWorldDelta, ik.LowerWorldDelta, world );
                    FkUtil.ToWorld( frames[f], rig, world );
                    var parent = rig[e].ParentIndex;
                    frames[f][e].Rot = Quaternion.Normalize( (parent < 0 ? Quaternion.Identity : Quaternion.Inverse( world[parent].Rot )) * attitude );
                }
                // The elbow or knee, by turning the limb about the line from its root to its end, which stays put.
                for ( var f = 0; f < frames.Count; f++ ) { FkUtil.ToWorld( frames[f], rig, world ); path[f] = world[b].Pos; }
                corrected += Positions( path, changed, 1 / unitsPerCm );
                for ( var f = 0; f < frames.Count; f++ )
                {
                    if ( !changed[f] ) continue;
                    FkUtil.ToWorld( frames[f], rig, world );
                    var line = world[e].Pos - world[a].Pos; if ( line.LengthSquared() < 1e-8f ) continue;
                    line = Vector3.Normalize( line );
                    var from = world[b].Pos - world[a].Pos; from -= line * Vector3.Dot( from, line );
                    var to = path[f] - world[a].Pos; to -= line * Vector3.Dot( to, line );
                    if ( from.LengthSquared() < 1e-8f || to.LengthSquared() < 1e-8f ) continue;
                    var turn = MathF.Atan2( Vector3.Dot( Vector3.Cross( from, to ), line ), Vector3.Dot( from, to ) );
                    Turn( frames[f], rig, world, a, Quaternion.CreateFromAxisAngle( line, turn ), e );
                }
            }
        // The hands' attitude in the world, which carries the fingers: a wrist that flicks while its arm moves smoothly
        // flicks every fingertip.
        var attitudes = new Quaternion[frames.Count];
        foreach ( var hand in new[] { BoneRole.HandL, BoneRole.HandR }.Select( target.BoneForRole ) )
        {
            if ( hand is not int h || rig[h].ParentIndex < 0 ) continue;
            for ( var pass = 0; pass < Passes; pass++ )
            {
                for ( var f = 0; f < frames.Count; f++ ) { FkUtil.ToWorld( frames[f], rig, world ); attitudes[f] = world[h].Rot; }
                corrected += Rotations( attitudes, changed );
                for ( var f = 0; f < frames.Count; f++ )
                {
                    if ( !changed[f] ) continue;
                    FkUtil.ToWorld( frames[f], rig, world );
                    frames[f][h].Rot = Quaternion.Normalize( Quaternion.Inverse( world[rig[h].ParentIndex].Rot ) * attitudes[f] );
                }
            }
        }
        // The head and each finger's middle and last joint, by turning the bone above toward the fitted place; the joint
        // keeps its attitude.
        var aimed = new List<int>();
        if ( target.BoneForRole( BoneRole.Head ) is int head ) aimed.Add( head );
        foreach ( var finger in new[] { "Thumb", "Index", "Middle", "Ring", "Pinky" } )
            foreach ( var side in new[] { "L", "R" } )
                foreach ( var joint in new[] { "Mid", "Dist" } )
                    if ( Enum.TryParse<BoneRole>( finger + joint + side, out var role ) && target.BoneForRole( role ) is int bone ) aimed.Add( bone );
        foreach ( var joint in aimed )
        {
            var above = rig[joint].ParentIndex; if ( above < 0 || rig[above].ParentIndex < 0 ) continue;
            for ( var pass = 0; pass < Passes; pass++ )
            {
                for ( var f = 0; f < frames.Count; f++ ) { FkUtil.ToWorld( frames[f], rig, world ); path[f] = world[joint].Pos; }
                corrected += Positions( path, changed, 1 / unitsPerCm );
                for ( var f = 0; f < frames.Count; f++ )
                {
                    if ( !changed[f] ) continue;
                    FkUtil.ToWorld( frames[f], rig, world );
                    var from = world[joint].Pos - world[above].Pos; var to = path[f] - world[above].Pos;
                    if ( from.LengthSquared() < 1e-10f || to.LengthSquared() < 1e-10f ) continue;
                    Turn( frames[f], rig, world, above, MathQ.FromTo( Vector3.Normalize( from ), Vector3.Normalize( to ) ), joint );
                }
            }
        }
        // Held toes back where they were, by turning their foot about the ankle.
        for ( var t = 0; t < toes.Length; t++ )
        {
            var foot = rig[toes[t]].ParentIndex;
            for ( var f = 0; f < frames.Count; f++ )
            {
                var w = toesHolding[t][f]; if ( w <= 0 ) continue;
                FkUtil.ToWorld( frames[f], rig, world );
                var from = world[toes[t]].Pos - world[foot].Pos; var to = Vector3.Lerp( world[toes[t]].Pos, toesHeld[t][f], w ) - world[foot].Pos;
                if ( from.LengthSquared() < 1e-10f || to.LengthSquared() < 1e-10f ) continue;
                Turn( frames[f], rig, world, foot, MathQ.FromTo( Vector3.Normalize( from ), Vector3.Normalize( to ) ), toes[t] );
            }
        }
        return corrected;
    }

    /// <summary>Turns <paramref name="bone"/> in the world by <paramref name="turn"/> about its own joint, keeping the
    /// world attitude of <paramref name="kept"/> below it.</summary>
    static void Turn( XForm[] frame, Skeleton.Skeleton rig, XForm[] world, int bone, Quaternion turn, int kept )
    {
        var attitude = world[kept].Rot; var parent = rig[bone].ParentIndex;
        frame[bone].Rot = Quaternion.Normalize( (parent < 0 ? Quaternion.Identity : Quaternion.Inverse( world[parent].Rot )) * Quaternion.Normalize( turn * world[bone].Rot ) );
        FkUtil.ToWorld( frame, rig, world );
        frame[kept].Rot = Quaternion.Normalize( Quaternion.Inverse( world[rig[kept].ParentIndex].Rot ) * attitude );
        FkUtil.ToWorld( frame, rig, world );
    }

    /// <summary>Despikes retargeted frames in place: every bone's local rotation and each root bone's position, in target
    /// units with <paramref name="unitsPerCm"/> to the centimetre. Returns the number of corrected samples; 0 for fewer
    /// than <see cref="Window"/> frames.</summary>
    public static int Apply( List<XForm[]> frames, Skeleton.Skeleton rig, float unitsPerCm )
    {
        if ( frames.Count < Window ) return 0;
        var corrected = 0; var changed = new bool[frames.Count];
        var rotations = new Quaternion[frames.Count]; var positions = new Vector3[frames.Count];
        for ( var pass = 0; pass < Passes; pass++ )
            for ( var bone = 0; bone < rig.Count; bone++ )
            {
                for ( var f = 0; f < frames.Count; f++ ) rotations[f] = frames[f][bone].Rot;
                corrected += Rotations( rotations, changed );
                for ( var f = 0; f < frames.Count; f++ ) if ( changed[f] ) frames[f][bone].Rot = rotations[f];
                if ( rig[bone].ParentIndex >= 0 ) continue;
                for ( var f = 0; f < frames.Count; f++ ) positions[f] = frames[f][bone].Pos;
                corrected += Positions( positions, changed, 1 / unitsPerCm );
                for ( var f = 0; f < frames.Count; f++ ) if ( changed[f] ) frames[f][bone].Pos = positions[f];
            }
        return corrected;
    }

    /// <summary>Savitzky-Golay weights for the centre of a cubic fitted over seven samples.</summary>
    static readonly double[] Weights = { -2 / 21d, 3 / 21d, 6 / 21d, 7 / 21d, 6 / 21d, 3 / 21d, -2 / 21d };

    /// <summary>Pulls the rotations that leave their neighbours' cubic onto it, fitting on the track as given; marks them in
    /// <paramref name="changed"/> and returns how many.</summary>
    static int Rotations( Quaternion[] track, bool[] changed )
    {
        var frames = track.Length; var half = Window / 2;
        // Sign-continuous so neighbouring quaternions average sensibly.
        var given = new Quaternion[frames];
        for ( var f = 0; f < frames; f++ )
        {
            given[f] = track[f];
            if ( f > 0 && Quaternion.Dot( given[f], given[f - 1] ) < 0 ) given[f] = Quaternion.Negate( given[f] );
        }
        Array.Clear( changed ); var count = 0;
        // Only frames with a whole window: repeating an end frame bends steady fast motion off its path.
        for ( var f = half; f < frames - half; f++ )
        {
            var fit = Quaternion.Zero;
            for ( var k = -half; k <= half; k++ )
            {
                var q = given[f + k]; var w = (float)Weights[k + half];
                fit = new Quaternion( fit.X + q.X * w, fit.Y + q.Y * w, fit.Z + q.Z * w, fit.W + q.W * w );
            }
            if ( fit.LengthSquared() < 1e-12f ) continue;
            fit = Quaternion.Normalize( fit );
            var off = 2 * MathF.Acos( Math.Clamp( MathF.Abs( Quaternion.Dot( fit, given[f] ) ), 0, 1 ) ) * 180 / MathF.PI;
            var pull = Math.Clamp( (off - StartDegrees) / (FullDegrees - StartDegrees), 0, 1 );
            if ( pull <= 0 ) continue;
            track[f] = Quaternion.Normalize( Quaternion.Slerp( given[f], fit, pull ) );
            changed[f] = true; count++;
        }
        return count;
    }

    /// <summary>As <see cref="Rotations"/>, for positions with <paramref name="cmPerUnit"/> centimetres to the unit.</summary>
    static int Positions( Vector3[] track, bool[] changed, float cmPerUnit )
    {
        var frames = track.Length; var half = Window / 2;
        var given = (Vector3[])track.Clone();
        Array.Clear( changed ); var count = 0;
        for ( var f = half; f < frames - half; f++ )
        {
            var fit = Vector3.Zero;
            for ( var k = -half; k <= half; k++ ) fit += given[f + k] * (float)Weights[k + half];
            var off = Vector3.Distance( fit, given[f] ) * cmPerUnit;
            var pull = Math.Clamp( (off - StartCm) / (FullCm - StartCm), 0, 1 );
            if ( pull <= 0 ) continue;
            track[f] = Vector3.Lerp( given[f], fit, pull );
            changed[f] = true; count++;
        }
        return count;
    }
}
