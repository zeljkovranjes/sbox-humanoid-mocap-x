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

/// <summary>Holds hands that rest on the floor in place, the way planted feet are held. Push-ups, crawling,
/// cartwheels and handstands put weight on the hands, but the capture has no contact track for them, so a hand on
/// the floor slid and hovered with the arm's small errors. After the floor is settled (CaptureGround, CaptureHover,
/// CaptureSeat), a hand whose lowest point (fingers included) stays within <see cref="NearFloorCm"/> of the floor
/// and moves slower than <see cref="StillMetresPerSecond"/> for at least <see cref="MinimumSeconds"/> rests on it:
/// through that stretch it is held where it landed, palm on the floor, re-solved with two-bone arm IK; its
/// orientation is kept. The slide the capture built up is let go over <see cref="BlendSeconds"/> after the hand lifts:
/// held at the stretch's middle instead, a hand that slid while resting jumped to it at up to 2 m/s. A hand that travels more than <see cref="MaximumTravelCm"/> while low is wiping or
/// sliding on purpose and is left alone, unless both feet are at least <see cref="SupportFeetCm"/> off the floor: then the
/// hand carries the body (handstands, freezes, one-handed spins) and cannot slide, however far the capture drifts. The body
/// is moved to stay over such a hand, a body on one hand is tilted about that wrist until its weight is over it, the
/// palm turns flat onto the floor and the fingers straighten along it.</summary>
public static class CaptureHandPlant
{
    /// <summary>A hand comes to rest within NearFloorCm of the floor and stays at rest until it rises past LiftCm: the capture
    /// lifted a hand planted behind a seated dancer 9 cm as he got up, and a single threshold let it float.</summary>
    public const float MaximumShoulderDegrees = 25, MaximumLeanDegrees = 15;
    public const float NearFloorCm = 8, LiftCm = 15, StillMetresPerSecond = 1, MinimumSeconds = .15f, MaximumTravelCm = 25, BlendSeconds = .25f;
    /// <summary>Both feet this far off the floor leave the hands carrying the body. The breakin clip's one-handed spin
    /// holds its feet 0.8–2 m up while the captured support hand slid up to 32 cm a frame.</summary>
    public const float SupportFeetCm = 40;
    /// <summary>Hands this close in height both count as the lowest; a higher hand under a raised body carries weight
    /// only while it rests still.</summary>
    public const float BothHandsCm = 2;
    static readonly BoneRole[][] Fingers =
    {
        new[] { BoneRole.IndexProxL, BoneRole.IndexMidL, BoneRole.IndexDistL, BoneRole.MiddleProxL, BoneRole.MiddleMidL, BoneRole.MiddleDistL,
            BoneRole.RingProxL, BoneRole.RingMidL, BoneRole.RingDistL, BoneRole.PinkyProxL, BoneRole.PinkyMidL, BoneRole.PinkyDistL },
        new[] { BoneRole.IndexProxR, BoneRole.IndexMidR, BoneRole.IndexDistR, BoneRole.MiddleProxR, BoneRole.MiddleMidR, BoneRole.MiddleDistR,
            BoneRole.RingProxR, BoneRole.RingMidR, BoneRole.RingDistR, BoneRole.PinkyProxR, BoneRole.PinkyMidR, BoneRole.PinkyDistR }
    };

    /// <returns>The number of hand samples held.</returns>
    /// <param name="observed">Frames in which the video shows each hand on the floor under a raised body, by hand role;
    /// such a hand is planted even where the capture lifted it.</param>
    public static int Apply( List<XForm[]> frames, TargetRig target, TargetUpAxis axis, float fps, IReadOnlyDictionary<BoneRole, bool[]> observed = null )
    {
        if ( frames.Count < 3 || fps <= 0 ) return 0;
        var rig = target.Skeleton; var up = axis == TargetUpAxis.YUpCm ? Vector3.UnitY : Vector3.UnitZ;
        var cm = axis == TargetUpAxis.ZUpEngine ? 1 / 2.54f : 1f;
        var arms = new[] { (BoneRole.UpperArmL, BoneRole.LowerArmL, BoneRole.HandL), (BoneRole.UpperArmR, BoneRole.LowerArmR, BoneRole.HandR) }
            .Select( r => target.BoneForRole( r.Item1 ) is int a && target.BoneForRole( r.Item2 ) is int b && target.BoneForRole( r.Item3 ) is int c ? (A: a, B: b, C: c) : ((int A, int B, int C)?)null )
            .Where( c => c is not null ).Select( c => c.Value ).ToArray();
        if ( arms.Length == 0 ) return 0;
        var world = new XForm[rig.Count];
        // Each hand's wrist and the height of its lowest part, per frame, before anything is held.
        var parts = arms.ToDictionary( a => a.C, a => CaptureSeat.HandParts( target, a.C ) );
        var feet = new[] { BoneRole.FootL, BoneRole.ToeL, BoneRole.FootR, BoneRole.ToeR }.Select( target.BoneForRole ).ToArray();
        var supported = new bool[frames.Count];
        // Finger chains (proximal, middle, distal for index to pinky) of each hand, where the rig has them all.
        var chains = arms.ToDictionary( a => a.C, a => Fingers[target.BoneForRole( BoneRole.HandL ) == a.C ? 0 : 1].Select( target.BoneForRole ).ToArray() is var bones
            && bones.All( b => b is int ) ? bones.Select( b => b!.Value ).ToArray() : null );
        var wrist = arms.ToDictionary( a => a.C, _ => new Vector3[frames.Count] );
        var lowest = arms.ToDictionary( a => a.C, _ => new float[frames.Count] );
        for ( var f = 0; f < frames.Count; f++ )
        {
            FkUtil.ToWorld( frames[f], rig, world );
            foreach ( var a in arms ) { wrist[a.C][f] = world[a.C].Pos; lowest[a.C][f] = parts[a.C].Min( b => Vector3.Dot( world[b].Pos, up ) ); }
            if ( feet[0] is int footL && feet[2] is int footR )
                supported[f] = Math.Min( Height( footL, feet[1] ), Height( footR, feet[3] ) ) > SupportFeetCm * cm;
        }
        float Height( int foot, int? toe ) => Math.Min( Vector3.Dot( world[foot].Pos, up ), toe is int t ? Vector3.Dot( world[t].Pos, up ) : float.MaxValue );
        var minimum = Math.Max( 3, (int)MathF.Ceiling( MinimumSeconds * fps ) ); var blend = Math.Max( 1, (int)MathF.Round( BlendSeconds * fps ) );
        var still = StillMetresPerSecond * 100 * cm / fps;
        var goals = arms.ToDictionary( a => a.C, _ => new (Vector3 Position, float Weight, bool Bearing)[frames.Count] );
        // Weight-bearing stretches (hand, first, last, landing point), to move the body over them below.
        var bearings = new List<(int Hand, int First, int Last, Vector3 Anchor)>();
        foreach ( var a in arms )
        {
            var p = wrist[a.C]; var low = lowest[a.C];
            bool Lowest( int f ) => arms.All( other => low[f] <= lowest[other.C][f] + BothHandsCm * cm );
            bool Still( int f ) => f == 0 || Flat( p[f] - p[f - 1], up ).Length() < still;
            // The lowest hand under a raised body rests however the capture drags it; any other hand must also be still,
            // so the free hand swinging low through a one-handed spin is not planted.
            bool Resting( int f, float height ) => low[f] < height * cm && (supported[f] && Lowest( f ) || Still( f ));
            for ( var f = 0; f < frames.Count; )
            {
                if ( !Resting( f, NearFloorCm ) ) { f++; continue; }
                var e = f; while ( e + 1 < frames.Count && Resting( e + 1, LiftCm ) ) e++;
                if ( e - f + 1 >= minimum )
                {
                    // Held where it landed: a hand that touches down stays there. The capture's slide builds up while it rests
                    // and is let go over BlendSeconds after the hand lifts, while it is moving anyway.
                    var length = e - f + 1; var anchor = Flat( p[f], up );
                    // The lower hand under a raised body carries it. A second hand that rests beside it is found below,
                    // once the body has been carried: until then the whole body slides with the capture.
                    var bearing = Enumerable.Range( f, length ).Count( i => supported[i] && Lowest( i ) ) * 2 >= length;
                    if ( bearing || Enumerable.Range( f, length ).All( i => (Flat( p[i], up ) - anchor).Length() <= MaximumTravelCm * cm ) )
                    {
                        if ( bearing ) bearings.Add( (a.C, f, e, anchor) );
                        // The palm settles onto the floor over a few frames rather than dropping in one.
                        var settle = Math.Max( 1, Math.Min( 3, (length - 1) / 3 ) );
                        for ( var i = f; i <= e; i++ )
                        {
                            // Palm on the floor: the wrist at the held place, as high as puts the hand's lowest point on the floor.
                            var w = Math.Clamp( (i - f + 1) / (float)(settle + 1), 0, 1 ); w = w * w * (3 - 2 * w);
                            var goal = anchor + up * (Vector3.Dot( p[i], up ) - low[i]);
                            if ( w > goals[a.C][i].Weight ) goals[a.C][i] = (goal, w, bearing);
                        }
                        // Everything held at the last frame, height included, is let go gradually.
                        var carried = anchor + up * (Vector3.Dot( p[e], up ) - low[e]) - p[e];
                        for ( var i = e + 1; i <= Math.Min( frames.Count - 1, e + blend ); i++ )
                        {
                            var w = 1 - (i - e) / (float)(blend + 1); w = w * w * (3 - 2 * w);
                            if ( w > goals[a.C][i].Weight ) goals[a.C][i] = (p[i] + carried, w, false);
                        }
                    }
                }
                f = e + 1;
            }
        }
        CarryBody( frames, rig, bearings, wrist, up, blend );
        var seen = arms.ToDictionary( a => a.C, a => observed?.GetValueOrDefault( target.BoneForRole( BoneRole.HandL ) == a.C ? BoneRole.HandL : BoneRole.HandR ) is { } mask
            && mask.Length == frames.Count ? mask : new bool[frames.Count] );
        SecondHands( frames, rig, arms, parts, supported, seen, goals, bearings, up, cm, still, minimum, blend );
        Balance( frames, target, bearings, goals, up );
        var held = 0;
        for ( var f = 0; f < frames.Count; f++ )
        {
            var frame = frames[f];
            // With both hands bearing weight, the second hand's small lean is followed by the first hand again, so neither
            // ends off the floor.
            var bothBearing = arms.Count( a => goals[a.C][f].Bearing && goals[a.C][f].Weight > 0 ) == arms.Length && arms.Length > 1;
            foreach ( var a in bothBearing ? arms.Concat( arms.Take( 1 ) ) : arms )
            {
                var (position, weight, bearing) = goals[a.C][f]; if ( weight <= 0 ) continue;
                FkUtil.ToWorld( frame, rig, world );
                var rotation = world[a.C].Rot;
                if ( bearing && chains[a.C] is { } chain )
                {
                    // A hand carrying the body lies flat: palm onto the floor, fingers straight along it, its lowest point on the floor.
                    var down = Quaternion.Slerp( Quaternion.Identity, FromTo( PalmNormal( world, chain, a.C, target.BoneForRole( BoneRole.HandL ) == a.C ), -up ), weight );
                    rotation = Quaternion.Normalize( down * rotation );
                    Straighten( frame, chain, weight );
                    var parent0 = rig[a.C].ParentIndex;
                    frame[a.C].Rot = Quaternion.Normalize( (parent0 < 0 ? Quaternion.Identity : Quaternion.Inverse( world[parent0].Rot )) * rotation );
                    FkUtil.ToWorld( frame, rig, world );
                    var depth = parts[a.C].Min( b => Vector3.Dot( world[b].Pos - world[a.C].Pos, up ) );
                    position = Flat( position, up ) + up * Math.Max( 0, -depth );
                }
                var goal = Vector3.Lerp( world[a.C].Pos, position, weight );
                // Out of the straight arm's reach (a seated dancer rising off a planted hand), the shoulder drops
                // toward the hand as a person's does: the clavicle turns, up to MaximumShoulderDegrees.
                var reach = Vector3.Distance( world[a.A].Pos, world[a.B].Pos ) + Vector3.Distance( world[a.B].Pos, world[a.C].Pos );
                // Turns bone about its own joint, up to limit degrees, just far enough to bring the shoulder within reach.
                void Lean( int bone, float limit )
                {
                    if ( bone < 0 || rig[bone].ParentIndex < 0 || Vector3.Distance( world[a.A].Pos, goal ) <= reach * .999f ) return;
                    var pivot = world[bone].Pos; var from = world[a.A].Pos - pivot; var to = goal - pivot;
                    var hinge = Vector3.Cross( from, to );
                    if ( hinge.LengthSquared() < 1e-10f || from.LengthSquared() < 1e-8f ) return;
                    hinge = Vector3.Normalize( hinge ); var turn = 0f;
                    for ( var step = 1; step <= 25; step++ )
                    {
                        turn = step * limit / 25 * MathF.PI / 180;
                        if ( Vector3.Distance( pivot + Vector3.Transform( from, Quaternion.CreateFromAxisAngle( hinge, turn ) ), goal ) <= reach * .999f ) break;
                    }
                    var turned = Quaternion.Normalize( Quaternion.CreateFromAxisAngle( hinge, turn ) * world[bone].Rot );
                    frame[bone].Rot = Quaternion.Normalize( Quaternion.Inverse( world[rig[bone].ParentIndex].Rot ) * turned );
                    FkUtil.ToWorld( frame, rig, world );
                }
                Lean( rig[a.A].ParentIndex, MaximumShoulderDegrees );
                // Still out of reach (pushing up off the hand), the body leans onto it from the base of the spine; only a third
                // as far while the other hand also bears weight, since the body is fixed on that hand.
                var otherBearing = arms.Any( o => o.C != a.C && goals[o.C][f].Bearing && goals[o.C][f].Weight > 0 );
                if ( target.BoneForRole( BoneRole.Spine0 ) is int spine ) Lean( spine, otherBearing ? MaximumLeanDegrees / 3 : MaximumLeanDegrees );
                var upper = world[a.A]; var middle = world[a.B]; var end = world[a.C];
                var bend = Vector3.Cross( middle.Pos - upper.Pos, end.Pos - middle.Pos );
                if ( bend.LengthSquared() < 1e-8f ) bend = Vector3.Transform( Vector3.UnitX, upper.Rot );
                var ik = TwoBoneIk.Solve( upper.Pos, middle.Pos, end.Pos, goal, soften: 0, stableBendAxis: bend );
                EffectorIk.ApplyWorldDeltas( frame, rig, a.A, a.B, a.C, ik.UpperWorldDelta, ik.LowerWorldDelta, world );
                FkUtil.ToWorld( frame, rig, world );
                var parent = rig[a.C].ParentIndex;
                frame[a.C].Rot = Quaternion.Normalize( (parent < 0 ? Quaternion.Identity : Quaternion.Inverse( world[parent].Rot )) * rotation );
            }
            held += arms.Count( a => goals[a.C][f].Weight > 0 );
        }
        return held;
    }
    static Vector3 Flat( Vector3 v, Vector3 up ) => v - up * Vector3.Dot( v, up );

    /// <summary>Moves the whole body so each weight-bearing hand stays where it landed, the way a support hand pins a
    /// handstand; the move eases out over <paramref name="blend"/> frames after the hand lifts. Where both hands carry
    /// the body, the one lower on average leads and the other is reached with the arm.</summary>
    static void CarryBody( List<XForm[]> frames, HumanoidMocapX.Core.Skeleton.Skeleton rig, List<(int Hand, int First, int Last, Vector3 Anchor)> bearings,
        Dictionary<int, Vector3[]> wrist, Vector3 up, int blend )
    {
        if ( bearings.Count == 0 ) return;
        var shift = new Vector3?[frames.Count];
        foreach ( var (hand, first, last, anchor) in bearings.OrderBy( s => Enumerable.Range( s.First, s.Last - s.First + 1 ).Average( i => Vector3.Dot( wrist[s.Hand][i], up ) ) ) )
        {
            for ( var i = first; i <= last; i++ ) shift[i] ??= anchor - Flat( wrist[hand][i], up );
            var carried = shift[last]!.Value;
            for ( var i = last + 1; i <= Math.Min( frames.Count - 1, last + blend ); i++ )
            {
                var w = 1 - (i - last) / (float)(blend + 1); w = w * w * (3 - 2 * w);
                shift[i] ??= carried * w;
            }
        }
        var roots = Enumerable.Range( 0, rig.Count ).Where( b => rig[b].ParentIndex < 0 ).ToArray();
        for ( var f = 0; f < frames.Count; f++ )
            if ( shift[f] is { } move ) foreach ( var root in roots ) frames[f][root].Pos += move;
    }

    /// <summary>Plants a hand that bears weight beside the one carrying the body: where the video shows it on the floor
    /// under a raised body, or where it rests still near the floor there. The breakin clip's closing headstand had both
    /// hands down in the video but the capture's right hand 22–26 cm up. Stillness is judged after the body was carried
    /// over the first hand, so the capture's slide does not hide it; the free hand sweeping low through a one-handed
    /// spin is never still and stays free. The hand is let go over <paramref name="blend"/> frames.</summary>
    static void SecondHands( List<XForm[]> frames, HumanoidMocapX.Core.Skeleton.Skeleton rig, (int A, int B, int C)[] arms, Dictionary<int, int[]> parts,
        bool[] supported, Dictionary<int, bool[]> seen, Dictionary<int, (Vector3 Position, float Weight, bool Bearing)[]> goals,
        List<(int Hand, int First, int Last, Vector3 Anchor)> bearings, Vector3 up, float cm, float still, int minimum, int blend )
    {
        var world = new XForm[rig.Count]; var hands = arms.Select( a => a.C ).ToArray();
        var wrist = hands.ToDictionary( h => h, _ => new Vector3[frames.Count] ); var low = hands.ToDictionary( h => h, _ => new float[frames.Count] );
        // Whether the floor under each wrist lies within the arm's reach of its shoulder.
        var reachable = hands.ToDictionary( h => h, _ => new bool[frames.Count] );
        for ( var f = 0; f < frames.Count; f++ )
        {
            FkUtil.ToWorld( frames[f], rig, world );
            foreach ( var (a, b, h) in arms )
            {
                wrist[h][f] = world[h].Pos; low[h][f] = parts[h].Min( p => Vector3.Dot( world[p].Pos, up ) );
                var floor = world[h].Pos - up * low[h][f];
                reachable[h][f] = Vector3.Distance( world[a].Pos, floor ) <= 1.02f * (Vector3.Distance( world[a].Pos, world[b].Pos ) + Vector3.Distance( world[b].Pos, world[h].Pos ));
            }
        }
        foreach ( var h in hands )
        {
            var p = wrist[h];
            bool Resting( int f, float height ) => !goals[h][f].Bearing && (seen[h][f]
                || supported[f] && low[h][f] < height * cm && (f == 0 || Flat( p[f] - p[f - 1], up ).Length() < still));
            for ( var f = 0; f < frames.Count; )
            {
                if ( !Resting( f, NearFloorCm ) ) { f++; continue; }
                var e = f; while ( e + 1 < frames.Count && Resting( e + 1, LiftCm ) ) e++;
                // A hand the video puts on the floor out of the arm's reach is a mismatch (SAM swaps hands upside down);
                // reaching for it would only drag the body off the hand carrying it.
                if ( e - f + 1 >= minimum && Enumerable.Range( f, e - f + 1 ).Count( i => reachable[h][i] ) * 2 >= e - f + 1 )
                {
                    var anchor = Flat( p[f], up ); var settle = Math.Max( 1, Math.Min( 3, (e - f) / 3 ) );
                    for ( var i = f; i <= e; i++ )
                    {
                        var w = Math.Clamp( (i - f + 1) / (float)(settle + 1), 0, 1 ); w = w * w * (3 - 2 * w);
                        if ( w > goals[h][i].Weight || !goals[h][i].Bearing ) goals[h][i] = (anchor + up * (Vector3.Dot( p[i], up ) - low[h][i]), w, true);
                    }
                    bearings.Add( (h, f, e, anchor) );
                    var carried = anchor + up * (Vector3.Dot( p[e], up ) - low[h][e]) - p[e];
                    for ( var i = e + 1; i <= Math.Min( frames.Count - 1, e + blend ); i++ )
                    {
                        var w = 1 - (i - e) / (float)(blend + 1); w = w * w * (3 - 2 * w);
                        if ( w > goals[h][i].Weight ) goals[h][i] = (p[i] + carried, w, false);
                    }
                }
                f = e + 1;
            }
        }
    }

    /// <summary>Segment mass fractions (de Leva 1996) between joint pairs, for the body's centre of mass.</summary>
    static readonly (BoneRole From, BoneRole To, float Mass)[] Segments =
    {
        (BoneRole.Hips, BoneRole.Neck, .43f), (BoneRole.Head, BoneRole.Head, .08f),
        (BoneRole.UpperArmL, BoneRole.LowerArmL, .027f), (BoneRole.UpperArmR, BoneRole.LowerArmR, .027f),
        (BoneRole.LowerArmL, BoneRole.HandL, .022f), (BoneRole.LowerArmR, BoneRole.HandR, .022f),
        (BoneRole.UpperLegL, BoneRole.LowerLegL, .14f), (BoneRole.UpperLegR, BoneRole.LowerLegR, .14f),
        (BoneRole.LowerLegL, BoneRole.FootL, .06f), (BoneRole.LowerLegR, BoneRole.FootR, .06f)
    };
    /// <summary>A body on one hand is tilted at most this far to bring its weight over the hand.</summary>
    public const float MaximumBalanceDegrees = 30;

    /// <summary>Tilts a body that rests on a single hand about that wrist until its centre of mass is over it: on one
    /// hand it cannot be anywhere else without falling. The breakin clip's one-handed spin leaned 14° off the hand
    /// in the capture where SAM saw 5°, which swung the free hand on a circle a quarter wider. Two hands or a hand and
    /// the head down share the weight, and the body is left as captured.</summary>
    static void Balance( List<XForm[]> frames, TargetRig target, List<(int Hand, int First, int Last, Vector3 Anchor)> bearings,
        Dictionary<int, (Vector3 Position, float Weight, bool Bearing)[]> goals, Vector3 up )
    {
        var rig = target.Skeleton; var segments = Segments.Select( s => (A: target.BoneForRole( s.From ), B: target.BoneForRole( s.To ), s.Mass) ).ToArray();
        if ( bearings.Count == 0 || segments.Any( s => s.A is null || s.B is null ) ) return;
        var roots = Enumerable.Range( 0, rig.Count ).Where( b => rig[b].ParentIndex < 0 ).ToArray();
        var world = new XForm[rig.Count];
        for ( var f = 0; f < frames.Count; f++ )
        {
            var on = bearings.Where( s => s.First <= f && f <= s.Last ).ToArray();
            if ( on.Length != 1 ) continue;
            var hand = on[0].Hand; var weight = goals[hand][f].Weight; if ( weight <= 0 ) continue;
            FkUtil.ToWorld( frames[f], rig, world );
            var centre = Vector3.Zero; var mass = 0f;
            foreach ( var (a, b, m) in segments ) { centre += (world[a!.Value].Pos + world[b!.Value].Pos) * .5f * m; mass += m; }
            var lever = centre / mass - world[hand].Pos;
            if ( Vector3.Dot( lever, up ) <= 0 ) continue;
            var tilt = FromTo( Vector3.Normalize( lever ), up );
            var angle = 2 * MathF.Acos( Math.Clamp( MathF.Abs( tilt.W ), 0, 1 ) );
            var limit = MaximumBalanceDegrees * MathF.PI / 180;
            if ( angle > limit ) tilt = Quaternion.Slerp( Quaternion.Identity, tilt, limit / angle );
            tilt = Quaternion.Normalize( Quaternion.Slerp( Quaternion.Identity, tilt, weight ) );
            var pivot = world[hand].Pos;
            foreach ( var root in roots )
            {
                frames[f][root].Pos = pivot + Vector3.Transform( frames[f][root].Pos - pivot, tilt );
                frames[f][root].Rot = Quaternion.Normalize( tilt * frames[f][root].Rot );
            }
        }
    }

    /// <summary>The direction the palm faces, from the wrist and the index, middle and pinky knuckles. The back of a
    /// right hand faces along (knuckle line × index-to-pinky); a left hand mirrors it.</summary>
    static Vector3 PalmNormal( XForm[] world, int[] chain, int hand, bool left )
    {
        var along = world[chain[3]].Pos - world[hand].Pos; var across = world[chain[0]].Pos - world[chain[9]].Pos;
        var normal = Vector3.Cross( along, across );
        if ( normal.LengthSquared() < 1e-10f ) return Vector3.Zero;
        normal = Vector3.Normalize( normal );
        return left ? normal : -normal;
    }

    /// <summary>Turns each finger joint toward straight, by <paramref name="weight"/>: each segment continues the one
    /// before, so a flat palm puts the fingers along the floor. Twist is kept.</summary>
    static void Straighten( XForm[] frame, int[] chain, float weight )
    {
        for ( var finger = 0; finger < 4; finger++ )
            for ( var segment = 0; segment < 3; segment++ )
            {
                var bone = chain[finger * 3 + segment];
                // The parent's bone runs along this joint's offset; this bone runs along its child's (the tip along its own).
                var inherited = frame[bone].Pos; var own = segment < 2 ? frame[chain[finger * 3 + segment + 1]].Pos : frame[bone].Pos;
                if ( inherited.LengthSquared() < 1e-10f || own.LengthSquared() < 1e-10f ) continue;
                var rotation = frame[bone].Rot;
                var straight = Quaternion.Normalize( FromTo( Vector3.Normalize( Vector3.Transform( own, rotation ) ), Vector3.Normalize( inherited ) ) * rotation );
                frame[bone].Rot = Quaternion.Normalize( Quaternion.Slerp( rotation, straight, weight ) );
            }
    }

    static Quaternion FromTo( Vector3 from, Vector3 to )
    {
        if ( from.LengthSquared() < 1e-10f || to.LengthSquared() < 1e-10f ) return Quaternion.Identity;
        return MathQ.FromTo( from, to );
    }
    static bool IsWithin( HumanoidMocapX.Core.Skeleton.Skeleton rig, int bone, int ancestor )
    {
        for ( var b = bone; b >= 0; b = rig[b].ParentIndex ) if ( b == ancestor ) return true;
        return false;
    }
}
