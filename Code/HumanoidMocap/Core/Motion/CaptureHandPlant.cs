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
        // Finger chains (proximal, middle, distal) of each hand: index and middle, then ring and pinky where the rig has them.
        // The Citizen has no pinky; without it none of its hands lay flat, and a planted wrist bobbed up to 21 cm as the
        // capture's long fingers swung under it.
        var chains = arms.ToDictionary( a => a.C, a =>
        {
            var fingers = Fingers[target.BoneForRole( BoneRole.HandL ) == a.C ? 0 : 1].Chunk( 3 ).Select( f => f.Select( target.BoneForRole ).ToArray() ).ToArray();
            if ( fingers.Take( 2 ).Any( f => f.Any( b => b is null ) ) ) return null;
            return fingers.Where( f => f.All( b => b is int ) ).SelectMany( f => f.Select( b => b!.Value ) ).ToArray();
        } );
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
        var goals = arms.ToDictionary( a => a.C, _ => new (Vector3 Position, float Weight, bool Bearing, bool Lifting)[frames.Count] );
        // Which weight-bearing stretch (index into bearings) wrote each goal, -1 for none.
        var owner = arms.ToDictionary( a => a.C, _ => Enumerable.Repeat( -1, frames.Count ).ToArray() );
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
                var e = f;
                while ( true )
                {
                    while ( e + 1 < frames.Count && Resting( e + 1, LiftCm ) ) e++;
                    // The lowest hand under a raised body still carries it while the capture lifts it for a moment: through the
                    // breakin clip's one-handed spin it lifted the support hand up to 22 cm for three to five frames, and each
                    // touch after that landed somewhere new. Up to BlendSeconds of that stays one hold.
                    // A hand that touches down again within ShortLiftFrames never really left: on the Citizen two of the breakin
                    // clip's touches, a frame apart, each landed elsewhere and the hand jumped 18–30 cm between them.
                    var g = e + 1; while ( g < frames.Count && g - e <= blend && (supported[g] && Lowest( g ) || g - e <= ShortLiftFrames) && !Resting( g, NearFloorCm ) ) g++;
                    if ( g == e + 1 || g >= frames.Count || g - e > blend || !Resting( g, NearFloorCm ) ) break;
                    e = g;
                }
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
                            if ( w > goals[a.C][i].Weight ) { goals[a.C][i] = (goal, w, bearing, false); owner[a.C][i] = bearing ? bearings.Count - 1 : -1; }
                        }
                        // Everything held at the last frame, height included, is let go gradually; a flat palm curls back
                        // over the same frames instead of snapping.
                        var carried = anchor + up * (Vector3.Dot( p[e], up ) - low[e]) - p[e];
                        for ( var i = e + 1; i <= Math.Min( frames.Count - 1, e + blend ); i++ )
                        {
                            var w = 1 - (i - e) / (float)(blend + 1); w = w * w * (3 - 2 * w);
                            if ( w > goals[a.C][i].Weight ) { goals[a.C][i] = (p[i] + carried, w, bearing, true); owner[a.C][i] = bearing ? bearings.Count - 1 : -1; }
                        }
                    }
                }
                f = e + 1;
            }
        }
        // Each stretch lands where the carried body puts its hand, and a hand being let go keeps its offset from where the
        // carried body puts it at the release, fading, rather than from the uncarried capture: while the carry eased out, a
        // Citizen's lifting hand was sent out of reach and its arm swung 45 cm.
        var (landed, moved) = CarryBody( frames, rig, bearings, wrist, up, blend );
        foreach ( var a in arms )
            for ( var i = 0; i < frames.Count; i++ )
                if ( owner[a.C][i] is var b && b >= 0 )
                    goals[a.C][i].Position += landed[b] + (goals[a.C][i].Lifting ? moved[i] - moved[bearings[b].Last] : Vector3.Zero);
        for ( var b = 0; b < bearings.Count; b++ ) bearings[b] = bearings[b] with { Anchor = bearings[b].Anchor + landed[b] };
        var seen = arms.ToDictionary( a => a.C, a => observed?.GetValueOrDefault( target.BoneForRole( BoneRole.HandL ) == a.C ? BoneRole.HandL : BoneRole.HandR ) is { } mask
            && mask.Length == frames.Count ? mask : new bool[frames.Count] );
        SecondHands( frames, rig, arms, parts, supported, seen, goals, bearings, up, cm, still, minimum, blend );
        Balance( frames, target, bearings, goals, up, fps );
        var heading = arms.ToDictionary( a => a.C, a => chains[a.C] is { } chain
            ? Headings( frames, rig, a.C, chain, target.BoneForRole( BoneRole.HandL ) == a.C, goals[a.C], up ) : null );
        var held = 0; var lift = arms.ToDictionary( a => a.C, _ => 0f );
        for ( var f = 0; f < frames.Count; f++ )
        {
            var frame = frames[f];
            // With both hands bearing weight, the second hand's small lean is followed by the first hand again, so neither
            // ends off the floor.
            var bothBearing = arms.Count( a => goals[a.C][f].Bearing && goals[a.C][f].Weight > 0 ) == arms.Length && arms.Length > 1;
            foreach ( var a in bothBearing ? arms.Concat( arms.Take( 1 ) ) : arms )
            {
                var (position, weight, bearing, lifting) = goals[a.C][f]; if ( weight <= 0 ) continue;
                FkUtil.ToWorld( frame, rig, world );
                var rotation = world[a.C].Rot;
                if ( bearing && chains[a.C] is { } chain )
                {
                    // A hand carrying the body lies flat: palm onto the floor, fingers straight along it, its lowest point on the floor.
                    var down = Quaternion.Slerp( Quaternion.Identity, FromTo( PalmNormal( world, chain, a.C, target.BoneForRole( BoneRole.HandL ) == a.C ), -up ), weight );
                    rotation = Quaternion.Normalize( down * rotation );
                    if ( heading[a.C] is { } turning && !double.IsNaN( turning[f] ) )
                    {
                        var turn = (float)Math.IEEERemainder( turning[f] - Heading( Vector3.Transform( frame[chain[3]].Pos, rotation ), up ), 2 * Math.PI ) * weight;
                        rotation = Quaternion.Normalize( Quaternion.CreateFromAxisAngle( up, turn ) * rotation );
                    }
                    Straighten( frame, chain, weight );
                    var parent0 = rig[a.C].ParentIndex;
                    frame[a.C].Rot = Quaternion.Normalize( (parent0 < 0 ? Quaternion.Identity : Quaternion.Inverse( world[parent0].Rot )) * rotation );
                    FkUtil.ToWorld( frame, rig, world );
                    var depth = parts[a.C].Min( b => Vector3.Dot( world[b].Pos - world[a.C].Pos, up ) );
                    // Down, the flat hand's lowest point is on the floor. Lifting, the hand rises as the capture lifts it from
                    // where it was held, never through the floor: forced down, a Citizen's lifting arm overreached and flipped.
                    if ( !lifting ) { lift[a.C] = Math.Max( 0, -depth ) - Vector3.Dot( position, up ); position = Flat( position, up ) + up * Math.Max( 0, -depth ); }
                    else position = Flat( position, up ) + up * Math.Max( Math.Max( 0, -depth ), Vector3.Dot( position, up ) + lift[a.C] );
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
    /// the body, the one lower on average leads and the other is reached with the arm. Returns, per stretch, how far
    /// its landing point moved with the carried body, and the body's move per frame.</summary>
    static (Vector3[] Landed, Vector3[] Moved) CarryBody( List<XForm[]> frames, HumanoidMocapX.Core.Skeleton.Skeleton rig, List<(int Hand, int First, int Last, Vector3 Anchor)> bearings,
        Dictionary<int, Vector3[]> wrist, Vector3 up, int blend )
    {
        var landed = new Vector3[bearings.Count];
        if ( bearings.Count == 0 ) return (landed, new Vector3[frames.Count]);
        // The move per frame, which stretch leads it, and how much of that stretch's move remains (1 while its hand is down).
        var shift = new Vector3?[frames.Count]; var lead = Enumerable.Repeat( -1, frames.Count ).ToArray(); var share = new float[frames.Count];
        // A hand that is down leads before any other hand's letting go.
        var order = Enumerable.Range( 0, bearings.Count ).OrderBy( s => Enumerable.Range( bearings[s].First, bearings[s].Last - bearings[s].First + 1 ).Average( i => Vector3.Dot( wrist[bearings[s].Hand][i], up ) ) ).ToArray();
        foreach ( var s in order )
        {
            var (hand, first, last, anchor) = bearings[s];
            for ( var i = first; i <= last; i++ ) if ( shift[i] is null ) { shift[i] = anchor - Flat( wrist[hand][i], up ); lead[i] = s; share[i] = 1; }
        }
        foreach ( var s in order )
        {
            var (hand, _, last, anchor) = bearings[s];
            var carried = anchor - Flat( wrist[hand][last], up );
            for ( var i = last + 1; i <= Math.Min( frames.Count - 1, last + blend ); i++ )
            {
                var w = 1 - (i - last) / (float)(blend + 1); w = w * w * (3 - 2 * w);
                if ( shift[i] is null ) { shift[i] = carried * w; lead[i] = s; share[i] = w; }
            }
        }
        // In time order, a stretch that takes over while another still moves the body carries that move on instead of
        // dropping it, and lands where the moved body puts its hand; so does a hand that only follows another's lead. On
        // the Citizen the breakin clip's left hand touched down again one frame after lifting and jumped 30 cm.
        foreach ( var s in Enumerable.Range( 0, bearings.Count ).OrderBy( s => bearings[s].First ) )
        {
            var first = bearings[s].First;
            if ( lead[first] != s ) { landed[s] = shift[first] ?? Vector3.Zero; continue; }
            if ( first == 0 || lead[first - 1] < 0 || lead[first - 1] == s ) continue;
            landed[s] = shift[first - 1]!.Value;
            for ( var i = 0; i < frames.Count; i++ ) if ( lead[i] == s ) shift[i] += landed[s] * share[i];
        }
        // Smoothed over time: the capture's frame-to-frame jitter at the hand would otherwise shake the whole body. The window
        // stays symmetric and inside the carried frames, shrinking toward their ends, so the body is not moved before the hand
        // lands (a hand landing on a 3 cm-a-frame slide settled 5 cm sideways) and a steady slide is followed exactly. The
        // arm's IK below keeps the hand itself exactly where it landed.
        var smoothed = new Vector3[frames.Count]; var radius = (int)MathF.Ceiling( 3 * CarrySigmaFrames );
        for ( var f = 0; f < frames.Count; f++ )
        {
            if ( shift[f] is null ) continue;
            var reach = 0;
            while ( reach < radius && f - reach - 1 >= 0 && f + reach + 1 < frames.Count && shift[f - reach - 1] is not null && shift[f + reach + 1] is not null ) reach++;
            var sum = Vector3.Zero; var total = 0f;
            for ( var j = f - reach; j <= f + reach; j++ )
            {
                var g = MathF.Exp( -.5f * (j - f) * (j - f) / (CarrySigmaFrames * CarrySigmaFrames) ); sum += shift[j]!.Value * g; total += g;
            }
            smoothed[f] = sum / total;
        }
        var roots = Enumerable.Range( 0, rig.Count ).Where( b => rig[b].ParentIndex < 0 ).ToArray();
        for ( var f = 0; f < frames.Count; f++ )
            if ( smoothed[f].LengthSquared() > 0 ) foreach ( var root in roots ) frames[f][root].Pos += smoothed[f];
        return (landed, smoothed);
    }

    /// <summary>Plants a hand that bears weight beside the one carrying the body: where the video shows it on the floor
    /// under a raised body, or where it rests still near the floor there. The breakin clip's closing headstand had both
    /// hands down in the video but the capture's right hand 22–26 cm up. Stillness is judged after the body was carried
    /// over the first hand, so the capture's slide does not hide it; the free hand sweeping low through a one-handed
    /// spin is never still and stays free. The hand is let go over <paramref name="blend"/> frames.</summary>
    static void SecondHands( List<XForm[]> frames, HumanoidMocapX.Core.Skeleton.Skeleton rig, (int A, int B, int C)[] arms, Dictionary<int, int[]> parts,
        bool[] supported, Dictionary<int, bool[]> seen, Dictionary<int, (Vector3 Position, float Weight, bool Bearing, bool Lifting)[]> goals,
        List<(int Hand, int First, int Last, Vector3 Anchor)> bearings, Vector3 up, float cm, float still, int minimum, int blend )
    {
        var world = new XForm[rig.Count]; var hands = arms.Select( a => a.C ).ToArray();
        var wrist = hands.ToDictionary( h => h, _ => new Vector3[frames.Count] ); var low = hands.ToDictionary( h => h, _ => new float[frames.Count] );
        // Whether the floor under each wrist lies within the arm's reach of its shoulder.
        var reachable = hands.ToDictionary( h => h, _ => new bool[frames.Count] );
        var shoulder = hands.ToDictionary( h => h, _ => new Vector3[frames.Count] ); var reach = hands.ToDictionary( h => h, _ => new float[frames.Count] );
        for ( var f = 0; f < frames.Count; f++ )
        {
            FkUtil.ToWorld( frames[f], rig, world );
            foreach ( var (a, b, h) in arms )
            {
                wrist[h][f] = world[h].Pos; low[h][f] = parts[h].Min( p => Vector3.Dot( world[p].Pos, up ) );
                var floor = world[h].Pos - up * low[h][f];
                shoulder[h][f] = world[a].Pos; reach[h][f] = 1.02f * (Vector3.Distance( world[a].Pos, world[b].Pos ) + Vector3.Distance( world[b].Pos, world[h].Pos ));
                reachable[h][f] = Vector3.Distance( world[a].Pos, floor ) <= reach[h][f];
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
                    // A hand that never leaves the floor is one touch: it keeps the place of a hold ending just before, and a hold
                    // starting just after carries on at this one's. The breakin clip's right hand, held through the closing freeze,
                    // jumped 66 cm to where the first pass had put it as the dancer pushed up.
                    var before = bearings.FindIndex( s => s.Hand == h && s.Last == f - 1 );
                    var anchor = before >= 0 ? bearings[before].Anchor : Flat( p[f], up ); var settle = Math.Max( 1, Math.Min( 3, (e - f) / 3 ) );
                    var after = bearings.FindIndex( s => s.Hand == h && s.First == e + 1 );
                    var last = after >= 0 ? bearings[after].Last : e;
                    if ( after >= 0 ) bearings.RemoveAt( after );
                    for ( var i = f; i <= last; i++ )
                    {
                        var goal = anchor + up * (Vector3.Dot( p[i], up ) - low[h][i]);
                        // Carried on only while the place stays within the arm's reach; as the body rises off it the hand lifts.
                        if ( i > e && Vector3.Distance( shoulder[h][i], goal ) > reach[h][i] )
                        {
                            for ( var j = i; j <= Math.Min( frames.Count - 1, last + blend ); j++ ) goals[h][j] = default;
                            last = i - 1; break;
                        }
                        var w = before >= 0 ? 1 : Math.Clamp( (i - f + 1) / (float)(settle + 1), 0, 1 ); w = w * w * (3 - 2 * w);
                        goals[h][i] = (goal, w, true, false);
                    }
                    e = last;
                    bearings.Add( (h, f, e, anchor) );
                    var carried = anchor + up * (Vector3.Dot( p[e], up ) - low[h][e]) - p[e];
                    for ( var i = e + 1; i <= Math.Min( frames.Count - 1, e + blend ); i++ )
                    {
                        var w = 1 - (i - e) / (float)(blend + 1); w = w * w * (3 - 2 * w);
                        if ( w >= goals[h][i].Weight ) goals[h][i] = (p[i] + carried, w, true, true);
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
    /// <summary>A body on one hand is tilted at most this far to bring its weight over the hand, only where it stays on
    /// that hand at least BalanceMinimumSeconds (a windmill's brief touches are not held long enough to tip the body
    /// over), and the tilt is smoothed over BalanceSigmaFrames so it eases in and out.</summary>
    public const float MaximumBalanceDegrees = 30, BalanceMinimumSeconds = .4f, BalanceSigmaFrames = 4;
    /// <summary>The body's move over a carrying hand is smoothed over this many frames.</summary>
    public const float CarrySigmaFrames = 3;
    /// <summary>A hand off the floor for at most this many frames between two touches stays where it was.</summary>
    public const int ShortLiftFrames = 2;
    /// <summary>Frames over which a flat hand's heading on the floor is smoothed.</summary>
    public const float HeadingSigmaFrames = 3;

    /// <summary>Tilts a body that rests on a single hand about that wrist until its centre of mass is over it: on one
    /// hand it cannot be anywhere else without falling. The breakin clip's one-handed spin leaned 14° off the hand
    /// in the capture where SAM saw 5°, which swung the free hand on a circle a quarter wider. Two hands or a hand and
    /// the head down share the weight, and the body is left as captured.</summary>
    static void Balance( List<XForm[]> frames, TargetRig target, List<(int Hand, int First, int Last, Vector3 Anchor)> bearings,
        Dictionary<int, (Vector3 Position, float Weight, bool Bearing, bool Lifting)[]> goals, Vector3 up, float fps )
    {
        var rig = target.Skeleton; var segments = Segments.Select( s => (A: target.BoneForRole( s.From ), B: target.BoneForRole( s.To ), s.Mass) ).ToArray();
        if ( bearings.Count == 0 || segments.Any( s => s.A is null || s.B is null ) ) return;
        var roots = Enumerable.Range( 0, rig.Count ).Where( b => rig[b].ParentIndex < 0 ).ToArray();
        var world = new XForm[rig.Count]; var minimum = (int)MathF.Ceiling( BalanceMinimumSeconds * fps );
        // The tilt each frame needs, as an axis times an angle, and the hand it turns about; zero where none is needed.
        var needed = new Vector3[frames.Count]; var pivotHand = new int[frames.Count]; Array.Fill( pivotHand, -1 );
        for ( var f = 0; f < frames.Count; f++ )
        {
            var on = bearings.Where( s => s.First <= f && f <= s.Last ).ToArray();
            if ( on.Length != 1 || on[0].Last - on[0].First + 1 < minimum ) continue;
            var hand = on[0].Hand; pivotHand[f] = hand;
            FkUtil.ToWorld( frames[f], rig, world );
            var centre = Vector3.Zero; var mass = 0f;
            foreach ( var (a, b, m) in segments ) { centre += (world[a!.Value].Pos + world[b!.Value].Pos) * .5f * m; mass += m; }
            var lever = centre / mass - world[hand].Pos;
            if ( Vector3.Dot( lever, up ) <= 0 ) continue;
            var axis = Vector3.Cross( Vector3.Normalize( lever ), up );
            if ( axis.LengthSquared() < 1e-12f ) continue;
            var angle = MathF.Min( MathF.Acos( Math.Clamp( Vector3.Dot( Vector3.Normalize( lever ), up ), -1, 1 ) ), MaximumBalanceDegrees * MathF.PI / 180 );
            needed[f] = Vector3.Normalize( axis ) * angle * goals[hand][f].Weight;
        }
        // Smoothed over time, zeros included, so the tilt eases in and out instead of switching with the hand.
        var radius = (int)MathF.Ceiling( 3 * BalanceSigmaFrames );
        for ( var f = 0; f < frames.Count; f++ )
        {
            var sum = Vector3.Zero; var total = 0f; var hand = -1; var nearest = int.MaxValue;
            for ( var j = Math.Max( 0, f - radius ); j <= Math.Min( frames.Count - 1, f + radius ); j++ )
            {
                var g = MathF.Exp( -.5f * (j - f) * (j - f) / (BalanceSigmaFrames * BalanceSigmaFrames) ); sum += needed[j] * g; total += g;
                if ( pivotHand[j] >= 0 && Math.Abs( j - f ) < nearest ) { nearest = Math.Abs( j - f ); hand = pivotHand[j]; }
            }
            var turn = sum / total;
            if ( hand < 0 || turn.LengthSquared() < 1e-10f ) continue;
            var tilt = Quaternion.CreateFromAxisAngle( Vector3.Normalize( turn ), turn.Length() );
            FkUtil.ToWorld( frames[f], rig, world );
            var pivot = world[hand].Pos;
            foreach ( var root in roots )
            {
                frames[f][root].Pos = pivot + Vector3.Transform( frames[f][root].Pos - pivot, tilt );
                frames[f][root].Rot = Quaternion.Normalize( tilt * frames[f][root].Rot );
            }
        }
    }

    /// <summary>The direction the palm faces, from the wrist and the index, middle and outermost knuckles. The back of a
    /// right hand faces along (knuckle line × index-to-outermost); a left hand mirrors it.</summary>
    static Vector3 PalmNormal( XForm[] world, int[] chain, int hand, bool left )
    {
        var along = world[chain[3]].Pos - world[hand].Pos; var across = world[chain[0]].Pos - world[chain[^3]].Pos;
        var normal = Vector3.Cross( along, across );
        if ( normal.LengthSquared() < 1e-10f ) return Vector3.Zero;
        normal = Vector3.Normalize( normal );
        return left ? normal : -normal;
    }

    /// <summary>Turns each finger joint toward straight, by <paramref name="weight"/>: each segment continues the one
    /// before, so a flat palm puts the fingers along the floor. Twist is kept.</summary>
    static void Straighten( XForm[] frame, int[] chain, float weight )
    {
        for ( var finger = 0; finger < chain.Length / 3; finger++ )
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

    /// <summary>Per frame of each hold, the heading a flat hand has on the floor (toward its middle knuckle), smoothed over
    /// <see cref="HeadingSigmaFrames"/> within the hold; NaN outside holds. Through the breakin clip's spin on one hand the
    /// support hand turned with the body, 30° a frame, which smoothing keeps; in the closing freeze the capture wobbled it
    /// 33° over three frames and back, which it removes.</summary>
    static double[] Headings( List<XForm[]> frames, HumanoidMocapX.Core.Skeleton.Skeleton rig, int hand, int[] chain, bool left,
        (Vector3 Position, float Weight, bool Bearing, bool Lifting)[] goals, Vector3 up )
    {
        var raw = new double[frames.Count]; Array.Fill( raw, double.NaN ); var world = new XForm[rig.Count];
        for ( var f = 0; f < frames.Count; f++ )
        {
            if ( !goals[f].Bearing || goals[f].Weight <= 0 ) continue;
            FkUtil.ToWorld( frames[f], rig, world );
            var flat = Quaternion.Normalize( FromTo( PalmNormal( world, chain, hand, left ), -up ) * world[hand].Rot );
            raw[f] = Heading( Vector3.Transform( frames[f][chain[3]].Pos, flat ), up );
            if ( f > 0 && !double.IsNaN( raw[f - 1] ) ) raw[f] = raw[f - 1] + Math.IEEERemainder( raw[f] - raw[f - 1], 2 * Math.PI );
        }
        var smoothed = new double[frames.Count]; var radius = (int)Math.Ceiling( 3 * HeadingSigmaFrames );
        for ( var f = 0; f < frames.Count; f++ )
        {
            smoothed[f] = double.NaN;
            if ( double.IsNaN( raw[f] ) ) continue;
            double sum = 0, total = 0;
            // Only within this hold: the unwrapped heading is continuous there.
            for ( var j = f; j >= Math.Max( 0, f - radius ) && !double.IsNaN( raw[j] ); j-- ) Add( j );
            for ( var j = f + 1; j <= Math.Min( frames.Count - 1, f + radius ) && !double.IsNaN( raw[j] ); j++ ) Add( j );
            smoothed[f] = sum / total;
            void Add( int j ) { var g = Math.Exp( -.5 * (j - f) * (j - f) / (HeadingSigmaFrames * HeadingSigmaFrames) ); sum += g * raw[j]; total += g; }
        }
        return smoothed;
    }

    /// <summary>The angle of <paramref name="v"/> on the floor, increasing with turns about <paramref name="up"/>.</summary>
    static float Heading( Vector3 v, Vector3 up ) => MathF.Atan2( Vector3.Dot( v, Vector3.Cross( up, Vector3.UnitX ) ), Vector3.Dot( v, Vector3.UnitX ) );

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
