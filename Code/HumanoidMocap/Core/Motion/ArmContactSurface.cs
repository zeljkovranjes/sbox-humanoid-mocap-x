using System;
using System.Numerics;
using HumanoidMocapX.Core.Maths;

namespace HumanoidMocapX.Core.Motion;
using Vector3 = System.Numerics.Vector3;

/// <summary>Bone-derived torso proxy with elliptical sides and rounded ends.</summary>
internal readonly struct ArmContactSurface
{
    readonly Vector3 bottom;
    readonly float height, width, depth, cap;
    internal readonly Vector3 Up, Across, Forward;

    ArmContactSurface(Vector3 bottom, Vector3 up, Vector3 across, float height, float span)
    {
        this.bottom = bottom; Up = up; Across = across; Forward = Vector3.Cross(across, up);
        this.height = height * .9f;
        width = (CaptureArmClearance.HalfWidth + CaptureArmClearance.ArmRadius) * span;
        depth = (CaptureArmClearance.HalfDepth + CaptureArmClearance.ArmRadius) * span;
        cap = CaptureArmClearance.ArmRadius * span;
    }

    internal static bool TryCreate(XForm[] world, int hips, int neck, int left, int right, out ArmContactSurface result)
    {
        result = default;
        var axis = world[neck].Pos - world[hips].Pos;
        var height = axis.Length();
        if (height < 1e-5f) return false;
        var up = axis / height;
        var shoulders = world[left].Pos - world[right].Pos;
        var span = shoulders.Length() * .5f;
        var across = shoulders - up * Vector3.Dot(shoulders, up);
        if (span < 1e-5f || across.LengthSquared() < 1e-10f) return false;
        result = new(world[hips].Pos, up, Vector3.Normalize(across), height, span);
        return true;
    }

    internal Vector3 ToLocalOffset(Vector3 value) => new(Vector3.Dot(value, Across), Vector3.Dot(value, Up), Vector3.Dot(value, Forward));
    internal Vector3 ToWorldOffset(Vector3 value) => Across * value.X + Up * value.Y + Forward * value.Z;
    internal Vector3 Outward(Vector3 point, Vector3 fallback)
    {
        var local = Normalized(point);
        var outward = Across * (local.X / width) + Forward * (local.Z / depth);
        return outward.LengthSquared() > 1e-8f ? outward : fallback;
    }
    Vector3 Normalized(Vector3 point)
    {
        var local = ToLocalOffset(point - bottom);
        return new(local.X / width, local.Y / cap, local.Z / depth);
    }

    /// <summary>Finds the deepest point of a segment, including the end caps, without sampling gaps.</summary>
    internal Vector3 ProjectSegment(Vector3 start, Vector3 end, Vector3 fallback, out float fraction)
    {
        var a = Normalized(start); var d = Normalized(end) - a; var top = height / cap;
        fraction = 0;
        var best = DistanceSquared(a, top);
        Consider(a, d, top, 1, ref fraction, ref best);
        var radial = d.X * d.X + d.Z * d.Z;
        if (radial > 1e-12f) Consider(a, d, top, -(a.X * d.X + a.Z * d.Z) / radial, ref fraction, ref best);
        var length = d.LengthSquared();
        if (length > 1e-12f)
        {
            Consider(a, d, top, -Vector3.Dot(a, d) / length, ref fraction, ref best);
            Consider(a, d, top, Vector3.Dot(new Vector3(0, top, 0) - a, d) / length, ref fraction, ref best);
        }
        if (MathF.Abs(d.Y) > 1e-12f)
        {
            Consider(a, d, top, -a.Y / d.Y, ref fraction, ref best);
            Consider(a, d, top, (top - a.Y) / d.Y, ref fraction, ref best);
        }
        if (best >= 1) return Vector3.Zero;
        var point = a + d * fraction;
        point.Y -= Math.Clamp(point.Y, 0, top);
        // All contacts on a hand use one escape direction, avoiding front/back flips
        // when a different finger or forearm point becomes the deepest contact.
        var localDirection = ToLocalOffset(fallback);
        var direction = new Vector3(localDirection.X / width, 0, localDirection.Z / depth);
        if (direction.LengthSquared() < 1e-12f) direction = Vector3.UnitZ;
        direction = Vector3.Normalize(direction);
        var along = Vector3.Dot(point, direction);
        var distance = -along + MathF.Sqrt(along * along + 1.0002f - best);
        var displacement = direction * distance;
        return ToWorldOffset(new(displacement.X * width, displacement.Y * cap, displacement.Z * depth));
    }

    static float DistanceSquared(Vector3 point, float top)
    {
        point.Y -= Math.Clamp(point.Y, 0, top);
        return point.LengthSquared();
    }

    static void Consider(Vector3 start, Vector3 delta, float top, float candidate, ref float fraction, ref float best)
    {
        candidate = Math.Clamp(candidate, 0, 1);
        var value = DistanceSquared(start + delta * candidate, top);
        if (value >= best) return;
        best = value; fraction = candidate;
    }
}
