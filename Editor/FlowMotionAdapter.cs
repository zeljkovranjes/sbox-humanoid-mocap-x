using System;
using System.Linq;
using System.Numerics;
using HumanoidMocapX.Core.Mapping;
using HumanoidMocapX.Core.Motion;

namespace HumanoidMocapX;

using Vector3 = System.Numerics.Vector3;

/// <summary>Editor integration with Humanoid Mocap's shared motion format.</summary>
public static class FlowMotionAdapter
{
    static readonly BoneRole[] Roles = {
        BoneRole.Hips, BoneRole.UpperLegL, BoneRole.UpperLegR, BoneRole.Spine0,
        BoneRole.LowerLegL, BoneRole.LowerLegR, BoneRole.Spine1, BoneRole.FootL,
        BoneRole.FootR, BoneRole.Spine2, BoneRole.ToeL, BoneRole.ToeR,
        BoneRole.Neck, BoneRole.ClavicleL, BoneRole.ClavicleR, BoneRole.Head,
        BoneRole.UpperArmL, BoneRole.UpperArmR, BoneRole.LowerArmL, BoneRole.LowerArmR,
        BoneRole.HandL, BoneRole.HandR,
        BoneRole.IndexProxL, BoneRole.IndexMidL, BoneRole.IndexDistL,
        BoneRole.MiddleProxL, BoneRole.MiddleMidL, BoneRole.MiddleDistL,
        BoneRole.PinkyProxL, BoneRole.PinkyMidL, BoneRole.PinkyDistL,
        BoneRole.RingProxL, BoneRole.RingMidL, BoneRole.RingDistL,
        BoneRole.ThumbProxL, BoneRole.ThumbMidL, BoneRole.ThumbDistL,
        BoneRole.IndexProxR, BoneRole.IndexMidR, BoneRole.IndexDistR,
        BoneRole.MiddleProxR, BoneRole.MiddleMidR, BoneRole.MiddleDistR,
        BoneRole.PinkyProxR, BoneRole.PinkyMidR, BoneRole.PinkyDistR,
        BoneRole.RingProxR, BoneRole.RingMidR, BoneRole.RingDistR,
        BoneRole.ThumbProxR, BoneRole.ThumbMidR, BoneRole.ThumbDistR
    };
    static readonly int[] ContactJoints = { 7, 10, 8, 11 };

    /// <summary>Converts local axis-angle rotations and world root travel to the shared
    /// motion format. Rejects invalid shapes, non-finite values and unsupported skeletons.</summary>
    public static MotionDocument ToDocument(this FlowMotion motion)
    {
        ArgumentNullException.ThrowIfNull(motion); motion.Validate();
        var document = new MotionDocument
        {
            Name = motion.Name,
            Backend = "FlowHMR",
            ModelVersion = motion.Model == FlowModel.Base ? "flowhmr_base" : "flowhmr_latest",
            SourceVideo = motion.Video,
            SourceFps = motion.Fps,
            Space = MotionSpace.WorldRelative
        };
        var ground = ContactJoints.Min(j => motion.Rest[j][1]);
        for (var j = 0; j < FlowMotion.JointCount; j++)
        {
            var offset = j == 0 ? MotionDocument.V(motion.Rest[j]) - Vector3.UnitY * ground
                : MotionDocument.V(motion.Rest[j]) - MotionDocument.V(motion.Rest[motion.Parents[j]]);
            document.Bones.Add(new MotionBone
            {
                Name = Roles[j].ToString(),
                Role = Roles[j],
                Parent = motion.Parents[j],
                RestPosition = MotionDocument.A(offset),
                Group = j >= 22 ? "fingers" : "body"
            });
        }
        for (var f = 0; f < motion.Poses.Length; f++) document.Frames.Add(Frame(motion, document, f));
        for (var c = 0; c < ContactJoints.Length && motion.Contacts.Length > 0; c++)
        {
            var probabilities = new float[motion.Poses.Length];
            for (var f = 0; f < probabilities.Length; f++) probabilities[f] = motion.Contacts[f][c];
            document.StationaryJoints.Add(new StationaryJointTrack
            {
                Bone = Roles[ContactJoints[c]].ToString(),
                Source = "FlowHMR predicted foot contacts",
                Probability = probabilities
            });
        }
        document.Diagnostics.Add("World motion, body shape and metric scale are model estimates, not calibrated measurements.");
        document.Diagnostics.Add(motion.Contacts.Length == 0 ? "No foot-contact predictions were returned; foot locking is unavailable."
            : "Foot locking uses predicted left ankle, left toe, right ankle and right toe contacts on the final target rig.");
        document.Validate();
        return document;
    }

    static MotionFrame Frame(FlowMotion motion, MotionDocument document, int f)
    {
        var frame = new MotionFrame
        {
            Time = motion.StartTime + f / motion.Fps,
            Positions = new float[FlowMotion.JointCount][],
            Rotations = new float[FlowMotion.JointCount][],
            Evidence = new JointEvidence[FlowMotion.JointCount]
        };
        for (var j = 0; j < FlowMotion.JointCount; j++)
        {
            // SMPL translation is added to the shaped pelvis; it is not the pelvis position itself.
            frame.Positions[j] = j == 0 ? MotionDocument.A(MotionDocument.V(motion.Rest[0]) + MotionDocument.V(motion.Translations[f]))
                : document.Bones[j].RestPosition.ToArray();
            var rotation = new Vector3(motion.Poses[f][j * 3], motion.Poses[f][j * 3 + 1], motion.Poses[f][j * 3 + 2]);
            var angle = rotation.Length();
            frame.Rotations[j] = MotionDocument.A(angle < 1e-8f ? Quaternion.Identity
                : Quaternion.CreateFromAxisAngle(rotation / angle, angle));
            frame.Evidence[j] = JointEvidence.Reconstructed;
        }
        return frame;
    }

}
