using System;
using System.IO;
using System.IO.Compression;
using System.Threading;
using HumanoidMocapX.Inference;

namespace HumanoidMocapX.EditorTools;

/// <summary>Reads only numeric SMPL-H arrays with the shared NPY reader; never
/// evaluates pickle objects. Run off the editor thread after verifying the asset.</summary>
internal static class SmplModel
{
    internal static SmplJoints Load(string path, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        using var archive = ZipFile.OpenRead(path);
        var hierarchy = Read("kintree_table", 2, SmplJoints.Joints);
        for (var joint = 0; joint < SmplJoints.Joints; joint++)
        {
            // Upstream replaces the unsigned root sentinel with -1.
            if (joint > 0 && hierarchy[joint] != FlowMotion.JointParents[joint]
                || hierarchy[SmplJoints.Joints + joint] != joint)
                throw new InvalidDataException("SMPL-H kintree_table is not the supported 52-joint hierarchy.");
        }
        return SmplJoints.Regress(Read("J_regressor", SmplJoints.Joints, SmplJoints.Vertices),
            Read("v_template", SmplJoints.Vertices, 3), Read("shapedirs", SmplJoints.Vertices, 3, SmplJoints.Shapes), cancellation);

        float[] Read(string name, params int[] shape)
        {
            var array = NumpyArray.Read(archive, name + ".npy", cancellation);
            if (!array.Shape.AsSpan().SequenceEqual(shape))
                throw new InvalidDataException("Incorrect SMPL-H array shape: " + name);
            return array.Values;
        }
    }
}
