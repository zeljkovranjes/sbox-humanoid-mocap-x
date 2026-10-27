using HumanoidMocapX.Inference;

static class SamCompactPoseTests
{
    internal static void Run()
    {
        var bodyAngles = DenoiserTests.Wave(133, .61f); var handAngles = DenoiserTests.Wave(27, .43f);
        var body = SamPoseFixture.Body(bodyAngles); var hand = SamPoseFixture.Hand(handAngles);
        var bodyResult = new float[133]; var handResult = new float[27];
        for (var i = 0; i < 4; i++) { SamCompactPose.Body(body, bodyResult); SamCompactPose.Hand(hand, handResult); }
        Compare(bodyAngles, bodyResult, 2e-7); Compare(handAngles, handResult, 2e-7);
        var before = GC.GetAllocatedBytesForCurrentThread(); SamCompactPose.Body(body, bodyResult); SamCompactPose.Hand(hand, handResult);
        MotionTests.Equal(0L, GC.GetAllocatedBytesForCurrentThread() - before);
        SamCompactPose.Body(new float[260], bodyResult); SamCompactPose.Hand(new float[54], handResult);
        MotionTests.Check(bodyResult.All(v => v == 0) && handResult.All(v => v == 0), "Clamped zero continuous rotations must stay finite");
        MotionTests.Throws<ArgumentException>(() => SamCompactPose.Body(body.AsSpan(1), bodyResult));
        MotionTests.Throws<ArgumentException>(() => SamCompactPose.Hand(hand, handResult.AsSpan(1)));
        MotionTests.Throws<ArgumentException>(() => SamCompactPose.Body(body, body.AsSpan(0, 133)));
        MotionTests.Throws<ArgumentException>(() => SamCompactPose.Hand(hand, hand.AsSpan(0, 27)));
        body[0] = float.NaN; hand[0] = float.PositiveInfinity;
        MotionTests.Throws<ArgumentException>(() => SamCompactPose.Body(body, bodyResult));
        MotionTests.Throws<ArgumentException>(() => SamCompactPose.Hand(hand, handResult));
        Rotations();
        Console.WriteLine("PASS: SAM continuous body/hand ordering, 6D columns, sine/cosine pairs, translations, global Euler conventions and gimbal locks");
    }

    static void Rotations()
    {
        var actual = new float[3]; var reconstructed = new float[9];
        foreach (var angles in new[] { new[] { .23, -.31, .47 }, new[] { 0.0, 0.0, Math.PI }, new[] { Math.PI, 0.0, 0.0 },
            new[] { .4, Math.PI / 2, -.6 }, new[] { .4, -Math.PI / 2, -.6 }, new[] { -.5, 1.1, .7 } })
        {
            var columns = SamPoseFixture.Columns(angles[0], angles[1], angles[2]);
            var reference = SamPoseFixture.Matrix(angles[0], angles[1], angles[2]);
            foreach (var global in new[] { false, true })
            {
                if (global) SamRotation.GlobalZYX(columns, actual); else SamRotation.CompactXYZ(columns, actual);
                var xyz = global ? actual.Reverse().ToArray() : actual;
                SamRotation.MatrixXYZ(xyz, reconstructed);
                for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++) MotionTests.Near(reference[r, c], reconstructed[r * 3 + c], 8e-7);
                if (Math.Abs(angles[1]) == Math.PI / 2) MotionTests.Equal(0f, actual[2]);
                else if (Math.Abs(angles[0]) < 3 && Math.Abs(angles[2]) < 3) Compare(angles.Select(a => (float)a).ToArray(), xyz, 3e-7);
            }
            // Non-unit columns and a parallel component must represent the same rotation.
            for (var i = 0; i < 3; i++) { columns[i + 3] = columns[i + 3] * .7f + columns[i] * .2f; columns[i] *= 1.7f; }
            SamRotation.GlobalZYX(columns, actual);
            SamRotation.MatrixXYZ(actual.Reverse().ToArray(), reconstructed);
            for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++) MotionTests.Near(reference[r, c], reconstructed[r * 3 + c], 8e-7);
        }
    }

    internal static void Compare(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, double tolerance)
    { MotionTests.Equal(expected.Length, actual.Length); for (var i = 0; i < expected.Length; i++) MotionTests.Near(expected[i], actual[i], tolerance); }
}
