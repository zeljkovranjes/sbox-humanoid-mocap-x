using System.Numerics;
using System.IO.Compression;
using System.Text;
using HumanoidMocapX.Core.Maths;
using HumanoidMocapX.Core.Motion;
using HumanoidMocapX.EditorTools.MotionBricks;
using HumanoidMocapX;
using HumanoidMocapX.EditorTools;
using HumanoidMocapX.Inference;

static class DecoderTests
{
    internal static void Run()
    {
        foreach (var frames in new[] { 1, 3, 150 }) Conversion(frames);
        RejectedInputs(); Checkpoint();
        Console.WriteLine("PASS: latent decoding, shape averaging, root travel, rotation conventions, contacts and decoder checkpoint buffers");
    }

    internal static void Released(string checkpoint, string bodyPath)
    {
        var joints = SmplModel.Load(bodyPath);
        MotionTests.Equal(156, joints.Template.Length); MotionTests.Equal(2496, joints.Directions.Length);
        foreach (var value in joints.Template.Concat(joints.Directions))
            MotionTests.Check(float.IsFinite(value), "Released SMPL-H regression produced a non-finite value");
        var motion = FlowCheckpoint.LoadDecoder(checkpoint, bodyPath).Decode(new float[3 * FlowSampler.Channels]);
        motion.Validate();
        Console.WriteLine("PASS: released " + Path.GetFileName(checkpoint) + " decoder with separate SMPL-H asset (mean latent, not inferred motion)");
    }

    static void Conversion(int frames)
    {
        var fixture = Fixture(frames);
        var decoder = new FlowDecoder(fixture.Mean, fixture.Deviation, fixture.Template, fixture.Directions);
        var motion = decoder.Decode(fixture.Latent);
        var document = motion.ToDocument();
        for (var joint = 0; joint < 52; joint++)
            for (var axis = 0; axis < 3; axis++)
            {
                var index = joint * 3 + axis; double expected = fixture.Template[index];
                for (var shape = 0; shape < 16; shape++)
                    expected += ((frames - 1) / 2d + shape * .1) * fixture.Directions[index * 16 + shape];
                MotionTests.Near(expected, motion.Rest[joint][axis], 2e-6);
            }
        for (var frame = 0; frame < frames; frame++)
        {
            var expectedX = -(.3f + .1f * frame + frame * (frame + 1) / 2f - motion.Rest[0][0]);
            var expectedY = 1.4f + .05f * frame - motion.Rest[0][1];
            var expectedZ = -(-.2f + .07f * frame - frame * (frame + 1) - motion.Rest[0][2]);
            MotionTests.Near(expectedX, motion.Translations[frame][0], .002);
            MotionTests.Near(expectedY, motion.Translations[frame][1]);
            MotionTests.Near(expectedZ, motion.Translations[frame][2], .004);
            MotionTests.Near(1.4f + .05f * frame, document.Frames[frame].Positions[0][1]);
            for (var joint = 0; joint < 52; joint++)
            {
                var expected = Rotation(joint);
                if (joint == 0) expected = new Quaternion(0, 1, 0, 0) * expected;
                var actual = MotionDocument.Q(document.Frames[frame].Rotations[joint]);
                MotionTests.Near(0, MathQ.AngleBetween(expected, actual), 1e-6);
            }
            MotionTests.Near(0, document.StationaryJoints[0].Probability[frame]);
            MotionTests.Near(.25, document.StationaryJoints[1].Probability[frame]);
            MotionTests.Near(.8, document.StationaryJoints[2].Probability[frame]);
            MotionTests.Near(1, document.StationaryJoints[3].Probability[frame]);
        }
        // Caller buffers and earlier outputs cannot change a cached decoder's next result.
        var expectedRest = motion.Rest[0][1]; motion.Rest[0][1] += 10;
        Array.Fill(fixture.Mean, float.NaN); Array.Fill(fixture.Deviation, float.NaN);
        Array.Fill(fixture.Template, float.NaN); Array.Fill(fixture.Directions, float.NaN);
        MotionTests.Near(expectedRest, decoder.Decode(fixture.Latent).Rest[0][1]);
    }

    static void RejectedInputs()
    {
        var fixture = Fixture(1);
        FlowDecoder Create() => new(fixture.Mean, fixture.Deviation, fixture.Template, fixture.Directions);
        var decoder = Create();
        MotionTests.Throws<ArgumentException>(() => decoder.Decode(Array.Empty<float>()));
        MotionTests.Throws<ArgumentException>(() => decoder.Decode(new float[492]));
        MotionTests.Throws<ArgumentException>(() => decoder.Decode(new float[5001 * 491]));
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        MotionTests.Throws<OperationCanceledException>(() => decoder.Decode(fixture.Latent, cancellation.Token));
        fixture.Latent[10] = float.NaN;
        MotionTests.Throws<InvalidDataException>(() => decoder.Decode(fixture.Latent));
        fixture = Fixture(1);
        for (var component = 0; component < 6; component++) Set(fixture, 0, 159 + component, 0);
        try { decoder.Decode(fixture.Latent); throw new Exception("Expected a degenerate rotation rejection"); }
        catch (InvalidDataException error) { MotionTests.Check(error.Message.Contains("frame 0, joint 0"), error.Message); }
        fixture = Fixture(1); fixture.Deviation[0] = 0;
        MotionTests.Throws<InvalidDataException>(() => Create());
        fixture = Fixture(1); fixture.Directions[200] = float.NaN;
        MotionTests.Throws<InvalidDataException>(() => Create());
        fixture = Fixture(1);
        MotionTests.Throws<InvalidDataException>(() => new FlowDecoder(fixture.Mean, fixture.Deviation, new float[3], fixture.Directions));
        fixture.Mean[0] = float.NaN; MotionTests.Throws<InvalidDataException>(() => Create());
        fixture = Fixture(2); fixture.Deviation[0] = float.MaxValue;
        decoder = Create(); fixture.Latent[491] = float.MaxValue;
        MotionTests.Throws<InvalidDataException>(() => decoder.Decode(fixture.Latent));
    }

    static void Checkpoint()
    {
        var fixture = Fixture(3); var buffers = Buffers(fixture);
        var folder = Path.Combine(Path.GetTempPath(), "flow-decoder-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(folder);
        try
        {
            var path = Path.Combine(folder, "decoder.ckpt"); CheckpointFixture.Write(path, buffers, prefix: "");
            var bodyPath = Path.Combine(folder, "body.npz"); WriteBody(bodyPath, fixture);
            var actual = FlowCheckpoint.LoadDecoder(path, bodyPath).Decode(fixture.Latent);
            var expected = new FlowDecoder(fixture.Mean, fixture.Deviation, fixture.Template, fixture.Directions).Decode(fixture.Latent);
            for (var frame = 0; frame < 3; frame++)
            {
                MotionTests.Check(actual.Poses[frame].AsSpan().SequenceEqual(expected.Poses[frame]), "Checkpoint rotation statistics changed");
                MotionTests.Check(actual.Translations[frame].AsSpan().SequenceEqual(expected.Translations[frame]), "Checkpoint root statistics changed");
                MotionTests.Check(actual.Contacts[frame].AsSpan().SequenceEqual(expected.Contacts[frame]), "Checkpoint contact statistics changed");
            }
            for (var joint = 0; joint < 52; joint++)
                MotionTests.Check(actual.Rest[joint].AsSpan().SequenceEqual(expected.Rest[joint]), "Checkpoint shaped rest changed");
            using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
            MotionTests.Throws<OperationCanceledException>(() => FlowCheckpoint.LoadDecoder(path, bodyPath, cancellation.Token));
            var index = 0;
            foreach (var corrupt in new Action<Dictionary<string, Tensor>>[] {
                b => b.Remove("rep_shapes_mean"), b => b["local_rot_data_std"].Data[0] = -1,
                b => b["smooth_root_vel_mean"].Data[0] = float.NaN
            })
            {
                buffers = Buffers(fixture); corrupt(buffers);
                var bad = Path.Combine(folder, "bad-" + index++ + ".ckpt"); CheckpointFixture.Write(bad, buffers, prefix: "");
                MotionTests.Throws<InvalidDataException>(() => FlowCheckpoint.LoadDecoder(bad, bodyPath));
            }
            foreach (var corruption in new[] { "hierarchy", "ids", "shape", "nan", "missing" })
            {
                var bad = Path.Combine(folder, corruption + ".npz"); WriteBody(bad, fixture, corruption);
                MotionTests.Throws<InvalidDataException>(() => FlowCheckpoint.LoadDecoder(path, bad));
            }
        }
        finally { Directory.Delete(folder, true); }
    }

    static Dictionary<string, Tensor> Buffers(Values fixture)
    {
        var buffers = new Dictionary<string, Tensor>();
        Pair("smooth_root_vel", new[] { fixture.Mean[0], 123f, fixture.Mean[1] }, new[] { fixture.Deviation[0], 456f, fixture.Deviation[1] }, 3);
        Pair("smooth_root_pos", new[] { 123f, fixture.Mean[2], 234f }, new[] { 345f, fixture.Deviation[2], 456f }, 3);
        Pair("local_joints_positions", fixture.Mean[3..159], fixture.Deviation[3..159], 52, 3);
        Pair("local_rot_data", fixture.Mean[159..471], fixture.Deviation[159..471], 52, 6);
        Pair("rep_shapes", fixture.Mean[471..487], fixture.Deviation[471..487], 16);
        Pair("rep_foot_contacts", fixture.Mean[487..491], fixture.Deviation[487..491], 4);
        return buffers;

        void Pair(string name, float[] mean, float[] deviation, params int[] shape)
        {
            var fullShape = new[] { 1, 1 }.Concat(shape).ToArray();
            Add(name + "_mean", mean, fullShape); Add(name + "_std", deviation, fullShape);
        }
        void Add(string name, float[] values, params int[] shape) => buffers.Add(name, new Tensor(name, values.ToArray(), shape.Reverse().ToArray()));
    }

    static void WriteBody(string path, Values fixture, string corruption = "")
    {
        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);
        var hierarchy = new double[104];
        for (var joint = 0; joint < 52; joint++) { hierarchy[joint] = FlowMotion.JointParents[joint]; hierarchy[52 + joint] = joint; }
        hierarchy[0] = uint.MaxValue;
        if (corruption == "hierarchy") hierarchy[2] = .5;
        if (corruption == "ids") hierarchy[53] = 2;
        Write("kintree_table", hierarchy, 2, 52);
        var regressor = new double[52 * 6890]; var template = new double[6890 * 3]; var directions = new double[6890 * 3 * 16];
        for (var joint = 0; joint < 52; joint++)
        {
            regressor[joint * 6890 + joint * 2] = .25; regressor[joint * 6890 + joint * 2 + 1] = .75;
            for (var axis = 0; axis < 3; axis++)
            {
                // Weighted regression must recover the expected joint, not select a vertex.
                template[joint * 6 + axis] = fixture.Template[joint * 3 + axis] * 4;
                for (var shape = 0; shape < 16; shape++)
                    directions[(joint * 6 + axis) * 16 + shape] = fixture.Directions[(joint * 3 + axis) * 16 + shape] * 4;
            }
        }
        if (corruption != "missing") Write("J_regressor", regressor, 52, 6890);
        if (corruption == "nan") template[0] = double.NaN;
        Write("v_template", template, 6890, 3);
        Write("shapedirs", directions, corruption == "shape" ? new[] { 6890, 16, 3 } : new[] { 6890, 3, 16 });

        void Write(string name, double[] values, params int[] shape)
        {
            using var stream = archive.CreateEntry(name + ".npy").Open(); using var writer = new BinaryWriter(stream, Encoding.ASCII);
            var header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + string.Join(", ", shape) + "), }\n";
            writer.Write(new byte[] { 0x93, 78, 85, 77, 80, 89, 1, 0 }); writer.Write((ushort)header.Length);
            writer.Write(Encoding.ASCII.GetBytes(header)); foreach (var value in values) writer.Write(value);
        }
    }

    sealed record Values(float[] Mean, float[] Deviation, float[] Template, float[] Directions, float[] Latent);

    static Values Fixture(int frames)
    {
        var fixture = new Values(Enumerable.Range(0, 491).Select(i => .01f * (i % 7 - 3)).ToArray(),
            Enumerable.Range(0, 491).Select(i => .5f + (i * 3 % 5) * .25f).ToArray(),
            MotionTests.Fixture().Rest.SelectMany(r => r).ToArray(), new float[52 * 3 * 16], new float[frames * 491]);
        fixture.Template[0] = .2f; fixture.Template[2] = -.1f;
        for (var coordinate = 0; coordinate < 156; coordinate++)
            for (var shape = 0; shape < 16; shape++) fixture.Directions[coordinate * 16 + shape] = .001f * (1 + (coordinate + shape) % 7);
        for (var frame = 0; frame < frames; frame++)
        {
            Set(fixture, frame, 0, frame == 0 ? 900 : frame * 30); Set(fixture, frame, 1, frame == 0 ? 800 : frame * -60);
            Set(fixture, frame, 2, 700); // This smooth-root Y must not be added to absolute pelvis Y.
            Set(fixture, frame, 3, .3f + .1f * frame); Set(fixture, frame, 4, 1.4f + .05f * frame); Set(fixture, frame, 5, -.2f + .07f * frame);
            for (var joint = 0; joint < 52; joint++)
            {
                var rotation = Rotation(joint);
                var first = Vector3.Transform(Vector3.UnitX, rotation);
                var second = Vector3.Transform(Vector3.UnitY, rotation) * 3 + first * .4f;
                first *= 2;
                float[] six = { first.X, second.X, first.Y, second.Y, first.Z, second.Z };
                for (var channel = 0; channel < 6; channel++) Set(fixture, frame, 159 + joint * 6 + channel, six[channel]);
            }
            for (var shape = 0; shape < 16; shape++) Set(fixture, frame, 471 + shape, frame + shape * .1f);
            float[] contacts = { -.2f, .25f, .8f, 1.4f };
            for (var contact = 0; contact < 4; contact++) Set(fixture, frame, 487 + contact, contacts[contact]);
        }
        return fixture;
    }

    static Quaternion Rotation(int joint)
    {
        var angle = (joint % 7) switch { 0 => .37f, 1 => 0, 2 => 1e-9f, 3 => MathF.PI, 4 => -1.7f, 5 => -MathF.PI + 1e-5f, _ => 4.1f };
        var axis = joint % 3 == 0 ? Vector3.UnitX : joint % 3 == 1 ? Vector3.UnitY : Vector3.UnitZ;
        return Quaternion.CreateFromAxisAngle(axis, angle);
    }

    static void Set(Values fixture, int frame, int channel, float physical)
        => fixture.Latent[frame * 491 + channel] = (physical - fixture.Mean[channel]) / fixture.Deviation[channel];
}
