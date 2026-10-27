// C# adaptation of YOLOX-L's inference architecture. See THIRD_PARTY_NOTICES.md.
using System;
using System.Collections.Generic;

namespace HumanoidMocapX.Inference;

internal enum YoloOperation { Input, Convolution, Silu, Add, Concat, Focus, MaxPool, Upsample }

internal sealed record YoloNode(YoloOperation Operation, int First, int Second, int Size, int Channels,
    string Weight = null, int Kernel = 1, int Stride = 1, bool Normalize = false)
{
    internal int Elements => Size * Size * Channels;
}

/// <summary>The fixed 640-pixel YOLOX-L inference graph, including all eighty classes.
/// Construction also assigns reusable storage after each tensor's last consumer.</summary>
internal sealed class YoloNetwork
{
    internal const int ImageSize = 640, Classes = 80, PredictionChannels = Classes + 5, Predictions = 8400;
    internal readonly YoloNode[] Nodes;
    internal readonly int[] Heads, Slots, Capacities;
    readonly List<YoloNode> building = new();

    internal YoloNetwork()
    {
        Add(new(YoloOperation.Input, -1, -1, ImageSize, 3));
        var stem = Conv("backbone.backbone.stem.conv", Add(new(YoloOperation.Focus, 0, -1, 320, 12)), 64, 3);
        var dark2 = Stage(stem, 2, 128, 3);
        var dark3 = Stage(dark2, 3, 256, 9);
        var dark4 = Stage(dark3, 4, 512, 9);
        var dark5 = Conv("backbone.backbone.dark5.0", dark4, 1024, 3, 2);
        var pooled = Conv("backbone.backbone.dark5.1.conv1", dark5, 512, 1);
        var pyramid = pooled;
        foreach (var kernel in new[] { 5, 9, 13 })
            pyramid = Concat(pyramid, Add(new(YoloOperation.MaxPool, pooled, -1, 20, 512, Kernel: kernel)));
        dark5 = Conv("backbone.backbone.dark5.1.conv2", pyramid, 1024, 1);
        dark5 = Csp("backbone.backbone.dark5.2", dark5, 1024, 3, false);
        var lateral = Conv("backbone.lateral_conv0", dark5, 512, 1);
        var top = Csp("backbone.C3_p4", Concat(Upsample(lateral), dark4), 512, 3, false);
        var reduced = Conv("backbone.reduce_conv1", top, 256, 1);
        var small = Csp("backbone.C3_p3", Concat(Upsample(reduced), dark3), 256, 3, false);
        var medium = Csp("backbone.C3_n3", Concat(Conv("backbone.bu_conv2", small, 256, 3, 2), reduced), 512, 3, false);
        var large = Csp("backbone.C3_n4", Concat(Conv("backbone.bu_conv1", medium, 512, 3, 2), lateral), 1024, 3, false);
        Heads = new[] { Head(small, 0), Head(medium, 1), Head(large, 2) };
        Nodes = building.ToArray(); building.Clear();
        (Slots, Capacities) = PlanStorage();
    }

    int Stage(int input, int stage, int channels, int depth)
    {
        var prefix = "backbone.backbone.dark" + stage;
        return Csp(prefix + ".1", Conv(prefix + ".0", input, channels, 3, 2), channels, depth, true);
    }

    int Csp(string prefix, int input, int channels, int depth, bool residual)
    {
        var first = Conv(prefix + ".conv1", input, channels / 2, 1);
        var second = Conv(prefix + ".conv2", input, channels / 2, 1);
        for (var index = 0; index < depth; index++)
        {
            var hidden = Conv(prefix + ".m." + index + ".conv1", first, channels / 2, 1);
            var next = Conv(prefix + ".m." + index + ".conv2", hidden, channels / 2, 3);
            first = residual ? Add(new(YoloOperation.Add, first, next, building[first].Size, channels / 2)) : next;
        }
        return Conv(prefix + ".conv3", Concat(first, second), channels, 1);
    }

    int Head(int input, int index)
    {
        var stem = Conv("head.stems." + index, input, 256, 1);
        var classification = Conv("head.cls_convs." + index + ".0", stem, 256, 3);
        classification = Conv("head.cls_convs." + index + ".1", classification, 256, 3);
        classification = Conv("head.cls_preds." + index, classification, Classes, 1, normalize: false);
        var regression = Conv("head.reg_convs." + index + ".0", stem, 256, 3);
        regression = Conv("head.reg_convs." + index + ".1", regression, 256, 3);
        var boxes = Conv("head.reg_preds." + index, regression, 4, 1, normalize: false);
        var objects = Conv("head.obj_preds." + index, regression, 1, 1, normalize: false);
        return Concat(Concat(boxes, objects), classification);
    }

    int Conv(string name, int input, int channels, int kernel, int stride = 1, bool normalize = true)
    {
        var size = (building[input].Size - 1) / stride + 1;
        var convolution = Add(new(YoloOperation.Convolution, input, -1, size, channels, name, kernel, stride, normalize));
        return normalize ? Add(new(YoloOperation.Silu, convolution, -1, size, channels)) : convolution;
    }

    int Concat(int first, int second)
    {
        if (building[first].Size != building[second].Size) throw new InvalidOperationException("YOLOX pyramid dimensions disagree.");
        return Add(new(YoloOperation.Concat, first, second, building[first].Size, building[first].Channels + building[second].Channels));
    }

    int Upsample(int input) => Add(new(YoloOperation.Upsample, input, -1, building[input].Size * 2, building[input].Channels));
    int Add(YoloNode node) { building.Add(node); return building.Count - 1; }

    (int[], int[]) PlanStorage()
    {
        var last = new int[Nodes.Length]; var slots = new int[Nodes.Length];
        for (var index = 0; index < Nodes.Length; index++)
        {
            if (Nodes[index].First >= 0) last[Nodes[index].First] = index;
            if (Nodes[index].Second >= 0) last[Nodes[index].Second] = index;
        }
        foreach (var head in Heads) last[head] = Nodes.Length;
        var capacities = new List<int>(); var availableAfter = new List<int>();
        for (var index = 0; index < Nodes.Length; index++)
        {
            var slot = -1; var count = Nodes[index].Elements;
            for (var candidate = 0; candidate < capacities.Count; candidate++)
                if (availableAfter[candidate] < index && capacities[candidate] >= count
                    && (slot < 0 || capacities[candidate] < capacities[slot])) slot = candidate;
            if (slot < 0) { slot = capacities.Count; capacities.Add(count); availableAfter.Add(0); }
            slots[index] = slot; availableAfter[slot] = last[index];
        }
        return (slots, capacities.ToArray());
    }
}
