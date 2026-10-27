// C# adaptation of YOLOX's inference-time batch normalization. See THIRD_PARTY_NOTICES.md.
using System;
using System.IO;
using System.Threading;

namespace HumanoidMocapX.Inference;

internal sealed record YoloFilter(float[] Weight, float[] Bias);

internal static class YoloWeights
{
    internal const float Epsilon = 1e-3f;

    /// <summary>Folds evaluation batch normalization into OIHW convolution weights in
    /// place and returns its bias. Rejects invalid statistics; discard weights on failure.</summary>
    internal static float[] Fold(float[] weights, ReadOnlySpan<float> scale, ReadOnlySpan<float> bias,
        ReadOnlySpan<float> mean, ReadOnlySpan<float> variance, CancellationToken cancellation = default)
    {
        if (scale.Length == 0 || weights.Length % scale.Length != 0 || weights.Length == 0
            || bias.Length != scale.Length || mean.Length != scale.Length || variance.Length != scale.Length)
            throw new InvalidDataException("YOLOX batch normalization dimensions disagree.");
        var outputBias = new float[scale.Length]; var width = weights.Length / scale.Length;
        for (var channel = 0; channel < scale.Length; channel++)
        {
            cancellation.ThrowIfCancellationRequested();
            if (!float.IsFinite(scale[channel]) || !float.IsFinite(bias[channel]) || !float.IsFinite(mean[channel])
                || !float.IsFinite(variance[channel]) || variance[channel] < 0)
                throw new InvalidDataException("YOLOX batch normalization contains invalid statistics.");
            var multiplier = scale[channel] / MathF.Sqrt(variance[channel] + Epsilon);
            outputBias[channel] = bias[channel] - mean[channel] * multiplier;
            if (!float.IsFinite(outputBias[channel])) throw new InvalidDataException("YOLOX folded bias overflowed.");
            for (var index = channel * width; index < (channel + 1) * width; index++)
            {
                weights[index] *= multiplier;
                if (!float.IsFinite(weights[index])) throw new InvalidDataException("YOLOX folded weight is not finite.");
            }
        }
        return outputBias;
    }
}
