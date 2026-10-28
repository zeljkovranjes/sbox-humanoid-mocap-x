HEADER
{
    Description = "Float32 activations, modulation and residual updates";
}
MODES
{
    Default();
}
COMMON
{
    #include "system.fxc"
}
CS
{
    #include "tensor-inference/common/dispatch.hlsl"
    StructuredBuffer<float> Input < Attribute("Input"); >;
    StructuredBuffer<float> Other < Attribute("Other"); >;
    StructuredBuffer<float> Parameters < Attribute("Parameters"); >;
    RWStructuredBuffer<float> Output < Attribute("Output"); >;
    uint Count < Attribute("Count"); >;
    uint Width < Attribute("Width"); >;
    uint Operation < Attribute("Operation"); >;
    uint FirstOffset < Attribute("FirstOffset"); >;
    uint SecondOffset < Attribute("SecondOffset"); >;
    float Amount < Attribute("Amount"); >;

    [numthreads(TENSOR_LANES, 1, 1)]
    void MainCs(uint3 thread : SV_DispatchThreadID)
    {
        uint index = TensorIndex(thread);
        if (index >= Count) return;
        float value = Input[index];
        uint channel = index % Width;
        if (Operation == 0) value /= 1 + exp(-value);
        else if (Operation == 1) value = .5 * value * (1 + tanh(.7978845608028654 * (value + .044715 * value * value * value)));
        else if (Operation == 2) value += Other[index];
        else if (Operation == 3) value = value * (1 + Parameters[SecondOffset + channel]) + Parameters[FirstOffset + channel];
        else if (Operation == 4) value += Other[index] * Parameters[FirstOffset + channel];
        else if (Operation == 5) value += Amount * (Other[index] - value);
        else if (Operation == 6)
        {
            float z = abs(value) * .7071067811865475;
            float t = 1 / (1 + .3275911 * z);
            // Abramowitz-Stegun 7.1.26; erfc keeps negative-tail precision.
            float erfc = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * exp(-z * z);
            value *= value >= 0 ? 1 - .5 * erfc : .5 * erfc;
        }
        else if (Operation == 7) value = (value / (1 + exp(-value))) * Other[index];
        Output[index] = value;
    }
}
