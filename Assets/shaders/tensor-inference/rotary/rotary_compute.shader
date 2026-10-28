HEADER
{
    Description = "In-place adjacent or split-half rotary embedding with precomputed factors";
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
    RWStructuredBuffer<float> Values < Attribute("Values"); >;
    StructuredBuffer<float> Factors < Attribute("Factors"); >;
    uint Pairs < Attribute("Pairs"); >;
    uint Width < Attribute("Width"); >;
    uint HeadWidth < Attribute("HeadWidth"); >;
    uint Positions < Attribute("Positions"); >;
    uint SplitHalves < Attribute("SplitHalves"); >;

    [numthreads(TENSOR_LANES, 1, 1)]
    void MainCs(uint3 thread : SV_DispatchThreadID)
    {
        uint pair = TensorIndex(thread);
        if (pair >= Pairs) return;
        uint index = pair * 2;
        uint position = (index / Width) % Positions;
        uint factor = position * HeadWidth + index % HeadWidth;
        float cosine = Factors[factor], sine = Factors[factor + 1];
        uint other = index + 1;
        if (SplitHalves != 0)
        {
            index = (index / HeadWidth) * HeadWidth + (index % HeadWidth) / 2;
            other = index + HeadWidth / 2;
        }
        float real = Values[index], imaginary = Values[other];
        Values[index] = real * cosine - imaginary * sine;
        Values[other] = real * sine + imaginary * cosine;
    }
}
