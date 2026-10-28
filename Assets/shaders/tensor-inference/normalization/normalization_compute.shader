HEADER
{
    Description = "Float32 layer and RMS normalization with optional affine parameters";
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
    StructuredBuffer<float> Input < Attribute("Input"); >;
    StructuredBuffer<float> Scale < Attribute("Scale"); >;
    StructuredBuffer<float> Bias < Attribute("Bias"); >;
    RWStructuredBuffer<float> Output < Attribute("Output"); >;
    uint Rows < Attribute("Rows"); >;
    uint Width < Attribute("Width"); >;
    uint HasScale < Attribute("HasScale"); >;
    uint HasBias < Attribute("HasBias"); >;
    uint Rms < Attribute("Rms"); >;
    float Epsilon < Attribute("Epsilon"); >;

    #define LANES 64
    #define ROWS_PER_GROUP_LINE 256
    groupshared float Sums[LANES];

    float ReduceSum(float value, uint lane)
    {
        Sums[lane] = value;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint stride = LANES / 2; stride > 0; stride >>= 1)
        {
            if (lane < stride) Sums[lane] += Sums[lane + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        float result = Sums[0];
        GroupMemoryBarrierWithGroupSync();
        return result;
    }

    [numthreads(LANES, 1, 1)]
    void MainCs(uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID)
    {
        uint row = group.y * ROWS_PER_GROUP_LINE + group.x;
        if (row >= Rows) return;
        uint offset = row * Width;
        float sum = 0;
        for (uint channel = local.x; channel < Width; channel += LANES) sum += Input[offset + channel];
        float mean = Rms != 0 ? 0 : ReduceSum(sum, local.x) / Width;
        float squares = 0;
        for (uint channel = local.x; channel < Width; channel += LANES)
        {
            float delta = Input[offset + channel] - mean;
            squares += delta * delta;
        }
        float inverse = rsqrt(ReduceSum(squares, local.x) / Width + Epsilon);
        for (uint channel = local.x; channel < Width; channel += LANES)
        {
            float value = (Input[offset + channel] - mean) * inverse;
            if (HasScale != 0) value *= Scale[channel];
            if (HasBias != 0) value += Bias[channel];
            Output[offset + channel] = value;
        }
    }
}
