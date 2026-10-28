HEADER
{
    Description = "Float32 attention with two visible key ranges per query and 64/128-channel heads";
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
    StructuredBuffer<float> Query < Attribute("Query"); >;
    StructuredBuffer<float> Key < Attribute("Key"); >;
    StructuredBuffer<float> Value < Attribute("Value"); >;
    StructuredBuffer<uint> KeyRanges < Attribute("KeyRanges"); >;
    RWStructuredBuffer<float> Output < Attribute("Output"); >;
    uint Tokens < Attribute("Tokens"); >;
    uint KeyTokens < Attribute("KeyTokens"); >;
    uint Heads < Attribute("Heads"); >;
    uint HeadWidth < Attribute("HeadWidth"); >;

    #define HEAD_WIDTH 64
    #define QUERIES_PER_GROUP_LINE 256
    groupshared float Products[HEAD_WIDTH];

    float Dot(float product, uint lane)
    {
        Products[lane] = product;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint stride = HEAD_WIDTH / 2; stride > 0; stride >>= 1)
        {
            if (lane < stride) Products[lane] += Products[lane + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        float result = Products[0];
        GroupMemoryBarrierWithGroupSync();
        return result;
    }

    [numthreads(HEAD_WIDTH, 1, 1)]
    void MainCs(uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID)
    {
        uint token = group.y * QUERIES_PER_GROUP_LINE + group.x;
        if (token >= Tokens) return;
        uint width = Heads * HeadWidth;
        uint channel = group.z * HeadWidth + local.x;
        bool wide = HeadWidth == 128;
        float query = Query[token * width + channel];
        float querySecond = wide ? Query[token * width + channel + HEAD_WIDTH] : 0;
        float maximum = -3.402823466e+38;
        float total = 0;
        float accumulated = 0;
        float accumulatedSecond = 0;
        // Online softmax avoids storing a token-by-token score matrix.
        for (uint range = 0; range < 2; range++)
        {
            uint at = token * 4 + range * 2;
            uint start = min(KeyRanges[at], KeyTokens);
            uint count = min(KeyRanges[at + 1], KeyTokens - start);
            for (uint index = 0; index < count; index++)
            {
                uint source = (start + index) * width + channel;
                float product = query * Key[source];
                if (wide) product += querySecond * Key[source + HEAD_WIDTH];
                float score = Dot(product, local.x) * rsqrt((float)HeadWidth);
                float nextMaximum = max(maximum, score);
                float previousWeight = exp(maximum - nextMaximum);
                float weight = exp(score - nextMaximum);
                accumulated = accumulated * previousWeight + Value[source] * weight;
                if (wide) accumulatedSecond = accumulatedSecond * previousWeight + Value[source + HEAD_WIDTH] * weight;
                total = total * previousWeight + weight;
                maximum = nextMaximum;
            }
        }
        Output[token * width + channel] = total > 0 ? accumulated / total : 0;
        if (wide) Output[token * width + channel + HEAD_WIDTH] = total > 0 ? accumulatedSecond / total : 0;
    }
}
