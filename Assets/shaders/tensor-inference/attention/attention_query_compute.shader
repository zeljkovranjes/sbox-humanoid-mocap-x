HEADER
{
    Description = "Float32 attention, one thread per query, for 64-channel heads whose queries all see the same key ranges";
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
    // Always 0. XOR with a value the compiler cannot see forces a product to be rounded, which
    // `precise` alone does not guarantee through this engine's shader compiler.
    uint Zero < Attribute("Zero"); >;

    #define HEAD_WIDTH 64
    #define QUERIES 64
    #define KEY_TILE 16
    groupshared float Keys[KEY_TILE * HEAD_WIDTH];
    groupshared float Values[KEY_TILE * HEAD_WIDTH];

    float Rounded(float value) { return asfloat(asuint(value) ^ Zero); }

    // Replays attention_compute's groupshared tree exactly: rounded products, then pairwise
    // sums at strides 32, 16, ..., 1. The reference cannot fuse a product into an add because its
    // products pass through memory, so these are rounded explicitly.
    float Dot(float query[HEAD_WIDTH], uint key)
    {
        float sums[HEAD_WIDTH / 2];
        [unroll] for (uint c = 0; c < HEAD_WIDTH / 2; c++)
        {
            float low = Rounded(query[c] * Keys[key + c]);
            float high = Rounded(query[c + HEAD_WIDTH / 2] * Keys[key + c + HEAD_WIDTH / 2]);
            sums[c] = low + high;
        }
        [unroll] for (uint stride = HEAD_WIDTH / 4; stride > 0; stride >>= 1)
            [unroll] for (uint i = 0; i < stride; i++) sums[i] = sums[i] + sums[i + stride];
        return sums[0];
    }

    // All queries share KeyRanges[0..3]; the caller guarantees it. Each thread owns one query and
    // applies the reference's online-softmax update per key, in the same key order.
    [numthreads(QUERIES, 1, 1)]
    void MainCs(uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID)
    {
        uint token = group.x * QUERIES + local.x;
        bool active = token < Tokens;
        uint width = Heads * HeadWidth;
        uint head = group.z * HEAD_WIDTH;
        float query[HEAD_WIDTH];
        float accumulated[HEAD_WIDTH];
        [unroll] for (uint c = 0; c < HEAD_WIDTH; c++)
        {
            query[c] = active ? Query[token * width + head + c] : 0;
            accumulated[c] = 0;
        }
        float maximum = -3.402823466e+38;
        float total = 0;
        for (uint range = 0; range < 2; range++)
        {
            uint start = min(KeyRanges[range * 2], KeyTokens);
            uint count = min(KeyRanges[range * 2 + 1], KeyTokens - start);
            for (uint tile = 0; tile < count; tile += KEY_TILE)
            {
                uint tileKeys = min(KEY_TILE, count - tile);
                GroupMemoryBarrierWithGroupSync();
                // 64 threads stage KEY_TILE keys and values, each thread one channel per key.
                for (uint k = 0; k < tileKeys; k++)
                {
                    uint source = (start + tile + k) * width + head + local.x;
                    Keys[k * HEAD_WIDTH + local.x] = Key[source];
                    Values[k * HEAD_WIDTH + local.x] = Value[source];
                }
                GroupMemoryBarrierWithGroupSync();
                for (uint k = 0; k < tileKeys; k++)
                {
                    float score = Dot(query, k * HEAD_WIDTH) * rsqrt((float)HeadWidth);
                    float nextMaximum = max(maximum, score);
                    float previousWeight = exp(maximum - nextMaximum);
                    float weight = exp(score - nextMaximum);
                    [unroll] for (uint c = 0; c < HEAD_WIDTH; c++)
                        accumulated[c] = accumulated[c] * previousWeight + Values[k * HEAD_WIDTH + c] * weight;
                    total = total * previousWeight + weight;
                    maximum = nextMaximum;
                }
            }
        }
        if (!active) return;
        [unroll] for (uint c = 0; c < HEAD_WIDTH; c++)
            Output[token * width + head + c] = total > 0 ? accumulated[c] / total : 0;
    }
}
