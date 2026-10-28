HEADER
{
    Description = "Dense float32 linear layer with PyTorch weight layout";
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
    StructuredBuffer<float> Weight < Attribute("Weight"); >;
    StructuredBuffer<float> Bias < Attribute("Bias"); >;
    RWStructuredBuffer<float> Output < Attribute("Output"); >;
    uint Rows < Attribute("Rows"); >;
    uint Inputs < Attribute("Inputs"); >;
    uint Outputs < Attribute("Outputs"); >;
    uint HasBias < Attribute("HasBias"); >;

    #define TILE 16
    #define VALUES 4
    #define OUTPUT_TILE (TILE * VALUES)
    groupshared float InputTile[OUTPUT_TILE][TILE + 1];
    groupshared float WeightTile[OUTPUT_TILE][TILE + 1];

    // Each thread reuses operands across sixteen outputs, preserving each dot product's order. The
    // next k tile is read into registers while the current one is multiplied, hiding load latency.
    [numthreads(TILE, TILE, 1)]
    void MainCs(uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID)
    {
        float4 sums[VALUES] = { 0, 0, 0, 0 };
        float nextInput[VALUES], nextWeight[VALUES];
        [unroll]
        for (uint part = 0; part < VALUES; part++)
        {
            uint row = local.y + part * TILE;
            uint inputRow = group.y * OUTPUT_TILE + row, weightRow = group.x * OUTPUT_TILE + row;
            nextInput[part] = inputRow < Rows && local.x < Inputs ? Input[inputRow * Inputs + local.x] : 0;
            nextWeight[part] = weightRow < Outputs && local.x < Inputs ? Weight[weightRow * Inputs + local.x] : 0;
        }
        for (uint start = 0; start < Inputs; start += TILE)
        {
            [unroll]
            for (uint part = 0; part < VALUES; part++)
            {
                InputTile[local.y + part * TILE][local.x] = nextInput[part];
                WeightTile[local.y + part * TILE][local.x] = nextWeight[part];
            }
            GroupMemoryBarrierWithGroupSync();
            uint column = start + TILE + local.x;
            [unroll]
            for (uint part = 0; part < VALUES; part++)
            {
                uint row = local.y + part * TILE;
                uint inputRow = group.y * OUTPUT_TILE + row, weightRow = group.x * OUTPUT_TILE + row;
                nextInput[part] = inputRow < Rows && column < Inputs ? Input[inputRow * Inputs + column] : 0;
                nextWeight[part] = weightRow < Outputs && column < Inputs ? Weight[weightRow * Inputs + column] : 0;
            }
            [unroll]
            for (uint inner = 0; inner < TILE; inner++)
            {
                float4 weights = float4(WeightTile[local.x][inner], WeightTile[local.x + TILE][inner],
                    WeightTile[local.x + 2 * TILE][inner], WeightTile[local.x + 3 * TILE][inner]);
                [unroll]
                for (uint row = 0; row < VALUES; row++)
                    sums[row] += InputTile[local.y + row * TILE][inner] * weights;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        [unroll]
        for (uint row = 0; row < VALUES; row++)
        {
            uint outputRow = group.y * OUTPUT_TILE + local.y + row * TILE;
            [unroll]
            for (uint column = 0; column < VALUES; column++)
            {
                uint outputColumn = group.x * OUTPUT_TILE + local.x + column * TILE;
                if (outputRow < Rows && outputColumn < Outputs)
                    Output[outputRow * Outputs + outputColumn] = sums[row][column] + (HasBias != 0 ? Bias[outputColumn] : 0);
            }
        }
    }
}
