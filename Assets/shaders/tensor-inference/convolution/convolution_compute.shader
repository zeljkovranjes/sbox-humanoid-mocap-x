HEADER
{
    Description = "Tiled float32 HWC convolution with OIHW weights";
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
    uint Height < Attribute("Height"); >;
    uint Width < Attribute("Width"); >;
    uint Channels < Attribute("Channels"); >;
    uint Outputs < Attribute("Outputs"); >;
    uint Kernel < Attribute("Kernel"); >;
    uint Stride < Attribute("Stride"); >;
    uint Padding < Attribute("Padding"); >;
    uint OutputWidth < Attribute("OutputWidth"); >;
    uint OutputPixels < Attribute("OutputPixels"); >;
    uint KernelElements < Attribute("KernelElements"); >;
    uint HasBias < Attribute("HasBias"); >;

    #define TILE 16
    #define VALUES 4
    #define OUTPUT_TILE (TILE * VALUES)
    groupshared float InputTile[OUTPUT_TILE][TILE + 1];
    groupshared float WeightTile[OUTPUT_TILE][TILE + 1];

    float ImageValue(uint pixel, uint column)
    {
        if (pixel >= OutputPixels || column >= KernelElements) return 0;
        uint channel = column / (Kernel * Kernel);
        uint ky = column / Kernel % Kernel;
        uint kx = column % Kernel;
        // Use unsigned coordinates until subtracting padding to avoid signed overflow.
        uint y = pixel / OutputWidth * Stride + ky;
        uint x = pixel % OutputWidth * Stride + kx;
        if (x < Padding || y < Padding) return 0;
        x -= Padding; y -= Padding;
        if (x >= Width || y >= Height) return 0;
        return Input[(y * Width + x) * Channels + channel];
    }

    // The linear kernel's tiling over generated patch rows: each thread keeps sixteen outputs and
    // every output still adds its patch products in ascending column order, then the bias.
    [numthreads(TILE, TILE, 1)]
    void MainCs(uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID)
    {
        float4 sums[VALUES] = { 0, 0, 0, 0 };
        float nextInput[VALUES], nextWeight[VALUES];
        [unroll]
        for (uint part = 0; part < VALUES; part++)
        {
            uint row = local.y + part * TILE;
            uint weightRow = group.x * OUTPUT_TILE + row;
            nextInput[part] = ImageValue(group.y * OUTPUT_TILE + row, local.x);
            nextWeight[part] = weightRow < Outputs && local.x < KernelElements ? Weight[weightRow * KernelElements + local.x] : 0;
        }
        for (uint start = 0; start < KernelElements; start += TILE)
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
                uint weightRow = group.x * OUTPUT_TILE + row;
                nextInput[part] = ImageValue(group.y * OUTPUT_TILE + row, column);
                nextWeight[part] = weightRow < Outputs && column < KernelElements ? Weight[weightRow * KernelElements + column] : 0;
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
            uint pixel = group.y * OUTPUT_TILE + local.y + row * TILE;
            [unroll]
            for (uint column = 0; column < VALUES; column++)
            {
                uint channel = group.x * OUTPUT_TILE + local.x + column * TILE;
                if (pixel < OutputPixels && channel < Outputs)
                    Output[pixel * Outputs + channel] = sums[row][column] + (HasBias != 0 ? Bias[channel] : 0);
            }
        }
    }
}
