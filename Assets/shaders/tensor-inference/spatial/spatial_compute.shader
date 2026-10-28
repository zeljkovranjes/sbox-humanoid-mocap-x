HEADER
{
    Description = "HWC focus, nearest upsampling and maximum pooling";
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
    RWStructuredBuffer<float> Output < Attribute("Output"); >;
    uint Height < Attribute("Height"); >;
    uint Width < Attribute("Width"); >;
    uint Channels < Attribute("Channels"); >;
    uint Count < Attribute("Count"); >;
    uint Operation < Attribute("Operation"); >;
    uint Kernel < Attribute("Kernel"); >;

    [numthreads(TENSOR_LANES, 1, 1)]
    void MainCs(uint3 thread : SV_DispatchThreadID)
    {
        uint index = TensorIndex(thread);
        if (index >= Count) return;
        if (Operation == 0)
        {
            uint channel = index % Channels;
            uint group = (index / Channels) % 4;
            uint pixel = index / (Channels * 4);
            uint x = (pixel % (Width / 2)) * 2 + group / 2;
            uint y = (pixel / (Width / 2)) * 2 + group % 2;
            Output[index] = Input[(y * Width + x) * Channels + channel];
        }
        else if (Operation == 1)
        {
            uint pixel = index / Channels;
            uint x = (pixel % (Width * 2)) / 2;
            uint y = (pixel / (Width * 2)) / 2;
            Output[index] = Input[(y * Width + x) * Channels + index % Channels];
        }
        else
        {
            int x = int((index / Channels) % Width);
            int y = int((index / Channels) / Width);
            int radius = int(Kernel / 2);
            float value = -asfloat(0x7f800000u);
            for (int row = max(0, y - radius); row <= min(int(Height) - 1, y + radius); row++)
                for (int column = max(0, x - radius); column <= min(int(Width) - 1, x + radius); column++)
                    value = max(value, Input[(uint(row) * Width + uint(column)) * Channels + index % Channels]);
            Output[index] = value;
        }
    }
}
