HEADER
{
    Description = "Float32 mean across tensor rows";
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
    uint Rows < Attribute("Rows"); >;
    uint Width < Attribute("Width"); >;

    [numthreads(TENSOR_LANES, 1, 1)]
    void MainCs(uint3 thread : SV_DispatchThreadID)
    {
        uint channel = TensorIndex(thread);
        if (channel >= Width) return;
        float sum = 0;
        for (uint row = 0; row < Rows; row++) sum += Input[row * Width + channel];
        Output[channel] = sum / Rows;
    }
}
