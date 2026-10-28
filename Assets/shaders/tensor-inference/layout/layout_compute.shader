HEADER
{
    Description = "Float32 column extraction and concatenation";
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
    StructuredBuffer<float> First < Attribute("First"); >;
    StructuredBuffer<float> Second < Attribute("Second"); >;
    RWStructuredBuffer<float> Output < Attribute("Output"); >;
    uint Count < Attribute("Count"); >;
    uint FirstWidth < Attribute("FirstWidth"); >;
    uint SecondWidth < Attribute("SecondWidth"); >;
    uint FirstColumn < Attribute("FirstColumn"); >;
    uint Columns < Attribute("Columns"); >;

    [numthreads(TENSOR_LANES, 1, 1)]
    void MainCs(uint3 thread : SV_DispatchThreadID)
    {
        uint index = TensorIndex(thread);
        if (index >= Count) return;
        uint row = index / Columns;
        uint column = index % Columns;
        Output[index] = SecondWidth == 0 ? First[row * FirstWidth + FirstColumn + column]
            : column < FirstWidth ? First[row * FirstWidth + column] : Second[row * SecondWidth + column - FirstWidth];
    }
}
