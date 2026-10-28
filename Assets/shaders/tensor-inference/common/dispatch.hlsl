#ifndef TENSOR_DISPATCH_HLSL
#define TENSOR_DISPATCH_HLSL

#define TENSOR_LANES 256
#define TENSOR_ELEMENTS_PER_LINE 65536

uint TensorIndex(uint3 thread)
{
    return thread.y * TENSOR_ELEMENTS_PER_LINE + thread.x;
}

#endif
