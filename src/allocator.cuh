#pragma once
#include <cuda.h>
#include <vector>

class GPUMemAllocator {
private:
    CUdeviceptr d_ptr;
    CUmemAllocationProp prop;
    CUmemAccessDesc accessDesc;
public:
    GPUMemAllocator();
    ~GPUMemAllocator();

    
};