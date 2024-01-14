#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; ++i)
    {   
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device %d name: %s\n", i, deviceProp.name);
        printf("Device %d compute capability: %d.%d\n", i, deviceProp.major, deviceProp.minor);
        printf("Device %d clock rate: %d\n", i, deviceProp.clockRate);
        printf("Device %d total global memory: %d\n", i, deviceProp.totalGlobalMem);
        printf("Device %d total constant memory: %d\n", i, deviceProp.totalConstMem);
        printf("Device %d multiprocessor count: %d\n", i, deviceProp.multiProcessorCount);
        printf("Device %d shared memory per block: %d\n", i, deviceProp.sharedMemPerBlock);
        printf("Device %d max threads per block: %d\n", i, deviceProp.maxThreadsPerBlock);
        printf("Device %d max threads dim: %d %d %d\n", i, deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Device %d max grid size: %d %d %d\n", i, deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Device %d warp size: %d\n", i, deviceProp.warpSize);
        printf("Device %d integrated: %d\n", i, deviceProp.integrated);
        printf("Device %d can map host memory: %d\n", i, deviceProp.canMapHostMemory);
        printf("Device %d compute mode: %d\n", i, deviceProp.computeMode);
        printf("Device %d max texture 1D: %d\n", i, deviceProp.maxTexture1D);
        printf("Device %d max texture 2D: %d %d\n", i, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
        printf("Device %d max texture 3D: %d %d %d\n", i, deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("Device %d max texture 1D linear: %d\n", i, deviceProp.maxTexture1DLinear);
        printf("Device %d max texture 2D linear: %d %d\n", i, deviceProp.maxTexture2DLinear[0], deviceProp.maxTexture2DLinear[1]);
        printf("Device %d max texture 2D gather: %d %d\n", i, deviceProp.maxTexture2DGather[0], deviceProp.maxTexture2DGather[1]);
        printf("Device %d max texture 3D depth: %d\n", i, deviceProp.maxTexture3DAlt[0]);
        printf("Device %d max texture 3D depth: %d %d %d\n", i, deviceProp.maxTexture3DAlt[0], deviceProp.maxTexture3DAlt[1], deviceProp.maxTexture3DAlt[2]);
        printf("Device %d max texture cubemap: %d\n", i, deviceProp.maxTextureCubemap);
        printf("Device %d max texture 1D mipmapped: %d\n", i, deviceProp.maxTexture1DMipmap);
        printf("Device %d max texture 2D mipmapped: %d %d\n", i, deviceProp.maxTexture2DMipmap[0], deviceProp.maxTexture2DMipmap[1]);
        // printf("Device %d max texture 3D mipmapped: %d %d %d\n", i, deviceProp.maxTexture3DMipmap[0], deviceProp.maxTexture3DMipmap[1], deviceProp.maxTexture3DMipmap[2]);
    }
    return 0;
}