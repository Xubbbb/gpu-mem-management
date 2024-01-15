#include "../src/allocator.cuh"
#include <stdio.h>

int main(){
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("free_mem: %f GB, total_mem: %f GB\n", (float)free_mem / 1024 / 1024 / 1024, (float)total_mem / 1024 / 1024 / 1024);


    GPUMemAllocator allocator;
    void *device_ptr = allocator.malloc(8 * sizeof(float));
    float *host_ptr = (float *)malloc(8 * sizeof(float));
    for (int i = 0; i < 8; i++){
        host_ptr[i] = i;
    }
    cudaMemcpy(device_ptr, host_ptr, 8 * sizeof(float), cudaMemcpyHostToDevice);
    float *host_ptr2 = (float *)malloc(8 * sizeof(float));
    cudaMemcpy(host_ptr2, device_ptr, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++){
        printf("%f\n", host_ptr2[i]);
    }

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("free_mem: %f GB, total_mem: %f GB\n", (float)free_mem / 1024 / 1024 / 1024, (float)total_mem / 1024 / 1024 / 1024);

    allocator.print_free_lists();

    allocator.free(device_ptr);
    allocator.print_free_lists();
    
    return 0;
}