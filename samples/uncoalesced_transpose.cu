#include <cuda_runtime.h>
#include <iostream>

/**
 * @file uncoalesced_transpose.cu
 * @brief A naive matrix transpose kernel designed to test the gemini-cuda audit engine.
 * * * INTENTIONAL BUGS FOR STATIC ANALYSIS:
 * 1. Uncoalesced Global Memory Writes: Threads in a warp write to disparate, 
 * strided memory locations in the `output` array, devastating memory bandwidth.
 */

#define TILE_DIM 32

__global__ void naive_transpose(float *odata, const float *idata, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < height) {
        // ⚡ BUG: UNCOALESCED MEMORY ACCESS
        // Reading from idata is coalesced (adjacent threads read adjacent memory),
        // but writing to odata is strided by 'height'. 
        // This will result in massive global memory transaction overhead.
        odata[x * height + y] = idata[y * width + x];
    }
}

int main() {
    std::cout << "Run: ./gemini-cuda samples/uncoalesced_transpose.cu" << std::endl;
    return 0;
}
