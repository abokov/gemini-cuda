#include <cuda_runtime.h>
#include <iostream>

/**
 * @file broken_reduction.cu
 * @brief A flawed parallel sum reduction kernel designed to test the gemini-cuda audit engine.
 * * INTENTIONAL BUGS FOR STATIC ANALYSIS:
 * 1. Missing __syncthreads() after loading global memory to shared memory.
 * 2. Missing __syncthreads() inside the reduction loop.
 * 3. Modulo operator (%) causes severe warp divergence.
 */

__global__ void buggy_sum_reduction(float *input, float *output, int n) {
    // Dynamically allocated shared memory for the block
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Load data from global memory into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;

    // ⚡ BUG 1: MISSING __syncthreads(); 
    // Threads will race ahead into the loop before all data is loaded.

    // 2. Perform the reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        
        // ⚡ BUG 2: WARP DIVERGENCE
        // The modulo operator forces threads in the same warp to branch differently.
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        
        // ⚡ BUG 3: MISSING __syncthreads(); 
        // Threads will read sdata[tid + s] before other threads have finished writing to it in the previous iteration.
    }

    // 3. Write the result for this block back to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    std::cout << "This is a sample file for the gemini-cuda audit engine." << std::endl;
    std::cout << "Run: ./gemini-cuda samples/broken_reduction.cu" << std::endl;
    return 0;
}
