#include <cuda_runtime.h>
#include <iostream>

/**
 * @file atomic_bottleneck.cu
 * @brief A histogram generation kernel designed to test the gemini-cuda audit engine.
 * * * INTENTIONAL BUGS FOR STATIC ANALYSIS:
 * 1. Extreme Atomic Serialization: Every thread across the entire grid attempts 
 * to atomically add to a single global memory address.
 */

__global__ void naive_global_counter(int *data, int *global_count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        if (data[i] > 100) {
            // ⚡ BUG: GLOBAL ATOMIC BOTTLENECK
            // If millions of threads hit this condition, they all queue up 
            // to update a single L2 cache line. The GPU essentially becomes 
            // a very slow single-core CPU.
            // Resolution should be block-level reduction using shared memory first.
            atomicAdd(global_count, 1);
        }
    }
}

int main() {
    std::cout << "Run: ./gemini-cuda samples/atomic_bottleneck.cu" << std::endl;
    return 0;
}
