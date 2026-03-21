#include <cuda_runtime.h>
#include <iostream>
#include <math.h>


/**
 * standart headers
 * @file naive_softmax.cu
 * @author Alexey Bokov <alex@bokov.net>
 * @brief A multi-pass Softmax kernel designed to test the llm-cuda-auditor engine.
 * * * INTENTIONAL BUGS FOR STATIC ANALYSIS:
 * 1. Global Memory Bandwidth Thrashing: This kernel reads the same global memory 
 * array three separate times (find max, compute sum, compute final prob).
 * In modern GenAI architectures, this causes massive memory-bound stalls.
 */

__global__ void naive_softmax(float *input, float *output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        // Pass 1: Find Max (Simplified for single thread to highlight the read pattern)
        float max_val = input[0];
        for (int i = 1; i < N; i++) {
            if (input[i] > max_val) max_val = input[i];
        }
        
        // Pass 2: Compute Sum of Exponentials
        float exp_sum = 0.0f;
        for (int i = 0; i < N; i++) {
            exp_sum += expf(input[i] - max_val);
        }
        
        // Pass 3: Compute Final Probability
        // ⚡ BUG: We have read the entire 'input' array from global memory THREE times.
        // Resolution: Fused kernel using shared memory/registers to keep data on-chip.
        output[tid] = expf(input[tid] - max_val) / exp_sum;
    }
}

int main() {
    std::cout << "Run: ./cuda-audit samples/naive_softmax.cu" << std::endl;
    return 0;
}


