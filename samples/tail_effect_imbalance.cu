#include <cuda_runtime.h>
#include <iostream>

/**
 * @file tail_effect_imbalance.cu
 * @author Alexey Bokov <alex@bokov.net>
 * @brief A workload distribution kernel designed to test the llm-cuda-auditor engine.
 * * * INTENTIONAL BUGS FOR STATIC ANALYSIS:
 * 1. Extreme Thread Imbalance (Tail Effect): Workload depends heavily on data values,
 * causing massive divergence where the warp execution time is bound by the slowest thread.
 */

__global__ void process_variable_workload(int *work_items, float *results, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        int my_work_count = work_items[tid];
        float local_result = 0.0f;
        
        // ⚡ BUG: THREAD IMBALANCE
        // If work_items[0] is 1,000,000 and work_items[1] through work_items[31] are 1,
        // threads 1-31 will finish instantly and sit idle, waiting for thread 0 to finish.
        // Resolution: Needs workload balancing on the CPU first, or a persistent thread 
        // queue model (work-stealing) on the GPU.
        for (int i = 0; i < my_work_count; i++) {
            local_result += sinf(i * 0.01f); 
        }
        
        results[tid] = local_result;
    }
}

int main() {
    std::cout << "Run: ./cuda-audit samples/tail_effect_imbalance.cu" << std::endl;
    return 0;
};

