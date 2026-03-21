#include <cuda_runtime.h>
#include <iostream>

/**
 * @file blocking_streams.cu
 * @author Alexey Bokov <alex@bokov.net>
 * @brief A data processing pipeline designed to test the llm-cuda-auditor engine.
 * * * INTENTIONAL BUGS FOR STATIC ANALYSIS:
 * 1. Pipeline Stalls / Blocking Transfers: Using synchronous cudaMemcpy inside a loop
 * forces the GPU to sit completely idle while waiting for the PCIe bus.
 */

__global__ void simple_compute(float *data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        data[tid] = data[tid] * 2.0f;
    }
}

void process_data_chunks(float **host_data, float **device_data, int num_chunks, int chunk_size) {
    size_t bytes = chunk_size * sizeof(float);

    for (int i = 0; i < num_chunks; i++) {
        // ⚡ BUG: Synchronous transfer blocks the CPU and GPU from overlapping work.
        // The compute engine does zero work during this transfer.
        cudaMemcpy(device_data[i], host_data[i], bytes, cudaMemcpyHostToDevice);
        
        // Kernel executes on the default stream (Stream 0), blocking other operations.
        simple_compute<<<(chunk_size + 255) / 256, 256>>>(device_data[i], chunk_size);
        
        // ⚡ BUG: Synchronous read-back blocks the next chunk from uploading.
        // Resolution: Use cudaMemcpyAsync and multiple cudaStream_t objects.
        cudaMemcpy(host_data[i], device_data[i], bytes, cudaMemcpyDeviceToHost);
    }
}

int main() {
    std::cout << "Run: ./cuda-audit samples/blocking_streams.cu" << std::endl;
    return 0;
};


