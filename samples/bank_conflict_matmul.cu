#include <cuda_runtime.h>
#include <iostream>

/**
 * @file bank_conflict_matmul.cu
 * @brief A tiled matrix multiplication kernel designed to test the gemini-cuda audit engine.
 * * * INTENTIONAL BUGS FOR STATIC ANALYSIS:
 * 1. Shared Memory Bank Conflicts: Naive allocation of 2D shared memory tiles 
 * leads to severe n-way bank conflicts during column-wise reads.
 */

#define TILE_SIZE 32

__global__ void tiled_matmul(float *A, float *B, float *C, int N) {
    // ⚡ BUG: BANK CONFLICTS
    // Accessing a column of this shared memory array (e.g., As[i][tx]) 
    // will cause all threads in a warp to hit the same memory banks.
    // Resolution should be adding padding: __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    float Cvalue = 0.0;

    for (int m = 0; m < (N - 1) / TILE_SIZE + 1; ++m) {
        if (Row < N && m * TILE_SIZE + tx < N)
            As[ty][tx] = A[Row * N + m * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0;

        if (Col < N && m * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + Col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx]; // Column-wise read on As causes bank conflicts
        }
        __syncthreads();
    }

    if (Row < N && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

int main() {
    std::cout << "Run: ./gemini-cuda samples/bank_conflict_matmul.cu" << std::endl;
    return 0;
}
