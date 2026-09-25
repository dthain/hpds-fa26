/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 2.4_matrix_multiplication
 *   ./2.4_matrix_multiplication
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 2.4_matrix_multiplication
 *   sbatch ../common/anvil_gpu.slurm ./2.4_matrix_multiplication
 *
 * Section 2.4: Naive matrix multiplication with one thread per output
 *
 * Matrix shapes:
 *
 *     A is M x K
 *     B is K x N
 *     C is M x N
 *
 * C(row,col) is the dot product of row row from A and column col from B:
 *
 *     C[row,col] = sum over k of A[row,k] * B[k,col]
 *
 * A 2D grid assigns one thread to each C element. The thread is parallel with
 * other output threads, but its K-term dot product remains a sequential loop.
 *
 * The CPU loop supplies an independent correctness reference. This file is a
 * mapping baseline, not a performance experiment; Section 4.4 compares this
 * global-memory algorithm with shared-memory tiling on the same GPU.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>


__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int k_size, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f; // A private per-thread scalar, normally a register.
        for (int k = 0; k < k_size; ++k) {
            /*
             * Row-major addresses:
             * A[row,k] -> row*k_size + k
             * B[k,col] -> k*n + col
             *
             * Moving along an A row changes addresses by one. Moving down a B
             * column changes addresses by n.
             */
            sum += a[row * k_size + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

void matmul_cpu(const float* a, const float* b, float* c, int m, int k_size, int n) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < k_size; ++k) {
                sum += a[row * k_size + k] * b[k * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

int main() {
    /*
     * The dimensions are not divisible by the 16x16 block shape. They also
     * exercise all row, column, and inner-dimension boundary conditions.
     */
    const int m = 513;
    const int k_size = 511;
    const int n = 515;

    static float a_h[m * k_size];
    static float b_h[k_size * n];
    static float expected_h[m * n];
    static float c_h[m * n];
    for (int i = 0; i < (m * k_size); ++i) {
        a_h[i] = (float)(i % 7) - 3.0f;
    }
    for (int i = 0; i < (k_size * n); ++i) {
        b_h[i] = (float)(i % 5) * 0.5f;
    }
    matmul_cpu(a_h, b_h, expected_h, m, k_size, n);

    float* a_d = NULL;
    float* b_d = NULL;
    float* c_d = NULL;
    const int a_bytes = (m * k_size) * sizeof(float);
    const int b_bytes = (k_size * n) * sizeof(float);
    const int c_bytes = (m * n) * sizeof(float);
    check_cuda(cudaMalloc((void**)&a_d, a_bytes), "cudaMalloc A");
    check_cuda(cudaMalloc((void**)&b_d, b_bytes), "cudaMalloc B");
    check_cuda(cudaMalloc((void**)&c_d, c_bytes), "cudaMalloc C");
    check_cuda(cudaMemcpy(a_d, a_h, a_bytes, cudaMemcpyHostToDevice), "copy A H2D");
    check_cuda(cudaMemcpy(b_d, b_h, b_bytes, cudaMemcpyHostToDevice), "copy B H2D");

    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(a_d, b_d, c_d, m, k_size, n);
    check_cuda(cudaGetLastError(), "launch matmul_kernel");
    check_cuda(cudaDeviceSynchronize(), "execute matmul_kernel");
    check_cuda(cudaMemcpy(c_h, c_d, c_bytes, cudaMemcpyDeviceToHost), "copy C D2H");
    check_cuda(cudaFree(a_d), "cudaFree A");
    check_cuda(cudaFree(b_d), "cudaFree B");
    check_cuda(cudaFree(c_d), "cudaFree C");

    float max_abs_error = 0.0f;
    for (int i = 0; i < (m * n); ++i) {
        max_abs_error = std::max(max_abs_error, std::fabs(c_h[i] - expected_h[i]));
    }
    printf("A: %dx%d, B: %dx%d, C: %dx%d\n", m, k_size, k_size, n, m, n);
    printf("Outputs checked against CPU reference: %d\n", (m * n));
    printf("Maximum absolute error: %.6g\n", max_abs_error);
    return max_abs_error <= 1.0e-4f ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*
 * This is a correct baseline, not a fast GEMM. Threads repeatedly load
 * overlapping A and B values from global memory. The memory-locality module
 * tiles these inputs into shared memory to reuse them within a block.
 */
