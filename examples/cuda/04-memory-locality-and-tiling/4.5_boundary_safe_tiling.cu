/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 4.5_boundary_safe_tiling
 *   ./4.5_boundary_safe_tiling
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 4.5_boundary_safe_tiling
 *   sbatch ../common/anvil_gpu.slurm ./4.5_boundary_safe_tiling
 *
 * Section 4.5: Boundary-safe tiled multiplication for rectangular matrices
 *
 * General shapes are A(MxK) * B(KxN) = C(MxN). M, K, and N need not be tile
 * multiples. Check the three boundaries separately:
 *
 * - A load is valid when row<M and a_col<K;
 * - B load is valid when b_row<K and col<N;
 * - C store is valid when row<M and col<N.
 *
 * Invalid input loads store zero into shared memory. Zero is the neutral value
 * for addition in the dot product and safely pads incomplete tiles.
 *
 * Threads with invalid C coordinates must still participate in tile loads and
 * both barriers. Returning early could deprive valid peers of inputs or violate
 * the block-wide barrier contract.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

const int kTile = 16;


__global__ void tiled_matmul_rectangular(const float* a, const float* b, float* c, int m, int k_size, int n) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;
    const int phases = (k_size + kTile - 1) / kTile;
    float sum = 0.0f;

    for (int phase = 0; phase < phases; ++phase) {
        const int a_col = phase * kTile + tx;
        const int b_row = phase * kTile + ty;

        a_tile[ty][tx] = (row < m && a_col < k_size) ? a[row * k_size + a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < k_size && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < kTile; ++k) {
            sum += a_tile[ty][k] * b_tile[k][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
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
    const int m = 37;
    const int k_size = 29;
    const int n = 41;
    static float a_h[m * k_size];
    static float b_h[k_size * n];
    static float expected_h[m * n];
    static float c_h[m * n];
    for (int i = 0; i < (m * k_size); ++i)
        a_h[i] = (i % 9) * 0.125f;
    for (int i = 0; i < (k_size * n); ++i)
        b_h[i] = (i % 7) * 0.25f - 0.5f;
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
    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    tiled_matmul_rectangular<<<grid, block>>>(a_d, b_d, c_d, m, k_size, n);
    check_cuda(cudaGetLastError(), "launch tiled_matmul_rectangular");
    check_cuda(cudaDeviceSynchronize(), "execute tiled_matmul_rectangular");
    check_cuda(cudaMemcpy(c_h, c_d, c_bytes, cudaMemcpyDeviceToHost), "copy C D2H");
    check_cuda(cudaFree(a_d), "cudaFree A");
    check_cuda(cudaFree(b_d), "cudaFree B");
    check_cuda(cudaFree(c_d), "cudaFree C");

    float max_error = 0.0f;
    for (int i = 0; i < (m * n); ++i) {
        max_error = std::max(max_error, std::fabs(c_h[i] - expected_h[i]));
    }
    const int output_threads = (int)(grid.x) * grid.y * block.x * block.y;
    printf("A: %dx%d, B: %dx%d\n", m, k_size, k_size, n);
    printf("Tile: %dx%d, phases: %d\n", kTile, kTile, (k_size + kTile - 1) / kTile);
    printf("Grid: %ux%u blocks\n", grid.x, grid.y);
    printf("Valid outputs: %d\n", m * n);
    printf("Threads outside C boundary: %d\n", output_threads - m * n);
    printf("All valid outputs checked against CPU reference.\n");
    printf("Maximum absolute error: %.6g\n", max_error);
    return max_error <= 1.0e-4f ? EXIT_SUCCESS : EXIT_FAILURE;
}
