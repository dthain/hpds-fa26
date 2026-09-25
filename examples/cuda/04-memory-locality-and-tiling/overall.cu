/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make overall
 *   ./overall
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make overall
 *   sbatch ../common/anvil_gpu.slurm ./overall
 *
 * Overall program: Naive versus tiled matrix multiplication
 *
 * Two kernels multiply the same rectangular matrices:
 *
 * - naive: every thread repeatedly loads its A row and B column from global;
 * - tiled: each block cooperatively stages reusable A/B tiles in shared memory.
 *
 * The output includes kernel time, modeled occupancy, and numerical error.
 * Tiling reduces global-memory traffic, but the speedup depends on the GPU.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

const int kTile = 16;


__global__ void naive_matmul(const float* a, const float* b, float* c, int m, int k_size, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < k_size; ++k) {
            sum += a[row * k_size + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void tiled_matmul(const float* a, const float* b, float* c, int m, int k_size, int n) {
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
        a_tile[ty][tx] = row < m && a_col < k_size ? a[row * k_size + a_col] : 0.0f;
        b_tile[ty][tx] = b_row < k_size && col < n ? b[b_row * n + col] : 0.0f;
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


float maximum_error(const float* actual, int elements, float expected) {
    float result = 0.0f;
    for (int i = 0; i < elements; ++i) {
        const float value = actual[i];
        result = std::max(result, std::fabs(value - expected));
    }
    return result;
}

int main() {
    // Odd rectangular dimensions exercise all three boundary conditions.
    const int m = 1601;
    const int k_size = 1603;
    const int n = 1607;
    const int repetitions = 5;
    const int measurement_rounds = 5;

    const int a_elements = m * k_size;
    const int b_elements = k_size * n;
    const int c_elements = m * n;
    static float a_h[a_elements];
    for (int i = 0; i < a_elements; ++i) {
        a_h[i] = 0.5f;
    }
    static float b_h[b_elements];
    for (int i = 0; i < b_elements; ++i) {
        b_h[i] = 0.25f;
    }
    static float naive_h[c_elements];
    static float tiled_h[c_elements];

    float* a_d = NULL;
    float* b_d = NULL;
    float* naive_d = NULL;
    float* tiled_d = NULL;
    const int a_bytes = a_elements * sizeof(float);
    const int b_bytes = b_elements * sizeof(float);
    const int c_bytes = c_elements * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&a_d, a_bytes));
    CUDA_CHECK(cudaMalloc((void**)&b_d, b_bytes));
    CUDA_CHECK(cudaMalloc((void**)&naive_d, c_bytes));
    CUDA_CHECK(cudaMalloc((void**)&tiled_d, c_bytes));
    CUDA_CHECK(cudaMemcpy(a_d, a_h, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b_h, b_bytes, cudaMemcpyHostToDevice));

    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);

    naive_matmul<<<grid, block>>>(a_d, b_d, naive_d, m, k_size, n);
    tiled_matmul<<<grid, block>>>(a_d, b_d, tiled_d, m, k_size, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float naive_samples[measurement_rounds];
    float tiled_samples[measurement_rounds];
    const auto measure_naive = [&](int round) {
        const float total_ms = time_cuda_ms([&] {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                naive_matmul<<<grid, block>>>(a_d, b_d, naive_d, m, k_size, n);
            }
            CUDA_CHECK(cudaGetLastError());
        });
        naive_samples[round] = total_ms / repetitions;
    };
    const auto measure_tiled = [&](int round) {
        const float total_ms = time_cuda_ms([&] {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                tiled_matmul<<<grid, block>>>(a_d, b_d, tiled_d, m, k_size, n);
            }
            CUDA_CHECK(cudaGetLastError());
        });
        tiled_samples[round] = total_ms / repetitions;
    };

    for (int round = 0; round < measurement_rounds; ++round) {
        if ((round & 1) == 0) {
            measure_naive(round);
            measure_tiled(round);
        } else {
            measure_tiled(round);
            measure_naive(round);
        }
    }

    CUDA_CHECK(cudaMemcpy(naive_h, naive_d, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tiled_h, tiled_d, c_bytes, cudaMemcpyDeviceToHost));

    int naive_blocks_per_sm = 0;
    int tiled_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&naive_blocks_per_sm, naive_matmul, kTile * kTile, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&tiled_blocks_per_sm, tiled_matmul, kTile * kTile, 0));

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(naive_d));
    CUDA_CHECK(cudaFree(tiled_d));

    const float expected = k_size * 0.5f * 0.25f;
    const float naive_error = maximum_error(naive_h, c_elements, expected);
    const float tiled_error = maximum_error(tiled_h, c_elements, expected);
    const float naive_ms = median_cuda_ms(naive_samples, measurement_rounds);
    const float tiled_ms = median_cuda_ms(tiled_samples, measurement_rounds);
    const double operations = 2.0 * m * k_size * n;
    const double naive_gflops = operations / (naive_ms * 1.0e6);
    const double tiled_gflops = operations / (tiled_ms * 1.0e6);
    const bool correct = naive_error <= 1.0e-3f && tiled_error <= 1.0e-3f;

    printf("A: %dx%d, B: %dx%d\n", m, k_size, k_size, n);
    printf("Tile: %dx%d, static shared/block: %zu bytes\n", kTile, kTile, 2 * kTile * kTile * sizeof(float));
    printf("Approximate global-load reduction from tiling: %dx\n", kTile);
    printf("Timing: median of %d rounds, %d launches per round; memory copies excluded.\n\n", measurement_rounds, repetitions);
    printf("%-12s%14s%14s%14s%14s\n", "GPU kernel", "Time (ms)", "GFLOP/s", "Blocks/SM", "Max error");
    printf("%-12s%14.6g%14.6g%14d%14.6g\n", "Naive", naive_ms, naive_gflops, naive_blocks_per_sm, naive_error);
    printf("%-12s%14.6g%14.6g%14d%14.6g\n", "Tiled", tiled_ms, tiled_gflops, tiled_blocks_per_sm, tiled_error);
    printf("Tiled speedup over naive GPU (naive / tiled): %.6gx\n", naive_ms / tiled_ms);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
