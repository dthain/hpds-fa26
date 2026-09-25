/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 4.4_tiled_matrix_multiplication
 *   ./4.4_tiled_matrix_multiplication
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 4.4_tiled_matrix_multiplication
 *   sbatch ../common/anvil_gpu.slurm ./4.4_tiled_matrix_multiplication
 *
 * Section 4.4: Tiled matrix multiplication with static shared memory
 *
 * The experiment compares two GPU kernels that perform the same arithmetic:
 * one reads every operand directly from global memory, while the other reuses
 * operands from shared-memory tiles. CPU performance is not part of this
 * comparison.
 *
 * Both kernels assume square matrices whose width is divisible by TILE_WIDTH.
 * Section 4.5 removes those simplifying assumptions.
 *
 * Each block computes one TILE_WIDTH x TILE_WIDTH tile of C. Each phase:
 *
 * 1. every thread loads one A element and one B element into shared memory;
 * 2. a barrier waits for the complete tiles;
 * 3. every thread reuses the tiles for TILE_WIDTH multiply-add iterations;
 * 4. a barrier waits before the shared arrays are overwritten next phase.
 *
 * This is strip-mining: split a long K loop into tile-sized phases.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

const int kTile = 16;

__global__ void naive_matmul_kernel(const float* a, const float* b, float* c, int width) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Every use reads A and B from global memory; there is no block-level reuse.
    for (int k = 0; k < width; ++k) {
        sum += a[row * width + k] * b[k * width + col];
    }
    c[row * width + col] = sum;
}

__global__ void tiled_matmul_kernel(const float* a, const float* b, float* c, int width) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;
    float sum = 0.0f;

    for (int phase = 0; phase < width / kTile; ++phase) {
        /*
         * The block's 256 threads cooperatively load 256 A values and 256 B
         * values. Neighboring tx values load neighboring row-major addresses,
         * giving a favorable coalesced global access pattern.
         */
        a_tile[ty][tx] = a[row * width + phase * kTile + tx];
        b_tile[ty][tx] = b[(phase * kTile + ty) * width + col];

        // Read-after-write: every tile element must be ready before reuse.
        __syncthreads();

        for (int k = 0; k < kTile; ++k) {
            sum += a_tile[ty][k] * b_tile[k][tx];
        }

        // Write-after-read: no thread may overwrite a tile still being consumed.
        __syncthreads();
    }
    c[row * width + col] = sum;
}

int main() {
    // A large matrix makes arithmetic and memory traffic dominate launch cost.
    const int width = 3200;
    const int repetitions = 5;
    const int measurement_rounds = 5;
    const int elements = width * width;
    const int bytes = elements * sizeof(float);
    static float a_h[elements];
    for (int i = 0; i < elements; ++i) {
        a_h[i] = 0.5f;
    }
    static float b_h[elements];
    for (int i = 0; i < elements; ++i) {
        b_h[i] = 0.25f;
    }
    static float naive_h[elements];
    static float tiled_h[elements];

    float* a_d = NULL;
    float* b_d = NULL;
    float* naive_d = NULL;
    float* tiled_d = NULL;
    check_cuda(cudaMalloc((void**)&a_d, bytes), "cudaMalloc A");
    check_cuda(cudaMalloc((void**)&b_d, bytes), "cudaMalloc B");
    check_cuda(cudaMalloc((void**)&naive_d, bytes), "cudaMalloc naive C");
    check_cuda(cudaMalloc((void**)&tiled_d, bytes), "cudaMalloc tiled C");
    check_cuda(cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice), "copy A H2D");
    check_cuda(cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice), "copy B H2D");

    const dim3 block(kTile, kTile);
    const dim3 grid(width / kTile, width / kTile);

    naive_matmul_kernel<<<grid, block>>>(a_d, b_d, naive_d, width);
    tiled_matmul_kernel<<<grid, block>>>(a_d, b_d, tiled_d, width);
    check_cuda(cudaGetLastError(), "launch warm-up naive_matmul_kernel");
    check_cuda(cudaDeviceSynchronize(), "execute warm-up matrix kernels");

    float naive_samples[measurement_rounds];
    float tiled_samples[measurement_rounds];
    const auto measure_naive = [&](int round) {
        const float total_ms = time_cuda_ms([&] {
            for (int run = 0; run < repetitions; ++run) {
                naive_matmul_kernel<<<grid, block>>>(a_d, b_d, naive_d, width);
            }
            check_cuda(cudaGetLastError(), "launch timed naive_matmul_kernel");
        });
        naive_samples[round] = total_ms / repetitions;
    };
    const auto measure_tiled = [&](int round) {
        const float total_ms = time_cuda_ms([&] {
            for (int run = 0; run < repetitions; ++run) {
                tiled_matmul_kernel<<<grid, block>>>(a_d, b_d, tiled_d, width);
            }
            check_cuda(cudaGetLastError(), "launch timed tiled_matmul_kernel");
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

    check_cuda(cudaMemcpy(naive_h, naive_d, bytes, cudaMemcpyDeviceToHost), "copy naive C D2H");
    check_cuda(cudaMemcpy(tiled_h, tiled_d, bytes, cudaMemcpyDeviceToHost), "copy tiled C D2H");

    check_cuda(cudaFree(a_d), "cudaFree A");
    check_cuda(cudaFree(b_d), "cudaFree B");
    check_cuda(cudaFree(naive_d), "cudaFree naive C");
    check_cuda(cudaFree(tiled_d), "cudaFree tiled C");

    const float expected = width * 0.5f * 0.25f;
    float naive_error = 0.0f;
    float tiled_error = 0.0f;
    for (int i = 0; i < elements; ++i) {
        naive_error = std::max(naive_error, std::fabs(naive_h[i] - expected));
        tiled_error = std::max(tiled_error, std::fabs(tiled_h[i] - expected));
    }

    const float naive_ms = median_cuda_ms(naive_samples, measurement_rounds);
    const float tiled_ms = median_cuda_ms(tiled_samples, measurement_rounds);
    const double operations = 2.0 * width * width * width;
    const double naive_gflops = operations / (naive_ms * 1.0e6);
    const double tiled_gflops = operations / (tiled_ms * 1.0e6);

    printf("Matrix: %d x %d\n", width, width);
    printf("Tile width: %d, phases: %d\n", kTile, width / kTile);
    printf("Shared memory/block: %zu bytes\n", 2 * kTile * kTile * sizeof(float));
    printf("Approximate global-load reduction from tiling: %dx\n", kTile);
    printf("Timing: median of %d rounds, %d launches per round; memory copies excluded.\n\n", measurement_rounds, repetitions);
    printf("%-22s%14s%14s%14s\n", "GPU kernel", "Time (ms)", "GFLOP/s", "Max error");
    printf("%-22s%14.6g%14.6g%14.6g\n", "Without tiling", naive_ms, naive_gflops, naive_error);
    printf("%-22s%14.6g%14.6g%14.6g\n\n", "With tiling", tiled_ms, tiled_gflops, tiled_error);
    printf("Speedup from tiling (without / with): %.6gx\n", naive_ms / tiled_ms);
    return naive_error <= 1.0e-4f && tiled_error <= 1.0e-4f ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*
 * In the naive kernel, overlapping global inputs are independently fetched by
 * many threads. Here each tile element is loaded once per block phase and reused
 * by TILE_WIDTH threads. The simplified reduction in global loads is therefore
 * approximately TILE_WIDTH, increasing arithmetic intensity.
 */
