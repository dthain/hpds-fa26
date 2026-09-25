/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 4.3_shared_memory
 *   ./4.3_shared_memory
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 4.3_shared_memory
 *   sbatch ../common/anvil_gpu.slurm ./4.3_shared_memory
 *
 * Section 4.3: Cooperative loading and reuse in shared memory
 *
 * This experiment compares two GPU kernels that do identical arithmetic.
 * Every block owns one input tile, and every thread sums every value in that
 * tile:
 *
 * - without reuse, every thread reads the complete tile from global memory;
 * - with reuse, the block cooperatively loads the tile once into shared memory.
 *
 * The source-level global-load count falls by a factor of TILE_WIDTH. Hardware
 * caches and warp broadcasts can reduce actual memory transactions, while the
 * shared-memory version pays for a barrier. Measured speedup must therefore be
 * observed rather than inferred from the load count alone.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

const int kTile = 256;

__global__ void global_reuse_kernel(const float* input, float* output) {
    const int tile_start = blockIdx.x * blockDim.x;
    float sum = 0.0f;

    // Every thread independently issues TILE_WIDTH global-memory loads.
    for (int k = 0; k < kTile; ++k) {
        sum += input[tile_start + k];
    }
    output[tile_start + threadIdx.x] = sum;
}

__global__ void shared_reuse_kernel(const float* input, float* output) {
    __shared__ float tile[kTile];
    const int tile_start = blockIdx.x * blockDim.x;

    // Cooperative load: TILE_WIDTH threads perform TILE_WIDTH global loads.
    tile[threadIdx.x] = input[tile_start + threadIdx.x];

    // Read-after-write barrier: the complete tile must be visible before reuse.
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < kTile; ++k) {
        sum += tile[k];
    }
    output[tile_start + threadIdx.x] = sum;
}

float maximum_difference(const float* a, const float* b, int elements) {
    float result = 0.0f;
    for (int i = 0; i < elements; ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

int main() {
    const int blocks = 4096;
    const int repetitions = 10;
    const int measurement_rounds = 5;
    const int elements = blocks * kTile;
    const int bytes = elements * sizeof(float);

    static float input_h[elements];
    static float global_h[elements];
    static float shared_h[elements];
    for (int i = 0; i < elements; ++i) {
        input_h[i] = (float)(i % 17) * 0.0625f;
    }

    float* input_d = NULL;
    float* global_d = NULL;
    float* shared_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&global_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&shared_d, bytes));
    CUDA_CHECK(cudaMemcpy(input_d, input_h, bytes, cudaMemcpyHostToDevice));

    global_reuse_kernel<<<blocks, kTile>>>(input_d, global_d);
    shared_reuse_kernel<<<blocks, kTile>>>(input_d, shared_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float global_samples[measurement_rounds];
    float shared_samples[measurement_rounds];

    const auto measure_global = [&](int round) {
        const float total_ms = time_cuda_ms([&] {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                global_reuse_kernel<<<blocks, kTile>>>(input_d, global_d);
            }
            CUDA_CHECK(cudaGetLastError());
        });
        global_samples[round] = total_ms / repetitions;
    };
    const auto measure_shared = [&](int round) {
        const float total_ms = time_cuda_ms([&] {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                shared_reuse_kernel<<<blocks, kTile>>>(input_d, shared_d);
            }
            CUDA_CHECK(cudaGetLastError());
        });
        shared_samples[round] = total_ms / repetitions;
    };

    // Alternate order to reduce systematic clock and thermal bias.
    for (int round = 0; round < measurement_rounds; ++round) {
        if ((round & 1) == 0) {
            measure_global(round);
            measure_shared(round);
        } else {
            measure_shared(round);
            measure_global(round);
        }
    }

    CUDA_CHECK(cudaMemcpy(global_h, global_d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(shared_h, shared_d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(global_d));
    CUDA_CHECK(cudaFree(shared_d));

    const float global_ms = median_cuda_ms(global_samples, measurement_rounds);
    const float shared_ms = median_cuda_ms(shared_samples, measurement_rounds);
    const float max_difference = maximum_difference(global_h, shared_h, elements);
    const double additions = (double)(elements) * kTile;
    const auto giga_additions_per_second = [=](float milliseconds) { return additions / (milliseconds * 1.0e6); };

    printf("Blocks: %d, threads/block: %d, output elements: %d\n", blocks, kTile, elements);
    printf("Modeled global loads/block without reuse: %d\n", kTile * kTile);
    printf("Modeled global loads/block with reuse: %d\n", kTile);
    printf("Source-level global-load reduction: %dx\n", kTile);
    printf("Timing: median of %d rounds, %d launches per round\n\n", measurement_rounds, repetitions);
    printf("%-22s%14s%18s\n", "GPU kernel", "Time (ms)", "Useful Gadd/s");
    printf("%-22s%14.6g%18.6g\n", "Global loads", global_ms, giga_additions_per_second(global_ms));
    printf("%-22s%14.6g%18.6g\n\n", "Shared-memory reuse", shared_ms, giga_additions_per_second(shared_ms));
    printf("Speedup from shared reuse (global / shared): %.6gx\n", global_ms / shared_ms);
    printf("Maximum absolute difference: %.6g\n", max_difference);
    return max_difference == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}
