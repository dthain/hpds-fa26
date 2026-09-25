/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 4.6_dynamic_shared_occupancy
 *   ./4.6_dynamic_shared_occupancy
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 4.6_dynamic_shared_occupancy
 *   sbatch ../common/anvil_gpu.slurm ./4.6_dynamic_shared_occupancy
 *
 * Section 4.6: Dynamic shared memory and occupancy tradeoffs
 *
 * Dynamic shared memory is declared without a compile-time size:
 *
 *     extern __shared__ float tile[];
 *
 * The third launch parameter selects the bytes reserved by every block:
 *
 *     kernel<<<grid, block, shared_bytes>>>(...);
 *
 * This controlled experiment keeps the kernel and useful work unchanged while
 * increasing only the per-block shared-memory reservation. Extra bytes are
 * intentionally unused: they create resource pressure without changing the
 * answer or operation count. The table places modeled occupancy beside measured
 * time. Occupancy is latency-hiding capacity, not a performance score, so the
 * fastest row need not be the row with the highest occupancy.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

__global__ void dynamic_shared_kernel(const float* input, float* output, int n) {
    extern __shared__ float tile[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = i < n ? input[i] : 0.0f;
    __syncthreads();
    if (i < n) {
        output[i] = tile[threadIdx.x] * 2.0f;
    }
}

const int measurement_rounds = 5;

struct Configuration {
    int shared_bytes;
    int blocks_per_sm;
    int active_warps;
    double occupancy;
    float samples[measurement_rounds];
};

int main() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const int block_size = 256;
    const int elements = 16 * 1024 * 1024;
    const int repetitions = 20;
    const int required_tile_bytes = block_size * sizeof(float);
    const int requested_reservations[] = {required_tile_bytes, 16 * 1024, 32 * 1024, 48 * 1024};
    const int maximum_warps = properties.maxThreadsPerMultiProcessor / properties.warpSize;
    const int warps_per_block = block_size / properties.warpSize;

    Configuration configurations[4];
    int configuration_count = 0;
    for (int index = 0; index < (int)(sizeof(requested_reservations) / sizeof(requested_reservations[0])); ++index) {
        const int shared_bytes = requested_reservations[index];
        if (shared_bytes > properties.sharedMemPerBlock) {
            continue;
        }
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, dynamic_shared_kernel, block_size, shared_bytes));
        const int active_warps = blocks_per_sm * warps_per_block;
        configurations[configuration_count++] = {shared_bytes, blocks_per_sm, active_warps, (double)(active_warps) / maximum_warps, {}};
    }

    if (configuration_count == 0) {
        fprintf(stderr, "No requested shared-memory reservation is supported by this GPU.\n");
        return EXIT_FAILURE;
    }

    const int bytes = elements * sizeof(float);
    static float input_h[elements];
    for (int i = 0; i < elements; ++i) {
        input_h[i] = 3.0f;
    }
    static float output_h[elements];
    float* input_d = NULL;
    float* output_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&output_d, bytes));
    CUDA_CHECK(cudaMemcpy(input_d, input_h, bytes, cudaMemcpyHostToDevice));
    const int grid_size = (elements + block_size - 1) / block_size;

    // Warm every launch configuration before collecting samples.
    for (int index = 0; index < configuration_count; ++index) {
        Configuration* configuration = &configurations[index];
        dynamic_shared_kernel<<<grid_size, block_size, configuration->shared_bytes>>>(input_d, output_d, elements);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto measure = [&](Configuration* configuration, int round) {
        const float total_ms = time_cuda_ms([&] {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                dynamic_shared_kernel<<<grid_size, block_size, configuration->shared_bytes>>>(input_d, output_d, elements);
            }
            CUDA_CHECK(cudaGetLastError());
        });
        configuration->samples[round] = total_ms / repetitions;
    };

    // Reverse alternate rounds to reduce systematic clock and thermal bias.
    for (int round = 0; round < measurement_rounds; ++round) {
        for (int step = 0; step < configuration_count; ++step) {
            const int index = (round & 1) == 0 ? step : configuration_count - 1 - step;
            measure(&configurations[index], round);
        }
    }

    CUDA_CHECK(cudaMemcpy(output_h, output_d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));

    float maximum_error = 0.0f;
    for (int i = 0; i < elements; ++i) {
        const float value = output_h[i];
        maximum_error = std::max(maximum_error, std::fabs(value - 6.0f));
    }

    printf("Device: %s\n", properties.name);
    printf("Shared memory/block limit: %zu bytes\n", properties.sharedMemPerBlock);
    printf("Elements: %d, block size: %d\n", elements, block_size);
    printf("Each row performs identical work; only reserved shared memory changes.\n");
    printf("Timing: median of %d rounds, %d launches per round; memory copies excluded.\n\n", measurement_rounds, repetitions);
    printf("%-18s%-12s%-12s%-14s%-14sBandwidth GB/s\n", "Reserved bytes", "Blocks/SM", "Warps/SM", "Occupancy %", "Time (ms)");

    float fastest_ms = 0.0f;
    int fastest_shared_bytes = 0;
    for (int index = 0; index < configuration_count; ++index) {
        Configuration* configuration = &configurations[index];
        const float kernel_ms = median_cuda_ms(configuration->samples, measurement_rounds);
        const double bandwidth_gbs = 2.0 * bytes / (kernel_ms * 1.0e6);
        if (fastest_shared_bytes == 0 || kernel_ms < fastest_ms) {
            fastest_ms = kernel_ms;
            fastest_shared_bytes = configuration->shared_bytes;
        }
        printf("%-18d%-12d%-12d%-14.1f%-14.4f%.4f\n", configuration->shared_bytes, configuration->blocks_per_sm, configuration->active_warps, 100.0 * configuration->occupancy, kernel_ms, bandwidth_gbs);
    }

    printf("\nFastest measured reservation: %d bytes/block\n", fastest_shared_bytes);
    printf("Occupancy estimates resident warp capacity; measured time decides performance.\n");
    printf("Maximum absolute error: %.4f\n", maximum_error);
    return maximum_error == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}
