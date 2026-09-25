/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 3.5_occupancy
 *   ./3.5_occupancy
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 3.5_occupancy
 *   sbatch ../common/anvil_gpu.slurm ./3.5_occupancy
 *
 * Section 3.5: Resource partitioning, residency, and occupancy
 *
 * Occupancy = active warps per SM / maximum warps per SM.
 *
 * Active blocks are limited simultaneously by:
 *
 * - maximum blocks per SM;
 * - maximum threads/warps per SM;
 * - registers required by each block;
 * - shared memory required by each block.
 *
 * More resident warps give the scheduler more alternatives while other warps
 * wait for memory or pipeline latency. This is latency hiding. Occupancy is not
 * an execution-efficiency percentage and 100% is not always required for speed.
 * The table puts theoretical occupancy beside measured time. Occupancy shows
 * how many warps may reside; timing shows which block size is faster here.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

__global__ void resource_kernel(float* output, int n) {
    extern __shared__ float shared[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Every thread contributes one value before the block-wide barrier.
    shared[threadIdx.x] = (float)(threadIdx.x);
    __syncthreads();

    if (i < n) {
        // value is a private automatic scalar and is normally held in a register.
        const float value = shared[threadIdx.x] * 2.0f + 1.0f;
        output[i] = value;
    }
}

int main() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get active device");
    cudaDeviceProp p{};
    check_cuda(cudaGetDeviceProperties(&p, device), "query device properties");

    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, resource_kernel), "query resource_kernel attributes");

    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    const int elements = 4 * 1024 * 1024;
    const int repetitions = 50;
    const int measurement_rounds = 5;
    const int output_bytes = elements * sizeof(float);
    static float output_h[elements];
    float* output_d = NULL;
    check_cuda(cudaMalloc((void**)&output_d, output_bytes), "allocate benchmark output");

    printf("Device: %s\n", p.name);
    printf("Maximum threads/SM: %d\n", p.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", p.warpSize);
    printf("Kernel registers/thread: %d\n", attributes.numRegs);
    printf("Elements: %d, timing: median of %d rounds, %d launches per round\n\n", elements, measurement_rounds, repetitions);
    printf("%-8s%-13s%-12s%-12s%-13s%-13s%-15sMax error\n", "Block", "Shared B", "Blocks/SM", "Warps/SM", "Occupancy %", "Kernel ms", "G elements/s");

    bool all_correct = true;
    int fastest_block = 0;
    float fastest_ms = FLT_MAX;
    double fastest_occupancy = 0.0;

    for (int index = 0; index < (int)(sizeof(block_sizes) / sizeof(block_sizes[0])); ++index) {
        const int block_size = block_sizes[index];
        if (block_size > p.maxThreadsPerBlock) {
            continue;
        }
        const int dynamic_shared = block_size * sizeof(float);
        int active_blocks = 0;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, resource_kernel, block_size, dynamic_shared), "calculate active blocks per SM");

        const int warps_per_block = (block_size + p.warpSize - 1) / p.warpSize;
        const int active_warps = active_blocks * warps_per_block;
        const int max_warps = p.maxThreadsPerMultiProcessor / p.warpSize;
        const double occupancy = (double)(active_warps) / max_warps;
        const int grid_size = (elements + block_size - 1) / block_size;

        resource_kernel<<<grid_size, block_size, dynamic_shared>>>(output_d, elements);
        check_cuda(cudaGetLastError(), "launch resource_kernel warm-up");
        check_cuda(cudaDeviceSynchronize(), "execute resource_kernel warm-up");

        float samples[measurement_rounds];
        for (int round = 0; round < measurement_rounds; ++round) {
            const float total_ms = time_cuda_ms([&] {
                for (int repetition = 0; repetition < repetitions; ++repetition) {
                    resource_kernel<<<grid_size, block_size, dynamic_shared>>>(output_d, elements);
                }
                check_cuda(cudaGetLastError(), "launch timed resource_kernel instances");
            });
            samples[round] = total_ms / repetitions;
        }

        const float kernel_ms = median_cuda_ms(samples, measurement_rounds);
        const double billion_elements_per_second = (double)(elements) / (kernel_ms * 1.0e6);
        check_cuda(cudaMemcpy(output_h, output_d, output_bytes, cudaMemcpyDeviceToHost), "copy benchmark output");

        float maximum_error = 0.0f;
        for (int i = 0; i < elements; ++i) {
            const float expected = (float)(i % block_size) * 2.0f + 1.0f;
            maximum_error = std::max(maximum_error, std::fabs(output_h[i] - expected));
        }
        all_correct = all_correct && maximum_error == 0.0f;

        if (kernel_ms < fastest_ms) {
            fastest_ms = kernel_ms;
            fastest_block = block_size;
            fastest_occupancy = occupancy;
        }

        printf("%-8d%-13d%-12d%-12d%-13.1f%-13.4f%-15.4f%.4f\n", block_size, dynamic_shared, active_blocks, active_warps, 100.0 * occupancy, kernel_ms, billion_elements_per_second, maximum_error);
    }

    check_cuda(cudaFree(output_d), "free benchmark output");

    /*
     * The occupancy API models resource residency; it does not time the kernel.
     * A lower-occupancy kernel can be faster if it gains more useful work per
     * thread, better locality, or fewer instructions. Measure after reasoning.
     */
    printf("\nFastest measured block size: %d threads (%.1f%% theoretical occupancy, %.4f ms)\n", fastest_block, 100.0 * fastest_occupancy, fastest_ms);
    printf("Measured time, not occupancy alone, decides which configuration is faster.\n");
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
