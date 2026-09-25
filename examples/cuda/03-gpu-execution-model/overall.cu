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
 * Overall program: Architecture-aware block-size experiment
 *
 * The program queries the device, then sweeps block sizes while it:
 *
 * - query SM, warp, thread, register, and shared-memory limits;
 * - partition each block into warps;
 * - use a block barrier for a shared producer/consumer phase;
 * - calculate resource-limited theoretical occupancy;
 * - sweep block sizes and measure steady-state kernel time;
 * - validate every configuration rather than assuming faster means correct.
 *
 * Occupancy and measured time are reported separately.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>


__global__ void architecture_kernel(const float* input, float* output, int n) {
    extern __shared__ float tile[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_in_block = threadIdx.x;

    // Every thread participates; rounded-up threads contribute neutral zero.
    tile[lane_in_block] = i < n ? input[i] : 0.0f;

    // Read-after-write barrier: neighbors must finish producing tile values.
    __syncthreads();

    if (i < n) {
        const int neighbor = lane_in_block == 0 ? 0 : lane_in_block - 1;
        output[i] = input[i] + tile[neighbor];
    }
}


int main() {
    const int n = 1000003;
    const int repetitions = 20;
    const int candidates[] = {32, 64, 128, 256, 512, 1024};
    const int bytes = n * sizeof(float);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, architecture_kernel));

    printf("Device: %s\n", p.name);
    printf("SMs: %d, warp size: %d, max threads/SM: %d\n", p.multiProcessorCount, p.warpSize, p.maxThreadsPerMultiProcessor);
    printf("Kernel registers/thread: %d\n\n", attributes.numRegs);

    static float input_h[n];
    static float output_h[n];
    for (int i = 0; i < n; ++i) {
        input_h[i] = (float)(i % 1009) * 0.001f;
    }

    float* input_d = NULL;
    float* output_d = NULL;
    CUDA_CHECK(cudaMalloc((void**)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&output_d, bytes));
    CUDA_CHECK(cudaMemcpy(input_d, input_h, bytes, cudaMemcpyHostToDevice));

    printf("%-8s%-8s%-12s%-12sKernel ms\n", "Block", "Warps", "Blocks/SM", "Occupancy");

    bool all_correct = true;
    for (int index = 0; index < (int)(sizeof(candidates) / sizeof(candidates[0])); ++index) {
        const int block_size = candidates[index];
        if (block_size > p.maxThreadsPerBlock) {
            continue;
        }
        const int grid_size = (n + block_size - 1) / block_size;
        const int shared_bytes = block_size * sizeof(float);

        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, architecture_kernel, block_size, shared_bytes));
        const int warps_per_block = (block_size + p.warpSize - 1) / p.warpSize;
        const int max_warps = p.maxThreadsPerMultiProcessor / p.warpSize;
        const double occupancy = (double)(blocks_per_sm * warps_per_block) / max_warps;

        architecture_kernel<<<grid_size, block_size, shared_bytes>>>(input_d, output_d, n);
        CUDA_CHECK(cudaDeviceSynchronize()); // warm-up

        const float total_ms = time_cuda_ms([&] {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                architecture_kernel<<<grid_size, block_size, shared_bytes>>>(input_d, output_d, n);
            }
            CUDA_CHECK(cudaGetLastError());
        });
        const float kernel_ms = total_ms / repetitions;
        CUDA_CHECK(cudaMemcpy(output_h, output_d, bytes, cudaMemcpyDeviceToHost));

        bool correct = true;
        for (int i = 0; i < n; ++i) {
            const int lane = i % block_size;
            const int neighbor_i = lane == 0 ? i : i - 1;
            const float expected = input_h[i] + input_h[neighbor_i];
            correct = correct && std::fabs(output_h[i] - expected) <= 1.0e-6f;
        }
        all_correct = all_correct && correct;

        printf("%-8d%-8d%-12d%-12.3f%.3f\n", block_size, warps_per_block, blocks_per_sm, 100.0 * occupancy, kernel_ms);
    }

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    printf("\nOccupancy estimates resident warp capacity; measured time decides performance.\n");
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
