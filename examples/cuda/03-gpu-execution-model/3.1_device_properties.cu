/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 3.1_device_properties
 *   ./3.1_device_properties
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 3.1_device_properties
 *   sbatch ../common/anvil_gpu.slurm ./3.1_device_properties
 *
 * Section 3.1: Query the hardware that will execute this program
 *
 * CUDA code is portable across devices, but device resources are not identical.
 * Query capabilities instead of hard-coding assumptions about block limits,
 * grid limits, SM count, warp size, registers, or shared memory.
 *
 * Compute capability major.minor identifies an NVIDIA architecture feature set.
 * It is not a direct speed rating. Performance also depends on SM count, clock,
 * memory system, instruction mix, and the application's bottleneck.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <stdlib.h>
#include <stdio.h>


int main() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices are visible\n");
        return EXIT_FAILURE;
    }

    printf("Visible CUDA devices: %d\n\n", device_count);
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp p{};
        check_cuda(cudaGetDeviceProperties(&p, device), "cudaGetDeviceProperties");

        const int max_warps_per_sm = p.maxThreadsPerMultiProcessor / p.warpSize;
        printf("Device %d: %s\n", device, p.name);
        printf("  Compute capability: %d.%d\n", p.major, p.minor);
        printf("  Streaming multiprocessors: %d\n", p.multiProcessorCount);
        printf("  Warp size: %d threads\n", p.warpSize);
        printf("  Maximum resident threads/SM: %d\n", p.maxThreadsPerMultiProcessor);
        printf("  Maximum resident warps/SM: %d\n", max_warps_per_sm);
        printf("  Maximum threads/block: %d\n", p.maxThreadsPerBlock);
        printf("  Maximum block dimensions: (%d,%d,%d)\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("  Maximum grid dimensions: (%d,%d,%d)\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        printf("  Registers/block limit: %d\n", p.regsPerBlock);
        printf("  Shared memory/block: %.6g KiB\n", p.sharedMemPerBlock / 1024.0);
        printf("  Shared memory/SM: %.6g KiB\n", p.sharedMemPerMultiprocessor / 1024.0);
        printf("  Global memory: %.2f GiB\n\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    /*
     * These are capacity limits, not promises that a kernel reaches them all at
     * once. Threads, blocks, registers, and shared memory are simultaneous
     * constraints. The tightest resource determines residency and occupancy.
     */
}
