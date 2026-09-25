/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 2.1_multidimensional_indices
 *   ./2.1_multidimensional_indices
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 2.1_multidimensional_indices
 *   sbatch ../common/anvil_gpu.slurm ./2.1_multidimensional_indices
 *
 * Section 2.1: Multidimensional grids, blocks, data, and flattening
 *
 * CUDA grids and blocks may each have up to three dimensions. dim3 stores the
 * x, y, and z extents. Unspecified dim3 dimensions default to one.
 *
 * Map CUDA coordinates to the data shape:
 *
 *     x -> column, y -> row, z -> plane.
 *
 * Memory remains flat. Coordinates organize the work; a row-major formula
 * converts (plane,row,col) into one offset:
 *
 *     offset = (plane * height + row) * width + col
 *
 * CPU and GPU compute the same offset array. This is a coordinate-mapping
 * example, not a performance experiment: eighteen elements are intentionally
 * small enough that every mapping can be printed and inspected.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <stdlib.h>
#include <stdio.h>


__global__ void flatten_3d_kernel(int* offsets, int width, int height, int depth) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int plane = blockIdx.z * blockDim.z + threadIdx.z;

    // Rounded-up grid dimensions create fringe threads, so all three logical coordinates need bounds checks.
    if (col < width && row < height && plane < depth) {
        const int offset = (plane * height + row) * width + col;
        offsets[offset] = offset;
    }
}

void flatten_3d_cpu(int* offsets, int width, int height, int depth) {
    for (int plane = 0; plane < depth; ++plane) {
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                const int offset = (plane * height + row) * width + col;
                offsets[offset] = offset;
            }
        }
    }
}

int main() {
    const int width = 3;
    const int height = 3;
    const int depth = 2;
    const int element_count = width * height * depth;
    const int bytes = element_count * sizeof(int);
    static int cpu_offsets[element_count];
    static int gpu_offsets[element_count];

    flatten_3d_cpu(cpu_offsets, width, height, depth);

    /*
     * block(2,2,2) contains 2*2*2=8 threads. The product of block dimensions,
     * not any one dimension alone, must respect maxThreadsPerBlock.
     */
    const dim3 block(2, 2, 2);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);

    int* device_offsets = NULL;
    check_cuda(cudaMalloc((void**)&device_offsets, bytes), "cudaMalloc offsets");

    flatten_3d_kernel<<<grid, block>>>(device_offsets, width, height, depth);
    check_cuda(cudaGetLastError(), "launch flatten_3d_kernel");
    check_cuda(cudaDeviceSynchronize(), "execute flatten_3d_kernel");
    check_cuda(cudaMemcpy(gpu_offsets, device_offsets, bytes, cudaMemcpyDeviceToHost), "copy offsets D2H");
    check_cuda(cudaFree(device_offsets), "cudaFree offsets");

    int maximum_difference = 0;
    for (int i = 0; i < element_count; ++i) {
        maximum_difference = std::max(maximum_difference, std::abs(gpu_offsets[i] - cpu_offsets[i]));
    }

    printf("Data shape (depth,height,width): (%d,%d,%d)\n", depth, height, width);
    printf("CUDA grid (x,y,z): (%u,%u,%u)\n", grid.x, grid.y, grid.z);
    printf("CUDA block (x,y,z): (%u,%u,%u)\n", block.x, block.y, block.z);
    for (int plane = 0; plane < depth; ++plane) {
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                const int offset = (plane * height + row) * width + col;
                printf("data=(plane=%d,row=%d,col=%d) -> offset=%d\n", plane, row, col, gpu_offsets[offset]);
            }
        }
    }
    printf("Coordinates checked: %d\n", element_count);
    printf("Maximum CPU/GPU offset difference: %d\n", maximum_difference);

    return maximum_difference == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
