/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 3.2_block_scheduling
 *   ./3.2_block_scheduling
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 3.2_block_scheduling
 *   sbatch ../common/anvil_gpu.slurm ./3.2_block_scheduling
 *
 * Section 3.2: Blocks are independent scheduling units
 *
 * The runtime assigns whole blocks to SMs. A block starts only when one SM has
 * enough thread slots, registers, shared memory, and a block slot for it. A grid
 * usually contains more blocks than can be resident simultaneously, so blocks
 * wait and are assigned as earlier blocks finish.
 *
 * Ordinary blocks may execute in any order. The same kernel can therefore use a
 * few SMs slowly or many SMs quickly without changing source code: transparent
 * scalability. Correctness must not depend on one block running before another.
 */

#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>

__global__ void independent_blocks_kernel(int* output, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Device printf is only a scheduling illustration; ordering is unspecified.
    if (threadIdx.x == 0) {
        printf("block %u began execution\n", blockIdx.x);
    }

    if (i < elements) {
        /*
         * This value depends only on i, never on another block's progress. That
         * independence allows blocks to execute concurrently or in any order.
         */
        output[i] = 3 * i + 1;
    }
}

int main() {
    const int elements = 1000;
    const int block_size = 64;
    const int grid_size = (elements + block_size - 1) / block_size;
    const int bytes = elements * sizeof(int);

    static int output_h[elements];
    int* output_d = NULL;
    cudaError_t error = cudaMalloc((void**)&output_d, bytes);
    if (error != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }

    independent_blocks_kernel<<<grid_size, block_size>>>(output_d, elements);
    error = cudaDeviceSynchronize();
    if (error == cudaSuccess) {
        error = cudaMemcpy(output_h, output_d, bytes, cudaMemcpyDeviceToHost);
    }
    cudaFree(output_d);
    if (error != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }

    bool correct = true;
    for (int i = 0; i < elements; ++i) {
        correct = correct && (output_h[i] == 3 * i + 1);
    }
    printf("Blocks launched: %d\n", grid_size);
    printf("The printed block order is not part of program semantics.\n");
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
