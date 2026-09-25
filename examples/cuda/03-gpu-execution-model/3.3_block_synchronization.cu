/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 3.3_block_synchronization
 *   ./3.3_block_synchronization
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 3.3_block_synchronization
 *   sbatch ../common/anvil_gpu.slurm ./3.3_block_synchronization
 *
 * Section 3.3: Block synchronization with __syncthreads()
 *
 * Threads in one block may cooperate because they are resident on the same SM.
 * __syncthreads() is a block-wide barrier: every thread waits until every other
 * thread in that block reaches the same barrier.
 *
 * This kernel has two phases:
 *
 * Phase 1: each thread produces one value in block-shared memory.
 * Barrier: every produced value must be visible before Phase 2 begins.
 * Phase 2: each thread consumes a value produced by its left neighbor.
 *
 * Without the barrier, a fast consumer could read before its neighbor writes.
 */

#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>

const int kBlockSize = 32;

__global__ void neighbor_exchange_kernel(int* output) {
    /*
     * One shared array exists per block. Every thread in the block can access
     * it. The memory-locality topic studies shared memory in detail; here it is
     * concrete location for demonstrating the synchronization contract.
     */
    __shared__ int produced[kBlockSize];

    const int lane = threadIdx.x;
    produced[lane] = 10 * lane; // Phase 1: each thread writes one element.

    /*
     * The barrier must be reached by every thread in the block. Placing it under
     * `if (lane < 16)` would be invalid because half the block could wait forever
     * for threads that never arrive. A barrier separates phases, not individual
     * threads.
     */
    __syncthreads();

    const int left_lane = (lane + kBlockSize - 1) % kBlockSize;
    output[lane] = produced[left_lane]; // Phase 2: consume neighbor's value.
}

int main() {
    static int output_h[kBlockSize];
    int* output_d = NULL;
    const int bytes = kBlockSize * sizeof(int);
    cudaError_t error = cudaMalloc((void**)&output_d, bytes);
    if (error == cudaSuccess) {
        neighbor_exchange_kernel<<<1, kBlockSize>>>(output_d);
        error = cudaDeviceSynchronize();
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(output_h, output_d, bytes, cudaMemcpyDeviceToHost);
    }
    cudaFree(output_d);
    if (error != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }

    bool correct = true;
    for (int lane = 0; lane < kBlockSize; ++lane) {
        const int left_lane = (lane + kBlockSize - 1) % kBlockSize;
        correct = correct && (output_h[lane] == 10 * left_lane);
    }
    for (int lane = 0; lane < kBlockSize; ++lane) {
        const int left_lane = (lane + kBlockSize - 1) % kBlockSize;
        printf("Thread %d received %d from thread %d\n", lane, output_h[lane], left_lane);
    }
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*
 * __syncthreads() does not synchronize different blocks. A normal kernel must
 * not assume all blocks are resident simultaneously. Global phase boundaries
 * are usually expressed as separate kernel launches: kernel completion is a
 * grid-wide boundary before a later kernel in the same stream begins.
 */
