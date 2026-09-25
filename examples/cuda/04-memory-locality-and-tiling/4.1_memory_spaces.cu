/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 4.1_memory_spaces
 *   ./4.1_memory_spaces
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 4.1_memory_spaces
 *   sbatch ../common/anvil_gpu.slurm ./4.1_memory_spaces
 *
 * Section 4.1: CUDA memory spaces, scope, and lifetime
 *
 * CUDA variable placement changes visibility, lifetime, latency, bandwidth, and
 * resource consumption. Names such as "local" describe programming scope, not
 * necessarily fast physical location.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

/*
 * __constant__ variables are declared at file scope. Host code initializes them
 * with cudaMemcpyToSymbol. Device code reads but cannot write them. Their values
 * persist across kernel launches for the application lifetime. Constant memory
 * is small and cached; it is especially effective when a warp reads one address.
 */
__constant__ float constant_scale;

/*
 * A file-scope __device__ variable resides in global memory, is visible to all
 * kernels, and persists across launches. Global communication still requires a
 * correct synchronization strategy; visibility alone does not prevent races.
 */
__device__ int device_launch_marker;


__global__ void memory_spaces_kernel(const float* input, float* output, int n, int marker) {
    /*
     * One shared array is created per block. All threads in that block access the
     * same array; other blocks have separate copies. Its lifetime is this launch.
     */
    __shared__ float tile[32];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x;

    /*
     * i, lane, and private_bias are automatic scalar variables. The compiler
     * normally places them in per-thread registers. Register allocation is not
     * guaranteed by syntax: excessive pressure may cause spilling to local
     * memory, which is backed by device memory and much slower.
     */
    const float private_bias = 1.0f;
    tile[lane] = i < n ? input[i] : 0.0f;
    __syncthreads();

    if (i < n) {
        output[i] = tile[lane] * constant_scale + private_bias;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        device_launch_marker = marker;
    }
}

int main() {
    const int n = 100;
    const int bytes = n * sizeof(float);
    static float input_h[n];
    static float output_h[n];
    static float expected_h[n];
    for (int i = 0; i < n; ++i) {
        input_h[i] = (float)(i);
    }

    const float scale_h = 2.5f;
    for (int i = 0; i < n; ++i) {
        expected_h[i] = input_h[i] * scale_h + 1.0f;
    }

    float* input_d = NULL;
    float* output_d = NULL;
    check_cuda(cudaMemcpyToSymbol(constant_scale, &scale_h, sizeof(scale_h)), "copy constant scale");
    check_cuda(cudaMalloc((void**)&input_d, bytes), "cudaMalloc input");
    check_cuda(cudaMalloc((void**)&output_d, bytes), "cudaMalloc output");
    check_cuda(cudaMemcpy(input_d, input_h, bytes, cudaMemcpyHostToDevice), "copy input H2D");
    memory_spaces_kernel<<<(n + 31) / 32, 32>>>(input_d, output_d, n, 2026);
    check_cuda(cudaGetLastError(), "launch memory_spaces_kernel");
    check_cuda(cudaDeviceSynchronize(), "execute memory_spaces_kernel");
    check_cuda(cudaMemcpy(output_h, output_d, bytes, cudaMemcpyDeviceToHost), "copy output D2H");

    int marker_h = 0;
    check_cuda(cudaMemcpyFromSymbol(&marker_h, device_launch_marker, sizeof(marker_h)), "copy device marker D2H");
    check_cuda(cudaFree(input_d), "cudaFree input");
    check_cuda(cudaFree(output_d), "cudaFree output");

    float max_abs_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_abs_error = std::max(max_abs_error, std::fabs(output_h[i] - expected_h[i]));
    }
    printf("Register: one private bias per thread\n");
    printf("Shared memory: one 32-float tile per block\n");
    printf("Constant memory scale: %.6g\n", scale_h);
    printf("Persistent global device marker: %d\n", marker_h);
    printf("Elements checked: %d\n", n);
    printf("Maximum absolute error: %.6g\n", max_abs_error);
    return marker_h == 2026 && max_abs_error == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*
 * Automatic arrays are logically private to a thread but may occupy local memory
 * if the compiler cannot keep them in registers. Use compiler resource reports
 * (for example nvcc --ptxas-options=-v) and profiling rather than inferring exact
 * physical placement solely from C++ source.
 */
