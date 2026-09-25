/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 3.4_warps_and_divergence
 *   ./3.4_warps_and_divergence
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 3.4_warps_and_divergence
 *   sbatch ../common/anvil_gpu.slurm ./3.4_warps_and_divergence
 *
 * Section 3.4: Warps, SIMD execution, and control divergence
 *
 * A block is partitioned into warps of consecutive linear thread indices. On
 * current NVIDIA hardware, warp size is 32.
 *
 * CUDA exposes SIMT semantics: programmers write independent threads, and each
 * thread has its own registers, program state, and data. The hardware does not,
 * however, issue a separate instruction for every thread. A warp scheduler
 * issues one instruction for a warp at a time, and all active lanes execute that
 * same instruction on their own data. This instruction issue is SIMD-like:
 * single instruction, multiple data lanes.
 *
 * Two groups of lanes in one warp cannot execute different instructions at the
 * same instant. If lanes disagree on an if/else condition,
 * the warp executes the required paths separately. Lanes not belonging to the
 * current path are masked inactive. This is control divergence. Threads retain
 * correct independent semantics, but the warp spends capacity executing each
 * path with only part of its SIMD-like lane width doing useful work.
 *
 * SIMT describes the programming model; SIMD-like describes warp instruction
 * issue.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

__global__ void classify_branches_kernel(int* interleaved_path, int* warp_aligned_path) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * Adjacent even/odd lanes disagree. Every warp containing both even and odd
     * lanes must execute both paths with half of its lanes inactive per pass.
     */
    if ((tid % 2) == 0) {
        interleaved_path[tid] = 100 + tid;
    } else {
        interleaved_path[tid] = 200 + tid;
    }

    /*
     * Threads 0..31 agree and threads 32..63 agree. Different warps may choose
     * different paths without intra-warp divergence. Divergence is a within-warp
     * property, not merely the existence of branches in a kernel.
     */
    const int warp = tid / warpSize;
    if ((warp % 2) == 0) {
        warp_aligned_path[tid] = 300 + tid;
    } else {
        warp_aligned_path[tid] = 400 + tid;
    }
}

const int kWorkIterations = 64;
const int kSeedCount = 1024;

/*
 * A one-instruction branch is often converted into predicated instructions,
 * making branch timing too small and noisy to teach divergence. These two
 * non-inlined functions create distinct, dependent instruction paths. Every
 * thread still performs the same amount of useful work.
 */
__device__ __noinline__ float path_a_work(float value) {
#pragma unroll 1
    for (int iteration = 0; iteration < kWorkIterations; ++iteration) {
        value = fmaf(value, 1.000001f, 0.000001f);
    }
    return value;
}

__device__ __noinline__ float path_b_work(float value) {
#pragma unroll 1
    for (int iteration = 0; iteration < kWorkIterations; ++iteration) {
        value = fmaf(value, 0.999999f, -0.000001f);
    }
    return value;
}

__global__ void divergent_work_kernel(float* output, int elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < elements) {
        const float seed = (float)(tid & (kSeedCount - 1)) * 0.001f;
        // Adjacent lanes disagree, so each warp must execute both call paths.
        output[tid] = (tid & 1) == 0 ? path_a_work(seed) : path_b_work(seed);
    }
}

__global__ void warp_aligned_work_kernel(float* output, int elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < elements) {
        const float seed = (float)(tid & (kSeedCount - 1)) * 0.001f;
        const int warp = tid / warpSize;
        // Every lane in a warp chooses the same path; neighboring warps alternate.
        output[tid] = (warp & 1) == 0 ? path_a_work(seed) : path_b_work(seed);
    }
}

float path_a_cpu(float value) {
    for (int iteration = 0; iteration < kWorkIterations; ++iteration) {
        value = std::fma(value, 1.000001f, 0.000001f);
    }
    return value;
}

float path_b_cpu(float value) {
    for (int iteration = 0; iteration < kWorkIterations; ++iteration) {
        value = std::fma(value, 0.999999f, -0.000001f);
    }
    return value;
}

int main() {
    const int classification_threads = 64;
    const int classification_bytes = classification_threads * sizeof(int);
    static int interleaved_h[classification_threads];
    static int aligned_h[classification_threads];
    int* interleaved_d = NULL;
    int* aligned_d = NULL;

    check_cuda(cudaMalloc((void**)&interleaved_d, classification_bytes), "allocate interleaved classification output");
    check_cuda(cudaMalloc((void**)&aligned_d, classification_bytes), "allocate aligned classification output");
    classify_branches_kernel<<<1, classification_threads>>>(interleaved_d, aligned_d);
    check_cuda(cudaGetLastError(), "launch classify_branches_kernel");
    check_cuda(cudaMemcpy(interleaved_h, interleaved_d, classification_bytes, cudaMemcpyDeviceToHost), "copy interleaved classification output");
    check_cuda(cudaMemcpy(aligned_h, aligned_d, classification_bytes, cudaMemcpyDeviceToHost), "copy aligned classification output");
    check_cuda(cudaFree(interleaved_d), "free interleaved classification output");
    check_cuda(cudaFree(aligned_d), "free aligned classification output");

    bool correct = true;
    for (int tid = 0; tid < classification_threads; ++tid) {
        const int expected_interleaved = (tid % 2 == 0 ? 100 : 200) + tid;
        const int expected_aligned = ((tid / 32) % 2 == 0 ? 300 : 400) + tid;
        correct = correct && interleaved_h[tid] == expected_interleaved;
        correct = correct && aligned_h[tid] == expected_aligned;
    }

    const int elements = 4 * 1024 * 1024;
    const int block_size = 256;
    const int repetitions = 20;
    const int measurement_rounds = 5;
    const int grid_size = (elements + block_size - 1) / block_size;
    const int output_bytes = elements * sizeof(float);

    int device = 0;
    check_cuda(cudaGetDevice(&device), "get active device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "query device properties");
    if (properties.warpSize != 32 || block_size > properties.maxThreadsPerBlock) {
        fprintf(stderr, "This experiment requires warp size 32 and support for 256-thread blocks\n");
        return EXIT_FAILURE;
    }

    float* divergent_d = NULL;
    float* aligned_output_d = NULL;
    check_cuda(cudaMalloc((void**)&divergent_d, output_bytes), "allocate divergent benchmark output");
    check_cuda(cudaMalloc((void**)&aligned_output_d, output_bytes), "allocate aligned benchmark output");

    // Warm up both kernels before timing to exclude one-time CUDA initialization.
    warp_aligned_work_kernel<<<grid_size, block_size>>>(aligned_output_d, elements);
    divergent_work_kernel<<<grid_size, block_size>>>(divergent_d, elements);
    check_cuda(cudaGetLastError(), "launch benchmark warm-up kernels");
    check_cuda(cudaDeviceSynchronize(), "execute benchmark warm-up kernels");

    float aligned_samples[measurement_rounds];
    float divergent_samples[measurement_rounds];

    const auto measure_aligned = [&](int round) {
        const float total_ms = time_cuda_ms([&] {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                warp_aligned_work_kernel<<<grid_size, block_size>>>(aligned_output_d, elements);
            }
            check_cuda(cudaGetLastError(), "launch timed warp-aligned kernels");
        });
        aligned_samples[round] = total_ms / repetitions;
    };
    const auto measure_divergent = [&](int round) {
        const float total_ms = time_cuda_ms([&] {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                divergent_work_kernel<<<grid_size, block_size>>>(divergent_d, elements);
            }
            check_cuda(cudaGetLastError(), "launch timed divergent kernels");
        });
        divergent_samples[round] = total_ms / repetitions;
    };

    // Alternate measurement order to reduce systematic clock and thermal bias.
    for (int round = 0; round < measurement_rounds; ++round) {
        if ((round & 1) == 0) {
            measure_aligned(round);
            measure_divergent(round);
        } else {
            measure_divergent(round);
            measure_aligned(round);
        }
    }

    static float divergent_h[elements];
    static float aligned_output_h[elements];
    check_cuda(cudaMemcpy(divergent_h, divergent_d, output_bytes, cudaMemcpyDeviceToHost), "copy divergent benchmark output");
    check_cuda(cudaMemcpy(aligned_output_h, aligned_output_d, output_bytes, cudaMemcpyDeviceToHost), "copy aligned benchmark output");
    check_cuda(cudaFree(divergent_d), "free divergent benchmark output");
    check_cuda(cudaFree(aligned_output_d), "free aligned benchmark output");

    static float expected_a[kSeedCount];
    static float expected_b[kSeedCount];
    for (int seed_index = 0; seed_index < kSeedCount; ++seed_index) {
        const float seed = (float)(seed_index) * 0.001f;
        expected_a[seed_index] = path_a_cpu(seed);
        expected_b[seed_index] = path_b_cpu(seed);
    }

    float maximum_error = 0.0f;
    for (int tid = 0; tid < elements; ++tid) {
        const int seed_index = tid & (kSeedCount - 1);
        const float expected_divergent = (tid & 1) == 0 ? expected_a[seed_index] : expected_b[seed_index];
        const float expected_aligned = ((tid / properties.warpSize) & 1) == 0 ? expected_a[seed_index] : expected_b[seed_index];
        maximum_error = std::max(maximum_error, std::fabs(divergent_h[tid] - expected_divergent));
        maximum_error = std::max(maximum_error, std::fabs(aligned_output_h[tid] - expected_aligned));
    }
    correct = correct && maximum_error <= 1.0e-5f;

    const float aligned_ms = median_cuda_ms(aligned_samples, measurement_rounds);
    const float divergent_ms = median_cuda_ms(divergent_samples, measurement_rounds);
    const auto nanoseconds_per_element = [](float milliseconds) { return milliseconds * 1.0e6 / elements; };
    const auto billion_elements_per_second = [](float milliseconds) { return (double)(elements) / (milliseconds * 1.0e6); };
    const auto useful_gflops = [](float milliseconds) { return (double)(elements) * (2 * kWorkIterations) / (milliseconds * 1.0e6); };

    printf("Device: %s\n", properties.name);
    printf("Warp size: %d, elements: %d, block size: %d\n", properties.warpSize, elements, block_size);
    printf("Timing: median of %d rounds, %d launches per round\n\n", measurement_rounds, repetitions);
    printf("SIMT view: threads have independent state and data.\n");
    printf("SIMD-like issue: one warp instruction is issued to all active lanes at a time.\n");
    printf("A divergent warp executes different branch paths separately, masking inactive lanes.\n\n");
    printf("Even/odd branch: each warp executes both paths with 16 active lanes per path\n");
    printf("Warp-aligned branch: each warp executes one path with all 32 lanes active\n\n");
    printf("%-22s%13s%15s%15s%17s\n", "Branch layout", "Kernel ms", "ns/element", "G elements/s", "Useful GFLOP/s");
    printf("%-22s%13.3f%15.3f%15.3f%17.3f\n", "Warp-aligned", aligned_ms, nanoseconds_per_element(aligned_ms), billion_elements_per_second(aligned_ms), useful_gflops(aligned_ms));
    printf("%-22s%13.3f%15.3f%15.3f%17.3f\n\n", "Even/odd divergent", divergent_ms, nanoseconds_per_element(divergent_ms), billion_elements_per_second(divergent_ms), useful_gflops(divergent_ms));
    printf("Divergent / warp-aligned time ratio: %.3fx\n", divergent_ms / aligned_ms);
    printf("Maximum absolute error: %.3f\n", maximum_error);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
