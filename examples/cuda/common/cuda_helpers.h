#pragma once

/* Shared error checks and CUDA-event timing used by the lesson programs. */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

inline void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

inline void check_cuda_at(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << expression << " failed: " << cudaGetErrorString(error) << " at " << file << ':' << line << '\n';
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) check_cuda_at((call), #call, __FILE__, __LINE__)

// CUDA events measure device-stream time, not allocation or memory copies.
template <typename Operation> float time_cuda_ms(Operation operation) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    operation();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return milliseconds;
}

inline float median_cuda_ms(std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

inline float median_cuda_ms(float* samples, int count) {
    std::sort(samples, samples + count);
    return samples[count / 2];
}
