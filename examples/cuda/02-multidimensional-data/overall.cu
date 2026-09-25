/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make overall
 *   ./overall bird.png overall_bird_grayscale.png overall_bird_blurred.png 2
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make overall
 *   sbatch ../common/anvil_gpu.slurm ./overall bird.png overall_bird_grayscale.png overall_bird_blurred.png 2
 *
 * Overall program: Image pipeline plus matrix multiplication
 *
 * This program runs two pieces of work:
 *
 * 1. read an RGB PNG and use a 2D grid to produce grayscale pixels;
 * 2. feed that device-resident grayscale array directly into a blur kernel;
 * 3. write the grayscale and blurred arrays as PNG files;
 * 4. use another 2D grid to multiply rectangular row-major matrices.
 *
 * Unlike Sections 2.2 and 2.3, it keeps the grayscale image on the GPU and feeds
 * it directly to the blur kernel:
 *
 *     PNG -> RGB host bytes -> GPU grayscale -> GPU blur -> output PNGs
 *
 * Default-stream ordering starts blur after grayscale. Keeping gray_d on the
 * device avoids an intermediate D2H and H2D pair. The CPU implementations are
 * correctness references; performance optimization begins in later topics.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include "png_io.h"

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>


int parse_radius(const char* text) {
    char* end = NULL;
    const long value = strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value < 0 || value > 32) {
        fprintf(stderr, "Blur radius must be an integer from 0 through 32\n");
        exit(EXIT_FAILURE);
    }
    return (int)(value);
}

__global__ void grayscale_kernel(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        const int pixel = row * width + col;
        const int rgb_offset = 3 * pixel;
        const unsigned int r = rgb[rgb_offset];
        const unsigned int g = rgb[rgb_offset + 1];
        const unsigned int b = rgb[rgb_offset + 2];
        gray[pixel] = (unsigned char)((21u * r + 72u * g + 7u * b) / 100u);
    }
}

__global__ void blur_kernel(const unsigned char* input, unsigned char* output, int width, int height, int radius) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        int sum = 0;
        int count = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const int neighbor_row = row + dy;
                const int neighbor_col = col + dx;
                if (neighbor_row >= 0 && neighbor_row < height && neighbor_col >= 0 && neighbor_col < width) {
                    sum += input[neighbor_row * width + neighbor_col];
                    ++count;
                }
            }
        }
        output[row * width + col] = (unsigned char)(sum / count);
    }
}

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int k_size, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < k_size; ++k) {
            sum += a[row * k_size + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

void grayscale_and_blur_cpu(const unsigned char* rgb, unsigned char* gray, unsigned char* blurred, int width, int height, int radius) {
    for (int pixel = 0; pixel < width * height; ++pixel) {
        const int rgb_offset = 3 * pixel;
        const unsigned int r = rgb[rgb_offset];
        const unsigned int g = rgb[rgb_offset + 1];
        const unsigned int b = rgb[rgb_offset + 2];
        gray[pixel] = (unsigned char)((21u * r + 72u * g + 7u * b) / 100u);
    }
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int sum = 0;
            int count = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int neighbor_row = row + dy;
                    const int neighbor_col = col + dx;
                    if (neighbor_row >= 0 && neighbor_row < height && neighbor_col >= 0 && neighbor_col < width) {
                        sum += gray[neighbor_row * width + neighbor_col];
                        ++count;
                    }
                }
            }
            blurred[row * width + col] = (unsigned char)(sum / count);
        }
    }
}

void matmul_cpu(const float* a, const float* b, float* c, int m, int k_size, int n) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < k_size; ++k) {
                sum += a[row * k_size + k] * b[k * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

int maximum_byte_difference(const unsigned char* actual, const unsigned char* expected, int pixels) {
    int maximum = 0;
    for (int i = 0; i < pixels; ++i) {
        maximum = std::max(maximum, std::abs((int)(actual[i]) - (int)(expected[i])));
    }
    return maximum;
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s INPUT.png GRAYSCALE.png BLURRED.png [RADIUS]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int radius = argc == 5 ? parse_radius(argv[4]) : 2;
    int width = 0;
    int height = 0;
    unsigned char* rgb_h = read_rgb_png(argv[1], &width, &height);
    const int pixels = width * height;
    const int rgb_bytes = 3 * pixels;
    const int gray_bytes = pixels;
    unsigned char* gray_reference = (unsigned char*)malloc(pixels);
    if (!gray_reference) {
        fprintf(stderr, "Cannot allocate image array\n");
        exit(EXIT_FAILURE);
    }
    unsigned char* blur_reference = (unsigned char*)malloc(pixels);
    if (!blur_reference) {
        fprintf(stderr, "Cannot allocate image array\n");
        exit(EXIT_FAILURE);
    }
    unsigned char* gray_gpu = (unsigned char*)malloc(pixels);
    if (!gray_gpu) {
        fprintf(stderr, "Cannot allocate image array\n");
        exit(EXIT_FAILURE);
    }
    unsigned char* blur_gpu = (unsigned char*)malloc(pixels);
    if (!blur_gpu) {
        fprintf(stderr, "Cannot allocate image array\n");
        exit(EXIT_FAILURE);
    }

    grayscale_and_blur_cpu(rgb_h, gray_reference, blur_reference, width, height, radius);

    unsigned char* rgb_d = NULL;
    unsigned char* gray_d = NULL;
    unsigned char* blur_d = NULL;
    check_cuda(cudaMalloc((void**)&rgb_d, rgb_bytes), "cudaMalloc RGB");
    check_cuda(cudaMalloc((void**)&gray_d, gray_bytes), "cudaMalloc grayscale");
    check_cuda(cudaMalloc((void**)&blur_d, gray_bytes), "cudaMalloc blur");
    check_cuda(cudaMemcpy(rgb_d, rgb_h, rgb_bytes, cudaMemcpyHostToDevice), "copy RGB H2D");

    const dim3 image_block(16, 16);
    const dim3 image_grid((width + image_block.x - 1) / image_block.x, (height + image_block.y - 1) / image_block.y);
    grayscale_kernel<<<image_grid, image_block>>>(rgb_d, gray_d, width, height);
    check_cuda(cudaGetLastError(), "launch grayscale_kernel");
    blur_kernel<<<image_grid, image_block>>>(gray_d, blur_d, width, height, radius);
    check_cuda(cudaGetLastError(), "launch blur_kernel");
    check_cuda(cudaDeviceSynchronize(), "execute image pipeline");
    check_cuda(cudaMemcpy(gray_gpu, gray_d, gray_bytes, cudaMemcpyDeviceToHost), "copy grayscale D2H");
    check_cuda(cudaMemcpy(blur_gpu, blur_d, gray_bytes, cudaMemcpyDeviceToHost), "copy blur D2H");
    check_cuda(cudaFree(rgb_d), "cudaFree RGB");
    check_cuda(cudaFree(gray_d), "cudaFree grayscale");
    check_cuda(cudaFree(blur_d), "cudaFree blur");

    const int gray_difference = maximum_byte_difference(gray_gpu, gray_reference, pixels);
    const int blur_difference = maximum_byte_difference(blur_gpu, blur_reference, pixels);
    write_grayscale_png(argv[2], gray_gpu, width, height);
    write_grayscale_png(argv[3], blur_gpu, width, height);

    printf("Image pipeline: %s (%dx%d), blur radius: %d\n", argv[1], width, height, radius);
    printf("Grayscale output: %s\n", argv[2]);
    printf("Blurred output: %s\n", argv[3]);
    printf("Device-resident handoff: grayscale output feeds blur without an intermediate host copy.\n");
    printf("Pixels checked against CPU references: %d\n", pixels);
    printf("Maximum grayscale difference: %d\n", gray_difference);
    printf("Maximum blur difference: %d\n", blur_difference);

    // Use a substantial rectangular problem whose dimensions also exercise the 16x16 grid's boundary guards.
    const int m = 513;
    const int k_size = 511;
    const int n = 515;
    static float a_h[m * k_size];
    static float b_h[k_size * n];
    static float c_reference[m * n];
    static float c_gpu[m * n];
    for (int i = 0; i < (m * k_size); ++i) {
        a_h[i] = (float)(i % 11) * 0.25f;
    }
    for (int i = 0; i < (k_size * n); ++i) {
        b_h[i] = (float)(i % 13) * 0.125f - 0.5f;
    }

    matmul_cpu(a_h, b_h, c_reference, m, k_size, n);

    float* a_d = NULL;
    float* b_d = NULL;
    float* c_d = NULL;
    check_cuda(cudaMalloc((void**)&a_d, (m * k_size) * sizeof(float)), "cudaMalloc A");
    check_cuda(cudaMalloc((void**)&b_d, (k_size * n) * sizeof(float)), "cudaMalloc B");
    check_cuda(cudaMalloc((void**)&c_d, (m * n) * sizeof(float)), "cudaMalloc C");
    check_cuda(cudaMemcpy(a_d, a_h, (m * k_size) * sizeof(float), cudaMemcpyHostToDevice), "copy A H2D");
    check_cuda(cudaMemcpy(b_d, b_h, (k_size * n) * sizeof(float), cudaMemcpyHostToDevice), "copy B H2D");

    const dim3 matrix_block(16, 16);
    const dim3 matrix_grid((n + matrix_block.x - 1) / matrix_block.x, (m + matrix_block.y - 1) / matrix_block.y);
    matmul_kernel<<<matrix_grid, matrix_block>>>(a_d, b_d, c_d, m, k_size, n);
    check_cuda(cudaGetLastError(), "launch matmul_kernel");
    check_cuda(cudaDeviceSynchronize(), "execute matmul_kernel");
    check_cuda(cudaMemcpy(c_gpu, c_d, (m * n) * sizeof(float), cudaMemcpyDeviceToHost), "copy C D2H");
    check_cuda(cudaFree(a_d), "cudaFree A");
    check_cuda(cudaFree(b_d), "cudaFree B");
    check_cuda(cudaFree(c_d), "cudaFree C");

    float matrix_max_error = 0.0f;
    for (int i = 0; i < (m * n); ++i) {
        matrix_max_error = std::max(matrix_max_error, std::fabs(c_gpu[i] - c_reference[i]));
    }
    printf("Matrix: %dx%d times %dx%d\n", m, k_size, k_size, n);
    printf("Outputs checked against CPU reference: %d\n", (m * n));
    printf("Maximum absolute error: %.6g\n", matrix_max_error);

    free(rgb_h);
    free(gray_reference);
    free(blur_reference);
    free(gray_gpu);
    free(blur_gpu);

    return gray_difference == 0 && blur_difference == 0 && matrix_max_error <= 1.0e-4f ? EXIT_SUCCESS : EXIT_FAILURE;
}
