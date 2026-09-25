/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 2.2_rgb_to_grayscale
 *   ./2.2_rgb_to_grayscale bird.png bird_grayscale.png
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 2.2_rgb_to_grayscale
 *   sbatch ../common/anvil_gpu.slurm ./2.2_rgb_to_grayscale bird.png bird_grayscale.png
 *
 * Section 2.2: Convert an RGB PNG to grayscale
 *
 * A two-dimensional CUDA thread coordinate selects one pixel and its three RGB
 * bytes.
 *
 * The host uses libpng through png_io.h to decode INPUT.png into this flat,
 * row-major byte layout:
 *
 *     R0 G0 B0 | R1 G1 B1 | R2 G2 B2 | ...
 *
 * The kernel never sees PNG compression. It receives decoded bytes, writes one
 * grayscale byte per pixel, and leaves PNG encoding to the host.
 *
 * Each thread follows these mappings:
 *
 *     col = blockIdx.x * blockDim.x + threadIdx.x
 *     row = blockIdx.y * blockDim.y + threadIdx.y
 *     pixel = row * width + col
 *     RGB byte offset = 3 * pixel
 *
 * Command-line arguments:
 *
 * 1. input PNG path;
 * 2. output grayscale PNG path.
 */

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include "png_io.h"

#include <algorithm>
#include <stdlib.h>
#include <stdio.h>


__global__ void rgb_to_grayscale_kernel(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ceiling division may launch threads past the image, so only valid coordinates may access memory.
    if (row < height && col < width) {
        const int pixel = row * width + col;
        const int rgb_offset = 3 * pixel;
        const unsigned int r = rgb[rgb_offset];
        const unsigned int g = rgb[rgb_offset + 1];
        const unsigned int b = rgb[rgb_offset + 2];

        /*
         * Human vision is more sensitive to green than red and more sensitive
         * to red than blue, so grayscale is a weighted sum rather than a plain
         * average. These integer weights approximate 0.21R + 0.72G + 0.07B.
         * They sum to 100, keep the result in [0, 255], and make CPU/GPU
         * comparison exactly reproducible for byte-valued input.
         */
        gray[pixel] = (unsigned char)((21u * r + 72u * g + 7u * b) / 100u);
    }
}

void grayscale_cpu(const unsigned char* rgb, unsigned char* gray, int pixels) {
    for (int pixel = 0; pixel < pixels; ++pixel) {
        const int rgb_offset = 3 * pixel;
        const unsigned int r = rgb[rgb_offset];
        const unsigned int g = rgb[rgb_offset + 1];
        const unsigned int b = rgb[rgb_offset + 2];
        gray[pixel] = (unsigned char)((21u * r + 72u * g + 7u * b) / 100u);
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
    if (argc != 3) {
        fprintf(stderr, "Usage: %s INPUT.png OUTPUT.png\n", argv[0]);
        return EXIT_FAILURE;
    }

    int width = 0;
    int height = 0;
    unsigned char* rgb_h = read_rgb_png(argv[1], &width, &height);
    const int pixels = width * height;
    const int rgb_bytes = 3 * pixels;
    const int gray_bytes = pixels;
    unsigned char* gray_gpu = (unsigned char*)malloc(pixels);
    if (!gray_gpu) {
        fprintf(stderr, "Cannot allocate image array\n");
        exit(EXIT_FAILURE);
    }
    unsigned char* gray_reference = (unsigned char*)malloc(pixels);
    if (!gray_reference) {
        fprintf(stderr, "Cannot allocate image array\n");
        exit(EXIT_FAILURE);
    }

    grayscale_cpu(rgb_h, gray_reference, pixels);

    /*
     * The _h and _d suffixes are common CUDA naming conventions:
     * _h means host memory and _d means device memory. They are not part
     * of the language, but they make each pointer's address space visible.
     */
    unsigned char* rgb_d = NULL;
    unsigned char* gray_d = NULL;
    check_cuda(cudaMalloc((void**)&rgb_d, rgb_bytes), "cudaMalloc RGB");
    check_cuda(cudaMalloc((void**)&gray_d, gray_bytes), "cudaMalloc grayscale");
    check_cuda(cudaMemcpy(rgb_d, rgb_h, rgb_bytes, cudaMemcpyHostToDevice), "copy RGB H2D");

    // A 16x16 block contains 256 threads and maps naturally to a two-dimensional image tile.
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgb_to_grayscale_kernel<<<grid, block>>>(rgb_d, gray_d, width, height);
    check_cuda(cudaGetLastError(), "launch rgb_to_grayscale_kernel");
    check_cuda(cudaDeviceSynchronize(), "execute rgb_to_grayscale_kernel");
    check_cuda(cudaMemcpy(gray_gpu, gray_d, gray_bytes, cudaMemcpyDeviceToHost), "copy grayscale D2H");
    check_cuda(cudaFree(rgb_d), "cudaFree RGB");
    check_cuda(cudaFree(gray_d), "cudaFree grayscale");

    const int maximum_difference = maximum_byte_difference(gray_gpu, gray_reference, pixels);
    write_grayscale_png(argv[2], gray_gpu, width, height);

    printf("Input: %s (%dx%d)\n", argv[1], width, height);
    printf("Block: %ux%u, grid: %ux%u\n", block.x, block.y, grid.x, grid.y);
    printf("Pixels checked against CPU reference: %d\n", pixels);
    printf("Output: %s\n", argv[2]);
    printf("Maximum CPU/GPU difference: %d\n", maximum_difference);

    free(rgb_h);
    free(gray_gpu);
    free(gray_reference);

    return maximum_difference == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
