/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 4.2_arithmetic_intensity
 *   ./4.2_arithmetic_intensity
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 4.2_arithmetic_intensity
 *   sbatch ../common/anvil_gpu.slurm ./4.2_arithmetic_intensity
 *
 * Section 4.2: Arithmetic intensity and a roofline-style bandwidth ceiling
 *
 * Arithmetic intensity = useful operations / bytes transferred from the memory
 * level being analyzed. It helps identify whether performance is likely limited
 * by memory bandwidth or compute throughput.
 *
 * Naive matrix multiplication inner iteration:
 *
 * - load A: 4 bytes;
 * - load B: 4 bytes;
 * - multiply and add: 2 FLOPs;
 * - intensity: 2/8 = 0.25 FLOP/byte.
 *
 * A tiled algorithm reuses each global load approximately TILE_WIDTH times, so
 * its simplified intensity grows to roughly 0.25*TILE_WIDTH FLOP/byte.
 */

#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>

int main() {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp p{};
    const cudaError_t error = cudaGetDeviceProperties(&p, device);
    if (error != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }

    /*
     * memoryClockRate is in kHz and memoryBusWidth is in bits. DDR transfers on
     * both clock edges, so a simple theoretical bandwidth estimate is:
     *
     * 2 * clock_hz * (bus_bits/8) bytes/second.
     *
     * This is a hardware peak, not an achievable application measurement.
     */
    const double peak_bandwidth_gbs = 2.0 * p.memoryClockRate * 1000.0 * (p.memoryBusWidth / 8.0) / 1.0e9;

    printf("Device: %s\n", p.name);
    printf("Estimated peak memory bandwidth: %.1f GB/s\n\n", peak_bandwidth_gbs);
    printf("%-12s%-18sBandwidth roof (GFLOP/s)\n", "Tile width", "Approx FLOP/B");

    const int tile_widths[] = {1, 8, 16, 32};
    for (int index = 0; index < (int)(sizeof(tile_widths) / sizeof(tile_widths[0])); ++index) {
        const int tile = tile_widths[index];
        const double intensity = 0.25 * tile;
        const double bandwidth_roof_gflops = peak_bandwidth_gbs * intensity;
        printf("%-12d%-18.2f%.1f\n", tile, intensity, bandwidth_roof_gflops);
    }

    printf("\nA roof is an upper bound. Poor coalescing, latency, instruction overhead, or insufficient parallelism can place real results well below it.\n");
}
