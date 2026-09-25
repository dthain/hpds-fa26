/*
 * From this topic directory (see ../README.md for details).
 *
 * HTCondor (Notre Dame CRC): build on the front end, run on a GPU worker
 *   module load cuda/12.1
 *   make 2.2_rgb_to_grayscale
 *   ./2.2_rgb_to_grayscale
 *
 * Slurm (Purdue Anvil): build on the login node, run with sbatch
 *   module load modtree/gpu cuda/12.8.0
 *   make 2.2_rgb_to_grayscale
 *   sbatch ../common/anvil_gpu.slurm
 *
 * Section 2.2: Convert an RGB PNG to grayscale
 *
 * A two-dimensional CUDA thread coordinate selects one pixel and its three RGB
 * bytes.  The RGB data comes like this:
 *
 *     R0 G0 B0 | R1 G1 B1 | R2 G2 B2 | ...
 *
 * And is converted into grayscale data like this:
 *
 *     G0 G1 G2 ...
*/

/*
NOTE: As given, this code shows the CPU version faster than the GPU.
Can you explain why, and make some changes that might improve things?
*/

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 1024

/*
Convert a matrix of 8-bit r,g,b values
into a matrix of 8-bit grayscale values.
Using the CPU, iterate over the entire array size,
and convert each one at a time.
*/

void rgb_to_grayscale_cpu( uint8_t *rgb, uint8_t *gray, int width, int height )
{
	for(int j=0; j<height; j++) {

		for(int i=0; i<width; i++) {

			int rgb_offset = 3 * (i+j*height);

			int r = rgb[rgb_offset];
		        int g = rgb[rgb_offset + 1];
		        int b = rgb[rgb_offset + 2];

			gray[i+j*height] = (21*r + 72*g + 7*b) / 100;
		}
	}
}

/*
Convert a matrix of 8-bit r,g,b values
into a matrix of 8-bit grayscale values.
Using the GPU, locate each thread's position
in the matrix, and convert one value.
*/

__global__ void rgb_to_grayscale_gpu( uint8_t* rgb, uint8_t* gray, int width, int height )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ( i<width && j<height ) {

		int rgb_offset = 3 * (i+j*height);

		int r = rgb[rgb_offset];
	        int g = rgb[rgb_offset + 1];
	        int b = rgb[rgb_offset + 2];

		gray[i+j*height] = (21*r + 72*g + 7*b) / 100;
		//printf("i %d j %d r %d g %d b %d gray %d\n",i,j,r,g,b,gray[i+j*height]);
	}
}

int main(int argc, char** argv)
{
	struct timeval start, stop, elapsed;
	int checksum;
	
	/* Allocate the host arrays for rgb and grayscale data */
	int rgb_size = SIZE * SIZE * sizeof(uint8_t) * 3;
	int gray_size = SIZE * SIZE * sizeof(uint8_t);

	uint8_t *rgb_h = (uint8_t *)malloc(rgb_size);
	uint8_t *gray_h = (uint8_t *)malloc(gray_size);

	/* Initialize the rgb data with something */
	for(int i=0; i<rgb_size; i++) {
		rgb_h[i] = i*i % 7;
	}

	/********************************************/
	/* Part 1: CPU Experiment.                  */
	/********************************************/
	   
	/* Mark the stop of the experiment. */
	gettimeofday(&start,0);

	rgb_to_grayscale_cpu(rgb_h,gray_h,SIZE,SIZE);

	/* Mark the stop of the experiment. */
	gettimeofday(&stop,0);

	/* Elapsed is the difference between the two */
	timersub(&stop,&start,&elapsed);

	/* Compute a final checksum of the results. */
	for(int i=0; i<gray_size; i++) {
		checksum += gray_h[i];
	}
	
	printf("cpu elapsed: %u.%0.6u checksum: %d\n",elapsed.tv_sec,elapsed.tv_usec,checksum);

	/********************************************/
	/* Part 2: GPU Experiment.                  */
	/********************************************/
	   
	/* Mark the stop of the experiment. */
	gettimeofday(&start,0);

	/* Allocate the device copies for rgb and grayscale */
	uint8_t *rgb_d = 0;
	uint8_t *gray_d = 0;
	
	CUDA_CHECK(cudaMalloc((void**)&rgb_d,rgb_size));
	CUDA_CHECK(cudaMalloc((void**)&gray_d,gray_size));

	/* Copy the rgb data over to the device */
	CUDA_CHECK(cudaMemcpy(rgb_d,rgb_h,rgb_size,cudaMemcpyHostToDevice));

	/* Each 16x16 block uses 256 threads. */
	/* Make the grid large enough to cover the image. */
        dim3 block(16, 16);
        dim3 grid(SIZE/block.x+1,SIZE/block.y+1);

	rgb_to_grayscale_gpu<<<grid,block>>>(rgb_d,gray_d,SIZE,SIZE);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	/* Copy the grayscale data back out. */
	CUDA_CHECK(cudaMemcpy(gray_h,gray_d,gray_size,cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(rgb_d));
	CUDA_CHECK(cudaFree(gray_d));
	
	/* Mark the stop of the experiment. */
	gettimeofday(&stop,0);

	/* Elapsed is the difference between the two */
	timersub(&stop,&start,&elapsed);

	/* Compute a final checksum of the results. */
	checksum = 0;
	for(int i=0; i<gray_size; i++) {
		checksum += gray_h[i];
	}
	
	printf("cuda elapsed: %u.%0.6u checksum: %d\n",elapsed.tv_sec,elapsed.tv_usec,checksum);

	return 0;
	
}
