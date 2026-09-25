/*
Simple example of using synchronization and atomic operations
to add up the elements of a vector.  Try different approaches
to shared storage and synchronization to see if you can improve
the performance.
*/

#include <cuda_runtime.h>

#include "../common/cuda_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 4096*4096*10

/* A device-wide global variable to capture the sum. */

__device__ float total_sum = 0;

/* BAD - vector_addup_v1 gets incorrect results. */

__global__ void vector_addup_v1( float *data, int length )
{
	int i = (blockIdx.x) * blockDim.x + threadIdx.x;

	if(i<length) {
		total_sum += data[i];
	}
}

/* GOOD - vector_addup_v2 synchronizes correctly. */
/* BUT - still a lot of atomics/sync.  Can you do better? */

__global__ void vector_addup_v2( float *data, int length )
{
	int i = (blockIdx.x) * blockDim.x + threadIdx.x;

	__shared__ float partial_sum;

	if(threadIdx.x==0) {
		partial_sum = 0;
	}
	
	if(i<length) {
		atomicAdd(&partial_sum,data[i]);
	}

	__syncthreads();

	if(threadIdx.x==0) {
		atomicAdd(&total_sum,partial_sum);
	}
		
}

int main()
{
	int bytes = SIZE * sizeof(float);
	
	float *data_h = (float *) malloc( bytes );
	float *data_d;

	for(int i=0; i<SIZE; i++) {
		data_h[i] = 1;
	}
		
	cudaMalloc((void**)&data_d,bytes);

	cudaMemcpy(data_d,data_h,bytes,cudaMemcpyHostToDevice);

	struct timeval start, stop, elapsed;

	/* Mark the stop of the experiment. */
	gettimeofday(&start,0);

	vector_addup_v2<<<SIZE/256,256>>>(data_d,SIZE);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	/* Mark the stop of the experiment. */
	gettimeofday(&stop,0);

	/* Elapsed is the difference between the two */
	timersub(&stop,&start,&elapsed);

	float total_sum_h;
	
	CUDA_CHECK(cudaMemcpyFromSymbol(&total_sum_h,total_sum,sizeof(float),0,cudaMemcpyDeviceToHost));

	printf("cuda elapsed: %u.%0.6u checksum: %f\n",elapsed.tv_sec,elapsed.tv_usec,total_sum_h);

	CUDA_CHECK(cudaFree(data_d));
}
