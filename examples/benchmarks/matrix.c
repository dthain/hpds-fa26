/*
Benchmark: Matrix Multiplication

Perform a multiplication of two random matrices, A and B,
such that each cell in the result C is the dot-product
of a row from A and a column from B.

The multiplication is coded in the most direct way,
which is three nested for-loops, resulting in an O(n^3)
algorithm.  There are many opportunities here to accelerate
the code by improving data layout, access patterns,
SIMD instructions, and parallelism with OpenMP.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef SIZE
#define SIZE 512
#endif

#ifndef ITER
#define ITER 100
#endif

/* Define a new type matrix_t which is a two dimensional array. */
typedef double matrix_t[SIZE][SIZE];

/* Initialize each element of the matrix to a single value. */

void matrix_init( matrix_t m, double value )
{
	for(int i=0; i<SIZE; i++) {
		for(int j=0; j<SIZE; j++) {
			m[i][j] = value;
		}
	}
}

/* Initialize each element of the matrix to a random value. */

void matrix_random( matrix_t m )
{
	for(int i=0; i<SIZE; i++) {
		for(int j=0; j<SIZE; j++) {
			m[i][j] = rand();
		}
	}
}

/* Sum up all the elements in the matrix. */

double matrix_addup( matrix_t m )
{
	double total = 0;
	for(int i=0; i<SIZE; i++) {
		for(int j=0; j<SIZE; j++) {
			total += m[i][j];
		}
	}
	return total;
}

/* Multiply matrix a into matrix b, in the straightfoward way. */

void matrix_multiply( matrix_t a, matrix_t b, matrix_t c )
{
	for(int i=0; i<SIZE; i++) {
		for(int j=0; j<SIZE; j++) {
			double total = 0;
			for(int k=0; k<SIZE; k++) {
				total += a[k][j]*b[i][k];
			}
			c[i][j] = total;
		}
	}
}

/* Declare three matrices */
matrix_t A, B, C;

int main( int argc, char *argv[] )
{
	/* Set the random seed for reproducibility. */
	srand(7);

	/* Initialize the matrices */
	matrix_random(A);
	matrix_random(B);

	/* Mark the start of the experiment. */
	struct timeval start;  
	gettimeofday(&start,0);

	/* Multiply the matrix K times... */
	for(int k=0; k<ITER; k++) {

		/* Sneak k into the data to prevent optimizing out. */		
		A[0][0] = k;

		matrix_multiply(A,B,C);
	}

	/* Mark the stop of the experiment. */
	struct timeval stop;  
	gettimeofday(&stop,0);

	/* Elapsed is the difference between the two */
	struct timeval elapsed;
	timersub(&stop,&start,&elapsed);

	/* Compute a final checksum of the results. */
	double checksum = matrix_addup(C);

	printf("elapsed: %u.%0.6u checksum: %lf\n",elapsed.tv_sec,elapsed.tv_usec,checksum);
}
