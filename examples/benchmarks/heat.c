/*
Benchmark: Heat Transfer

This program performs a simple 2D heat transfer simulation.
A half-circle object in the middle of the field is maintained
at 100C, while the boundaries of the field are maintained at 25C.
Heat propagates from the object to the surrounding field according
to Fourier's law (Q = m*c*dT/dx), which essentially states that
heat transfer is proportional to temperature difference.

The simulation is a grid of points, each tracking the current
temperature at that point.  Assuming a constant material throughout
the field, the new temperature can be computed as the average
of the current point with its neighbors.

However, we cannot just update each cell of one array iteratively,
because the computations at each point would interfere with each other.
Instead, we maintain a current grid (A) and a next grid (B).
The temperature at each point in the B grid is computed from the values
in the A grid, allowing each point to be independent.

The final field is emitted as a PNM image, which can be viewed with
`eog` or converted into a PNG with `convert`.

As with the other benchmarks, this is the simplest possible formulation
of the problem.  You could make the simulation more realistic by adding
different material types, extending to 3 dimensions, etc.  And, you
can accelerate by reorganizing data structures, exploiting SIMD instructions,
parallelizing with OpenMP, etc...
*/

#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#ifndef SIZE
#define SIZE 256
#endif

#ifndef ITER
#define ITER 1000
#endif

typedef double matrix_t[SIZE][SIZE];

/* Clear a grid by setting all values to zero. */

void grid_clear( matrix_t A )
{
	for(int i=0;i<SIZE;i++) {
		for(int j=0;j<SIZE;j++) {
			A[i][j] = 0;
		}
	}
	
}

/*
Set up some interesting boundary conditions on a grid.
Set the outside borders to 25C.
Then place a half-circle in the middle at 100C.
*/

void grid_boundary_conditions( matrix_t A )
{
	for(int i=0;i<SIZE;i++) {
		A[0][i] = 25.0;
		A[SIZE-1][i] = 25.0;
		A[i][0] = 25.0;
		A[i][SIZE-1] = 25.0;
	}

	double radius = SIZE/4;
	
	for(double theta=0; theta<M_PI; theta+=0.01 ) {
		int i = SIZE/2 + radius * cos( theta );
		int j = SIZE/2 + radius * sin( theta );
		A[i][j] = 100;
	}
}

/*
Advance the simulation by one timestep.
A contains data from the current timestep, and B is the next.
Each point is computed as the average of the 4 neighbors and itself.
Adjust alpha to control how fast heat moves.
*/

void grid_timestep( matrix_t A, matrix_t B )
{
	double alpha = 0.5;

	/* Note that we skip points on the edges, to avoid neighbor wraparound */
	for(int i=1;i<SIZE-1;i++) {
		for(int j=1;j<SIZE-1;j++) {
			double neighbors = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])/4;
			B[i][j] = (alpha*neighbors + (1-alpha)*A[i][j]);
		}
	}
}

/*
Copy the contents of matrix B into matrix A,
so as to get ready for the next timestep.
(With some careful thought, you can make this much more efficient.)
*/

void grid_copy( matrix_t A, matrix_t B )
{
	for(int i=0;i<SIZE;i++) {
		for(int j=0;j<SIZE;j++) {
			A[i][j] = B[i][j];
		}
	}
}

/*
Save a matrix as a simple pnm bitmap file.
Grid values between 0-100C are scaled to black (0) and white (255).
*/

void grid_save_pnm( matrix_t M, const char *filename )
{
	double scale = 2.55;
  
	FILE *f = fopen(filename,"wb");
	fprintf(f, "P6\n%d %d\n255\n",SIZE,SIZE);

	for(int j=0;j<SIZE;j++){
		for(int i=0;i<SIZE;i++){
			unsigned char rgb[3];
			rgb[0] = M[i][j]*scale;
			rgb[1] = M[i][j]*scale;
			rgb[2] = M[i][j]*scale;
			fwrite(rgb,1,sizeof(rgb),f);
		}
	}

	fclose(f);
}

/* Sum up all the elements in the matrix. */

double grid_addup( matrix_t m )
{
	double total = 0;
	for(int i=0; i<SIZE; i++) {
		for(int j=0; j<SIZE; j++) {
			total += m[i][j];
		}
	}
	return total;
}

/*
Declare two matrices for the simulation:
A contains the current time step, and
B receives the values for the next time step.
*/

matrix_t A, B;

int main()
{
	/* Set both matrices to zero. */  
	grid_clear(A);
	grid_clear(B);
	
	/* Mark the start of the experiment. */
	struct timeval start;  
	gettimeofday(&start,0);

	/* For each iteration, set the boundaries, compute the timestep, and copy back */
	for(int k=0;k<ITER;k++) {
		grid_boundary_conditions(A);
		grid_timestep(A,B);
		grid_copy(A,B);
	}

	grid_boundary_conditions(A);

	/* Mark the stop of the experiment. */
	struct timeval stop;  
	gettimeofday(&stop,0);

	/* Elapsed is the difference between the two */
	struct timeval elapsed;
	timersub(&stop,&start,&elapsed);

	/* Write out the final grid as an image. */
	grid_save_pnm(A,"heat.pnm");
	
	/* Compute a final checksum of the results. */
	double checksum = grid_addup(A);

	printf("elapsed: %u.%0.6u checksum: %lf\n",elapsed.tv_sec,elapsed.tv_usec,checksum);
	return 0;
}
