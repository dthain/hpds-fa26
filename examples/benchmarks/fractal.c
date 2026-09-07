/*
Benchmark: Mandelbrot Fractal

Compute an image of the Mandelbrot fractal in the complex plane.
For each pixel in a SIZE x SIZE array, run the escape time algorithm
with a maximum of ITER iterations, then color that pixel according
to the number of iterations.

Note that the image is written out in the (simple) PNM file format.
You can display this image with the `eog` tool on Linux, or use
`convert` to convert it into a png or more convenient style.

The fractal computation is "pleasantly parallel" in that each
pixel can be computed independently without data sharing.
*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <sys/time.h>

#ifndef SIZE
#define SIZE 1024
#endif

#ifndef ITER
#define ITER 500
#endif

typedef int matrix_t[SIZE][SIZE];

/*
Compute the number of iterations at point x, y
in the complex space, up to a maximum of maxiter.
Return the number of iterations at that point.

This example computes the Mandelbrot fractal:
z = z^2 + alpha

Where z is initially zero, and alpha is the location x + iy
in the complex plane.  Note that we are using the "complex"
numeric type in C, which has the special functions cabs()
and cpow() to compute the absolute values and powers of
complex values.
*/

static int fractal_compute_point( double x, double y, int max )
{
	double complex z = 0;
	double complex alpha = x + I*y;

	int iter = 0;

	while( cabs(z)<4 && iter < max ) {
		z = cpow(z,2) + alpha;
		iter++;
	}

	return iter;
}

/*
Compute an entire image, writing each point to the given bitmap.
Scale the image to the range (xmin-xmax,ymin-ymax).
*/

void fractal_compute( matrix_t M, double xmin, double xmax, double ymin, double ymax, int maxiter )
{
	int i,j;

	// For every pixel i,j, in the image...

	for(i=0;i<SIZE;i++) {
		for(j=0;j<SIZE;j++) {

			// Scale from pixels i,j to coordinates x,y
			double x = xmin + i*(xmax-xmin)/SIZE;
			double y = ymin + j*(ymax-ymin)/SIZE;

			// Compute the iterations at x,y
			M[i][j] = fractal_compute_point(x,y,maxiter);
		}
	}
}

/*
Save a fractal matrix as a simple pnm bitmap file.
Any point that escapes on the first iteration is black,
while any point that reaches max iterations is white,
everything else between grayscale.
*/

void fractal_save_pnm( matrix_t M, const char *filename )
{
	double scale = 255.0/ITER;
  
	FILE *f = fopen(filename,"wb");
	fprintf(f, "P6\n%d %d\n255\n",SIZE,SIZE);

	for(int i=0;i<SIZE;i++){
		for(int j=0;j<SIZE;j++){
			unsigned char rgb[3];
			rgb[0] = M[i][j]*scale;
			rgb[1] = M[i][j]*scale;
			rgb[2] = M[i][j]*scale;
			fwrite(rgb,1,sizeof(rgb),f);
		}
	}

	fclose(f);
}

int matrix_addup( matrix_t M )
{
	int total=0;
	for(int i=0;i<SIZE;i++){
		for(int j=0;j<SIZE;j++) {
			total += M[i][j];
		}
	}
	return total;
}

matrix_t M;

int main( int argc, char *argv[] )
{
	// The initial boundaries of the fractal image in x,y space.
	double xmin=-1.5, xmax=0.5, ymin=-1.0, ymax=1.0;

	// Experiment with these to get a different image.
	//double xmin= 0.286682, xmax=0.287182, ymin=0.014037, ymax=0.014537;

	/* Mark the start of the experiment. */
	struct timeval start;  
	gettimeofday(&start,0);

	fractal_compute(M,xmin,xmax,ymin,ymax,ITER);

	/* Mark the stop of the experiment. */
	struct timeval stop;  
	gettimeofday(&stop,0);

	/* Elapsed is the difference between the two */
	struct timeval elapsed;
	timersub(&stop,&start,&elapsed);

	fractal_save_pnm(M,"fractal.pnm");

	/* Compute a final checksum of the results. */
	double checksum = matrix_addup(M);

	printf("elapsed: %u.%0.6u checksum: %lf\n",elapsed.tv_sec,elapsed.tv_usec,checksum);
	return 0;
}
