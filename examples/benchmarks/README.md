# Benchmark Codes for CSE 60772

This is a selection of simple benchmark codes provided
to give you some initial experience in measuring and
optimizing parallel codes.  Each one represents a different
common computational pattern found in scientific computing,
and quickly becomes time consuming as the problem size is increased.

Note that these are "toys" used to illustrate key problems in
high performance computing, and there are a number of ways
in which they can be improved, and perhaps dramatically so.
You are welcome and encouraged to experiment with compiler
optimizations, manual SIMD instructions, OpenMP directives,
rearranging data structures, and so forth.  Go nuts!

To get started:
```
git clone https://github.com/dthain/hpds-fa26
cd hpds-fa26/examples/benchmarks
make
```

(If you have cloned this repository before,
then just pull to get the latest version of the code:)

```
git pull origin main
```

- [fractal.c](fractal.c) computes an image from the Mandelbrot set,
using the "escape time algorithm" computed at each point of the pixel.
The entire image is computed internally in a matrix, and then written
out as a PNM image.  SIZE gives the dimension of the image, and ITER
indicates the maximum number of iterations.

- [matrix.c](matrix.c) performs a conventional matrix multiplication
using the basic O(n^3) algorithm.  SIZE gives the dimension of the
matricies, and ITER indicates how many multiplications to perform.

- [heat.c](heat.c) performs a simulation of a 2D space in which
a curved shape is held at a constant temperature, and heat propagates
to the surrounding area via Fourier's law.  SIZE gives the dimension
of the grid, and ITER indicates how many timesteps to compute.

- [nbody.c](nbody.c) performs a simulation of N bodies moving
through 2D space, attracted by Netwon's law of gravitation.
SIZE gives the total number of bodies, and DELTAT indiates the timestep granularity.

- [nqueens.c](nqueens.c) searches for solutions to the problem
of placing N queens on an NxN chessboard.  SIZE is the dimension
of the chessboard.  (ITER is not used.)

<table>
<tr>
<td>Fractal
<td>N-Body
<td>Heat
<tr>
<td><img width=1024px src=images/fractal.png>
<td><img width=1024px src=images/nbody.png>
<td><img width=1024px src=images/heat.png>
</table>


