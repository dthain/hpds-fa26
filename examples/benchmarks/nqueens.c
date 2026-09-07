/*
Simple Benchmark: N-Queens Placement

Find solutions to the problem of placing N queens
on an NxN chessboard, such that none occupy the same
row, column, or diagonal.  This benchmark is representative
of codes that are searching for solutions to combinatoric
problems.

There are many ways to code up a solution to this problem.
Here we take a simple brute-force approach.
Because there can be no more than one queen per row anyway,
we work by placing one queen in the 0th row, then the 1st
row, etc up to the SIZE-1 row.

The current (attempted) solution is represented by an array
`queenxpos[SIZE]`, such that `queenxpos[ypos]` gives the `xpos`
location of the queen in row `ypos`.  The function `nqueens_search`
works by attempting to place a queen at all `xpos` values in row `ypos`,
assuming that all rows<ypos have already been filled.

This benchmark can be parallelized in a number of ways,
but the fundamental challenge is that the workload is recursive.
How can you organize the work to divide up different branches of the tree?
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef SIZE
#define SIZE 16
#endif

int total_solutions = 0;
int max_solutions_printed = 16;

/*
Return true if any queens in rows<ypos
are in the same column as xpos,ypos.
*/

int any_in_same_column( int queenxpos[SIZE], int xpos, int ypos )
{
	for(int j=0;j<ypos;j++) {
		if(queenxpos[j]==xpos) return 1;
	}	

	return 0;
}

/*
Return true if any queens in rows<ypos
are in the same diagonal as xpos,ypos.
*/

int any_in_same_diagonal( int queenxpos[SIZE], int xpos, int ypos )
{
	for(int j=0;j<ypos;j++) {
		int xdist = xpos - queenxpos[j];
	  	int ydist = ypos - j;

		if(xdist<0) xdist=-xdist;
		
		if(xdist==ydist) return 1;
	}	

	return 0;
}

/* Print out one selected solution indicated by queenxpos. */

void nqueens_print( int queenxpos[SIZE] )
{
	if(total_solutions<max_solutions_printed) {
		printf("solution %d:\n",total_solutions);
		for(int y=0;y<SIZE;y++) {
			int x = queenxpos[y];
			for(int i=0;i<x;i++) printf(". ");
			printf("# ");
			for(int i=(x+1);i<SIZE;i++) printf(". ");
			printf("\n");
		}
		printf("\n");
	} else if(total_solutions==max_solutions_printed) {
		printf("large number of solutions, stopped printing...\n");
	}
	total_solutions++;
}

/*
Recursively search for a solution, given a partial
solution queenxpos, completed up to row<ypos.
*/

void nqueens_search( int queenxpos[SIZE], int ypos )
{
	/* Consider every xpos in the row for this ypos: */
	for( int xpos=0; xpos<SIZE; xpos++ ) {

		/* Consider a queen placed at xpos, ypos. */
		/* Skip unsafe locations. */
	  
		if(any_in_same_column(queenxpos,xpos,ypos)) continue;
		if(any_in_same_diagonal(queenxpos,xpos,ypos)) continue;

		/* Then place a queen here */
		queenxpos[ypos] = xpos;

		/* Did we make it to the last row? */
		if( ypos==SIZE-1 ) {
			/* Yes - display one solution. */
			nqueens_print(queenxpos);
		} else {
			/* Not yet - work the next row. */
			nqueens_search(queenxpos,ypos+1);
		}
	}
}

int main( int argc, char *argv[] )
{
	/* Mark the start of the experiment. */
	struct timeval start;  
	gettimeofday(&start,0);

	int queenxpos[SIZE];
	nqueens_search(queenxpos,0);

	/* Mark the stop of the experiment. */
	struct timeval stop;  
	gettimeofday(&stop,0);

	/* Elapsed is the difference between the two */
	struct timeval elapsed;
	timersub(&stop,&start,&elapsed);

	printf("elapsed: %u.%0.6u solutions: %d\n",elapsed.tv_sec,elapsed.tv_usec,total_solutions);

	return 0;
}
