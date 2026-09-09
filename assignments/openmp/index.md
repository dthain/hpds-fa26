# A2: OpenMP Assignment

First review the [general instructions](../../general) for assignments.

1 - Explore the OpenMP reference materials on the course web page.
Select three interesting OpenMP directives that we didn't discuss in class.
Describe each one in a paragraph that describes what it does, why it is interesting, and how you might use it.

2 - Explore the [benchmarks](https://github.com/dthain/hpds-fa26/tree/main/examples/benchmarks)
presented in class.  Select one of the benchmark codes (except fractal) to work with, and read it carefully
to understand the fundamental operation.  Adjust SIZE, ITER, DELTAT so that the benchmark
runs in about 60s on a single core, without excessive output.

3 - Create an improved version of this benchmark that uses OpenMP for parallelization.
Be creative and make use of appropriate directives and capabilities that we did not discuss in class.
Take care to understand which data should be private/shared within threads.
Test your solution using **1 to 4 threads** (but no more) on the CRC Front End machine.
Double check that the parallelized version produces correct output.

4 - Evaluate the performance of the benchmark on 1-64 cores, but 
**don't run on the front end**.  Instead, submit each execution
using HTCondor, so that the job runs on a node in the cluster.
For example, create a submit script `benchmark.8.submit` something like this:

```
universe = vanilla

executable = ./benchmark
environment = OMP_NUM_THREADS=8

output = benchmark.8.out
error = benchmark.8.err
log = benchmark.8.log

request_cpus = 8
request_memory = 1024MB
request_disk = 100MB

should_transfer_files = yes
when_to_transfer_output = on_exit

queue 1
```

Then submit the job like this:
```console
condor_submit benchmark.8.submit
```

Remember the job ID printed by `condor_submit`, such as `12345.0`. To see whether it is still in the queue, replace `NETID` with your NetID:

```console
condor_q NETID
```

5 - Evaluate the performance of the benchmark
on 64 cores as SIZE starts small and increases by powers of two.
Stop if your runtime exceeds 30 minutes on 64 cores.
As above, run these jobs on the cluster, not the front end node.

6 - Plot your results from step 4 (vary cores) and step 5 (vary size)
and discuss the results, being sure to point out and explain any unexpected behaviors.

7 - Repeat steps 2 through 6 on a second benchmark of your choice.  (Again, not fractal.)

## Turning In

Commit all of your code, scripts, data, and plots to your course repository
in a directory called `openmp`. Include a `README.md` that ties everything together and addresses the points above.
In all things, show insight, curiosity, and craftsmanship.
Be sure to push everything to GitHub!
Turn in your work by submitting the URL of your repository to the corresponding
assignment page in Canvas.
