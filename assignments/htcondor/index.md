# HTCondor Assignment

## Objective

Use HTCondor to render a one-minute long video showing a gradual zoom
into the Mandelbrot set, making use of both multicore and distributed
resources to minimize the total rendering time.

(This will be a simpler assignment to get you familiar with HTCondor
while still leaving time to work on your course project.)

1 - Establish a sequental baseline.  Take the mandelbrot fractal benchmark,
modify it to generate a 4K HDTV resolution image of 2880x2160,
and to accept the xmin,xmax,ymin,ymax coordinates on the command line.
Measure the time to execute the sequential version, and note the size of the
output file produced.

2 - Use this online [mandelbrot explorer](https://math.hws.edu/eck/js/mandelbrot/MB.html) to select an interesting portion
of the mandelbrot set.  (Zoom in at least 10 times or more, until the tool
starts to show poor resolution.)  Use the "Show XML" button to extract the
boundaries of the zoomed in image.

3 - Write a short script or program to submit HTCondor jobs in order to
run the fractal program once for each frame in the video (60s at 24fps),
gradually zooming in from the whole view to your desired destination.
Use `ffmpeg` to join all the images into a single mp4 video.
(`module load ffmpeg`).

4 - Submit your user-log file to the [condor log analyer](http://condorlog.cse.nd.edu)
and save the resulting URL.  Upload your resulting video to Google Drive, mark it
as "readable by anyone with the link", and save the URL.
Commit all of your code, scripts, and logfile to your course repository.
**But do not add any image files or videos to the repository.**

5 - Write a short README.md that provides links to your resulting movie and log analyzer page.
Briefly discuss your observations from the log analyzer results -- did anything surprising happen?
How does using the shared cluster compare to the "ideal" speedup that would result
from having dedicated machines?









