# Course Project - CSE 60772 Fall 2026

## Project Starting Points

Here are the [project starting points](projects/index.md) for bidding.

## Project Overview

The course project will be an extensive undertaking in the second half of the semester, in which you will either analyze and improve an existing scientific application, or develop a new application in your area of expertise.  Either way, the project will provide an opportunity for you to develop and demonstrate skills in
analyzing, optmizing, and describing a complex application.

The primary project deliverable will consist
of an **extensive technical paper** that explains the scientific purpose,
technical architecture, local parallelism, distributed parallelism,
and performance results of your work.
This document will be developed piecewise over the
course of the project, and revised through multiple steps, resulting in
a high quality paper suitable for submission to a conference or journal.
You will also turn in the code, data, and scripts used to generate
your results.

Each project will be conducted in **teams of two** so that you have ample
opportunity to work together on software development, writing and revising,
and solving technical problems.  Here are the project stages and deadlines:

- **Project Bids** (Oct 12)  Identify a partner you would like to work with,
read over the brief project descriptions and links, and send an ordered
list of *three* projects that you might like to work on together.
(Prof. Thain will make sure each group gets a different project.)
Alternatively, if you have a specific application in your area of research
that you would like to create or improve, then provide a one-page description
of your objectives, the current state of the application, and your proposed
frameworks for parallelism.

- **Project Interviews** - Make contact with the sponsor of the project
and set up a one-hour block to interview them.  Use this time to develop
a good understanding of the *scientific purpose* of the application
and its *technical architecture* as noted in the paper outline below.
Take extensive notes, and use this time to develop a thoughtful approach
for your upcoming measurements and proposed improvements.

**Please Note:** Your sponsor has agreed to talk with you and give you
some starting points, but cannot act as daily technical support to get
things going.  You will have to figure out technical problems yourselves,
or ask the TA or Prof Thain for advice.

- **Checkpoint 1** - Nov 2 - Turn in the first draft of your paper
and the current state of your code repository.  At this point,
sections 2 and 3 should be largely complete, and section 4 should be
underway.  It is ok and expected that other parts will be incomplete,
and perhaps some parts may reflect questions or uncertainties.
Just show the progress that you have made.

- **Checkpoint 2** - Nov 16 - Turn in the second draft of your paper.
At this point, sections 2, 3, and 4 should be largely complete, and
any comments received from the first checkpoint should have been addressed.

- **Class Presentation** - Nov 30 - Dec 7 - Give a highly polished
15-minute in-class presentation with slides that gives an overview of the scientific objective, technical architecture, local performance, and distributed performance of your application.  This presentation will require preparation and practice
to result in a clear and informative performance.
(Your technical results should be largely complete at this point,
although it is ok if there are a few loose ends remaining.)

- **Final Submission** - Dec 9 at 5:00PM - Turn in the final version of your paper
in PDF form along with a link to your project repository containing all
code, data, scripts, and graphs.

## Paper Organization

Your final paper should be a highly polished technical document
formatted using the [ACM Conference Proceedings template](https://www.overleaf.com/latex/templates/acm-conference-proceedings-primary-article-template/wbvnghjbzwpc)

Your paper should be written for an audience that is scientifically literate
and familar with computing, but is not already familiar with your specific
research area.

**Your words must be your own.  You may not use AI of any kind to generate
text in this paper.  Expect to spend time drafting, revising, and revising again.**

Your writing style should be straightforward and focused.
There is no specific length requirement: explain what needs to be explained
and don't add "fluff" to make it more verbose.
With that said: a paper of less than 8 pages is probably not adequately
explaining the details, while a paper of more than 20 pages is probably too long winded.

It should consist of the following major sections:

- **0. Abstract**. (write this last)
A one-paragraph summary of the whole paper that describes each section
of the paper in one sentence each, highlighting the bottom-line performance
improvements.  (e.g. By using technique Z, the application runs 2.3X faster.)

- **1. Introduction** (write this second-to-last)
A one-page summary of the whole paper that descrives each following section
of the paper in one paragraph each, giving the reader a tour of the whole
paper from beginning to end.

- **2. Scientific Objective**
A description of the scientific purpose of the application, giving an
overview of the scientific experiment, fundamental questions, data sources,
and general structure of the activity.
(Rely heavily on your interview with the sponsor for this.)

- **3. Technical Architecture**
A detailed, careful description of the structure of the application.
This section absolutely must include one or more beautiful, handcrafted, detailed,
informative diagrams that shows the relationships of the key components
of the application.  Describe the technologies to be used (languages, libraries,
parallel frameworks) and the overall parallelization strategy.

- **4. Local Parallelism Study**
Establish a baseline performance of one component of the application
running on a single node with a reasonable problem size (or data size)
to finish in a manageable time.  Parallelize the baseline performance
(or improve the parallelization) and measure it at varying scales
against the baseline.  Report on the success and/or limitations of your results.

- **5. Distributed Parallelism Study**
Establish a baseline performance of the whole application on a cluster
with a reasonable problem size (or data size) to finish in a manageable time.
Parallelize the basline performance across the cluster (or improve the parallelization) and measure it at varying scales against the baseline.
Report on the success and/or limitations of your results.

- **6. Conclusions**.  Briefly restate the objectives of the project,
and discuss the success, limitations, and significance of your results.
Suggest ideas for future improvements to the application.

## Presentation Organization

(more information to follow)
