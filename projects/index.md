# Course Project Starting Points

Each of these is a "starting point" for an application, drawn from
research groups at Notre Dame.   Once you have matched with a project,
it will be necessary to make an appointment with the project sponsor
to develop a 

For many of these, we have some examples code, a small input, and scripts for
a minimal run on CRC. That run is only a starting point. What to scale up,
what to measure, and what to improve is for you to develop.

Clone the course repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/dthain/hpds-fa26.git
```

If you already cloned it without them:

```bash
git submodule update --init --recursive
```

## Project List

- Name: [CM1LP](cm1lp/)
  Contact: David Richter (`David.Richter.26@nd.edu`)
  Purpose: The CM1 atmospheric model extended with particles that represent rain, cloud droplets, or spray. Fortran with MPI and OpenACC.
  Code: <https://github.com/RichterLab/CM1LP> (`codex-branch`), included as a submodule.
  Data: `cm1lp/data/namelist.input`, a 128³ Pi Chamber case.

- Name: [PyFlowCL](pyflowcl/)
  Contact: Jon MacArt (`jmacart@nd.edu`)
  Purpose: Compressible and incompressible CFD on structured meshes. Python with `mpi4py` and PyTorch, plus some C++ kernels.
  Code: included in `pyflowcl/code/`. There is no public repository.
  Data: none needed for the 2D shear layer case, which builds its grid in code. Setup and usage guides are in `pyflowcl/data/`.

- Name: [MFDn Transitions](mfdn-transitions/)
  Contact: Mark Caprio (`mcaprio@nd.edu`)
  Purpose: Postprocessing of nuclear wave functions from MFDn to compute transition matrix elements. Fortran with hybrid MPI/OpenMP.
  Code: <https://github.com/nd-nuclear-theory/mfdn-transitions>, included as a submodule.
  Data: the repository's small `example-transitions-01` case. Production inputs are much larger, ask Mark if you want one.

- Name: [QTL Workflow](qtl-workflow/)
  Contact: Tarrick Qahash (`tqahash@nd.edu`)
  Purpose: QTL mapping of drug response in a *Plasmodium falciparum* genetic cross, then a search for genes and mutations inside the QTL intervals. R.
  Code: `qtl-workflow/code/QTL_Code.R`.
  Data: genotype and phenotype CSVs plus the full VCF in `qtl-workflow/data/`.

- Name: [GBCF nf-core RNA-seq](gbcf-nf-core-rnaseq/)
  Contact: Bharat Mishra (`bmishra2@nd.edu`)
  Purpose: Running a production RNA-seq pipeline (Nextflow with containers) on CRC, looking at scheduling, resource use, and scaling.
  Code: <https://github.com/nf-core/rnaseq> (release 3.26.0), included as a submodule.
  Data: the pipeline's small built-in `test` profile. Bharat has larger real datasets and will share one on the CRC shared file system.

- Name: [BSA and Variant Analysis](bsa-variant-analysis/)
  Contact: Nirjhar Bhattacharyya (`nbhattac@nd.edu`)
  Purpose: Bulk segregant analysis of a *Plasmodium falciparum* genetic cross, computing and smoothing allele frequencies across about 100 bulk samples. R.
  Code: <https://github.com/NirjharBhattacharyya/Bulk_Segregation_Analysis_Ferdig_Lab>. The repository is private, ask Nirjhar for access.
  Data: a SNP table with 12,803 variants in `bsa-variant-analysis/data/`.

- [NA61 ROOT Analysis](na61-root-analysis/): C++/ROOT event selection for the NA61 experiment at CERN, used for NuMI neutrino-flux estimation. Contact Michael Quintieri (`mquintie@nd.edu`) and Laura Fields (`lfields2@nd.edu`) for data and code access.

- [Subgraph Isomorphism](subgraph-isomorphism/): subgraph-isomorphism codes used to compare an FPGA-oriented system with conventional HPC systems. Contact Peter Kogge (`Peter.M.Kogge.1@nd.edu`) for data and code access.

- [Persistent Homology / TDA](persistent-homology/): high-performance topological data analysis. Peter Kogge (`Peter.M.Kogge.1@nd.edu`) for data and code access.
