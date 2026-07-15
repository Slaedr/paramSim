ParamSim
========

A set of programs to solve partial differential equations by finite element methods under different parameters. This is based on [DEAL-II library](https://dealii.org) and some example codes from that project.

The code focuses on solving a fixed PDE but for different types of parameters, esp. boundary data. To that end, all case-specific parameters can be set on the command line, so that parameter values can be chosen externally as needed and the simulations run accordingly.

## Running the code

Currently, `run_case` is the main executable. Type the command `run_case --help` to see the common options for all PDEs and cases.
For PDE- and case-specific physical options, see the scripts/examples directory. For an example of how to run an ensemble of simulations on a HPC cluster, see the slurm\_scripts/examples directory.
