ParamSim
========

A set of programs to solve partial differential equations by finite element methods under different parameters. This is based on [DEAL-II library](https://dealii.org) and some example codes from that project.

The code focuses on solving a fixed PDE with different parameters, especially
boundary data. Case-specific parameters can be supplied through command-line
options or referenced parameter files, allowing parameter values to be chosen
externally for each simulation.

Copyright (C) 2023-2026 Oak Ridge National Laboratory.
ParamSim is distributed under the terms described in [the 'NTCL 1.0' license](License.md).

## Running the code

Currently, `run_case` is the main executable. Type the command `run_case --help` to see the common options for all PDEs and cases.
For the cube case parameter-file formats and ensemble range schemas, see the
[case parameter guide](cases/Readme.md). Ready-to-run configurations are in
the [local examples](scripts/examples) and
[cluster examples](slurm_scripts/examples) directories.

Every run must select spatial dimension 2 or 3. For example:

```sh
run_case --dimension 3 --pde poisson_cg --case poisson_verify \
    --initial_resolution 2 --refine_levels 1 --output_prefix field
```
Each refinement cycle writes the complete solution volume to `<output-prefix>-<cycle>.vtk`.

In addition, ensembles of simulations for input parameters drawn from a random distribtion can be run from `scripts/run.py` and `scripts/run_cluster.py`. Invoke them with `--help` for more details. The scripts pass randomly-generated inpts to `run_case` internally.

## Available PDEs

The following PDEs are currently available with continuous Galerkin FEM discretization and Dirichlet boundary data:
1. Poisson equation. It can be selected by `--pde poisson_cg`.
2. Minimum surface equation. This is a nonlinaer elliptic PDE, frequently called "minimal surface" or "minsurf" in the code. Even for a nonlinear elliptic PDE, this is a difficult PDE because of the nonlinear gradient-dependent diffusivity. It can be selected by `--pde minimal_surface`.
3. Gelfand equation. This is a nonlinear reaction-diffusion PDE, but the nonlinearity only comes from an exponential source term. It can be selected by `--pde gelfand`.

