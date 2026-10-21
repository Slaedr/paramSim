ParamSim
========

A set of programs to solve partial differential equations by finite element methods under different parameters. This is based on [DEAL-II library](https://dealii.org) and some example codes from that project.

The code focuses on solving a fixed PDE with different parameters, especially
boundary data. Case-specific parameters can be supplied through command-line
options or referenced parameter files, allowing parameter values to be chosen
externally for each simulation.

Copyright (C) 2023-2026 Oak Ridge National Laboratory.
ParamSim is distributed under the terms described in [the 'NTCL 1.0' license](License.md).

## Dependencies

The C++ simulation code (`run_case`) requires a recent Deal.II installation, CMake and GoogleTest.

The Python-based ensemble simulations require MPI, mpi4py, meshio and h5py.
The parallel dataset generator `scripts/ex_datacombine_mpi.py` requires parallel HDF5 and parallel
h5py. Once parallel HDF5 is installed, parallel h4py can be installed using
```bash
export CC=mpicc
export HDF5_MPI=ON
export HDF5_DIR=/path/to/parallel/hdf5
pip install --no-binary=h5py h5py
```

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
By default, after the finest solve converges, the program writes its solution on
the finest mesh and interpolates it onto selected retained coarse meshes. Files
are named `<output-prefix>-<original-level>.vtk`. Set
`--coarse_grid_solution native_simulation` to write each selected level's own
converged solution immediately after that level finishes. If a later solve
fails, already written native coarse files remain and no file is written for
the failed level; interpolation mode writes no files unless the finest solve
converges. By default all levels are written;
`--coarse_output_levels N` limits output to the finest mesh and its `N`
immediately preceding meshes. Use `0` to write only the finest level. Valid
values are `-1` (all coarse levels) through `--refine_levels - 1`. The finest
level is always the direct converged solution in either mode.

In addition, ensembles of simulations for input parameters drawn from a random distribtion can be run from `scripts/run.py` and `scripts/run_cluster.py`. Invoke them with `--help` for more details. The scripts pass randomly-generated inpts to `run_case` internally.

## Available PDEs

The following PDEs are currently available with continuous Galerkin FEM discretization and Dirichlet boundary data:
1. Poisson equation. It can be selected by `--pde poisson_cg`.
2. Minimum surface equation. This is a nonlinaer elliptic PDE, frequently called "minimal surface" or "minsurf" in the code. Even for a nonlinear elliptic PDE, this is a difficult PDE because of the nonlinear gradient-dependent diffusivity. It can be selected by `--pde minimal_surface`.
3. Gelfand equation. This is a nonlinear reaction-diffusion PDE, but the nonlinearity only comes from an exponential source term. It can be selected by `--pde gelfand`.

## Postprocessing simulation outputs

The converters `scripts/ex_datacombine.py` (serial) and
`scripts/ex_datacombine_mpi.py` (MPI with parallel h5py) combine one or more
ensemble directory trees. Each numeric `sim` directory contains files named
`<output_path>-<refinement_cycle>.vtk` (or `.vtu`), following the solver's output
convention. Prefixes may contain hyphens and differ between trees; the final
numeric suffix identifies the resolution.

Every requested sample must contain the same resolution IDs. IDs are sorted
numerically, may start above zero, and may have shared gaps. Each resolution
produces an HDF5 file containing `mesh` and `fields`. The mesh and field shapes
can differ between resolutions; all samples within a resolution must use the
same field shape and mesh coordinates.

For one resolution, `--outpath dataset.h5` creates exactly `dataset.h5`.
For multiple resolutions, IDs 2 and 10 with either `--outpath dataset` or
`--outpath dataset.h5` create `dataset-2.h5` and `dataset-10.h5`.
Samples are interleaved by input tree in blocks of `--batch-size`, including a
partial final block. Their ordering is identical in every resolution output.
`--nsamples` selects sample IDs zero through the requested count minus one from
each tree; additional samples do not affect the selected resolutions.

For example, to combine five samples from each of two trees with two MPI ranks:

```bash
mpiexec -n 2 python scripts/ex_datacombine_mpi.py \
  --inpaths ensemble-a ensemble-b --outpath dataset.h5 \
  --dimension 2 --nsamples 5 --batch-size 2
```

The Python entry point `ensemble_dir_to_hdf5` in
`scripts/vtk_to_hdf5_dataset.py` applies the same naming and validation rules to
all samples in a single tree. `VTKToHDF5` accepts `resolution_id` to select one
level explicitly when working with an already open HDF5 file.
