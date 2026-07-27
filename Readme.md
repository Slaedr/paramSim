ParamSim
========

A set of programs to solve partial differential equations by finite element methods under different parameters. This is based on [DEAL-II library](https://dealii.org) and some example codes from that project.

The code focuses on solving a fixed PDE with different parameters, especially
boundary data. Case-specific parameters can be supplied through command-line
options or referenced parameter files, allowing parameter values to be chosen
externally for each simulation.

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

Ensemble JSON files likewise require an integer `"dimension": 2` or
`"dimension": 3` field. The scripts pass this value to `run_case`.

Each refinement cycle writes the complete solution volume to
`<output-prefix>-<cycle>.vtk`. Boundary data is not written to a separate file.

### Current 3D limitations

The manufactured minimal-surface verification cases support both 2D and 3D.
Existing parameterized boundary profiles are extruded in the third direction;
true two-coordinate B2D parameterizations, beginning with `cube_exponential`,
are future work.
