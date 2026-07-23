ParamSim
========

A set of programs to solve partial differential equations by finite element methods under different parameters. This is based on [DEAL-II library](https://dealii.org) and some example codes from that project.

The code focuses on solving a fixed PDE but for different types of parameters, esp. boundary data. To that end, all case-specific parameters can be set on the command line, so that parameter values can be chosen externally as needed and the simulations run accordingly.

## Running the code

Currently, `run_case` is the main executable. Type the command `run_case --help` to see the common options for all PDEs and cases.
For PDE- and case-specific physical options, see the scripts/examples directory. For an example of how to run an ensemble of simulations on a HPC cluster, see the slurm\_scripts/examples directory.

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

The `minimal_surface_ball_verify` and `minimal_surface_cube_verify` cases remain
2D-only because their manufactured right-hand side has not yet been derived in
3D. Existing parameterized boundary profiles are extruded in the third
direction; true two-coordinate B2D parameterizations, beginning with
`cube_exponential`, are future work.
