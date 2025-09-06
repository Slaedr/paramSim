#!/usr/bin/env sh

./run_case --pde poisson_cg --case poisson_verify --initial_resolution 4 --refine_levels 2 \
    --fe_degree 1 -o check --is_adaptive false --tolerance 1e-6

meshes="0 1"
for imesh in $meshes; do
    if [ ! -f check-$imesh.vtk ]; then
        echo "FAIL: Volume output check-$imesh.vtk not created!"
        return -1
    fi
    if [ ! -f check-$imesh-boundary.vtk ]; then
        echo "FAIL: Boundary output $imesh not created!"
        return -2
    fi
    rm check-$imesh.vtk
    rm check-$imesh-boundary.vtk
done
