#!/usr/bin/env sh
set -eu

for dimension in 2 3; do
    prefix="check-${dimension}d"
    ./run_case --dimension "$dimension" --pde poisson_cg \
        --case poisson_verify --initial_resolution 4 --refine_levels 2 \
        --fe_degree 1 -o "$prefix" --is_adaptive false --tolerance 1e-6

    for imesh in 0 1; do
        volume_file="${prefix}-${imesh}.vtk"
        if [ ! -f "$volume_file" ]; then
            echo "FAIL: Volume output $volume_file not created!"
            exit 1
        fi
        rm "$volume_file"
    done
done
