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

prefix="check-finest-only"
./run_case --dimension 2 --pde poisson_cg --case poisson_verify \
    --initial_resolution 2 --refine_levels 3 --coarse_output_levels 0 \
    --fe_degree 1 -o "$prefix" --tolerance 1e-6
test -f "${prefix}-2.vtk"
test ! -e "${prefix}-0.vtk"
test ! -e "${prefix}-1.vtk"
rm "${prefix}-2.vtk"

prefix="check-limited"
./run_case --dimension 2 --pde poisson_cg --case poisson_verify \
    --initial_resolution 2 --refine_levels 3 --coarse_output_levels 1 \
    --fe_degree 1 -o "$prefix" --tolerance 1e-6
test ! -e "${prefix}-0.vtk"
test -f "${prefix}-1.vtk"
test -f "${prefix}-2.vtk"
rm "${prefix}-1.vtk" "${prefix}-2.vtk"
