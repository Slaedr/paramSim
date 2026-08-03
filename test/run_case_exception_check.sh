#!/usr/bin/env sh
# The libEnsemble drivers (scripts/ensemble.py) record a simulation as
# failed only when the run_case process exits with a nonzero status.
# This checks that a thrown C++ exception actually produces that: the
# "Exception on processing:" banner on stderr and a nonzero exit code.
set -u

check_failure() {
    description="$1"
    expected_message="$2"
    shift 2

    ./run_case "$@" > exception_check.out 2> exception_check.err
    status=$?

    if [ "$status" -eq 0 ]; then
        echo "FAIL: run_case returned 0 for $description"
        cat exception_check.err
        exit 1
    fi
    if ! grep -q "Exception on processing:" exception_check.err; then
        echo "FAIL: no exception banner on stderr for $description"
        cat exception_check.err
        exit 1
    fi
    if ! grep -q "$expected_message" exception_check.err; then
        echo "FAIL: stderr for $description lacks '$expected_message'"
        cat exception_check.err
        exit 1
    fi
    rm -f exception_check.out exception_check.err
}

check_failure "an unknown case name" "Non-existent case!" \
    --dimension 2 --pde poisson_cg --case nonexistent_case \
    --initial_resolution 4

check_failure "an unknown PDE solver" "Unsupported PDE solver!" \
    --dimension 2 --pde nonexistent_pde --case poisson_verify \
    --initial_resolution 4
