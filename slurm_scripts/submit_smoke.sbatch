#!/bin/bash
#SBATCH -A PROJECT_ID
#SBATCH -J paramsim-smoke
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# Wrapper around the production script with a small, inexpensive run.
export RANKS_PER_NODE="${RANKS_PER_NODE:-8}"
export PARAMSIM_NUM_SAMPLES="${PARAMSIM_NUM_SAMPLES:-16}"

# Set these before sbatch or edit and uncomment them here:
# export PROJECT_ID=yyyXXX
# export PARAMSIM_ROOT=/path/to/paramSim-sources
# export MODULE_AND_PYTHON_ENV_FILE=airis_bashrc.sh
# export OUTPUT_BASE=<proj>/paramsim_runs

exec "${PARAMSIM_ROOT:?Set PARAMSIM_ROOT}/slurm_scripts/submit.sbatch"
