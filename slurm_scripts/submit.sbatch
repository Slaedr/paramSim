#!/bin/bash
#SBATCH -A PROJECT_ID
#SBATCH -J paramsim
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o %x-%j.out

# Production launcher: one Slurm job step for the entire ensemble.
set -euo pipefail

# ----------------------- EDIT THESE PATHS -----------------------
PROJECT_ID="${PROJECT_ID:-${SLURM_JOB_ACCOUNT:-PROJECT_ID}}"
PARAMSIM_ROOT="${PARAMSIM_ROOT:-/path/to/paramSim}"
CASE_FILE="${CASE_FILE:-${PARAMSIM_ROOT}/frontier/cases/poisson_exp_frontier.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${MEMBERWORK}/${PROJECT_ID}/paramsim_runs}"
# and set MODULE_AND_PYTHON_ENV_FILE
# ------------------------VARIABLES TO SET------------------------
# RANKS_PER_NODE
# PARAMSIM_NUM_SAMPLES or "num_samples" in JSON

if [[ -z "${MODULE_AND_PYTHON_ENV_FILE}" ]]; then
    echo "Need env var MODULE_AND_PYTHON_ENV_FILE"
    exit 3
fi

# Load appropriate modules for paramSim
# Load python environment with libensemble, mpi4py
source ${MODULE_AND_PYTHON_ENV_FILE}

export PARAMSIM_EXEC="${PARAMSIM_EXEC:-${PARAMSIM_BUILD_DIR}/run_case}"
export DEAL_II_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Default: 56 allocatable CPU cores per node. Start with 8-16
# ranks per node for a smoke test, then increase toward 56 after checking memory
# and I/O behavior. One rank is the manager; all remaining ranks are sims.
RANKS_PER_NODE="${RANKS_PER_NODE:-56}"
NTASKS=$((SLURM_NNODES * RANKS_PER_NODE))
if (( NTASKS < 2 )); then
  echo "Need at least two ranks: one manager and one worker" >&2
  exit 2
fi

RUN_ROOT="${OUTPUT_BASE}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
cp "$CASE_FILE" "$RUN_ROOT/case.json"
module -t list >"$RUN_ROOT/modules.txt" 2>&1 || true
python3 -m pip freeze >"$RUN_ROOT/python-freeze.txt"
{
  echo "job_id=$SLURM_JOB_ID"
  echo "nodes=$SLURM_NNODES"
  echo "ranks_per_node=$RANKS_PER_NODE"
  echo "total_ranks=$NTASKS"
  echo "simulation_workers=$((NTASKS - 1))"
  echo "paramsim_exec=$PARAMSIM_EXEC"
  echo "case_file=$CASE_FILE"
  echo "start_utc=$(date -u +%FT%TZ)"
} >"$RUN_ROOT/run-metadata.txt"

export PARAMSIM_OUTPUT_DIR="$RUN_ROOT/ensemble"
cd "$RUN_ROOT"

SECONDS=0

set +e
srun --unbuffered \
  -N "$SLURM_NNODES" \
  -n "$NTASKS" \
  --ntasks-per-node="$RANKS_PER_NODE" \
  -c 1 \
  --cpu-bind=threads \
  --threads-per-core=1 \
  -m block:cyclic \
  python3 -u "$PARAMSIM_ROOT/scripts/run_cluster.py" \
    --comms mpi \
    -e "$RUN_ROOT/case.json"

duration=$SECONDS
hours=$(($duration / 3600))
rem_seconds=$(($duration % 3600))
minutes=$(($rem_seconds / 60))
seconds=$(($rem_seconds % 60))

rc=$?
set -e
echo "end_utc=$(date -u +%FT%TZ)" >>"$RUN_ROOT/run-metadata.txt"
echo Elapsed time = $hours:$minutes:$seconds hours,minutes,seconds >> "$RUN_ROOT/run-metadata.txt"
echo "exit_code=$rc" >>"$RUN_ROOT/run-metadata.txt"
exit "$rc"
