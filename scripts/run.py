#!/usr/bin/env python3
"""Run local or MPI paramSim ensembles.

Examples:
    python scripts/run.py -e case.json
    python scripts/run.py -e case.json --nworkers 4
    srun -n 5 python scripts/run.py -e case.json --comms mpi
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run or restart a local or MPI paramSim ensemble.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Environment variables:
  Nonempty values override the corresponding case JSON settings.

  PARAMSIM_EXEC
    Simulation executable; overrides simulation_exec_path.
    Required if simulation_exec_path is absent.
    Relative paths are resolved from the case JSON directory.

  PARAMSIM_OUTPUT_DIR
    Simulation output directory; overrides ensemble_dir_path.
    Default: ensemble. Relative paths use the launch directory.

  PARAMSIM_NUM_SAMPLES
    Positive integer specifying the total number of simulations;
    overrides num_samples. Required if num_samples is absent.

  PARAMSIM_RANDOM_SEED
    Random seed for fresh runs; overrides random_seed. Default: 0.
    Restarts use fresh random streams.
""",
    )
    parser.add_argument(
        "-e",
        "--ensemble_params_file",
        required=True,
        help="JSON file containing ensemble and parameter-family settings",
    )
    parser.add_argument(
        "--restart-history",
        help="saved libEnsemble history to resume; relative to launch directory",
    )
    parser.add_argument(
        "--comms",
        choices=("local", "mpi"),
        default="local",
        help="manager/worker communication mode (default: local)",
    )
    parser.add_argument(
        "-n",
        "--nworkers",
        type=int,
        help="positive local worker count (default: 1); invalid in MPI mode",
    )
    args = parser.parse_args()
    if args.comms == "mpi" and args.nworkers is not None:
        parser.error("--nworkers is only valid with --comms local")
    if args.nworkers is not None and args.nworkers < 1:
        parser.error("--nworkers must be positive")

    import ensemble

    ensemble.run_ensemble(
        args.ensemble_params_file,
        restart_history=args.restart_history,
        comms=args.comms,
        nworkers=args.nworkers,
    )
