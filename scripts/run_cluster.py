#!/usr/bin/env python3
"""Command-line entry point for a paramSim ensemble on a compute cluster."""

import argparse

import ensemble_cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
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
        help="saved libEnsemble history to resume",
    )
    args, _unknown = parser.parse_known_args()
    ensemble_cluster.run_ensemble(
        args.ensemble_params_file, restart_history=args.restart_history
    )
