#!/usr/bin/env python3
"""Command-line entry point for a paramSim ensemble on a compute cluster."""

import argparse

import ensemble_cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--ensemble_params_file",
        required=True,
        help="JSON file containing ensemble and parameter-family settings",
    )
    args, _unknown = parser.parse_known_args()
    ensemble_cluster.run_ensemble(args.ensemble_params_file)
