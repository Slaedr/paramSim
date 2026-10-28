#!/usr/bin/env python3

import argparse

#from context import scripts
import ensemble

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Environment variables:
  PARAMSIM_OUTPUT_DIR
    Simulation output directory. A nonempty value overrides
    ensemble_dir_path in the case JSON. Default: ensemble.
    Relative paths are resolved from the launch directory.
""",
)
parser.add_argument("-e", "--ensemble_params_file", required=True,
        help="JSON file containing settings for case parameters")
parser.add_argument(
    "--restart-history",
    help="saved libEnsemble history to resume",
)

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    ensemble.run_ensemble(
        args.ensemble_params_file, restart_history=args.restart_history
    )
