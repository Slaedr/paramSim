#!/usr/bin/env python3

import argparse

from context import scripts
from scripts import ensemble

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--ensemble_params_file", type=str,
        help="JSON file containing settings for case parameters")
args, unknown = parser.parse_known_args()

ensemble.run_ensemble(args.ensemble_params_file)
