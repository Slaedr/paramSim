#!/usr/bin/env python3

import argparse

from context import scripts
from scripts import ensemble

parser = argparse.ArgumentParser()
parser.add_argument("case_params", type=str,
        help="JSON file containing settings for case parameters")
args, unknown = parser.parse_known_args()

ensemble.run_ensemble(args.case_params)
