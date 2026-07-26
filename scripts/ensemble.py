import json
from pathlib import Path

import numpy as np
import libensemble as libe
from libensemble.executors.executor import Executor
from libensemble.executors import Executor
from libensemble.libE import libE
import libensemble.tools

from setup_case import (
    get_case_parameter_args,
    get_common_args_str,
    load_case_parameter_ranges,
    sample_parameter_values,
    setup_case,
    write_case_parameter_file,
)

def gen_random_samples(H_in, persis_info, gen_specs):

    user_specs = gen_specs["user"]
    batch_size = user_specs["gen_batch_size"]

    out = np.zeros(batch_size, dtype=gen_specs["out"])

    # Iterate over types of parameters to generate
    for paramset in gen_specs["out"]:
        paramname = paramset[0]
        lower = user_specs["lower"][paramname]
        upper = user_specs["upper"][paramname]
        integer = paramname in user_specs.get("integer_parameters", ())
        out[paramname] = sample_parameter_values(
            persis_info["rand_stream"],
            lower,
            upper,
            batch_size,
            integer=integer,
        )

    # Send back our output and persis_info
    return out, persis_info

def run_case(H_in, persis_info, sim_specs, libE_info):
    for ibatch in range(len(H_in)):
        parameter_file = write_case_parameter_file(
            sim_specs["user"]["case_type"],
            sim_specs["user"]["dimension"],
            H_in[ibatch],
        )
        all_args = (
            sim_specs["user"]["common_args"]
            + get_case_parameter_args(parameter_file)
        )

        # Submit our app for execution
        exctr = Executor.executor
        task = exctr.submit(app_name="run_fem_case", app_args=all_args)
        # Block until the task finishes
        task.wait()
    return 0

def run_ensemble(case_file_path):

    nworkers, is_manager, libE_specs, _ = libe.tools.parse_args()
    libe.logger.set_level("INFO")
    exctr = Executor()

    # Read params for this ensemble-case
    case_path = Path(case_file_path).expanduser().resolve()
    with case_path.open("r", encoding="utf-8") as case_file:
        case_data = json.load(case_file)
    case_parameter_ranges = load_case_parameter_ranges(
        case_data, case_path
    )

    # Register simulation executable with executor
    exctr.register_app(full_path=case_data["simulation_exec_path"], app_name="run_fem_case")

    common_arg_str = get_common_args_str(case_data)

    gen_specs = {
        "gen_f" : gen_random_samples,
        "out" : [],
        "user" : {
            "lower" : {},
            "upper" : {},
            "gen_batch_size" : case_data["batch_size"],
        },
    }
    sim_specs = {
        "sim_f" : run_case,
        "in" : [],
        "out" : [("success", int)],
        "user" : {
            "case_type" : case_data["case_type"],
            "dimension" : case_data["dimension"],
            "common_args" : common_arg_str,
            "batch_size" : case_data["batch_size"]
        }
    }

    setup_case(case_data, gen_specs, sim_specs, case_parameter_ranges)

    # Create and work inside separate per-simulation directories
    libE_specs["sim_dirs_make"] = True

    exit_criteria = {"sim_max": case_data["num_samples"]}

    # Seed random streams for each worker for gen_f
    persis_info = libe.tools.add_unique_random_streams({}, nworkers + 1)

    # Launch libEnsemble
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info, \
                                libE_specs=libE_specs)

    if is_manager:
        libe.tools.save_libE_output(H, persis_info, __file__, nworkers)
