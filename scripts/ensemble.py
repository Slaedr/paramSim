import json

import numpy as np
import libensemble as libe
from libensemble.executors.executor import Executor
from libensemble.executors import Executor
from libensemble.libE import libE
import libensemble.tools

from setup_case import setup_case, get_args_str

def gen_random_samples(H_in, persis_info, gen_specs):

    user_specs = gen_specs["user"]
    batch_size = user_specs["gen_batch_size"]

    out = np.zeros(batch_size, dtype=gen_specs["out"])

    # Iterate over types of parameters to generate
    for paramset in gen_specs["out"]:
        print("Generating random values for parameter set " + str(paramset))
        paramname = paramset[0]
        paramtype = paramset[1]
        paramsizes = paramset[2]
        lower = user_specs["lower"][paramname]
        upper = user_specs["upper"][paramname]
        assert(lower.shape == upper.shape)

        # Set the "x" output field to contain random numbers, using random stream
        out[paramname] = persis_info["rand_stream"].uniform(lower, upper,
                                                            (batch_size,) + lower.shape)

    # Send back our output and persis_info
    return out, persis_info

def run_case(H_in, persis_info, sim_specs, libE_info):
    for ibatch in range(len(H_in)):
        case_argstr = get_args_str(sim_specs["user"]["case_type"], H_in[ibatch])
        all_args = sim_specs["user"]["common_args"] + case_argstr

        # Submit our app for execution
        exctr = Executor.executor
        task = exctr.submit(app_name="run_fem_case", app_args=all_args)
        # Block until the task finishes
        task.wait()
    return 0

def run_ensemble(case_file_path):

    nworkers, is_manager, libE_specs, _ = libe.tools.parse_args()
    libe.logger.set_level("DEBUG")
    exctr = Executor()

    # Read params for this ensemble-case
    with open(case_file_path, 'r') as f:
        casefilecontents = f.read()
    case_data = json.loads(casefilecontents)

    # Register simulation executable with executor
    exctr.register_app(full_path=case_data["simulation_exec_path"], app_name="run_fem_case")

    common_arg_str = "--solver " + case_data["solver_type"] + " --case " + case_data["case_type"] \
        + " --refine_levels 1 --initial_resolution " + str(case_data["resolution"])

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
            "common_args" : common_arg_str,
            "batch_size" : case_data["batch_size"]
        }
    }

    setup_case(case_data, gen_specs, sim_specs)

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

