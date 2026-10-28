import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import libensemble as libe
from libensemble.executors import Executor
from libensemble.libE import libE
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE
import libensemble.tools

from ensemble_restart import load_restart_history, prepare_output_directory

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
    batch_size = min(
        int(user_specs["gen_batch_size"]),
        max(0, int(user_specs["num_samples"]) - len(H_in)),
    )

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
    output = np.zeros(len(H_in), dtype=sim_specs["out"])
    executor = libE_info.get("executor", Executor.executor)
    calc_status = WORKER_DONE

    for ibatch, history_row in enumerate(H_in):
        parameter_file = write_case_parameter_file(
            sim_specs["user"]["case_type"],
            sim_specs["user"]["dimension"],
            history_row,
        )
        all_args = (
            sim_specs["user"]["common_args"]
            + get_case_parameter_args(parameter_file)
        )

        start = time.monotonic()
        task = executor.submit(
            app_name="run_fem_case",
            app_args=all_args,
            stdout=f"run_case_{ibatch}.out",
            stderr=f"run_case_{ibatch}.err",
        )
        task.wait()
        elapsed = time.monotonic() - start

        succeeded = bool(getattr(task, "success", False))
        return_code = getattr(task, "errcode", None)
        if return_code is None:
            return_code = 0 if succeeded else -1

        output["success"][ibatch] = int(succeeded)
        output["return_code"][ibatch] = int(return_code)
        output["runtime_sec"][ibatch] = elapsed
        output["hostname"][ibatch] = (
            socket.gethostname().encode("utf-8")[:63]
        )

        if not succeeded:
            calc_status = TASK_FAILED

    return output, persis_info, calc_status

def run_ensemble(case_file_path, restart_history=None):
    """Configure and run or restart a local paramSim ensemble.

    Args:
        case_file_path: JSON file containing ensemble configuration.
        restart_history: Saved history path, or None for a fresh ensemble.

    Returns:
        None.
    """
    # Read params for this ensemble-case
    case_path = Path(case_file_path).expanduser().resolve()
    with case_path.open("r", encoding="utf-8") as case_file:
        case_data = json.load(case_file)
    case_parameter_ranges = load_case_parameter_ranges(
        case_data, case_path
    )

    num_samples = int(case_data["num_samples"])
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    output_dir = Path(
        os.environ.get("PARAMSIM_OUTPUT_DIR")
        or case_data.get("ensemble_dir_path", "ensemble")
    ).expanduser().resolve()

    comms = case_data.get("comms", "local")
    nworkers = case_data.get("nworkers", 1)
    is_manager = True
    #nworkers, is_manager, libE_specs, _ = libe.tools.parse_args()
    libE_specs = dict()
    libE_specs["comms"] = comms
    libE_specs["nworkers"] = nworkers
    # Create and work inside separate per-simulation directories
    libE_specs["sim_dirs_make"] = True
    libe.logger.set_level("INFO")

    exctr = Executor()

    # Register simulation executable with executor
    exctr.register_app(full_path=case_data["simulation_exec_path"], app_name="run_fem_case")

    common_arg_str = get_common_args_str(case_data)

    gen_specs = {
        "gen_f" : gen_random_samples,
        "in" : ["sim_id"],
        "out" : [],
        "user" : {
            "lower" : {},
            "upper" : {},
            "gen_batch_size" : case_data["batch_size"],
            "num_samples" : num_samples,
        },
    }
    sim_specs = {
        "sim_f" : run_case,
        "in" : [],
        "out" : [
            ("success", np.int32),
            ("return_code", np.int32),
            ("runtime_sec", np.float64),
            ("hostname", "S64"),
        ],
        "user" : {
            "case_type" : case_data["case_type"],
            "dimension" : case_data["dimension"],
            "common_args" : common_arg_str,
            "batch_size" : case_data["batch_size"]
        }
    }

    setup_case(case_data, gen_specs, sim_specs, case_parameter_ranges)

    history_path = (
        Path(restart_history).expanduser().resolve()
        if restart_history is not None else None
    )
    history = None
    remaining = num_samples
    if history_path is not None:
        history, remaining = load_restart_history(
            history_path, gen_specs, sim_specs, num_samples
        )

    directory_width = max(4, len(str(num_samples - 1)))
    archive_path = None
    if remaining:
        directory_width, archive_path = prepare_output_directory(
            output_dir, history, directory_width
        )

    retries = (
        int(np.count_nonzero(
            ~history["sim_ended"] & ~history["cancel_requested"]
        ))
        if history is not None else 0
    )
    print(
        f"paramSim/libEnsemble: history={history_path}, "
        f"completed={num_samples - remaining}, retries={retries}, "
        f"new_samples={remaining - retries}, "
        f"random_streams={'fresh' if history_path is not None else 'default'}",
        flush=True,
    )
    if archive_path is not None:
        print(f"Partial outputs archived in {archive_path}", flush=True)
    if remaining == 0:
        print("Requested ensemble is already complete.", flush=True)
        return

    libE_specs.update(
        {
            "ensemble_dir_path": str(output_dir),
            "reuse_output_dir": True,
            "calc_dir_id_width": directory_width,
            "save_every_k_sims": int(case_data.get(
                "checkpoint_every", max(1, min(256, num_samples))
            )),
            "save_H_on_completion": False,
            "save_H_and_persis_on_abort": True,
        }
    )
    exit_criteria = {"sim_max": remaining}

    # Seed random streams for each worker for gen_f
    if history_path is not None:
        persis_info = libe.tools.add_unique_random_streams(
            {}, nworkers + 1, seed=None
        )
    else:
        persis_info = libe.tools.add_unique_random_streams({}, nworkers + 1)

    # Launch libEnsemble
    H, persis_info, flag = libE(
        sim_specs,
        gen_specs,
        exit_criteria,
        persis_info=persis_info,
        libE_specs=libE_specs,
        H0=history,
    )

    if is_manager:
        libe.tools.save_libE_output(H, persis_info, "paramsim", nworkers)
