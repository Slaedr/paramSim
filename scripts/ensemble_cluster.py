#!/usr/bin/env python3
"""Cluster-oriented libEnsemble driver for paramSim.

Design:
  * libEnsemble manager/worker communication uses MPI (one srun total).
  * Every libEnsemble worker is pinned to one CPU core.
  * The base Executor starts one local, serial run_case subprocess per worker.
  * libEnsemble creates an isolated directory for each simulation.

This file is intended to live beside paramSim's existing setup_case.py.
"""

#from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import libensemble as libe
import numpy as np
from libensemble.executors import Executor
from libensemble.libE import libE
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE
from libensemble.tools import add_unique_random_streams, save_libE_output

from setup_case import get_args_str, get_common_args_str, setup_case


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _env_or_case(env_name: str, case_data: dict[str, Any], key: str, default: Any = None) -> Any:
    value = os.environ.get(env_name)
    if value is not None and value != "":
        return value
    return case_data.get(key, default)


def gen_random_samples(H_in, persis_info, gen_specs):
    """Generate a batch of uniformly distributed parameter samples."""
    del H_in
    user_specs = gen_specs["user"]
    batch_size = int(user_specs["gen_batch_size"])
    out = np.zeros(batch_size, dtype=gen_specs["out"])

    for param_spec in gen_specs["out"]:
        param_name = param_spec[0]
        lower = user_specs["lower"][param_name]
        upper = user_specs["upper"][param_name]
        if lower.shape != upper.shape:
            raise ValueError(f"Mismatched bounds for {param_name}: {lower.shape} != {upper.shape}")
        out[param_name] = persis_info["rand_stream"].uniform(
            lower,
            upper,
            (batch_size,) + lower.shape,
        )

    return out, persis_info


def run_case(H_in, persis_info, sim_specs, libE_info):
    """Launch one serial paramSim executable for every received history row."""
    output = np.zeros(len(H_in), dtype=sim_specs["out"])
    executor = libE_info.get("executor", Executor.executor)
    calc_status = WORKER_DONE

    for ibatch, history_row in enumerate(H_in):
        case_args = get_args_str(sim_specs["user"]["case_type"], history_row)
        app_args = sim_specs["user"]["common_args"] + case_args

        start = time.monotonic()
        task = executor.submit(
            app_name="run_fem_case",
            app_args=app_args,
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
        output["hostname"][ibatch] = socket.gethostname().encode("utf-8")[:63]

        if not succeeded:
            calc_status = TASK_FAILED

    return output, persis_info, calc_status


def run_ensemble(case_file_path: str) -> None:
    """Configure and run a paramSim ensemble from a JSON case file."""
    nworkers, is_manager, libE_specs, _ = libe.tools.parse_args()
    if nworkers < 1:
        raise RuntimeError("At least one libEnsemble worker is required")

    libe.logger.set_level("INFO")

    case_path = Path(case_file_path).expanduser().resolve()
    with case_path.open("r", encoding="utf-8") as case_file:
        case_data = json.load(case_file)

    executable_value = _env_or_case(
        "PARAMSIM_EXEC",
        case_data,
        "simulation_exec_path",
    )
    if not executable_value:
        raise ValueError("Set simulation_exec_path in the case JSON or PARAMSIM_EXEC")
    executable = _resolve_path(str(executable_value), case_path.parent)
    if not executable.is_file():
        raise FileNotFoundError(f"paramSim executable not found: {executable}")
    if not os.access(executable, os.X_OK):
        raise PermissionError(f"paramSim executable is not executable: {executable}")

    output_value = _env_or_case(
        "PARAMSIM_OUTPUT_DIR",
        case_data,
        "ensemble_dir_path",
        "ensemble",
    )
    output_dir = _resolve_path(str(output_value), Path.cwd())
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    num_samples = int(
        _env_or_case("PARAMSIM_NUM_SAMPLES", case_data, "num_samples")
    )
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    configured_batch = int(case_data.get("batch_size", 0))
    generation_batch = configured_batch if configured_batch > 0 else \
            min(num_samples, max(64, 4 * nworkers))

    random_seed = int(
        _env_or_case("PARAMSIM_RANDOM_SEED", case_data, "random_seed", 0)
    )
    checkpoint_every = int(case_data.get("checkpoint_every", max(1, min(256, num_samples))))

    executor = Executor()
    executor.register_app(full_path=str(executable), app_name="run_fem_case")

    common_args = get_common_args_str(case_data)

    gen_specs = {
        "gen_f": gen_random_samples,
        "out": [],
        "user": {
            "lower": {},
            "upper": {},
            "gen_batch_size": generation_batch,
        },
    }

    sim_specs = {
        "sim_f": run_case,
        "in": [],
        "out": [
            ("success", np.int32),
            ("return_code", np.int32),
            ("runtime_sec", np.float64),
            ("hostname", "S64"),
        ],
        "user": {
            "common_args": common_args,
            "case_type": case_data["case_type"],
        },
    }

    # Reuse the case-family definitions already present in paramSim.
    setup_case(case_data, gen_specs, sim_specs)

    # Each MPI worker is already co-located with and bound to the CPU core on
    # which its serial subprocess should execute. No nested srun is required.
    libE_specs.update(
        {
            "sim_dirs_make": True,
            "ensemble_dir_path": str(output_dir),
            "reuse_output_dir": False,
            "calc_dir_id_width": max(6, len(str(num_samples - 1))),
            "save_every_k_sims": checkpoint_every,
            "save_H_on_completion": False,
            "disable_resource_manager": True,
            "gen_on_manager": True,
        }
    )

    exit_criteria = {"sim_max": num_samples}
    persis_info = add_unique_random_streams({}, nworkers + 1, seed=random_seed)

    H, persis_info, flag = libE(
        sim_specs,
        gen_specs,
        exit_criteria,
        persis_info,
        libE_specs=libE_specs,
    )

    if is_manager:
        # This writes NumPy history and persistent-state files in the job's
        # current working directory, alongside ensemble.log/libE_stats.txt.
        save_libE_output(H, persis_info, "paramsim", nworkers)
        completed = int(np.count_nonzero(H["sim_ended"])) if "sim_ended" in H.dtype.names else len(H)
        successful = int(np.count_nonzero(H["success"])) if "success" in H.dtype.names else -1
        print(
            f"paramSim/libEnsemble finished: flag={flag}, generated={len(H)}, "
            f"completed={completed}, successful={successful}, output={output_dir}",
            flush=True,
        )
