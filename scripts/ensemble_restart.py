"""Shared history preparation for restarting paramSim ensembles."""

import tempfile
from pathlib import Path

import numpy as np


def load_restart_history(history_path, gen_specs, sim_specs, num_samples):
    """Load and prepare saved history for another libEnsemble invocation.

    Args:
        history_path: Saved history file, relative to the launch directory.
        gen_specs: Generator specification defining parameter fields.
        sim_specs: Simulation specification defining result fields.
        num_samples: Requested total number of ended simulations.

    Returns:
        Prepared history and the number of additional evaluations required.

    Raises:
        OSError: If the history file cannot be read.
        ValueError: If the history is incompatible or the target is invalid.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    path = Path(history_path).expanduser().resolve()
    history = np.load(path, allow_pickle=False)
    if (
        not isinstance(history, np.ndarray)
        or history.ndim != 1
        or history.dtype.names is None
    ):
        raise ValueError(f"History must be a structured 1D array: {path}")

    names = history.dtype.names
    expected = np.dtype(
        gen_specs["out"]
        + sim_specs["out"]
        + [
            ("sim_started", np.bool_),
            ("sim_ended", np.bool_),
            ("cancel_requested", np.bool_),
        ]
    )
    missing = set(expected.names).union({"sim_id"}).difference(names)
    if missing:
        raise ValueError(
            f"History {path} is missing fields: {', '.join(sorted(missing))}"
        )
    for field in expected.names:
        if history.dtype[field] != expected[field]:
            raise ValueError(
                f"Incompatible history field {field}: "
                f"{history.dtype[field]} != {expected[field]}"
            )

    if (
        history.dtype["sim_id"].kind not in "iu"
        or not np.array_equal(history["sim_id"], np.arange(len(history)))
    ):
        raise ValueError(
            "History sim_id values must match consecutive rows starting at zero"
        )
    if np.any(history["sim_ended"] & ~history["sim_started"]):
        raise ValueError("History contains simulations ended without starting")

    completed = int(np.count_nonzero(history["sim_ended"]))
    remaining = num_samples - completed
    if remaining < 0:
        raise ValueError(
            f"num_samples={num_samples} is below the completed count {completed}"
        )

    unfinished = ~history["sim_ended"]
    history["sim_started"][unfinished] = False
    history["cancel_requested"][unfinished] = True
    retry_rows = np.flatnonzero(unfinished)[:remaining]
    history["cancel_requested"][retry_rows] = False

    reset_values = {
        "sim_worker": 0,
        "sim_started_time": np.inf,
        "sim_ended_time": 0.0,
        "gen_informed": False,
        "gen_informed_time": np.inf,
        "kill_sent": False,
    }
    for field, value in reset_values.items():
        if field in names:
            history[field][unfinished] = value
    for field in np.dtype(sim_specs["out"]).names:
        history[field][unfinished] = np.zeros(
            (), dtype=history.dtype[field]
        )

    return history, remaining


def prepare_output_directory(output_dir, history, default_width):
    """Protect completed outputs and archive partial simulation directories.

    Args:
        output_dir: Directory containing libEnsemble simulation directories.
        history: Prepared restart history, or None for a fresh ensemble.
        default_width: Simulation-number padding when no directories exist.

    Returns:
        Simulation-number padding and the archive path, or None if unused.

    Raises:
        FileExistsError: If a fresh run would reuse simulation directories.
        ValueError: If existing directory IDs or padding are ambiguous.
        OSError: If directories cannot be inspected or archived.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Not an ensemble directory: {output_dir}")

    directories = {}
    for directory in sorted(output_dir.glob("sim*")):
        suffix = directory.name.removeprefix("sim")
        if not directory.is_dir() or not suffix.isdigit():
            continue
        sim_id = int(suffix)
        if sim_id in directories:
            raise ValueError(f"Multiple directories for simulation {sim_id}")
        directories[sim_id] = directory

    if history is None:
        if directories:
            raise FileExistsError(
                f"Simulation directories already exist in {output_dir}; "
                "use --restart-history to resume"
            )
        return default_width, None

    width = min(
        (len(path.name.removeprefix("sim")) for path in directories.values()),
        default=default_width,
    )
    for sim_id, directory in directories.items():
        if directory.name != f"sim{sim_id:0{width}d}":
            raise ValueError("Simulation directories use inconsistent padding")

    partial_directories = [
        directory
        for sim_id, directory in directories.items()
        if sim_id >= len(history) or not history["sim_ended"][sim_id]
    ]
    if not partial_directories:
        return width, None

    archive_root = output_dir / "restart_archive"
    archive_root.mkdir(exist_ok=True)
    archive_path = Path(
        tempfile.mkdtemp(prefix="attempt-", dir=archive_root)
    )
    for directory in partial_directories:
        directory.rename(archive_path / directory.name)

    return width, archive_path
