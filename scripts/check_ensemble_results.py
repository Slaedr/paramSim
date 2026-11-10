#!/usr/bin/env python3
"""Summarize a completed paramSim/libEnsemble run directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


HISTORY_PATTERN = "paramsim_history_*.npy"
REQUIRED_FIELDS = (
    "sim_id",
    "sim_started",
    "sim_ended",
    "success",
    "return_code",
)


def exit_incompatible(message: str) -> None:
    """Report an unusable history and exit with status two."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


def find_newest_history(run_root: Path) -> Path:
    """Return the newest paramSim final-history file."""
    candidates = list(run_root.glob(HISTORY_PATTERN))
    if not candidates:
        exit_incompatible(
            f"No compatible libEnsemble history found under {run_root}"
        )
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def load_history(history_path: Path) -> np.ndarray:
    """Load a history and verify its required durable fields."""
    try:
        history = np.load(history_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        exit_incompatible(
            f"Incompatible history {history_path}: {error}"
        )

    names = history.dtype.names or ()
    missing = [
        field for field in REQUIRED_FIELDS if field not in names
    ]
    if missing:
        exit_incompatible(
            f"Incompatible history {history_path}: missing required "
            f"field(s): {', '.join(missing)}"
        )
    return history


def find_simulation_directory(
    ensemble_dir: Path,
    sim_id: int,
) -> Path:
    """Find a simulation directory regardless of suffix padding."""
    for candidate in sorted(ensemble_dir.glob("sim*")):
        suffix = candidate.name.removeprefix("sim")
        if (
            candidate.is_dir()
            and suffix.isdigit()
            and int(suffix) == sim_id
        ):
            return candidate
    return ensemble_dir / f"sim{sim_id:06d}"


def path_with_status(path: Path) -> str:
    """Format a diagnostic path and identify missing files."""
    suffix = "" if path.exists() else " (missing)"
    return f"{path}{suffix}"


def delete_simulation_directory(
    simulation_dir: Path,
    ensemble_dir: Path,
) -> bool:
    """Delete a failed simulation directory inside the ensemble tree.

    Missing directories and symbolic links are left untouched, as is
    anything that is not a direct child of the ensemble directory.

    Args:
        simulation_dir: Directory of the failed simulation.
        ensemble_dir: Resolved simulation output directory.

    Returns:
        True if the directory was deleted, False otherwise.
    """
    if not simulation_dir.exists() and not simulation_dir.is_symlink():
        print(f"not deleting {simulation_dir}: directory is missing")
        return False
    if simulation_dir.is_symlink() or not simulation_dir.is_dir():
        print(
            f"not deleting {simulation_dir}: not a regular directory",
            file=sys.stderr,
        )
        return False
    if simulation_dir.parent.resolve() != ensemble_dir:
        print(
            f"not deleting {simulation_dir}: outside {ensemble_dir}",
            file=sys.stderr,
        )
        return False
    try:
        shutil.rmtree(simulation_dir)
    except OSError as error:
        print(
            f"failed to delete {simulation_dir}: {error}",
            file=sys.stderr,
        )
        return False
    print(f"deleted {simulation_dir}")
    return True


def renumber_simulation_directories(ensemble_dir: Path) -> bool:
    """Rename simulation directories to consecutive integers from 0.

    Directories named ``sim<digits>`` are ordered by their numeric
    suffix and renamed in ascending order, keeping the widest zero
    padding already in use. Because every new number is no greater
    than the old one, a rename never overwrites a directory that has
    not yet been moved. Symbolic links are left untouched; if one
    occupies a target name, renumbering stops.

    Args:
        ensemble_dir: Resolved simulation output directory.

    Returns:
        True if renumbering completed, False if it stopped early.
    """
    found = []
    for candidate in ensemble_dir.glob("sim*"):
        suffix = candidate.name.removeprefix("sim")
        if (
            suffix.isdigit()
            and candidate.is_dir()
            and not candidate.is_symlink()
        ):
            found.append((int(suffix), len(suffix), candidate))
    if not found:
        return True
    found.sort(key=lambda item: (item[0], item[1]))
    width = max(item[1] for item in found)

    renamed = 0
    for new_id, (_, _, old_path) in enumerate(found):
        new_path = ensemble_dir / f"sim{new_id:0{width}d}"
        if new_path == old_path:
            continue
        if new_path.exists() or new_path.is_symlink():
            print(
                f"not renumbering: {new_path} already exists",
                file=sys.stderr,
            )
            return False
        try:
            old_path.rename(new_path)
        except OSError as error:
            print(
                f"failed to rename {old_path}: {error}",
                file=sys.stderr,
            )
            return False
        print(f"renamed {old_path} -> {new_path}")
        renamed += 1
    print(f"renumbered {renamed} of {len(found)} simulation directories")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--ensemble-dir",
        type=Path,
        default=Path("ensemble"),
        help="simulation output directory, relative to run_root by default",
    )
    parser.add_argument(
        "--delete-failed",
        action="store_true",
        help="delete the simulation directories of failed simulations",
    )
    parser.add_argument(
        "--renumber-dirs",
        action="store_true",
        help=(
            "after deleting failed simulations, renumber the remaining "
            "simulation directories consecutively from 0 "
            "(requires --delete-failed)"
        ),
    )
    args = parser.parse_args()
    if args.renumber_dirs and not args.delete_failed:
        parser.error("--renumber-dirs requires --delete-failed")

    run_root = args.run_root.expanduser().resolve()
    ensemble_dir = args.ensemble_dir.expanduser()
    if not ensemble_dir.is_absolute():
        ensemble_dir = run_root / ensemble_dir
    ensemble_dir = ensemble_dir.resolve()

    history_path = find_newest_history(run_root)
    H = load_history(history_path)
    names = H.dtype.names or ()
    started = H["sim_started"].astype(bool)
    ended = H["sim_ended"].astype(bool)
    success = H["success"].astype(bool)

    ended_without_start = ended & ~started
    if np.any(ended_without_start):
        exit_incompatible(
            f"History {history_path} contains "
            f"{int(np.count_nonzero(ended_without_start))} "
            "simulation(s) marked ended without being started"
        )

    unfinished = started & ~ended
    if np.any(unfinished):
        exit_incompatible(
            f"History {history_path} contains "
            f"{int(np.count_nonzero(unfinished))} "
            "unfinished simulation(s)"
        )

    failed = (~success) & ended

    print(f"history: {history_path}")
    print(f"rows generated: {len(H)}")
    print(f"simulations ended: {int(np.count_nonzero(ended))}")
    print(f"successful: {int(np.count_nonzero(success & ended))}")
    print(f"failed: {int(np.count_nonzero(failed))}")
    if "runtime_sec" in names and np.any(ended):
        runtimes = H["runtime_sec"][ended]
        print(
            "runtime seconds min/median/max: "
            f"{runtimes.min():.3f} / "
            f"{np.median(runtimes):.3f} / "
            f"{runtimes.max():.3f}"
        )

    for row in H[failed]:
        sim_id = int(row["sim_id"])
        simulation_dir = find_simulation_directory(
            ensemble_dir,
            sim_id,
        )
        case_params = simulation_dir / "case_params.txt"
        stderr = simulation_dir / "run_case_0.err"
        print(
            f"simulation {sim_id} failed: "
            f"return_code={int(row['return_code'])}; "
            f"simulation_dir={path_with_status(simulation_dir)}; "
            f"case_params={path_with_status(case_params)}; "
            f"stderr={path_with_status(stderr)}"
        )
        if args.delete_failed:
            delete_simulation_directory(simulation_dir, ensemble_dir)

    if args.renumber_dirs and not renumber_simulation_directories(
        ensemble_dir
    ):
        raise SystemExit(2)
    if np.any(failed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
