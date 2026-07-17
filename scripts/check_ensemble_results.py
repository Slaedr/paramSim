#!/usr/bin/env python3
"""Summarize a completed paramSim/libEnsemble run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    candidates = sorted(args.run_root.glob("paramsim_history_*.npy"))
    if not candidates:
        candidates = sorted(args.run_root.glob("*History*.npy"))
    if not candidates:
        raise SystemExit(f"No libEnsemble history .npy found under {args.run_root}")

    history_path = candidates[-1]
    H = np.load(history_path, allow_pickle=False)
    names = H.dtype.names or ()
    ended = H["sim_ended"] if "sim_ended" in names else np.ones(len(H), dtype=bool)
    success = H["success"] if "success" in names else np.zeros(len(H), dtype=bool)

    print(f"history: {history_path}")
    print(f"rows generated: {len(H)}")
    print(f"simulations ended: {int(np.count_nonzero(ended))}")
    print(f"successful: {int(np.count_nonzero(success & ended))}")
    print(f"failed: {int(np.count_nonzero((~success.astype(bool)) & ended))}")
    if "runtime_sec" in names and np.any(ended):
        runtimes = H["runtime_sec"][ended]
        print(f"runtime seconds min/median/max: {runtimes.min():.3f} / {np.median(runtimes):.3f} / {runtimes.max():.3f}")


if __name__ == "__main__":
    main()
