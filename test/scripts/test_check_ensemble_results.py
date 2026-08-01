import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPOSITORY_ROOT / "scripts" / "check_ensemble_results.py"
REQUIRED_DTYPE = [
    ("sim_id", np.int64),
    ("sim_started", np.bool_),
    ("sim_ended", np.bool_),
    ("success", np.int32),
    ("return_code", np.int32),
    ("runtime_sec", np.float64),
]


def save_history(path, rows, dtype=REQUIRED_DTYPE):
    history = np.zeros(len(rows), dtype=dtype)
    for index, row in enumerate(rows):
        if (
            "sim_started" in history.dtype.names
            and "sim_started" not in row
        ):
            history["sim_started"][index] = row.get("sim_ended", False)
        for field, value in row.items():
            history[field][index] = value
    np.save(path, history)


def run_checker(run_root, *arguments):
    return subprocess.run(
        [sys.executable, str(CHECKER), str(run_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


class CheckEnsembleResultsTests(unittest.TestCase):
    def test_all_successes_use_newest_supported_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            old_history = root / "paramsim_history_old.npy"
            new_history = root / "paramsim_history_new.npy"
            save_history(
                old_history,
                [{
                    "sim_id": 0,
                    "sim_ended": True,
                    "success": 0,
                    "return_code": 9,
                }],
            )
            save_history(
                new_history,
                [{
                    "sim_id": 0,
                    "sim_ended": True,
                    "success": 1,
                    "return_code": 0,
                    "runtime_sec": 1.25,
                }],
            )
            os.utime(old_history, (1, 1))
            os.utime(new_history, (2, 2))

            result = run_checker(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"history: {new_history}", result.stdout)
            self.assertIn("rows generated: 1", result.stdout)
            self.assertIn("simulations ended: 1", result.stdout)
            self.assertIn("successful: 1", result.stdout)
            self.assertIn("failed: 0", result.stdout)

    def test_lists_each_failure_and_resolves_directory_suffixes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            save_history(
                root / "paramsim_history_failures.npy",
                [
                    {
                        "sim_id": 7,
                        "sim_ended": True,
                        "success": 0,
                        "return_code": 3,
                    },
                    {
                        "sim_id": 42,
                        "sim_ended": True,
                        "success": 0,
                        "return_code": 17,
                    },
                    {
                        "sim_id": 105,
                        "sim_ended": True,
                        "success": 0,
                        "return_code": 23,
                    },
                ],
            )
            ensemble_dir = root / "results"
            sim_7 = ensemble_dir / "sim0007"
            sim_42 = ensemble_dir / "sim000042"
            sim_7.mkdir(parents=True)
            sim_42.mkdir()
            (sim_7 / "case_params.txt").write_text(
                "parameters\n", encoding="utf-8"
            )
            (sim_7 / "run_case_0.err").write_text(
                "solver failed\n", encoding="utf-8"
            )

            result = run_checker(
                root, "--ensemble-dir", "results"
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("failed: 3", result.stdout)
            self.assertIn(
                "simulation 7 failed: return_code=3; "
                f"simulation_dir={sim_7}; "
                f"case_params={sim_7 / 'case_params.txt'}; "
                f"stderr={sim_7 / 'run_case_0.err'}",
                result.stdout,
            )
            self.assertIn(
                "simulation 42 failed: return_code=17; "
                f"simulation_dir={sim_42}; "
                f"case_params={sim_42 / 'case_params.txt'} (missing); "
                f"stderr={sim_42 / 'run_case_0.err'} (missing)",
                result.stdout,
            )
            missing_sim = ensemble_dir / "sim000105"
            self.assertIn(
                "simulation 105 failed: return_code=23; "
                f"simulation_dir={missing_sim} (missing); "
                f"case_params={missing_sim / 'case_params.txt'} "
                "(missing); "
                f"stderr={missing_sim / 'run_case_0.err'} (missing)",
                result.stdout,
            )

    def test_unstarted_surplus_rows_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            save_history(
                root / "paramsim_history_surplus.npy",
                [
                    {
                        "sim_id": 0,
                        "sim_started": True,
                        "sim_ended": True,
                        "success": 1,
                        "return_code": 0,
                    },
                    {
                        "sim_id": 1,
                        "sim_started": False,
                        "sim_ended": False,
                        "success": 0,
                        "return_code": 0,
                    },
                ],
            )

            result = run_checker(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("rows generated: 2", result.stdout)
            self.assertIn("simulations ended: 1", result.stdout)

    def test_unfinished_history_returns_status_two(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            save_history(
                root / "paramsim_history_unfinished.npy",
                [{
                    "sim_id": 5,
                    "sim_started": True,
                    "sim_ended": False,
                    "success": 0,
                    "return_code": 0,
                }],
            )

            result = run_checker(root)

            self.assertEqual(result.returncode, 2)
            self.assertIn("unfinished", result.stderr.lower())

    def test_missing_required_fields_return_status_two(self):
        required_fields = (
            "sim_id",
            "sim_started",
            "sim_ended",
            "success",
            "return_code",
        )
        for missing_field in required_fields:
            with self.subTest(missing_field=missing_field):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    dtype = [
                        field
                        for field in REQUIRED_DTYPE
                        if field[0] != missing_field
                    ]
                    save_history(
                        root / "paramsim_history_incompatible.npy",
                        [{}],
                        dtype=dtype,
                    )

                    result = run_checker(root)

                    self.assertEqual(result.returncode, 2)
                    self.assertIn(missing_field, result.stderr)

    def test_absent_history_returns_status_two(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_checker(Path(temporary_directory))

            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "No compatible libEnsemble history",
                result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
