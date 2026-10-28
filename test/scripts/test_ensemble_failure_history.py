"""End-to-end test: a failing run_case is recorded as a failure in the
libEnsemble history saved by scripts/ensemble.py.

Runs the real production entry point, scripts/run.py, as a subprocess against
a small (--comms local --nworkers 1) ensemble, so libEnsemble genuinely owns
process start-up, working directories and logging the way it does in normal
use. Most tests substitute a stub "run_case" executable that mimics the C++
binary's failure contract -- a "Exception on processing:" banner on stderr
and a nonzero exit code (see run_case.cpp) -- which keeps the test fast and
independent of the C++ build. test_real_run_case_exception_is_recorded_as_failure
repeats the key assertion against the actual built binary, whose path is
supplied by test/CMakeLists.txt via the PARAMSIM_EXEC environment
variable (using the $<TARGET_FILE:run_case> generator expression), and is
skipped when that variable is not set.
"""

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import libensemble  # noqa: F401
except ImportError:
    # No libEnsemble in this Python environment: nothing here is runnable.
    # Matches SKIP_RETURN_CODE in test/CMakeLists.txt.
    sys.exit(77)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_PY = REPOSITORY_ROOT / "scripts" / "run.py"
RUN_CLUSTER_PY = REPOSITORY_ROOT / "scripts" / "run_cluster.py"
CHECKER = REPOSITORY_ROOT / "scripts" / "check_ensemble_results.py"
REAL_RUN_CASE = os.environ.get("PARAMSIM_EXEC")


STUB_SOURCE = '''#!{python}
"""Stub standing in for run_case: mimics the real binary's failure
contract (an exception banner on stderr and a nonzero exit code) for a
configurable set of sim_ids, and otherwise "succeeds"."""
import os
import sys
from pathlib import Path

FAIL_IDS = {{
    int(token)
    for token in os.environ.get("PARAMSIM_STUB_FAIL_IDS", "").split(",")
    if token
}}

# libEnsemble runs each simulation inside <ensemble_dir>/simNNNN, where NNNN
# is the sim_id -- use that to decide, deterministically, whether this
# invocation should fail.
digits = "".join(char for char in Path.cwd().name if char.isdigit())
simulation_id = int(digits) if digits else -1

Path("stub_invocation.txt").write_text(
    str(simulation_id) + " " + " ".join(sys.argv[1:]) + "\\n",
    encoding="utf-8",
)

print("Solving")
if simulation_id in FAIL_IDS:
    print(
        "\\n----------------------------------------------------\\n"
        "Exception on processing: \\nNon-existent case!\\nAborting!\\n"
        "----------------------------------------------------",
        file=sys.stderr,
    )
    sys.exit(1)

Path("field-0.vtk").write_text("stub solution\\n", encoding="utf-8")
sys.exit(0)
'''


def write_stub(root):
    """Write and chmod +x a fake run_case executable, return its path."""
    stub_path = root / "stub_run_case.py"
    stub_path.write_text(
        STUB_SOURCE.format(python=sys.executable), encoding="utf-8"
    )
    stub_path.chmod(stub_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return stub_path


def write_case_files(root, executable, num_samples=5):
    """Write a case JSON + ranges JSON modeled on scripts/examples, return
    the case JSON path."""
    ranges_path = root / "case_ranges.json"
    ranges_path.write_text(
        json.dumps(
            {
                "num_centers_range": [1, 3],
                "coordinate_bounds": [-1.0, 1.0],
                "coeff_bounds": [0.1, 0.9],
                "width_bounds": [0.2, 0.7],
            }
        ),
        encoding="utf-8",
    )
    case_path = root / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "num_samples": num_samples,
                "batch_size": 1,
                "simulation_exec_path": str(executable),
                "dimension": 2,
                "resolution": 4,
                "refine_levels": 1,
                "pde": "poisson_cg",
                "case_type": "cube_exponential",
                "case_params_ranges_file": "case_ranges.json",
            }
        ),
        encoding="utf-8",
    )
    return case_path


def run_ensemble_in(
    root, case_file, fail_ids=(), *, entrypoint=RUN_PY,
    restart_history=None, nworkers=1, output_dir=None,
):
    """Launch an ensemble subprocess with isolated paramSim overrides.

    Args:
        root: Launch directory.
        case_file: Ensemble configuration file.
        fail_ids: Simulation IDs for which the stub should fail.
        entrypoint: Local or cluster launch script.
        restart_history: Optional history path passed to the launch script.
        nworkers: Worker count passed on the command line.
        output_dir: Simulation directory, defaulting to root/ensemble.

    Returns:
        Completed subprocess with captured output and exit status.
    """
    environment = dict(os.environ)
    for key in (
        "PARAMSIM_EXEC", "PARAMSIM_NUM_SAMPLES", "PARAMSIM_RANDOM_SEED",
    ):
        environment.pop(key, None)
    environment["PARAMSIM_OUTPUT_DIR"] = str(
        output_dir if output_dir is not None else root / "ensemble"
    )
    environment["PARAMSIM_STUB_FAIL_IDS"] = ",".join(
        str(sim_id) for sim_id in fail_ids
    )
    command = [
        sys.executable,
        str(entrypoint),
        "-e",
        str(case_file),
        "--comms",
        "local",
        "--nworkers",
        str(nworkers),
    ]
    if restart_history is not None:
        command.extend(["--restart-history", str(restart_history)])
    return subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def find_history(root):
    return list(root.glob("paramsim_history_*.npy"))


def run_checker(root):
    return subprocess.run(
        [sys.executable, str(CHECKER), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


class StubDrivenEnsembleTests(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tempdir.name)
        self.stub = write_stub(self.root)

    def tearDown(self):
        self._tempdir.cleanup()

    def run_and_load_history(self, fail_ids, num_samples=5):
        case_file = write_case_files(self.root, self.stub, num_samples)
        result = run_ensemble_in(self.root, case_file, fail_ids)
        histories = find_history(self.root)
        self.assertEqual(
            len(histories),
            1,
            "expected exactly one history file; "
            f"found {histories}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        history_path = histories[0]
        self.assertIn(
            f"evals={num_samples}_workers=1",
            history_path.name,
            f"unexpected history filename {history_path.name}",
        )
        history = np.load(history_path, allow_pickle=False)
        self.assertEqual(
            len(history),
            num_samples,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return history, result

    def test_history_marks_stub_exceptions_as_failed(self):
        fail_ids = (1, 3)
        history, result = self.run_and_load_history(fail_ids)

        self.assertTrue(
            history["sim_started"].all(),
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertTrue(history["sim_ended"].all())

        success = history["success"].astype(bool)
        failed = ~success
        failed_ids = sorted(int(i) for i in history["sim_id"][failed])
        self.assertEqual(failed_ids, list(fail_ids))

        self.assertEqual(
            sorted(int(rc) for rc in history["return_code"][failed]),
            [1] * len(fail_ids),
        )
        self.assertTrue((history["return_code"][success] == 0).all())
        self.assertTrue((history["runtime_sec"] > 0.0).all())
        self.assertTrue(all(history["hostname"]))

        ensemble_dir = self.root / "ensemble"
        failed_sim_dir = ensemble_dir / "sim0001"
        stderr_file = failed_sim_dir / "run_case_0.err"
        self.assertTrue(
            stderr_file.exists(), f"missing {stderr_file}"
        )
        self.assertIn(
            "Exception on processing:", stderr_file.read_text(encoding="utf-8")
        )
        self.assertTrue((failed_sim_dir / "case_params.txt").exists())

        invocation = (failed_sim_dir / "stub_invocation.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("--case_params_file case_params.txt", invocation)
        self.assertIn("--dimension 2", invocation)

    def test_all_successes_produce_a_clean_history(self):
        history, result = self.run_and_load_history(fail_ids=())

        self.assertTrue(
            (history["success"] == 1).all(),
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertTrue((history["return_code"] == 0).all())

        checker_result = run_checker(self.root)
        self.assertEqual(checker_result.returncode, 0, checker_result.stderr)
        self.assertIn("failed: 0", checker_result.stdout)

    def test_results_checker_reports_the_recorded_failures(self):
        self.run_and_load_history(fail_ids=(1, 3))

        result = run_checker(self.root)

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("failed: 2", result.stdout)
        self.assertIn(
            "simulation 1 failed: return_code=1", result.stdout
        )
        stderr_path = self.root / "ensemble" / "sim0001" / "run_case_0.err"
        self.assertIn(f"stderr={stderr_path}", result.stdout)
        self.assertNotIn(f"stderr={stderr_path} (missing)", result.stdout)


class RestartEnsembleTests(unittest.TestCase):
    def test_surplus_rows_and_invalid_restart_requests(self):
        """Limit queued retries and protect outputs from invalid launches."""
        for entrypoint in (RUN_PY, RUN_CLUSTER_PY):
            with (
                self.subTest(entrypoint=entrypoint.name),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                root = Path(temporary_directory)
                case_file = write_case_files(
                    root, write_stub(root), num_samples=5
                )
                initial = run_ensemble_in(
                    root, case_file, entrypoint=entrypoint
                )
                self.assertEqual(
                    initial.returncode, 0, initial.stdout + initial.stderr
                )
                original = np.load(find_history(root)[0], allow_pickle=False)
                saved = original.copy()
                saved["sim_ended"][2:] = False
                np.save(root / "restart.npy", saved, allow_pickle=False)
                np.save(
                    root / "bad.npy",
                    np.zeros(1, dtype=[("sim_id", np.int64)]),
                    allow_pickle=False,
                )
                case_data = json.loads(case_file.read_text(encoding="utf-8"))
                case_data.update(num_samples=3, batch_size=8)
                case_file.write_text(json.dumps(case_data), encoding="utf-8")
                output_dir = root / "ensemble"
                invocations = {
                    path: path.stat().st_mtime_ns
                    for path in output_dir.glob("sim*/stub_invocation.txt")
                }

                for history_path, message in (
                    (None, "--restart-history"),
                    (Path("missing.npy"), "No such file"),
                    (Path("bad.npy"), "missing fields"),
                ):
                    with self.subTest(restart_history=history_path):
                        rejected = run_ensemble_in(
                            root, case_file, entrypoint=entrypoint,
                            restart_history=history_path,
                        )
                        self.assertNotEqual(rejected.returncode, 0)
                        self.assertIn(message, rejected.stderr)
                        self.assertEqual(
                            {
                                path: path.stat().st_mtime_ns
                                for path in output_dir.glob(
                                    "sim*/stub_invocation.txt"
                                )
                            },
                            invocations,
                        )
                        self.assertFalse(
                            (output_dir / "restart_archive").exists()
                        )

                resumed = run_ensemble_in(
                    root, case_file, entrypoint=entrypoint,
                    restart_history=Path("restart.npy"),
                )
                self.assertEqual(
                    resumed.returncode, 0, resumed.stdout + resumed.stderr
                )
                self.assertIn(
                    "completed=2, retries=1, new_samples=0", resumed.stdout
                )
                latest_path = max(
                    find_history(root), key=lambda path: path.stat().st_mtime_ns
                )
                limited = np.load(latest_path, allow_pickle=False)
                self.assertEqual(len(limited), 5)
                np.testing.assert_array_equal(
                    limited["sim_ended"], [True, True, True, False, False]
                )
                self.assertTrue(limited["cancel_requested"][3:].all())
                self.assertFalse(limited["sim_started"][3:].any())

                case_data["num_samples"] = 6
                case_file.write_text(json.dumps(case_data), encoding="utf-8")
                extended = run_ensemble_in(
                    root, case_file, entrypoint=entrypoint,
                    restart_history=latest_path,
                )
                self.assertEqual(
                    extended.returncode, 0, extended.stdout + extended.stderr
                )
                self.assertIn(
                    "completed=3, retries=2, new_samples=1", extended.stdout
                )
                final_path = max(
                    find_history(root), key=lambda path: path.stat().st_mtime_ns
                )
                finished = np.load(final_path, allow_pickle=False)
                self.assertEqual(len(finished), 6)
                self.assertTrue(finished["sim_ended"].all())
                np.testing.assert_array_equal(
                    finished["center_coordinates"][:5],
                    original["center_coordinates"],
                )

    def test_restart_preserves_results_and_completes_missing_work(self):
        """Retry saved inputs, draw new samples, and preserve ended results."""
        parameter_fields = (
            "num_centers", "center_coordinates",
            "center_coefficients", "center_widths",
        )
        for entrypoint in (RUN_PY, RUN_CLUSTER_PY):
            with (
                self.subTest(entrypoint=entrypoint.name),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                root = Path(temporary_directory)
                stub = write_stub(root)
                case_file = write_case_files(root, stub, num_samples=5)
                output_dir = root / "custom-ensemble"
                initial = run_ensemble_in(
                    root, case_file, fail_ids=(1,),
                    entrypoint=entrypoint, output_dir=output_dir,
                )
                self.assertEqual(
                    initial.returncode, 0, initial.stdout + initial.stderr
                )
                original = np.load(find_history(root)[0], allow_pickle=False)
                directories = {
                    int(path.name.removeprefix("sim")): path
                    for path in output_dir.glob("sim*")
                    if path.is_dir()
                }
                completed_files = {
                    path: (path.read_bytes(), path.stat().st_mtime_ns)
                    for sim_id in (0, 1)
                    for path in directories[sim_id].iterdir()
                    if path.is_file()
                }
                retry_inputs = {
                    sim_id: (directories[sim_id] / "case_params.txt").read_bytes()
                    for sim_id in (2, 3)
                }
                (directories[2] / "stale.vtk").write_text(
                    "partial output", encoding="utf-8"
                )

                saved = original[:4].copy()
                saved["sim_ended"][2:] = False
                saved["sim_started"][3] = False
                np.save(root / "restart.npy", saved, allow_pickle=False)
                case_data = json.loads(case_file.read_text(encoding="utf-8"))
                case_data.update(
                    num_samples=6, nworkers=2, batch_size=4, checkpoint_every=1
                )
                case_file.write_text(json.dumps(case_data), encoding="utf-8")

                resumed = run_ensemble_in(
                    root, case_file, entrypoint=entrypoint, nworkers=2,
                    output_dir=output_dir, restart_history=Path("restart.npy"),
                )
                self.assertEqual(
                    resumed.returncode, 0, resumed.stdout + resumed.stderr
                )
                self.assertIn(
                    "completed=2, retries=2, new_samples=2", resumed.stdout
                )
                latest_path = max(
                    find_history(root), key=lambda path: path.stat().st_mtime_ns
                )
                finished = np.load(latest_path, allow_pickle=False)
                self.assertEqual(len(finished), 6)
                self.assertTrue(finished["sim_ended"].all())
                np.testing.assert_array_equal(finished["sim_id"], np.arange(6))
                self.assertEqual(finished["success"][1], 0)
                self.assertEqual(
                    finished["return_code"][1], original["return_code"][1]
                )
                for field in parameter_fields:
                    np.testing.assert_array_equal(
                        finished[field][:4], saved[field]
                    )
                self.assertFalse(any(
                    all(
                        np.array_equal(new[field], old[field])
                        for field in parameter_fields
                    )
                    for new in finished[4:]
                    for old in original
                ))
                for path, expected in completed_files.items():
                    self.assertEqual(
                        (path.read_bytes(), path.stat().st_mtime_ns), expected
                    )
                for sim_id, expected in retry_inputs.items():
                    self.assertEqual(
                        (directories[sim_id] / "case_params.txt").read_bytes(),
                        expected,
                    )
                self.assertFalse((directories[2] / "stale.vtk").exists())
                archives = list((output_dir / "restart_archive").iterdir())
                self.assertEqual(len(archives), 1)
                self.assertEqual(
                    {path.name for path in archives[0].iterdir()},
                    {directories[index].name for index in (2, 3, 4)},
                )
                self.assertTrue(
                    (archives[0] / directories[2].name / "stale.vtk").is_file()
                )
                self.assertTrue(list(root.glob(
                    "libE_history*after_sim_*.npy"
                )))
                np.testing.assert_array_equal(
                    np.load(root / "restart.npy", allow_pickle=False), saved
                )

                invocations = {
                    path: path.stat().st_mtime_ns
                    for path in output_dir.glob("sim*/stub_invocation.txt")
                }
                repeated = run_ensemble_in(
                    root, case_file, entrypoint=entrypoint, nworkers=2,
                    output_dir=output_dir, restart_history=latest_path,
                )
                self.assertEqual(
                    repeated.returncode, 0, repeated.stdout + repeated.stderr
                )
                self.assertIn("already complete", repeated.stdout)
                self.assertEqual(
                    {
                        path: path.stat().st_mtime_ns
                        for path in output_dir.glob("sim*/stub_invocation.txt")
                    },
                    invocations,
                )
                self.assertEqual(
                    list((output_dir / "restart_archive").iterdir()), archives
                )


@unittest.skipUnless(
    REAL_RUN_CASE, "set PARAMSIM_EXEC to the built run_case binary"
)
class RealRunCaseEnsembleTests(unittest.TestCase):
    def test_real_run_case_exception_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            ranges_path = root / "case_ranges.json"
            ranges_path.write_text(
                json.dumps(
                    {
                        "num_centers_range": [1, 3],
                        "coordinate_bounds": [-1.0, 1.0],
                        "coeff_bounds": [0.1, 0.9],
                        "width_bounds": [0.2, 0.7],
                    }
                ),
                encoding="utf-8",
            )
            case_path = root / "case.json"
            case_path.write_text(
                json.dumps(
                    {
                        "num_samples": 2,
                        "batch_size": 1,
                        "simulation_exec_path": REAL_RUN_CASE,
                        "dimension": 2,
                        "resolution": 4,
                        "refine_levels": 1,
                        # An unsupported PDE solver makes run_case throw
                        # after argument parsing, before any mesh work.
                        "pde": "nonexistent_pde",
                        "case_type": "cube_exponential",
                        "case_params_ranges_file": "case_ranges.json",
                    }
                ),
                encoding="utf-8",
            )

            result = run_ensemble_in(root, case_path, fail_ids=())
            histories = find_history(root)
            self.assertEqual(
                len(histories),
                1,
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            history = np.load(histories[0], allow_pickle=False)
            self.assertEqual(len(history), 2)
            self.assertTrue(history["sim_ended"].all())
            self.assertTrue(
                (history["success"] == 0).all(),
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertTrue((history["return_code"] == 1).all())

            stderr_file = root / "ensemble" / "sim0000" / "run_case_0.err"
            self.assertTrue(stderr_file.exists())
            stderr_text = stderr_file.read_text(encoding="utf-8")
            self.assertIn("Exception on processing:", stderr_text)
            self.assertIn("Unsupported PDE solver!", stderr_text)


if __name__ == "__main__":
    unittest.main()
