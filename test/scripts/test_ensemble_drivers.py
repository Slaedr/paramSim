import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ensemble_restart import load_restart_history, prepare_output_directory


def load_driver(module_name):
    module_path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"test_{module_name}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTask:
    def __init__(self, success, return_code):
        self.success = success
        self.errcode = return_code
        self.waited = False

    def wait(self):
        self.waited = True


class FakeExecutor:
    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.submissions = []

    def submit(self, **kwargs):
        self.submissions.append(kwargs)
        return self.tasks[len(self.submissions) - 1]


class EnsembleDriverTests(unittest.TestCase):
    def test_generators_respect_remaining_sample_slots(self):
        """Limit new samples using all rows already present in history."""
        gen_specs = {
            "out": [("parameter", np.float32)],
            "user": {
                "lower": {"parameter": -1.0},
                "upper": {"parameter": 1.0},
                "gen_batch_size": 4,
                "num_samples": 5,
            },
        }
        for driver in self.drivers:
            for existing_count, expected_count in (
                (0, 4),
                (3, 2),
                (5, 0),
                (7, 0),
            ):
                with self.subTest(
                    driver=driver.__name__, existing_count=existing_count
                ):
                    history = np.zeros(
                        existing_count, dtype=[("sim_id", np.int64)]
                    )
                    samples, _ = driver.gen_random_samples(
                        history,
                        {"rand_stream": np.random.default_rng(1234)},
                        gen_specs,
                    )
                    self.assertEqual(len(samples), expected_count)

    def test_restart_random_streams_generate_new_samples(self):
        """Check fresh restart samples against parameters saved previously."""
        sampling_script = """
import sys

import numpy as np
from libensemble.tools import add_unique_random_streams

seed = None if sys.argv[1] == "restart" else 1234
streams = add_unique_random_streams({}, 2, seed=seed)
samples = streams[0]["rand_stream"].uniform(
    -1.0, 1.0, size=(2, 3)
)
np.save(sys.argv[2], samples, allow_pickle=False)
"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for phase in ("initial", "repeat", "restart"):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        sampling_script,
                        phase,
                        str(root / f"{phase}.npy"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    result.stdout + result.stderr,
                )

            initial = np.load(root / "initial.npy", allow_pickle=False)
            repeated = np.load(root / "repeat.npy", allow_pickle=False)
            restarted = np.load(root / "restart.npy", allow_pickle=False)

        np.testing.assert_array_equal(initial, repeated)
        self.assertEqual(initial.shape, (2, 3))
        self.assertEqual(restarted.shape, (2, 3))

        matches = np.all(
            restarted[:, None, :] == initial[None, :, :],
            axis=2,
        )
        self.assertFalse(
            matches.any(),
            "A restart sample duplicated a previously saved sample",
        )

    @classmethod
    def setUpClass(cls):
        cls.drivers = (
            load_driver("ensemble"),
            load_driver("ensemble_cluster"),
        )

    @staticmethod
    def sim_specs():
        return {
            "out": [
                ("success", np.int32),
                ("return_code", np.int32),
                ("runtime_sec", np.float64),
                ("hostname", "S64"),
            ],
            "user": {
                "case_type": "cube_exponential",
                "dimension": 3,
                "common_args": "--dimension 3",
            },
        }

    def run_driver(self, driver, tasks):
        executor = FakeExecutor(tasks)
        history = np.zeros(len(tasks), dtype=[("sample", np.int32)])
        persistent_info = {"marker": object()}

        with (
            mock.patch.object(
                driver,
                "write_case_parameter_file",
                return_value=Path("case_params.txt"),
            ),
            mock.patch.object(
                driver,
                "get_case_parameter_args",
                return_value=" --case_params_file case_params.txt",
            ),
        ):
            result = driver.run_case(
                history,
                persistent_info,
                self.sim_specs(),
                {"executor": executor},
            )

        return result, executor, persistent_info

    def test_successful_tasks_return_complete_output(self):
        for driver in self.drivers:
            with self.subTest(driver=driver.__name__):
                task = FakeTask(True, 0)
                result, executor, persistent_info = self.run_driver(
                    driver, [task]
                )
                output, returned_info, status = result

                self.assertIs(returned_info, persistent_info)
                self.assertEqual(status, WORKER_DONE)
                self.assertEqual(output["success"].tolist(), [1])
                self.assertEqual(output["return_code"].tolist(), [0])
                self.assertGreaterEqual(output["runtime_sec"][0], 0.0)
                self.assertTrue(output["hostname"][0])
                self.assertTrue(task.waited)
                self.assertEqual(
                    executor.submissions[0]["stdout"], "run_case_0.out"
                )
                self.assertEqual(
                    executor.submissions[0]["stderr"], "run_case_0.err"
                )

    def test_nonzero_task_marks_calculation_failed(self):
        for driver in self.drivers:
            with self.subTest(driver=driver.__name__):
                tasks = [FakeTask(True, 0), FakeTask(False, 17)]
                result, executor, _ = self.run_driver(driver, tasks)
                output, _, status = result

                self.assertEqual(status, TASK_FAILED)
                self.assertEqual(output["success"].tolist(), [1, 0])
                self.assertEqual(output["return_code"].tolist(), [0, 17])
                self.assertEqual(
                    executor.submissions[1]["stdout"], "run_case_1.out"
                )
                self.assertEqual(
                    executor.submissions[1]["stderr"], "run_case_1.err"
                )


class RestartHistoryTests(unittest.TestCase):
    def test_incompatible_histories_are_rejected(self):
        """Reject invalid field layouts, simulation IDs, and completion flags."""
        wrong_shape = np.zeros(
            4,
            dtype=[
                ("parameters", np.float32, (2,))
                if field[0] == "parameters" else field
                for field in self.original.dtype.descr
            ],
        )
        duplicate_ids = self.original.copy()
        duplicate_ids["sim_id"][2] = 1
        invalid_flags = self.original.copy()
        invalid_flags["sim_started"][0] = False

        for label, history, message in (
            ("plain", np.zeros(4), "structured 1D array"),
            ("missing", self.original[["sim_id"]], "missing fields"),
            ("shape", wrong_shape, "Incompatible history field parameters"),
            ("IDs", duplicate_ids, "consecutive rows"),
            ("flags", invalid_flags, "ended without starting"),
        ):
            with self.subTest(history=label):
                np.save(self.history_path, history, allow_pickle=False)
                with self.assertRaisesRegex(ValueError, message):
                    load_restart_history(
                        self.history_path, self.gen_specs, self.sim_specs, 6
                    )

    def test_output_directory_archives_only_unfinished_results(self):
        """Preserve ended outputs and archive partial or unrecorded outputs."""
        output_dir = self.history_path.parent / "ensemble"
        output_dir.mkdir()
        for name in ("sim0000", "sim0001", "sim0002", "sim0004"):
            directory = output_dir / name
            directory.mkdir()
            (directory / "result.txt").write_text(name, encoding="utf-8")

        width, archive = prepare_output_directory(
            output_dir, self.original, default_width=6
        )

        self.assertEqual(width, 4)
        self.assertIsNotNone(archive)
        self.assertEqual(
            {path.name for path in archive.iterdir()},
            {"sim0002", "sim0004"},
        )
        for name in ("sim0000", "sim0001"):
            self.assertEqual(
                (output_dir / name / "result.txt").read_text(encoding="utf-8"),
                name,
            )
        for name in ("sim0002", "sim0004"):
            self.assertFalse((output_dir / name).exists())
            self.assertEqual(
                (archive / name / "result.txt").read_text(encoding="utf-8"),
                name,
            )

    def test_fresh_run_refuses_existing_simulation_directories(self):
        """Allow an empty output directory but protect existing simulations."""
        output_dir = self.history_path.parent / "ensemble"
        output_dir.mkdir()
        self.assertEqual(
            prepare_output_directory(output_dir, None, default_width=6),
            (6, None),
        )
        simulation_dir = output_dir / "sim000000"
        simulation_dir.mkdir()

        with self.assertRaisesRegex(FileExistsError, "--restart-history"):
            prepare_output_directory(output_dir, None, default_width=6)

        self.assertTrue(simulation_dir.is_dir())
        self.assertFalse((output_dir / "restart_archive").exists())

    def setUp(self):
        """Save history containing success, failure, and unfinished rows."""
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.history_path = Path(temporary_directory.name) / "history.npy"
        self.gen_specs = {"out": [("parameters", np.float32, (3,))]}
        self.sim_specs = {
            "out": [
                ("success", np.int32),
                ("return_code", np.int32),
            ]
        }
        self.original = np.zeros(
            4,
            dtype=self.gen_specs["out"] + self.sim_specs["out"] + [
                ("sim_id", np.int64),
                ("sim_started", np.bool_),
                ("sim_ended", np.bool_),
                ("cancel_requested", np.bool_),
                ("kill_sent", np.bool_),
                ("gen_informed", np.bool_),
                ("sim_worker", np.int64),
                ("sim_started_time", np.float64),
                ("sim_ended_time", np.float64),
                ("gen_informed_time", np.float64),
            ],
        )
        self.original["sim_id"] = np.arange(4)
        self.original["parameters"] = np.arange(12).reshape(4, 3)
        self.original["sim_started"] = [True, True, True, False]
        self.original["sim_ended"] = [True, True, False, False]
        self.original["success"] = [1, 0, 1, 1]
        self.original["return_code"] = [0, 17, 99, 99]
        self.original["cancel_requested"][2:] = True
        self.original["kill_sent"][2:] = True
        self.original["gen_informed"] = True
        self.original["sim_worker"] = 4
        self.original["sim_started_time"] = 10.0
        self.original["sim_ended_time"] = 20.0
        self.original["gen_informed_time"] = 30.0
        np.save(self.history_path, self.original, allow_pickle=False)

    def test_restart_preserves_inputs_and_completed_results(self):
        """Retain ended failures and reset unfinished execution state."""
        history, remaining = load_restart_history(
            self.history_path, self.gen_specs, self.sim_specs, 6
        )

        self.assertEqual(remaining, 4)
        np.testing.assert_array_equal(history[:2], self.original[:2])
        np.testing.assert_array_equal(
            history["parameters"], self.original["parameters"]
        )
        np.testing.assert_array_equal(
            history["sim_id"], self.original["sim_id"]
        )
        for field in (
            "sim_started", "sim_ended", "cancel_requested",
            "kill_sent", "gen_informed",
        ):
            self.assertFalse(history[field][2:].any(), field)
        for field in (
            "success", "return_code", "sim_worker", "sim_ended_time",
        ):
            self.assertTrue((history[field][2:] == 0).all(), field)
        self.assertTrue(np.isposinf(history["sim_started_time"][2:]).all())
        self.assertTrue(np.isposinf(history["gen_informed_time"][2:]).all())
        np.testing.assert_array_equal(
            np.load(self.history_path, allow_pickle=False), self.original
        )

    def test_retry_count_respects_target(self):
        """Cancel surplus queued rows and handle an already-met target."""
        for target, expected_cancellations in (
            (3, [False, True]),
            (2, [True, True]),
        ):
            with self.subTest(target=target):
                history, remaining = load_restart_history(
                    self.history_path, self.gen_specs, self.sim_specs, target
                )
                self.assertEqual(remaining, target - 2)
                np.testing.assert_array_equal(
                    history["cancel_requested"][2:], expected_cancellations
                )
                np.testing.assert_array_equal(history[:2], self.original[:2])

    def test_target_below_completed_count_is_rejected(self):
        """Reject a target smaller than the saved completed count."""
        with self.assertRaisesRegex(ValueError, "below the completed count"):
            load_restart_history(
                self.history_path, self.gen_specs, self.sim_specs, 1
            )


if __name__ == "__main__":
    unittest.main()
