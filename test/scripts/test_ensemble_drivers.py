import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


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


if __name__ == "__main__":
    unittest.main()
