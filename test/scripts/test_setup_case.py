import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from setup_case import get_common_args_str


class CommonArgsTests(unittest.TestCase):
    def base_case(self):
        return {
            "dimension": 3,
            "pde": "poisson_cg",
            "case_type": "cube_exponential",
            "refine_levels": 2,
            "resolution": 8,
        }

    def test_dimension_is_forwarded(self):
        self.assertEqual(
            get_common_args_str(self.base_case()),
            "--dimension 3 --pde poisson_cg --case cube_exponential "
            "--refine_levels 2 --initial_resolution 8 --output_prefix field",
        )

    def test_dimension_is_required(self):
        case_data = self.base_case()
        del case_data["dimension"]

        with self.assertRaisesRegex(ValueError, "must define dimension"):
            get_common_args_str(case_data)

    def test_dimension_must_be_two_or_three(self):
        for value in (1, 4, True, "3"):
            with self.subTest(value=value):
                case_data = self.base_case()
                case_data["dimension"] = value
                with self.assertRaisesRegex(ValueError, "integer 2 or 3"):
                    get_common_args_str(case_data)


if __name__ == "__main__":
    unittest.main()
