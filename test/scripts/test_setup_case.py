import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from setup_case import (
    get_case_parameter_args,
    get_common_args_str,
    load_case_parameter_ranges,
    polynomial_coefficient_count,
    sample_parameter_values,
    setup_case,
    validate_case_parameter_ranges,
    write_case_parameter_file,
)


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


class RangeFileLoadingTests(unittest.TestCase):
    def test_resolves_range_file_relative_to_common_json(self):
        ranges = {
            "num_centers_range": [1, 5],
            "coordinate_bounds": [-1.0, 1.0],
            "coeff_bounds": [0.1, 0.9],
            "width_bounds": [0.2, 0.7],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            common_path = root / "configuration" / "ensemble.json"
            range_path = root / "configuration" / "ranges" / "exponential.json"
            range_path.parent.mkdir(parents=True)
            common_path.write_text("{}", encoding="utf-8")
            range_path.write_text(json.dumps(ranges), encoding="utf-8")

            case_data = {
                "case_type": "cube_exponential",
                "case_params_ranges_file": "ranges/exponential.json",
            }

            self.assertEqual(
                load_case_parameter_ranges(case_data, common_path),
                ranges,
            )

    def test_requires_range_file_reference(self):
        with self.assertRaisesRegex(ValueError, "case_params_ranges_file"):
            load_case_parameter_ranges(
                {"case_type": "cube_exponential"},
                "ensemble.json",
            )

    def test_validates_loaded_range_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            common_path = root / "ensemble.json"
            range_path = root / "invalid.json"
            range_path.write_text(
                json.dumps(
                    {
                        "num_modes_range": [0, 2],
                        "a_bounds": [-1.0, 1.0],
                        "b_bounds": [-1.0, 1.0],
                        "constant_bounds": [0.5, 1.5],
                        "wavelength_bounds": [0.25, 1.0],
                    }
                ),
                encoding="utf-8",
            )

            case_data = {
                "case_type": "cube_fourier",
                "case_params_ranges_file": "invalid.json",
            }

            with self.assertRaisesRegex(ValueError, "num_modes_range"):
                load_case_parameter_ranges(case_data, common_path)


class RangeValidationTests(unittest.TestCase):
    @staticmethod
    def valid_ranges():
        return {
            "cube_exponential": {
                "num_centers_range": [1, 5],
                "coordinate_bounds": [-1.0, 1.0],
                "coeff_bounds": [0.1, 0.9],
                "width_bounds": [0.2, 0.7],
            },
            "cube_fourier": {
                "num_modes_range": [1, 5],
                "a_bounds": [-1.0, 1.0],
                "b_bounds": [-1.0, 1.0],
                "constant_bounds": [0.5, 1.5],
                "wavelength_bounds": [0.25, 1.0],
            },
            "cube_polynomial": {
                "num_degree_levels_range": [1, 5],
                "center_coordinate_bounds": [-0.5, 0.5],
                "coeff_bounds": [-1.0, 1.0],
            },
        }

    def test_accepts_all_case_schemas_and_equal_bounds(self):
        for case_type, ranges in self.valid_ranges().items():
            constant_ranges = {
                name: [bounds[0], bounds[0]]
                for name, bounds in ranges.items()
            }
            with self.subTest(case_type=case_type):
                validate_case_parameter_ranges(
                    case_type,
                    constant_ranges,
                )
                validate_case_parameter_ranges(
                    case_type,
                    ranges
                )

    def test_rejects_invalid_count_ranges(self):
        invalid_ranges = (
            [0, 2],
            [3, 2],
            [1.0, 2],
            [True, 2],
            [1],
        )

        for count_range in invalid_ranges:
            with self.subTest(count_range=count_range):
                ranges = dict(self.valid_ranges()["cube_exponential"])
                ranges["num_centers_range"] = count_range
                with self.assertRaisesRegex(ValueError, "num_centers_range"):
                    validate_case_parameter_ranges(
                        "cube_exponential", ranges
                    )

    def test_rejects_missing_ranges(self):
        ranges = dict(self.valid_ranges()["cube_fourier"])
        del ranges["b_bounds"]

        with self.assertRaisesRegex(KeyError, "b_bounds"):
            validate_case_parameter_ranges("cube_fourier", ranges)

    def test_rejects_nonfinite_and_reversed_bounds(self):
        ranges = dict(self.valid_ranges()["cube_fourier"])
        ranges["constant_bounds"] = [0.0, math.inf]
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_case_parameter_ranges("cube_fourier", ranges)

        ranges = dict(self.valid_ranges()["cube_polynomial"])
        ranges["coeff_bounds"] = [1.0, -1.0]
        with self.assertRaisesRegex(ValueError, "lower <= upper"):
            validate_case_parameter_ranges("cube_polynomial", ranges)

    def test_rejects_nonpositive_widths_and_wavelengths(self):
        ranges = dict(self.valid_ranges()["cube_exponential"])
        ranges["width_bounds"] = [0.0, 0.7]
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_case_parameter_ranges("cube_exponential", ranges)

        ranges = dict(self.valid_ranges()["cube_fourier"])
        ranges["wavelength_bounds"] = [-0.25, 1.0]
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_case_parameter_ranges("cube_fourier", ranges)

    def test_rejects_exponential_coordinates_outside_domain(self):
        for coordinate_bounds in ([-1.1, 0.5], [-0.5, 1.1]):
            with self.subTest(coordinate_bounds=coordinate_bounds):
                ranges = dict(self.valid_ranges()["cube_exponential"])
                ranges["coordinate_bounds"] = coordinate_bounds
                with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
                    validate_case_parameter_ranges(
                        "cube_exponential", ranges
                    )

    def test_rejects_unknown_case_type_and_nonobject_data(self):
        with self.assertRaisesRegex(ValueError, "Unsupported case type"):
            validate_case_parameter_ranges("unknown", {})

        with self.assertRaisesRegex(ValueError, "must contain an object"):
            validate_case_parameter_ranges("cube_exponential", [])


def empty_specs():
    return (
        {
            "out": [],
            "user": {
                "lower": {},
                "upper": {},
            },
        },
        {"in": []},
    )


class HistoryFieldTests(unittest.TestCase):
    def test_exponential_fields_use_maximum_center_count(self):
        case_data = {
            "case_type": "cube_exponential",
            "dimension": 2,
        }
        ranges = {
            "num_centers_range": [1, 4],
            "coordinate_bounds": [-1.0, 1.0],
            "coeff_bounds": [0.1, 0.9],
            "width_bounds": [0.2, 0.7],
        }
        gen_specs, sim_specs = empty_specs()

        setup_case(case_data, gen_specs, sim_specs, ranges)

        history_dtype = np.dtype(gen_specs["out"])
        self.assertEqual(history_dtype["num_centers"].shape, ())
        self.assertEqual(
            history_dtype["center_coordinates"].shape, (4, 3)
        )
        self.assertEqual(
            history_dtype["center_coefficients"].shape, (4,)
        )
        self.assertEqual(history_dtype["center_widths"].shape, (4,))
        self.assertEqual(
            gen_specs["user"]["integer_parameters"],
            ["num_centers"],
        )
        np.testing.assert_array_equal(
            gen_specs["user"]["lower"]["center_coordinates"],
            np.full((4, 3), -1.0, dtype=np.float32),
        )
        self.assertEqual(
            sim_specs["in"],
            [
                "num_centers",
                "center_coordinates",
                "center_coefficients",
                "center_widths",
            ],
        )

    def test_fourier_fields_use_maximum_mode_count(self):
        case_data = {
            "case_type": "cube_fourier",
            "dimension": 3,
        }
        ranges = {
            "num_modes_range": [2, 6],
            "a_bounds": [-1.0, 1.0],
            "b_bounds": [-2.0, 2.0],
            "constant_bounds": [0.5, 1.5],
            "wavelength_bounds": [0.25, 1.0],
        }
        gen_specs, sim_specs = empty_specs()

        setup_case(case_data, gen_specs, sim_specs, ranges)

        history_dtype = np.dtype(gen_specs["out"])
        self.assertEqual(history_dtype["num_modes"].shape, ())
        self.assertEqual(
            history_dtype["mode_coefficients"].shape, (6, 2)
        )
        self.assertEqual(history_dtype["constant"].shape, ())
        self.assertEqual(history_dtype["wavelength"].shape, ())
        np.testing.assert_array_equal(
            gen_specs["user"]["lower"]["mode_coefficients"][:, 0],
            np.full(6, -1.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            gen_specs["user"]["lower"]["mode_coefficients"][:, 1],
            np.full(6, -2.0, dtype=np.float32),
        )
        self.assertEqual(
            sim_specs["in"],
            [
                "num_modes",
                "mode_coefficients",
                "constant",
                "wavelength",
            ],
        )

    def test_polynomial_storage_is_dimension_aware(self):
        ranges = {
            "num_degree_levels_range": [1, 4],
            "center_coordinate_bounds": [-0.5, 0.5],
            "coeff_bounds": [-1.0, 1.0],
        }

        for dimension, expected_coefficients in ((2, 10), (3, 20)):
            with self.subTest(dimension=dimension):
                case_data = {
                    "case_type": "cube_polynomial",
                    "dimension": dimension,
                }
                gen_specs, sim_specs = empty_specs()

                setup_case(case_data, gen_specs, sim_specs, ranges)

                history_dtype = np.dtype(gen_specs["out"])
                self.assertEqual(
                    history_dtype["num_degree_levels"].shape, ()
                )
                self.assertEqual(
                    history_dtype["center_coordinates"].shape, (3,)
                )
                self.assertEqual(
                    history_dtype["coefficients"].shape,
                    (expected_coefficients,),
                )
                self.assertEqual(
                    polynomial_coefficient_count(dimension, 4),
                    expected_coefficients,
                )
                self.assertEqual(
                    sim_specs["in"],
                    [
                        "num_degree_levels",
                        "center_coordinates",
                        "coefficients",
                    ],
                )


class SamplingTests(unittest.TestCase):
    def test_integer_sampling_is_inclusive(self):
        class RecordingRandomStream:
            def __init__(self):
                self.arguments = None

            def integers(self, lower, upper, size, endpoint):
                self.arguments = (lower, upper, size, endpoint)
                return np.asarray([lower, upper], dtype=np.int32)

        stream = RecordingRandomStream()
        samples = sample_parameter_values(
            stream,
            np.asarray(1, dtype=np.int32),
            np.asarray(5, dtype=np.int32),
            2,
            integer=True,
        )

        self.assertEqual(stream.arguments, (1, 5, (2,), True))
        np.testing.assert_array_equal(samples, [1, 5])

    def test_repeated_float_fields_are_sampled_independently(self):
        class DistinctRandomStream:
            def uniform(self, lower, upper, size):
                fractions = (
                    np.arange(1, np.prod(size) + 1, dtype=np.float64)
                    / (np.prod(size) + 1)
                ).reshape(size)
                return lower + (upper - lower) * fractions

        samples = sample_parameter_values(
            DistinctRandomStream(),
            np.full((4, 3), -1.0, dtype=np.float32),
            np.full((4, 3), 1.0, dtype=np.float32),
            2,
        )

        self.assertEqual(samples.shape, (2, 4, 3))
        self.assertNotEqual(samples[0, 0, 0], samples[0, 0, 1])
        self.assertNotEqual(samples[0, 0, 0], samples[0, 1, 0])

    def test_equal_bounds_remain_constant_across_samples(self):
        random_stream = np.random.default_rng(42)

        float_samples = sample_parameter_values(
            random_stream,
            np.full((2, 3), 1.25),
            np.full((2, 3), 1.25),
            10,
        )
        integer_samples = sample_parameter_values(
            random_stream,
            np.asarray(3, dtype=np.int32),
            np.asarray(3, dtype=np.int32),
            10,
            integer=True,
        )

        np.testing.assert_array_equal(
            float_samples,
            np.full((10, 2, 3), 1.25),
        )
        np.testing.assert_array_equal(
            integer_samples,
            np.full(10, 3),
        )


class ParameterFileTests(unittest.TestCase):
    def test_writes_only_active_exponential_centers(self):
        args = {
            "num_centers": 2,
            "center_coordinates": np.asarray(
                [
                    [-1.0, -0.5, 0.25],
                    [0.5, 1.0, -0.25],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "center_coefficients": np.asarray([0.1, -0.3, 99.0]),
            "center_widths": np.asarray([0.2, 0.4, 99.0]),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "case_params.txt"
            returned_path = write_case_parameter_file(
                "cube_exponential", 2, args, path
            )

            self.assertEqual(returned_path, path)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "2\n"
                "-1.0 -0.5 0.25 0.1 0.2\n"
                "0.5 1.0 -0.25 -0.3 0.4\n",
            )

    def test_writes_only_active_fourier_modes(self):
        args = {
            "num_modes": 2,
            "mode_coefficients": np.asarray(
                [[1.0, -2.0], [3.0, -4.0], [99.0, 99.0]]
            ),
            "constant": 1.5,
            "wavelength": 0.75,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "case_params.txt"
            write_case_parameter_file("cube_fourier", 3, args, path)

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "2\n"
                "1.5 0.75\n"
                "1.0 -2.0\n"
                "3.0 -4.0\n",
            )

    def test_writes_dimension_specific_polynomial_rows(self):
        expected_rows = {
            2: [
                ["1.0"],
                ["2.0", "3.0"],
                ["4.0", "5.0", "6.0"],
            ],
            3: [
                ["1.0"],
                ["2.0", "3.0", "4.0"],
                ["5.0", "6.0", "7.0", "8.0", "9.0", "10.0"],
            ],
        }

        for dimension, coefficient_rows in expected_rows.items():
            with self.subTest(dimension=dimension):
                args = {
                    "num_degree_levels": 3,
                    "center_coordinates": np.asarray([0.5, -0.5, 0.25]),
                    "coefficients": np.arange(1.0, 21.0),
                }

                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "case_params.txt"
                    write_case_parameter_file(
                        "cube_polynomial", dimension, args, path
                    )

                    lines = path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    self.assertEqual(lines[0], "3")
                    self.assertEqual(lines[1], "0.5 -0.5 0.25")
                    self.assertEqual(
                        [line.split() for line in lines[2:]],
                        coefficient_rows,
                    )

    def test_argument_uses_parameter_file(self):
        arguments = get_case_parameter_args("case_params.txt")

        self.assertEqual(
            arguments,
            " --case_params_file case_params.txt",
        )


if __name__ == "__main__":
    unittest.main()
