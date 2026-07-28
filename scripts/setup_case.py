import json
import math
from pathlib import Path

import numpy as np


def _validate_count_range(range_data: dict, name: str):
    value = range_data[name]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integer endpoints")
    if any(type(endpoint) is not int for endpoint in value):
        raise ValueError(f"{name} endpoints must be integers")

    lower, upper = value
    if lower < 1 or lower > upper:
        raise ValueError(f"{name} must satisfy 1 <= lower <= upper")


def _validate_float_bounds(range_data: dict, name: str):
    value = range_data[name]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numeric endpoints")
    if any(type(endpoint) not in (int, float) for endpoint in value):
        raise ValueError(f"{name} endpoints must be numeric")

    lower, upper = (float(endpoint) for endpoint in value)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{name} endpoints must be finite")
    if lower > upper:
        raise ValueError(f"{name} must satisfy lower <= upper")
    return lower, upper


def validate_case_parameter_ranges(case_type: str, range_data: dict):
    """Validate case-specific parameter ranges loaded from JSON.

    Args:
        case_type: Name of the cube case whose schema should be used.
        range_data: Case-specific count and floating-point bounds.

    Raises:
        KeyError: If a required range is absent.
        ValueError: If a range or case type is invalid.
    """
    if not isinstance(range_data, dict):
        raise ValueError("The case parameter ranges JSON must contain an object")

    if case_type == "cube_exponential":
        _validate_count_range(range_data, "num_centers_range")
        coordinate_lower, coordinate_upper = _validate_float_bounds(
            range_data, "coordinate_bounds"
        )
        _validate_float_bounds(range_data, "coeff_bounds")
        width_lower, _ = _validate_float_bounds(range_data, "width_bounds")

        if coordinate_lower < -1.0 or coordinate_upper > 1.0:
            raise ValueError("coordinate_bounds must lie within [-1, 1]")
        if width_lower <= 0.0:
            raise ValueError("width_bounds must be positive")

    elif case_type == "cube_fourier":
        _validate_count_range(range_data, "num_modes_range")
        _validate_float_bounds(range_data, "a_bounds")
        _validate_float_bounds(range_data, "b_bounds")
        _validate_float_bounds(range_data, "constant_bounds")
        wavelength_lower, _ = _validate_float_bounds(
            range_data, "wavelength_bounds"
        )

        if wavelength_lower <= 0.0:
            raise ValueError("wavelength_bounds must be positive")

    elif case_type == "cube_polynomial":
        _validate_count_range(range_data, "num_degree_levels_range")
        _validate_float_bounds(range_data, "center_coordinate_bounds")
        _validate_float_bounds(range_data, "coeff_bounds")

    else:
        raise ValueError(
            f"Unsupported case type for parameter ranges: {case_type}"
        )


def load_case_parameter_ranges(case_data: dict, case_file_path):
    """Load and validate ranges referenced by a common ensemble JSON.

    Relative range-file paths are resolved from the directory containing the
    common ensemble JSON.

    Args:
        case_data: Common ensemble configuration.
        case_file_path: Path of the common ensemble JSON.

    Returns:
        The validated case-specific range configuration.
    """
    reference = case_data.get("case_params_ranges_file")
    if type(reference) is not str or not reference.strip():
        raise ValueError(
            "The case JSON must define a nonempty case_params_ranges_file"
        )

    common_path = Path(case_file_path).expanduser().resolve()
    range_path = Path(reference).expanduser()
    if not range_path.is_absolute():
        range_path = common_path.parent / range_path
    range_path = range_path.resolve()

    with range_path.open("r", encoding="utf-8") as range_file:
        range_data = json.load(range_file)

    validate_case_parameter_ranges(case_data.get("case_type"), range_data)
    return range_data


def sample_parameter_values(
    random_stream,
    lower,
    upper,
    sample_count: int,
    integer: bool = False,
):
    """Sample one fixed-shape history field.

    Args:
        random_stream: NumPy random generator used for sampling.
        lower: Scalar or array of lower bounds.
        upper: Scalar or array of upper bounds.
        sample_count: Number of history rows to generate.
        integer: Whether to use inclusive discrete-uniform sampling.

    Returns:
        An array whose leading dimension is ``sample_count``.
    """
    lower_values = np.asarray(lower)
    upper_values = np.asarray(upper)
    if lower_values.shape != upper_values.shape:
        raise ValueError(
            f"Mismatched bound shapes: "
            f"{lower_values.shape} != {upper_values.shape}"
        )

    sample_shape = (sample_count,) + lower_values.shape
    if integer:
        if lower_values.shape:
            raise ValueError("Integer history fields must be scalar")
        return random_stream.integers(
            int(lower_values),
            int(upper_values),
            size=sample_shape,
            endpoint=True,
        )

    return random_stream.uniform(
        lower_values,
        upper_values,
        sample_shape,
    )


def polynomial_coefficient_count(dimension: int, degree_levels: int) -> int:
    """Return the number of total-degree coefficients through all levels.

    Args:
        dimension: Active spatial dimension, either 2 or 3.
        degree_levels: Number of degree levels beginning with degree zero.

    Returns:
        Total number of monomial coefficients.

    Raises:
        ValueError: If the dimension is unsupported.
    """
    if dimension == 2:
        return degree_levels * (degree_levels + 1) // 2
    if dimension == 3:
        return degree_levels * (degree_levels + 1) * (degree_levels + 2) // 6
    raise ValueError("Polynomial coefficient counts require dimension 2 or 3")


def get_common_args_str(case_data : dict) -> str:
    """Construct command-line arguments shared by every simulation.

    Args:
        case_data: Common ensemble configuration.

    Returns:
        Command-line fragment containing the common run settings.

    Raises:
        ValueError: If the configured dimension is invalid or absent.
    """
    if "dimension" not in case_data:
        raise ValueError("The case JSON must define dimension as 2 or 3")

    dimension = case_data["dimension"]
    if type(dimension) is not int or dimension not in (2, 3):
        raise ValueError("dimension must be the integer 2 or 3")

    refine_levels = case_data.get("refine_levels", 1)
    return (
        "--dimension " + str(dimension)
        + " --pde " + case_data["pde"]
        + " --case " + case_data["case_type"]
        + " --refine_levels " + str(refine_levels)
        + " --initial_resolution " + str(case_data["resolution"])
        + " --output_prefix field"
    )

#TODO: Replace the if-blocks in this file with a set of classes

def setup_case(
    case_data: dict,
    gen_specs: dict,
    sim_specs: dict,
    case_parameter_ranges: dict,
):
    """Populate history fields and sampling bounds for one case family.

    Args:
        case_data: Common ensemble configuration.
        gen_specs: libEnsemble generator specification to update.
        sim_specs: libEnsemble simulation specification to update.
        case_parameter_ranges: Validated case-specific parameter ranges.
    """
    case_type = case_data["case_type"]
    validate_case_parameter_ranges(case_type, case_parameter_ranges)
    integer_parameters = gen_specs["user"].setdefault(
        "integer_parameters", []
    )

    if case_type == "cube_exponential":
        count_lower, count_upper = case_parameter_ranges[
            "num_centers_range"
        ]
        max_centers = count_upper
        coordinate_lower, coordinate_upper = case_parameter_ranges[
            "coordinate_bounds"
        ]
        coefficient_lower, coefficient_upper = case_parameter_ranges[
            "coeff_bounds"
        ]
        width_lower, width_upper = case_parameter_ranges["width_bounds"]

        gen_specs["out"].extend(
            [
                ("num_centers", np.int32),
                (
                    "center_coordinates",
                    np.float32,
                    (max_centers, 3),
                ),
                (
                    "center_coefficients",
                    np.float32,
                    (max_centers,),
                ),
                ("center_widths", np.float32, (max_centers,)),
            ]
        )
        sim_specs["in"].extend(
            [
                "num_centers",
                "center_coordinates",
                "center_coefficients",
                "center_widths",
            ]
        )
        integer_parameters.append("num_centers")
        gen_specs["user"]["lower"]["num_centers"] = np.asarray(
            count_lower, dtype=np.int32
        )
        gen_specs["user"]["upper"]["num_centers"] = np.asarray(
            count_upper, dtype=np.int32
        )
        gen_specs["user"]["lower"]["center_coordinates"] = np.full(
            (max_centers, 3), coordinate_lower, dtype=np.float32
        )
        gen_specs["user"]["upper"]["center_coordinates"] = np.full(
            (max_centers, 3), coordinate_upper, dtype=np.float32
        )
        gen_specs["user"]["lower"]["center_coefficients"] = np.full(
            max_centers, coefficient_lower, dtype=np.float32
        )
        gen_specs["user"]["upper"]["center_coefficients"] = np.full(
            max_centers, coefficient_upper, dtype=np.float32
        )
        gen_specs["user"]["lower"]["center_widths"] = np.full(
            max_centers, width_lower, dtype=np.float32
        )
        gen_specs["user"]["upper"]["center_widths"] = np.full(
            max_centers, width_upper, dtype=np.float32
        )

    elif case_type == "cube_fourier":
        count_lower, count_upper = case_parameter_ranges["num_modes_range"]
        max_modes = count_upper
        a_lower, a_upper = case_parameter_ranges["a_bounds"]
        b_lower, b_upper = case_parameter_ranges["b_bounds"]
        constant_lower, constant_upper = case_parameter_ranges[
            "constant_bounds"
        ]
        wavelength_lower, wavelength_upper = case_parameter_ranges[
            "wavelength_bounds"
        ]

        gen_specs["out"].extend(
            [
                ("num_modes", np.int32),
                ("mode_coefficients", np.float32, (max_modes, 2)),
                ("constant", np.float32),
                ("wavelength", np.float32, (case_data["dimension"],)),
            ]
        )
        sim_specs["in"].extend(
            [
                "num_modes",
                "mode_coefficients",
                "constant",
                "wavelength",
            ]
        )
        integer_parameters.append("num_modes")
        gen_specs["user"]["lower"]["num_modes"] = np.asarray(
            count_lower, dtype=np.int32
        )
        gen_specs["user"]["upper"]["num_modes"] = np.asarray(
            count_upper, dtype=np.int32
        )
        lower_coefficients = np.empty((max_modes, 2), dtype=np.float32)
        upper_coefficients = np.empty((max_modes, 2), dtype=np.float32)
        lower_coefficients[:, 0] = a_lower
        lower_coefficients[:, 1] = b_lower
        upper_coefficients[:, 0] = a_upper
        upper_coefficients[:, 1] = b_upper
        gen_specs["user"]["lower"][
            "mode_coefficients"
        ] = lower_coefficients
        gen_specs["user"]["upper"][
            "mode_coefficients"
        ] = upper_coefficients
        gen_specs["user"]["lower"]["constant"] = np.asarray(
            constant_lower, dtype=np.float32
        )
        gen_specs["user"]["upper"]["constant"] = np.asarray(
            constant_upper, dtype=np.float32
        )
        gen_specs["user"]["lower"]["wavelength"] = np.full(
            case_data["dimension"], wavelength_lower, dtype=np.float32
        )
        gen_specs["user"]["upper"]["wavelength"] = np.full(
            case_data["dimension"], wavelength_upper, dtype=np.float32
        )

    elif case_type == "cube_polynomial":
        count_lower, count_upper = case_parameter_ranges[
            "num_degree_levels_range"
        ]
        max_degree_levels = count_upper
        max_coefficients = polynomial_coefficient_count(
            case_data["dimension"], max_degree_levels
        )
        center_lower, center_upper = case_parameter_ranges[
            "center_coordinate_bounds"
        ]
        coefficient_lower, coefficient_upper = case_parameter_ranges[
            "coeff_bounds"
        ]

        gen_specs["out"].extend(
            [
                ("num_degree_levels", np.int32),
                ("center_coordinates", np.float32, (3,)),
                ("coefficients", np.float32, (max_coefficients,)),
            ]
        )
        sim_specs["in"].extend(
            ["num_degree_levels", "center_coordinates", "coefficients"]
        )
        integer_parameters.append("num_degree_levels")
        gen_specs["user"]["lower"]["num_degree_levels"] = np.asarray(
            count_lower, dtype=np.int32
        )
        gen_specs["user"]["upper"]["num_degree_levels"] = np.asarray(
            count_upper, dtype=np.int32
        )
        gen_specs["user"]["lower"]["center_coordinates"] = np.full(
            3, center_lower, dtype=np.float32
        )
        gen_specs["user"]["upper"]["center_coordinates"] = np.full(
            3, center_upper, dtype=np.float32
        )
        gen_specs["user"]["lower"]["coefficients"] = np.full(
            max_coefficients, coefficient_lower, dtype=np.float32
        )
        gen_specs["user"]["upper"]["coefficients"] = np.full(
            max_coefficients, coefficient_upper, dtype=np.float32
        )

    else:
        raise ValueError(f"Invalid case type: {case_type}")

def _setup_legacy_case(case_data: dict, gen_specs: dict, sim_specs: dict):
    """ Depending on the case type, adds case-specific ensemble run parameters to
        libensemble dicts.
        
        For the case cube_exponential, this needs an array "centers" of length 3, each having
        dict "coords_bounds" (lower and upper bounds for y-coordinates, so array of length 2),
        dict "coeff_bounds" (lower and upper bounds for coefficients, so array of length 2).
        In addition, a key "width_bounds" with a 2-array as value, having lower and upper bounds
        for the width of each exponential hill.

        @param[in] case_data  Dict of ensemble options supplied in the ensemble settings JSON.
        @param[in,out] gen_specs  Parameter bounds are populated in this libEnsemble dict.
    """
    if case_data["case_type"] == "cube_exponential":
        ncenters = 3
        ndim = 1
        # Output of generator include 2 centers, each with x-coord, y-coord and coefficient
        gen_specs["out"].append( ("centers", np.float32, (ncenters, ndim+1)) )
        # ..and width of the hills
        gen_specs["out"].append( ("width", np.float32, (1,)) )
        sim_specs["in"].append("centers")
        sim_specs["in"].append("width")

        cparams = case_data["centers"]
        l_cbounds = np.zeros((ncenters,ndim+1), dtype=np.float32)
        u_cbounds = np.zeros((ncenters,ndim+1), dtype=np.float32)
        for ic in range(ncenters):
            l_cbounds[ic][0] = cparams[ic]["coords_bounds"][0]
            l_cbounds[ic][1] = cparams[ic]["coeff_bounds"][0]
            u_cbounds[ic][0] = cparams[ic]["coords_bounds"][1]
            u_cbounds[ic][1] = cparams[ic]["coeff_bounds"][1]
        gen_specs["user"]["lower"]["centers"] = l_cbounds
        gen_specs["user"]["upper"]["centers"] = u_cbounds
        
        l_wbound = case_data["width_bounds"][0]
        u_wbound = case_data["width_bounds"][1]
        gen_specs["user"]["lower"]["width"] = np.array([l_wbound], dtype=np.float32)
        gen_specs["user"]["upper"]["width"] = np.array([u_wbound], dtype=np.float32)

    elif case_data["case_type"] == "cube_polynomial":
        nterms = 4
        ndim = 1
        gen_specs["out"].append( ("coeffs", np.float32, (nterms,)) )
        gen_specs["out"].append( ("center_y", np.float32, (1,)) )
        sim_specs["in"].append("coeffs")
        sim_specs["in"].append("center_y")

        cparams = case_data["coeffs_bounds"]
        l_cbounds = np.zeros((nterms,), dtype=np.float32)
        u_cbounds = np.zeros((nterms,), dtype=np.float32)
        for it in range(nterms):
            l_cbounds[it] = cparams[it][0]
            u_cbounds[it] = cparams[it][1]
        gen_specs["user"]["lower"]["coeffs"] = l_cbounds
        gen_specs["user"]["upper"]["coeffs"] = u_cbounds

        l_ybound = case_data["center_y_bounds"][0]
        u_ybound = case_data["center_y_bounds"][1]
        gen_specs["user"]["lower"]["center_y"] = np.array([l_ybound], dtype=np.float32)
        gen_specs["user"]["upper"]["center_y"] = np.array([u_ybound], dtype=np.float32)

    elif case_data["case_type"] == "cube_fourier":
        nmodes = 2
        ndim = 1
        # Output of generator include 2 centers, each with x-coord, y-coord and coefficient
        gen_specs["out"].append( ("amplitudes", np.float32, (nmodes, 2)) )
        # ..and width of the hills
        gen_specs["out"].append( ("constant", np.float32, (1,)) )
        gen_specs["out"].append( ("wavelength", np.float32, (1,)) )
        sim_specs["in"].append("amplitudes")
        sim_specs["in"].append("constant")
        sim_specs["in"].append("wavelength")

        cparams = case_data["amplitudes"]
        l_ampbounds = np.zeros((nmodes,2), dtype=np.float32)
        u_ampbounds = np.zeros((nmodes,2), dtype=np.float32)
        for ic in range(nmodes):
            l_ampbounds[ic,0] = cparams[ic]["a_bounds"][0]
            l_ampbounds[ic,1] = cparams[ic]["b_bounds"][0]
            u_ampbounds[ic,0] = cparams[ic]["a_bounds"][1]
            u_ampbounds[ic,1] = cparams[ic]["b_bounds"][1]
        gen_specs["user"]["lower"]["amplitudes"] = l_ampbounds
        gen_specs["user"]["upper"]["amplitudes"] = u_ampbounds

        l_cbound = case_data["constant_bounds"][0]
        u_cbound = case_data["constant_bounds"][1]
        gen_specs["user"]["lower"]["constant"] = np.array([l_cbound], dtype=np.float32)
        gen_specs["user"]["upper"]["constant"] = np.array([u_cbound], dtype=np.float32)

        l_wbound = case_data["wavelength_bounds"][0]
        u_wbound = case_data["wavelength_bounds"][1]
        gen_specs["user"]["lower"]["wavelength"] = np.array([l_wbound], dtype=np.float32)
        gen_specs["user"]["upper"]["wavelength"] = np.array([u_wbound], dtype=np.float32)
    else:
        raise "Invalid case type!"

def write_case_parameter_file(
    case_type: str,
    dimension: int,
    args,
    file_path="case_params.txt",
):
    """Write one sampled history row using the selected case schema.

    Args:
        case_type: Name of the cube case.
        dimension: Active spatial dimension, either 2 or 3.
        args: Sampled libEnsemble history row or equivalent mapping.
        file_path: Destination parameter-file path.

    Returns:
        The destination as a ``Path`` object.

    Raises:
        ValueError: If the case type or spatial dimension is unsupported.
    """
    path = Path(file_path)
    with path.open("w", encoding="utf-8") as parameter_file:
        if case_type == "cube_exponential":
            count = int(args["num_centers"])
            parameter_file.write(f"{count}\n")
            for index in range(count):
                values = [
                    *args["center_coordinates"][index],
                    args["center_coefficients"][index],
                    args["center_widths"][index],
                ]
                parameter_file.write(
                    " ".join(str(value) for value in values) + "\n"
                )

        elif case_type == "cube_fourier":
            count = int(args["num_modes"])
            parameter_file.write(f"{count}\n")
            header = [
                args["constant"],
                *args["wavelength"][:dimension],
            ]
            parameter_file.write(
                " ".join(str(value) for value in header) + "\n"
            )
            for index in range(count):
                parameter_file.write(
                    " ".join(
                        str(value)
                        for value in args["mode_coefficients"][index]
                    )
                    + "\n"
                )

        elif case_type == "cube_polynomial":
            degree_levels = int(args["num_degree_levels"])
            parameter_file.write(f"{degree_levels}\n")
            parameter_file.write(
                " ".join(
                    str(value) for value in args["center_coordinates"]
                )
                + "\n"
            )

            coefficient_offset = 0
            for degree in range(degree_levels):
                next_offset = polynomial_coefficient_count(
                    dimension, degree + 1
                )
                degree_coefficients = args["coefficients"][
                    coefficient_offset:next_offset
                ]
                parameter_file.write(
                    " ".join(
                        str(value) for value in degree_coefficients
                    )
                    + "\n"
                )
                coefficient_offset = next_offset

        else:
            raise ValueError(f"Invalid case type: {case_type}")

    return path


def get_case_parameter_args(file_path="case_params.txt") -> str:
    """Return the run_case argument for a generated parameter file.

    Args:
        file_path: Parameter-file path passed to ``run_case``.

    Returns:
        Command-line fragment containing ``--case_params_file``.
    """
    return f" --case_params_file {file_path}"
