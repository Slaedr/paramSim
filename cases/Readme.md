# Cube case parameters

The `cube_exponential`, `cube_fourier`, and `cube_polynomial` cases apply a
configurable Dirichlet profile to the cube face at \(x=-1\). The remaining
faces use the case's constant boundary value.

## Direct `run_case` usage

Pass the selected case's text file with `--case_params_file`:

```sh
./build/run_case --dimension 2 --pde poisson_cg \
    --case cube_exponential \
    --case_params_file cases/examples/cube_exponential_case_params.txt \
    --initial_resolution 8 --refine_levels 1 --output_prefix field
```

If `--case_params_file` is omitted, the selected case uses its built-in
defaults. Parameter files are strict, line-oriented, whitespace-separated
text files. Counts must be positive integers, numeric values must be finite,
and each row must contain exactly the documented number of values.

### Exponential parameters

The `cube_exponential` format is:

```text
<number_of_centers>
<x> <y> <z> <coefficient> <width>
...one row per center
```

All three coordinates are required and must lie in `[-1, 1]`. Each width must
be positive. Coefficients may have either sign. Each center has its own
coefficient and width.

The current profile evaluates normalized Gaussians using the x-y distance in
both spatial dimensions. The z coordinate is parsed and validated but does
not currently affect the profile, so the 3D boundary data is extruded in z.

See
[cube_exponential_case_params.txt](examples/cube_exponential_case_params.txt)
for a complete file.

### Fourier parameters

The `cube_fourier` format is:

```text
<number_of_modes>
<constant> <fundamental_wavelength>
<cosine_coefficient_1> <sine_coefficient_1>
...one row per mode
```

The fundamental wavelength must be positive. For mode \(n\), the profile adds

\[
a_n \cos(2\pi n y/\lambda) + b_n \sin(2\pi n y/\lambda).
\]

Rows are frequencies `1` through `number_of_modes`; the constant is present
only once and is not a mode row. The profile depends on y in both 2D and 3D.

See [cube_fourier_case_params.txt](examples/cube_fourier_case_params.txt) for
a complete file.

### Polynomial parameters

The `cube_polynomial` format is:

```text
<number_of_degree_levels>
<center_x> <center_y> <center_z>
<degree-0 coefficients>
<degree-1 coefficients>
...one row through degree number_of_degree_levels-1
```

The three center coordinates are always required. In 2D, `center_z` is parsed
but ignored. The polynomial uses shifted coordinates

- \(X=x-\mathrm{center}_x\)
- \(Y=y-\mathrm{center}_y\)
- \(Z=z-\mathrm{center}_z\) in 3D

Each row contains every monomial of that total degree. Terms are ordered by
descending x exponent, then descending y exponent:

| Degree | 2D order | 3D order |
|---:|---|---|
| 0 | \(1\) | \(1\) |
| 1 | \(X, Y\) | \(X, Y, Z\) |
| 2 | \(X^2, XY, Y^2\) | \(X^2, XY, XZ, Y^2, YZ, Z^2\) |
| 3 | \(X^3, X^2Y, XY^2, Y^3\) | \(X^3, X^2Y, X^2Z, XY^2, XYZ, XZ^2, Y^3, Y^2Z, YZ^2, Z^3\) |

For degree \(d\), a 2D row has \(d+1\) coefficients and a 3D row has
\((d+1)(d+2)/2\). Therefore, \(N\) degree levels contain \(N(N+1)/2\)
coefficients in 2D or \(N(N+1)(N+2)/6\) coefficients in 3D.

Here, `number_of_degree_levels = N` means degrees `0` through `N-1`; it does
not mean the total number of monomials.

See
[cube_polynomial_2d_case_params.txt](examples/cube_polynomial_2d_case_params.txt)
and
[cube_polynomial_3d_case_params.txt](examples/cube_polynomial_3d_case_params.txt)
for complete files.

## Ensemble range configuration

An ensemble uses two JSON files:

1. A common configuration containing the PDE, dimension, resolution, and
   case name.
2. A case-specific range configuration referenced from the common file:

```json
{
  "case_type": "cube_exponential",
  "case_params_ranges_file": "poisson_exp_ranges.json"
}
```

A relative `case_params_ranges_file` path is resolved relative to the common
JSON file. The ensemble driver samples the ranges and writes a local
`case_params.txt` for each simulation.

Every range is a two-element `[lower, upper]` array. Integer count ranges are
sampled inclusively. Floating-point bounds must be finite and ordered.
Writing identical endpoints, such as `"coeff_bounds": [1.0, 1.0]`, keeps that
parameter constant across the entire ensemble.

Repeated values are sampled independently: individual centers, coordinates,
widths, Fourier coefficients, polynomial center coordinates, and polynomial
monomial coefficients do not share one sampled value merely because they use
the same bounds.

### Exponential ranges

```json
{
  "num_centers_range": [1, 5],
  "coordinate_bounds": [-1.0, 1.0],
  "coeff_bounds": [0.1, 0.9],
  "width_bounds": [0.2, 0.7]
}
```

- `num_centers_range`: Inclusive positive-integer center-count range.
- `coordinate_bounds`: Bounds shared by every x, y, and z coordinate; they
  must lie within `[-1, 1]`.
- `coeff_bounds`: Bounds shared by every center coefficient.
- `width_bounds`: Positive bounds shared by every center width.

See [poisson_exp.json](../scripts/examples/poisson_exp.json) and
[poisson_exp_ranges.json](../scripts/examples/poisson_exp_ranges.json).

### Fourier ranges

```json
{
  "num_modes_range": [1, 5],
  "a_bounds": [-1.0, 1.0],
  "b_bounds": [-1.0, 1.0],
  "constant_bounds": [0.5, 1.5],
  "wavelength_bounds": [0.25, 1.0]
}
```

- `num_modes_range`: Inclusive positive-integer mode-count range.
- `a_bounds`: Bounds shared by cosine coefficients.
- `b_bounds`: Bounds shared by sine coefficients.
- `constant_bounds`: Bounds for the single constant term.
- `wavelength_bounds`: Positive fundamental-wavelength bounds.

See [poisson_fourier.json](../scripts/examples/poisson_fourier.json) and
[poisson_fourier_ranges.json](../scripts/examples/poisson_fourier_ranges.json).

### Polynomial ranges

```json
{
  "num_terms_range": [1, 5],
  "center_coordinate_bounds": [-0.5, 0.5],
  "coeff_bounds": [-1.0, 1.0]
}
```

- `num_terms_range`: Inclusive positive-integer range of degree levels. Despite
  its public name, this is not a total monomial count.
- `center_coordinate_bounds`: Bounds shared by the three center coordinates.
- `coeff_bounds`: Bounds shared by every monomial coefficient.

See [poisson_polynomial.json](../scripts/examples/poisson_polynomial.json) and
[poisson_polynomial_ranges.json](../scripts/examples/poisson_polynomial_ranges.json).
