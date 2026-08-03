# Cube case parameters

The `cube_exponential`, `cube_fourier`, and `cube_polynomial` cases apply a
configurable Dirichlet profile, evaluated over the cube's full coordinate
range, as the boundary condition on every face of the cube.

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

The profile evaluates normalized Gaussians using every active coordinate.
The z coordinate is ignored in 2D and participates in the distance and
normalization in 3D.

See
[cube_exponential_case_params.txt](examples/cube_exponential_case_params.txt)
for a complete file.

### Fourier parameters

In 2D, the `cube_fourier` format is:

```text
<number_of_modes>
<constant> <wavelength_x> <wavelength_y>
<cc_1> <cs_1> <sc_1> <ss_1>
...one row per mode
```

In 3D, the header has one additional directional wavelength:

```text
<number_of_modes>
<constant> <wavelength_x> <wavelength_y> <wavelength_z>
<ccc_1> <ccs_1> <csc_1> <css_1> <scc_1> <scs_1> <ssc_1> <sss_1>
...one row per mode
```

Every directional fundamental wavelength must be positive. For spatial
dimension \(D\), mode \(n\) adds every tensor product

\[
\sum_{\boldsymbol{q}\in\{c,s\}^D} a_{n,\boldsymbol{q}}
\prod_{d=1}^{D} f_{q_d}(2\pi n x_d/\lambda_d),
\qquad f_c(\theta)=\cos(\theta),\quad f_s(\theta)=\sin(\theta).
\]

Coefficients are ordered lexicographically with cosine before sine in each
direction: `CC, CS, SC, SS` in 2D and
`CCC, CCS, CSC, CSS, SCC, SCS, SSC, SSS` in 3D. Thus, each mode row has four
values in 2D and eight values in 3D.

Rows are frequencies `1` through `number_of_modes`; the constant is present
only once and is not a mode row.

The built-in defaults preserve the earlier profile: the all-cosine and
all-sine coefficients are 1 for each of two modes, while all mixed
coefficients are 0.

See the
[2D example](examples/cube_fourier_case_params.txt) and
[3D example](examples/cube_fourier_3d_case_params.txt).

#### Resolving the profile

The cube spans \([-1, 1]^D\), so a fundamental wavelength of \(\lambda\) puts
\(2/\lambda\) periods along each edge, and mode \(n\) multiplies that by \(n\).
The initial grid has to resolve the highest mode: with `--initial_resolution R`
the cell size is \(2/R\) and mode \(n\) has period \(\lambda/n\), giving
\(R\lambda/(2n)\) cells per period.

Under-resolving the profile is not merely inaccurate. On a grid too coarse to
see the oscillation, the solver converges quickly to the solution of an
effectively smoother, aliased problem; the first refinement then exposes the
true profile, and the interpolated coarse solution is a poor starting point for
it. This matters most for `minimal_surface`, whose coefficient
\(1/\sqrt{1+|\nabla u|^2}\) becomes small and strongly varying wherever
boundary gradients are large — the defaults (two modes, unit all-cosine and
all-sine coefficients, \(\lambda = 1\)) reach gradients of order
\(2\pi + 4\pi \approx 19\). The
nonlinear solve can then stall, and `run_case` will report a failure to
converge rather than write an unconverged volume.

With the 3D defaults, `--initial_resolution 16` gives 4 cells per period of the
highest mode and solves cleanly, while `--initial_resolution 8` gives 2 and is
known to stall for `minimal_surface`. Treat 4 cells per period of the highest
mode as the minimum. Longer wavelengths or smaller mode coefficients relax the
requirement.

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

1. A common configuration containing the PDE, dimension, resolution, case
   name, and an optional initialization PDE.
2. A case-specific range configuration referenced from the common file:

```json
{
  "pde": "minimal_surface",
  "dimension": 2,
  "resolution": 63,
  "case_type": "cube_exponential",
  "init_pde": "poisson_cg",
  "case_params_ranges_file": "poisson_exp_ranges.json"
}
```

A common configuration may omit `init_pde` to use the per-PDE default.
Setting it to `"none"` disables the initialization solve. Any other
non-empty value is passed to `run_case` through `--init_pde`.

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
  "coeff_bounds": [-1.0, 1.0],
  "constant_bounds": [0.5, 1.5],
  "wavelength_bounds": [0.25, 1.0]
}
```

- `num_modes_range`: Inclusive positive-integer mode-count range.
- `coeff_bounds`: Bounds shared by all four 2D or eight 3D coefficients in
  every mode. Every coefficient is sampled independently.
- `constant_bounds`: Bounds for the single constant term.
- `wavelength_bounds`: Positive bounds used to sample each active direction's
  fundamental wavelength independently.

See [poisson_fourier.json](../scripts/examples/poisson_fourier.json) and
[poisson_fourier_ranges.json](../scripts/examples/poisson_fourier_ranges.json).

### Polynomial ranges

```json
{
  "num_degree_levels_range": [1, 5],
  "center_coordinate_bounds": [-0.5, 0.5],
  "coeff_bounds": [-1.0, 1.0]
}
```

- `num_degree_levels_range`: Inclusive positive-integer range of polynomial
  degree levels.
- `center_coordinate_bounds`: Bounds shared by the three center coordinates.
- `coeff_bounds`: Bounds shared by every monomial coefficient.

See [poisson_polynomial.json](../scripts/examples/poisson_polynomial.json) and
[poisson_polynomial_ranges.json](../scripts/examples/poisson_polynomial_ranges.json).
