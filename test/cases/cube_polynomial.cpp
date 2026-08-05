#include <gtest/gtest.h>

#include "../../cases/cube/polynomial.hpp"
#include "../../pdes/nonlinear_elliptic/gelfand.hpp"
#include "../../pdes/nonlinear_elliptic/minimal_surface.hpp"
#include "../../pdes/poisson/poisson_cg.hpp"
#include "../utils/temporary_parameter_file.hpp"
#include "solve_grid_sequence.hpp"

namespace {

using namespace paramsim::test;

constexpr char gelfand_parameters_2d[] =
    "4\n"
    "0.1 -0.2 0.3\n"
    "-0.25\n"
    "1.2 0.5\n"
    "-0.7 0.9 -0.4\n"
    "0.6 -1.1 0.3 0.8\n";

constexpr char gelfand_parameters_3d[] =
    "4\n"
    "0.1 -0.2 0.3\n"
    "-0.25\n"
    "1.2 0.5 -0.8\n"
    "-0.7 0.9 -0.4 0.6 -1.1 0.3\n"
    "0.8 -0.6 1.0 -0.5 0.7 -0.9 0.4 1.1 -0.3 0.55\n";

TEST(CubePolynomialPoisson, SolverConvergesWithP1In2D)
{
    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubePolynomial,
                                   paramsim::pde::PoissonCG>(
                  32, 3, "poisson_cg", poisson_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubePolynomialPoisson, SolverConvergesWithP1In3D)
{
    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubePolynomial,
                                   paramsim::pde::PoissonCG>(
                  8, 3, "poisson_cg", poisson_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubePolynomialGelfand, SolverConvergesWithP1In2D)
{
    const TemporaryParameterFile parameter_file(gelfand_parameters_2d);

    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubePolynomial,
                                   paramsim::pde::Gelfand>(
                  16, 3, "gelfand", 12, "default", parameter_file.path())),
              nonlinear_tolerance);
}

TEST(CubePolynomialGelfand, SolverConvergesWithP1In3D)
{
    const TemporaryParameterFile parameter_file(gelfand_parameters_3d);

    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubePolynomial,
                                   paramsim::pde::Gelfand>(
                  8, 3, "gelfand", 12, "default", parameter_file.path())),
              nonlinear_tolerance);
}

TEST(CubePolynomialMinimalSurface, SolverConvergesWithP1In2D)
{
    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubePolynomial,
                                   paramsim::pde::MinimalSurface>(
                  16, 3, "minimal_surface",
                  minimal_surface_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubePolynomialMinimalSurface, SolverConvergesWithP1In3D)
{
    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubePolynomial,
                                   paramsim::pde::MinimalSurface>(
                  16, 2, "minimal_surface",
                  minimal_surface_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubePolynomialMinimalSurface,
     DefaultInitialGuessMatchesExplicitPoissonIn2D)
{
    const double default_residual =
        solve_grid_sequence<2, paramsim::cases::cube::CubePolynomial,
                            paramsim::pde::MinimalSurface>(
            8, 2, "minimal_surface", 15);
    const double explicit_residual =
        solve_grid_sequence<2, paramsim::cases::cube::CubePolynomial,
                            paramsim::pde::MinimalSurface>(
            8, 2, "minimal_surface", 15, "none");

    EXPECT_NEAR(default_residual, explicit_residual, 1e-9);
    EXPECT_LT(default_residual, nonlinear_tolerance);
}

} // namespace
