#include <gtest/gtest.h>

#include "../../cases/cube/fourier.hpp"
#include "../../pdes/nonlinear_elliptic/minimal_surface.hpp"
#include "../../pdes/poisson/poisson_cg.hpp"
#include "solve_grid_sequence.hpp"

namespace {

using namespace paramsim::test;

TEST(CubeFourierPoisson, SolverConvergesWithP1In2D)
{
    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::PoissonCG>(
                  32, 3, "poisson_cg", poisson_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeFourierPoisson, SolverConvergesWithP1In3D)
{
    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::PoissonCG>(
                  8, 3, "poisson_cg", poisson_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeFourierMinimalSurface, SolverConvergesWithP1In2D)
{
    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::MinimalSurface>(
                  16, 3, "minimal_surface",
                  minimal_surface_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeFourierMinimalSurface, SolverConvergesWithP1In3D)
{
    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::MinimalSurface>(
                  16, 2, "minimal_surface",
                  minimal_surface_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeFourierMinimalSurface, DefaultInitialGuessMatchesExplicitPoissonIn2D)
{
    const double default_residual =
        solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                            paramsim::pde::MinimalSurface>(
            4, 2, "minimal_surface", 15);
    const double explicit_residual =
        solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                            paramsim::pde::MinimalSurface>(
            4, 2, "minimal_surface", 15);

    EXPECT_DOUBLE_EQ(default_residual, explicit_residual);
    EXPECT_LT(default_residual, nonlinear_tolerance);
}

} // namespace
