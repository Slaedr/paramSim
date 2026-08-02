#include <gtest/gtest.h>

#include "../../cases/cube/exponential.hpp"
#include "../../pdes/nonlinear_elliptic/minimal_surface.hpp"
#include "../../pdes/poisson/poisson_cg.hpp"
#include "solve_grid_sequence.hpp"

namespace {

using namespace paramsim::test;

TEST(InitializationPDE, HasPerPDEDefaults)
{
    EXPECT_EQ(paramsim::solver::default_init_pde("minimal_surface"),
              "poisson_cg");
    EXPECT_EQ(paramsim::solver::default_init_pde("poisson_cg"),
              paramsim::solver::none_init_pde);
}

TEST(CubeExponentialPoisson, SolverConvergesWithP1In2D)
{
    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubeExponential,
                                   paramsim::pde::PoissonCG>(
                  32, 3, "poisson_cg", poisson_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeExponentialPoisson, SolverConvergesWithP1In3D)
{
    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubeExponential,
                                   paramsim::pde::PoissonCG>(
                  8, 3, "poisson_cg", poisson_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeExponentialMinimalSurface, SolverConvergesWithP1In2D)
{
    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubeExponential,
                                   paramsim::pde::MinimalSurface>(
                  16, 3, "minimal_surface",
                  minimal_surface_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeExponentialMinimalSurface, SolverConvergesWithP1In3D)
{
    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubeExponential,
                                   paramsim::pde::MinimalSurface>(
                  8, 3, "minimal_surface",
                  minimal_surface_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeExponentialMinimalSurface,
     DefaultInitialGuessMatchesExplicitPoissonIn2D)
{
    const double default_residual =
        solve_grid_sequence<2, paramsim::cases::cube::CubeExponential,
                            paramsim::pde::MinimalSurface>(
            16, 1, "minimal_surface", minimal_surface_maximum_newton_iterations);
    const double explicit_residual =
        solve_grid_sequence<2, paramsim::cases::cube::CubeExponential,
                            paramsim::pde::MinimalSurface>(
            16, 1, "minimal_surface", minimal_surface_maximum_newton_iterations,
            "none");

    EXPECT_NEAR(default_residual, explicit_residual, 1e-8);
    EXPECT_LT(default_residual, nonlinear_tolerance);
}

} // namespace
