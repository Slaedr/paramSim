#include <memory>
#include <string>

#include <boost/program_options/variables_map.hpp>
#include <gtest/gtest.h>

#include "../../cases/cube/exponential.hpp"
#include "../../pdes/nonlinear_elliptic/minimal_surface.hpp"
#include "../../pdes/poisson/poisson_cg.hpp"
#include "../../solvers/newton.hpp"

namespace {

constexpr double nonlinear_tolerance = 1e-6;
constexpr int poisson_maximum_newton_iterations = 2;
constexpr int minimal_surface_maximum_newton_iterations = 10;

enum class InitialGuess {
    zero,
    poisson,
};

/**
 * @brief Computes a Poisson solution to use as a nonlinear initial guess.
 *
 * Interestingly, keeping the RHS and BCs the same, initializing a minimal
 * surface solve with a Poisson solution is a much better starting point than
 * all zeros.
 *
 * @tparam dim Active spatial dimension.
 * @param test_case Cube exponential case shared with the nonlinear PDE.
 * @param pde_parameters Grid and finite-element parameters.
 * @param solution Destination for the computed Poisson solution.
 */
template <int dim>
void initialize_with_poisson(
    const std::shared_ptr<paramsim::cases::cube::CubeExponential<dim>>&
        test_case,
    const paramsim::PDEParams& pde_parameters,
    paramsim::DiscretePDEBase::vector_type& solution)
{
    paramsim::PDEParams poisson_parameters = pde_parameters;
    poisson_parameters.pde_solver = "poisson_cg";
    poisson_parameters.refine_levels = 1;

    auto poisson_pde = std::make_shared<paramsim::pde::PoissonCG<dim>>(
        test_case, poisson_parameters);
    const paramsim::SolverParams poisson_solver_parameters{
        nonlinear_tolerance,
        poisson_maximum_newton_iterations,
    };
    paramsim::solver::NewtonSolver poisson_solver(poisson_pde,
                                                  poisson_solver_parameters);
    poisson_pde->allocate_solution_vector(solution);
    poisson_solver.solve(solution);
}

/**
 * @brief Solves a cube exponential PDE problem on three uniform grids.
 *
 * @tparam dim Active spatial dimension.
 * @tparam PDE Discrete PDE implementation to solve.
 * @param initial_resolution Number of cells per direction on the first grid.
 * @param pde_name PDE name stored in its runtime parameters.
 * @param maximum_iterations Maximum Newton iterations on each grid.
 * @param initial_guess Method used to initialize the first-grid solution.
 * @return Nonlinear residual norm following the final solve.
 */
template <int dim, template <int> class PDE>
double solve_three_grid_sequence(const unsigned initial_resolution,
                                 const int grid_count,
                                 const std::string& pde_name,
                                 const int maximum_iterations,
                                 const InitialGuess initial_guess)
{
    auto test_case =
        std::make_shared<paramsim::cases::cube::CubeExponential<dim>>();
    test_case->initialize(boost::program_options::variables_map{});

    const paramsim::PDEParams pde_parameters{
        pde_name, 1, initial_resolution, grid_count, false, "_",
    };
    const paramsim::SolverParams solver_parameters{
        nonlinear_tolerance,
        maximum_iterations,
    };
    auto discrete_pde = std::make_shared<PDE<dim>>(test_case, pde_parameters);
    paramsim::solver::NewtonSolver solver(discrete_pde, solver_parameters);

    paramsim::DiscretePDEBase::vector_type solution;
    discrete_pde->allocate_solution_vector(solution);
    if (initial_guess == InitialGuess::poisson) {
        initialize_with_poisson(test_case, pde_parameters, solution);
    }

    double final_residual = 0.0;

    for (int grid = 0; grid < grid_count; ++grid) {
        solver.reinit();
        solver.solve(solution);

        paramsim::DiscretePDEBase::vector_type residual;
        discrete_pde->allocate_solution_vector(residual);
        discrete_pde->evaluate_residual(solution, residual);
        discrete_pde->apply_zero_boundary_values(residual);
        final_residual = discrete_pde->compute_lp_norm(residual, 2);

        if (grid + 1 < grid_count) {
            discrete_pde->refine_mesh_and_interpolate_solution(solution);
        }
    }

    return final_residual;
}

TEST(CubeExponentialPoisson, ConvergesWithP1In2D)
{
    EXPECT_LT((solve_three_grid_sequence<2, paramsim::pde::PoissonCG>(
                  32, 3, "poisson_cg", poisson_maximum_newton_iterations,
                  InitialGuess::zero)),
              nonlinear_tolerance);
}

TEST(CubeExponentialPoisson, ConvergesWithP1In3D)
{
    EXPECT_LT((solve_three_grid_sequence<3, paramsim::pde::PoissonCG>(
                  16, 3, "poisson_cg", poisson_maximum_newton_iterations,
                  InitialGuess::zero)),
              nonlinear_tolerance);
}

TEST(CubeExponentialMinimalSurface, ConvergesWithP1In2D)
{
    EXPECT_LT(
        (solve_three_grid_sequence<2, paramsim::pde::MinimalSurface>(
            16, 3, "minimal_surface", minimal_surface_maximum_newton_iterations,
            InitialGuess::poisson)),
        nonlinear_tolerance);
}

TEST(CubeExponentialMinimalSurface, ConvergesWithP1In3D)
{
    EXPECT_LT(
        (solve_three_grid_sequence<3, paramsim::pde::MinimalSurface>(
            8, 3, "minimal_surface", minimal_surface_maximum_newton_iterations,
            InitialGuess::poisson)),
        nonlinear_tolerance);
}

} // namespace
