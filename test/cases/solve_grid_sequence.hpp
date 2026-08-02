#ifndef PARAMSIM_TEST_CASES_SOLVE_GRID_SEQUENCE_HPP_
#define PARAMSIM_TEST_CASES_SOLVE_GRID_SEQUENCE_HPP_

#include <memory>
#include <string>

#include <boost/program_options/variables_map.hpp>

#include "../../pdes/pdebase.hpp"
#include "../../solvers/grid_refinement_solve.hpp"
#include "../../solvers/newton.hpp"

namespace paramsim {
namespace test {

constexpr double nonlinear_tolerance = 1e-6;
constexpr int poisson_maximum_newton_iterations = 2;
constexpr int minimal_surface_maximum_newton_iterations = 10;

/**
 * @brief Solves a cube case on a sequence of uniformly refined grids.
 *
 * @tparam dim Active spatial dimension.
 * @tparam Case Cube case implementation supplying boundary and source data.
 * @tparam PDE Discrete PDE implementation to solve.
 * @param initial_resolution Number of cells per direction on the first grid.
 * @param grid_count Number of grids in the refinement sequence.
 * @param pde_name PDE name stored in its runtime parameters.
 * @param maximum_iterations Maximum Newton iterations on each grid.
 * @return Nonlinear residual norm following the final solve.
 */
template <int dim, template <int> class Case, template <int> class PDE>
double solve_grid_sequence(const unsigned initial_resolution,
                           const int grid_count, const std::string& pde_name,
                           const int maximum_iterations,
                           const std::string init_pde_name = "default")
{
    auto test_case = std::make_shared<Case<dim>>();
    test_case->initialize(boost::program_options::variables_map{});

    const PDEParams pde_parameters{
        pde_name, 1, initial_resolution, grid_count, false, "_",
    };
    const SolverParams solver_parameters{
        nonlinear_tolerance,
        maximum_iterations,
    };
    auto discrete_pde = std::make_shared<PDE<dim>>(test_case, pde_parameters);
    solver::NewtonSolver nonlinear_solver(discrete_pde, solver_parameters);
    const SolverParams initialization_solver_parameters{
        nonlinear_tolerance,
        poisson_maximum_newton_iterations,
    };

    auto solution = solver::run_grid_refinement<dim>(
        pde_parameters, solver_parameters, test_case, discrete_pde,
        init_pde_name == "default" ? solver::default_init_pde(pde_name)
                                   : init_pde_name,
        initialization_solver_parameters);

    DiscretePDEBase::vector_type residual;
    discrete_pde->allocate_solution_vector(residual);
    discrete_pde->evaluate_residual(solution, residual);
    discrete_pde->apply_zero_boundary_values(residual);
    return discrete_pde->compute_lp_norm(residual, 2);
}

} // namespace test
} // namespace paramsim

#endif
