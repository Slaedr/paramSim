#ifndef PARAMSIM_SOLVERS_GRID_REFINEMENT_SOLVE_HPP_
#define PARAMSIM_SOLVERS_GRID_REFINEMENT_SOLVE_HPP_

#include <memory>
#include <string>

#include "../cases/case.hpp"
#include "../pdes/pdebase.hpp"

namespace paramsim {
namespace solver {

/**
 * Solves a (nonlinear) discrete PDE on a sequence of grids.
 *
 * @tparam dim  The spatial dimensionality of the PDE solve.
 *
 * @param params  PDE discretization parameters.
 * @param sparams Solver parameters for the main PDE.
 * @param tcase  The test case to be solve (RHS, BCs).
 * @param pdeb  The discrete PDE object.
 * @param init_pde  Name of a PDE to solve to initialize the coarse grid
 *                  solution. Can be "none" for zero-initialization.
 *                  @see default_init_pde
 * @param init_solver_params  Solver parameters for the initialization PDE.
 *
 * @return  The computed solution on the finest grid.
 */
template <int dim>
DiscretePDEBase::vector_type
run_grid_refinement(const PDEParams& params, const SolverParams& sparams,
                    std::shared_ptr<const Case<dim>> tcase,
                    std::shared_ptr<DiscretePDEBase> pdeb,
                    const std::string& init_pde,
                    const SolverParams& init_solver_params);

/// The value of an initialization PDE name that disables the initialization
/// solve.
constexpr const char *none_init_pde = "none";

/**
 * Returns the PDE whose solution makes a good initial guess for the given PDE.
 *
 * Interestingly, keeping the RHS and BCs the same, initializing a minimal
 * surface solve with a Poisson solution is a much better starting point than
 * all zeros.
 *
 * @param pde_solver  Name of the PDE that is to be solved.
 *
 * @return Name of the PDE to solve first, or @ref none_init_pde if solving a
 *   simpler PDE first is not worthwhile.
 */
std::string default_init_pde(const std::string& pde_solver);

/**
 * Computes an initial guess for the main solve by solving a simpler PDE first.
 *
 * The initialization PDE is solved on the initial grid using the same case, and
 * therefore the same right-hand side, geometry and Dirichlet boundary
 * conditions, as the main PDE. It is skipped if the requested PDE resolves to
 * @ref none_init_pde or to the main PDE itself.
 *
 * @tparam dim  Active spatial dimension.
 *
 * @param test_case  Case shared with the main PDE.
 * @param params  Parameters of the main PDE; only the PDE name is overridden.
 * @param requested_init_pde  Name of the initialization PDE. If empty, the
 *   default for @c params.pde_solver is used.
 * @param init_solver_params  Convergence settings for the initialization solve.
 * @param solution  Solution vector of the main PDE, overwritten with the
 *   initialization solution. Must already be allocated by the main PDE.
 */
template <int dim>
void compute_initial_guess(std::shared_ptr<const Case<dim>> test_case,
                           const PDEParams& params,
                           const std::string& requested_init_pde,
                           const SolverParams& init_solver_params,
                           DiscretePDEBase::vector_type& solution);

} // namespace solver
} // namespace paramsim

#endif // PARAMSIM_SOLVERS_GRID_REFINEMENT_SOLVE_HPP_
