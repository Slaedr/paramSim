#include "grid_refinement_solve.hpp"

#include <iostream>
#include <stdexcept>

#include <deal.II/fe/fe_q.h>

#include "newton.hpp"

namespace paramsim {
namespace solver {

template <int dim>
DiscretePDEBase::vector_type
run_grid_refinement(const PDEParams& params, const SolverParams& sparams,
                    std::shared_ptr<const Case<dim>> tcase,
                    std::shared_ptr<DiscretePDEBase> pdeb,
                    const std::string& init_pde,
                    const SolverParams& init_solver_params)
{
    auto pde =
        std::dynamic_pointer_cast<DiscretePDE<dim, dealii::FE_Q<dim>>>(pdeb);
    solver::NewtonSolver solver(pde, sparams);
    using vector_type = typename DiscretePDEBase::vector_type;
    vector_type u;
    pde->allocate_solution_vector(u);
    paramsim::solver::compute_initial_guess<dim>(tcase, params, init_pde,
                                                 init_solver_params, u);

    unsigned resolution = params.initial_resolution;
    for (int imesh = 0; imesh < params.refine_levels;
         imesh++, resolution *= 2) {
        solver.reinit();
        const auto result = solver.solve(u);
        pde->output_results(imesh, u);
        if (!result.converged) {
            // On an intermediate level a partially converged state is still a
            // usable starting point for the next, finer grid. On the final
            // level it is the answer, and an unconverged answer must not be
            // mistaken for data.
            if (imesh == params.refine_levels - 1) {
                throw std::runtime_error(
                    "The nonlinear solve did not converge on the final "
                    "refinement level (" +
                    std::to_string(imesh) + "): residual norm " +
                    std::to_string(result.final_residual_norm) + " after " +
                    std::to_string(result.iterations) + " iterations.");
            }
            std::cout << "WARNING: the nonlinear solve did not converge on "
                         "refinement level "
                      << imesh << " (residual norm "
                      << result.final_residual_norm << "). Continuing to the "
                      << "next level." << std::endl;
        }
        if (imesh < params.refine_levels - 1) {
            pde->refine_mesh_and_interpolate_solution(u);
        }
    }

    return u;
}

template DiscretePDEBase::vector_type run_grid_refinement<2>(
    const PDEParams& params, const SolverParams& sparams,
    std::shared_ptr<const Case<2>> tcase, std::shared_ptr<DiscretePDEBase> pdeb,
    const std::string& init_pde, const SolverParams& init_solver_params);
template DiscretePDEBase::vector_type run_grid_refinement<3>(
    const PDEParams& params, const SolverParams& sparams,
    std::shared_ptr<const Case<3>> tcase, std::shared_ptr<DiscretePDEBase> pdeb,
    const std::string& init_pde, const SolverParams& init_solver_params);

std::string default_init_pde(const std::string& pde_solver)
{
    if (pde_solver == "minimal_surface") {
        return "poisson_cg";
    } else {
        return none_init_pde;
    }
}

template <int dim>
void compute_initial_guess(std::shared_ptr<const Case<dim>> test_case,
                           const PDEParams& params,
                           const std::string& requested_init_pde,
                           const SolverParams& init_solver_params,
                           DiscretePDEBase::vector_type& solution)
{
    const std::string init_pde = requested_init_pde.empty()
                                     ? default_init_pde(params.pde_solver)
                                     : requested_init_pde;

    if (init_pde == none_init_pde) {
        std::cout << "No initialization solve for PDE '" << params.pde_solver
                  << "'." << std::endl;
        return;
    }
    if (init_pde == params.pde_solver) {
        std::cout << "Skipping initialization solve: the initialization PDE '"
                  << init_pde << "' is the PDE being solved." << std::endl;
        return;
    }

    std::cout << "Initializing the '" << params.pde_solver << "' solve with a '"
              << init_pde << "' solution." << std::endl;

    PDEParams init_params = params;
    init_params.pde_solver = init_pde;

    std::shared_ptr<DiscretePDEBase> init_pdeb =
        create_discrete_pde(test_case, init_params);

    DiscretePDEBase::vector_type init_solution;
    init_pdeb->allocate_solution_vector(init_solution);

    if (init_solution.size() != solution.size()) {
        throw std::runtime_error(
            "Initialization PDE '" + init_pde + "' has " +
            std::to_string(init_solution.size()) +
            " DoFs, but the PDE being solved, '" + params.pde_solver +
            "', has " + std::to_string(solution.size()) +
            ". The initialization solution cannot be transferred.");
    }

    NewtonSolver init_solver(init_pdeb, init_solver_params);
    init_solver.solve(init_solution);

    solution = init_solution;
}

template void compute_initial_guess<2>(std::shared_ptr<const Case<2>>,
                                       const PDEParams&, const std::string&,
                                       const SolverParams&,
                                       DiscretePDEBase::vector_type&);
template void compute_initial_guess<3>(std::shared_ptr<const Case<3>>,
                                       const PDEParams&, const std::string&,
                                       const SolverParams&,
                                       DiscretePDEBase::vector_type&);

} // namespace solver
} // namespace paramsim
