#include "newton.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

namespace paramsim {
namespace solver {

void NewtonSolver::linear_solve(const int i_iter)
{
    // Start every linear solve from zero. The iterative solvers below treat the
    // incoming vector as an initial guess, and du_ otherwise still holds the
    // previous Newton iteration's direction.
    du_ = 0;

    const auto max_its = std::min(
        1000, static_cast<int>(i_max_its_ * std::pow(r_base_, i_iter)));
    const double i_tol_exp = std::log10(i_tol_);
    const double tol =
        std::max(1e-10, std::pow(10, i_tol_exp * std::pow(r_base_, i_iter)));

    dealii::SolverControl solver_control(max_its, tol);
    if (lstype_ == lin_sys_type::spd) {
        dealii::SolverCG<vector_type> solver(solver_control);
        dealii::PreconditionSSOR<matrix_type> prec;
        prec.initialize(system_matrix_, 1.1);
        solver.solve(system_matrix_, du_, rhs_, prec);
    } else {
        dealii::SolverGMRES<vector_type> solver(
            solver_control,
            dealii::SolverGMRES<vector_type>::AdditionalData{30});
        dealii::PreconditionSOR<matrix_type> prec;
        prec.initialize(system_matrix_, 1.0);
        solver.solve(system_matrix_, du_, rhs_, prec);
    }
    std::cout << "  Newton: linear solver: converged in "
              << solver_control.last_step() << " iterations." << std::endl;
}

NewtonSolver::NewtonSolver(std::shared_ptr<const DiscretePDEBase> pde,
                           const SolverParams& params)
    : pde_{pde}, sparams_{params},
      lstype_{pde->is_symm_positive_definite() ? lin_sys_type::spd
                                               : lin_sys_type::gen}
{
    static_assert(
        std::is_same_v<typename DiscretePDEBase::vector_type, vector_type>,
        "Inconsistent vector types between PDE and solver!");
    reinit();
}

void NewtonSolver::reinit()
{
    // initialize vectors and sparsity pattern. Zero the vectors.
    pde_->setup_system(false, rhs_, du_, sparsity_pattern_);
    // initialize system matrix and zero its values
    system_matrix_.reinit(sparsity_pattern_);
}

NewtonResult NewtonSolver::solve(vector_type& u)
{
    double cur_norm = 1.0;
    int i_iter = 0;
    pde_->set_boundary_values(u);

    for (i_iter = 0; i_iter < sparams_.max_its; i_iter++) {
        std::cout << "  Newton: iteration " << i_iter << ", ";

        // assemble Jacobian and residual
        pde_->assemble_system(AssemblyOptions{false}, u, system_matrix_, rhs_);

        // apply boundary conditions to system before norm computation and
        // linear solve
        pde_->apply_zero_boundary_values(du_, system_matrix_, rhs_);

        // compute and report residual norm
        cur_norm = pde_->compute_lp_norm(rhs_, 2);
        std::cout << "current norm = " << cur_norm << std::endl;

        // check convergence
        if (cur_norm < sparams_.tolerance) {
            std::cout << "  Newton: converged." << std::endl;
            return {true, i_iter, cur_norm};
        }

        // linear solve
        linear_solve(i_iter);
        pde_->impose_constraints(du_);

        // update, but only along a direction that actually reduces the residual
        const auto step = determine_step_length(u, cur_norm);
        if (!step.success) {
            std::cout << "  Newton: no acceptable step found; stopping at a "
                         "residual norm of "
                      << cur_norm << "." << std::endl;
            return {false, i_iter, cur_norm};
        }
        u.add(step.lambda, du_);
        cur_norm = step.norm;
    }

    // The final step of the loop above may itself have reached the tolerance,
    // in which case the state being returned is converged even though the
    // iteration ran out of its budget before it could re-check.
    if (cur_norm < sparams_.tolerance) {
        std::cout << "  Newton: converged." << std::endl;
        return {true, i_iter, cur_norm};
    }

    std::cout << "  Newton: reached the maximum of " << sparams_.max_its
              << " iterations without converging." << std::endl;
    return {false, i_iter, cur_norm};
}

NewtonSolver::StepSearchResult
NewtonSolver::determine_step_length(const vector_type& u, const double rnorm_0)
{
    /* Backtracking line search with an Armijo-type sufficient-decrease
     * condition. Only step lengths that have actually been evaluated are ever
     * returned, and a search that finds nothing reports failure instead of
     * handing back a step that increases the residual.
     * Can use CP line search from Brune et al., SIAM Review, 2025.
     */
    double lambda = 1.0;
    dealii::Vector<scalar_type> y(u.size());
    for (int i = 0; i < max_backtracks_; i++) {
        // compute new point: y <- du
        y = du_;
        // y <- u + ly
        y.sadd(lambda, u);

        pde_->evaluate_residual(y, rhs_);
        pde_->apply_zero_boundary_values(rhs_);

        const double trial_norm = pde_->compute_lp_norm(rhs_, 2);
        std::cout << "  Newton:     line search: step length " << lambda
                  << ", current norm = " << trial_norm << std::endl;

        if (trial_norm <= (1.0 - c_armijo_ * lambda) * rnorm_0) {
            std::cout << "  Newton:   step length: " << lambda << std::endl;
            return {lambda, true, trial_norm};
        }
        lambda *= backtrack_factor_;
    }

    std::cout << "  Newton: Line search failed! No step length down to "
              << lambda << " gave sufficient decrease." << std::endl;
    return {0.0, false, rnorm_0};
}

} // namespace solver
} // namespace paramsim
