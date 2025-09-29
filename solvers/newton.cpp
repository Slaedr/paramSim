#include "newton.hpp"

#include <cmath>
#include <algorithm>
#include <type_traits>

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

namespace paramsim {
namespace solver {


void NewtonSolver::linear_solve(const int i_iter)
{
    const auto max_its = std::min(1000, static_cast<int>(i_max_its_*std::pow(r_base_, i_iter)));
    const double i_tol_exp = std::log10(i_tol_);
    const double tol = std::max(1e-12, std::pow(10, i_tol_exp*std::pow(r_base_, i_iter)));

    dealii::SolverControl solver_control (max_its, tol);
    if(lstype_ == lin_sys_type::spd) {
        dealii::SolverCG<vector_type> solver(solver_control);
        dealii::PreconditionSSOR<matrix_type> prec;
        prec.initialize(system_matrix_, 1.1);
        solver.solve(system_matrix_, du_, rhs_, prec);
    } else {
        dealii::SolverGMRES<vector_type> solver(
            solver_control, dealii::SolverGMRES<vector_type>::AdditionalData{30});
        dealii::PreconditionSOR<matrix_type> prec;
        prec.initialize(system_matrix_, 1.0);
        solver.solve(system_matrix_, du_, rhs_, prec);
    }
}


NewtonSolver::NewtonSolver(std::shared_ptr<const DiscretePDEBase> pde, const SolverParams& params)
    : pde_{pde}, sparams_{params}, lstype_{pde->is_symm_positive_definite() ? lin_sys_type::spd :
                                           lin_sys_type::gen}
{
    static_assert(std::is_same_v<typename DiscretePDEBase::vector_type, vector_type>,
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

void NewtonSolver::solve(vector_type& u)
{
    double cur_norm = 1.0;
    pde_->set_boundary_values(u);

    for(int i_iter = 0; i_iter < sparams_.max_its; i_iter++) {
        std::cout << "  Newton: iteration " << i_iter << std::endl;
        pde_->assemble_system(AssemblyOptions{false}, u, system_matrix_, rhs_);
        pde_->apply_zero_boundary_values(du_, system_matrix_, rhs_);
        cur_norm = pde_->compute_lp_norm(rhs_, 2);
        if(cur_norm < sparams_.tolerance) {
            std::cout << "  Newton: converged." << std::endl;
            break;
        }
        linear_solve(i_iter);
        pde_->impose_constraints(du_);
        const double alpha = determine_step_length(cur_norm);
        u.add(alpha, du_);
    }
}

double NewtonSolver::determine_step_length(const double rnorm_0) const
{
    constexpr int max_its = 10;
    double steplen = 1.0;
    for(int i = 0; i < max_its; i++) {
        //
    }
    return steplen;
}

}
}
