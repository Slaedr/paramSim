#ifndef PARAMSIM_SOLVERS_NEWTON_H
#define PARAMSIM_SOLVERS_NEWTON_H

#include <memory>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include "../pdes/pdebase.hpp"


namespace paramsim {
namespace solver {


enum class lin_sys_type {
    gen,
    spd
};


class NewtonSolver
{
public:
    using vector_type = dealii::Vector<double>;
    using matrix_type = dealii::SparseMatrix<double>;

    NewtonSolver(std::shared_ptr<const DiscretePDEBase> pde, const SolverParams& params);

    /// Re-allocate matrix and vectors after a mesh change.
    void reinit();

    /**
     * Solve nonlinear system starting with an initial guess.
     *
     * @param u  Initial solution guess.
     */
    void solve(vector_type& u);

protected:
    std::shared_ptr<const DiscretePDEBase> pde_;
    SolverParams sparams_;
    int i_max_its_{500};
    double i_tol_{1e-8};
    double r_base_{1.1};
    lin_sys_type lstype_;

    vector_type du_;
    vector_type rhs_;
    dealii::SparsityPattern sparsity_pattern_;
    matrix_type system_matrix_;

    // Solve linear system at a given nonlinear iteration.
    void linear_solve(int i_iter);

    // Determine nonlinear update step length, given the current residual norm.
    double determine_step_length(double orig_norm_2) const;
};


}
}


#endif // PARAMSIM_SOLVERS_NEWTON_H
