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


/// Outcome of a nonlinear solve.
struct NewtonResult {
    /// Whether the residual norm reached SolverParams::tolerance.
    bool converged;
    /// Number of nonlinear iterations performed.
    int iterations;
    /// Residual norm of the returned state.
    double final_residual_norm;
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
     * If no acceptable step can be found, the iteration stops rather than
     * applying a step that increases the residual; the returned result reports
     * that the solve did not converge.
     *
     * @param u  Initial solution guess; overwritten with the final state.
     */
    NewtonResult solve(vector_type& u);

protected:
    /// Outcome of the backtracking line search.
    struct StepSearchResult {
        /// Accepted step length; zero when no acceptable step was found.
        double lambda;
        /// Whether a step satisfying the sufficient-decrease condition was found.
        bool success;
        /// Residual norm at the accepted step; the starting norm on failure.
        double norm;
    };

    std::shared_ptr<const DiscretePDEBase> pde_;
    SolverParams sparams_;
    int i_max_its_{500};
    /// Hard ceiling on linear iterations, whatever the schedule below asks for.
    int max_linear_its_{5000};
    /** Forcing term of the first nonlinear iteration.
     *
     * The linear solve tolerance is this fraction of the current residual
     * norm, tightened geometrically as the Newton iteration proceeds. It is
     * relative rather than absolute: an absolute target is either unreachable
     * once the Jacobian is ill-conditioned, or already satisfied on entry --
     * in which case the linear solve silently becomes a no-op.
     */
    double eta_0_{1e-2};
    /// Tightest forcing term, so the linear solves stay affordable.
    double eta_min_{1e-8};
    /// Floor on the absolute tolerance, for when the residual is already tiny.
    double absolute_floor_{1e-14};
    double r_base_{1.2};
    lin_sys_type lstype_;

    /// Maximum number of backtracking steps in the line search.
    int max_backtracks_{5};
    /** Factor by which the step length shrinks per backtracking step.
     *
     * Deliberately shallow: the residual norm can have a sharp minimum in the
     * step length, so sampling it finely matters more than reaching small step
     * lengths quickly.
     */
    double backtrack_factor_{0.66};
    /// Coefficient of the Armijo sufficient-decrease condition.
    double c_armijo_{1e-4};

    vector_type du_;
    vector_type rhs_;
    dealii::SparsityPattern sparsity_pattern_;
    matrix_type system_matrix_;

    /** Solve the linear system at a given nonlinear iteration.
     *
     * @return  False if the linear solver stopped without reducing the linear
     *   residual at all, in which case du_ is not a usable direction.
     */
    bool linear_solve(int i_iter);

    // Determine nonlinear update step length, given the current residual norm.
    StepSearchResult determine_step_length(const vector_type& state, double orig_norm_2);
};


}
}


#endif // PARAMSIM_SOLVERS_NEWTON_H
