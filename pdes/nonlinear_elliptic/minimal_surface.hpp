#ifndef PARAMSIM_MINIMAL_SURFACE_HPP_
#define PARAMSIM_MINIMAL_SURFACE_HPP_

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>

#include "../pdebase.hpp"
#include "../../cases/case.hpp"

namespace paramsim {
namespace pde {

using namespace dealii;

/** Describes the FEM procedures for the minimal surface equation, a nonlinear elliptic PDE.
 *
 * Adapted from DEAL.II's step 15 example.
 */
template <int dim>
class MinimalSurface : public DiscretePDE<dim>
{
public:
    MinimalSurface(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params,
                   const SolverParams& solver_params);

    void run() override;

protected:
    /** Sets up the system.
     *
     * \param initial_step  Whether this is the first time it is called or not.
     *   The difference is that the first time around we need to distribute
     *   the degrees of freedom and set the solution vector for $u^n$ to
     *   the correct size. The following times, the function is called after
     *   we have already done these steps as part of refining the mesh.
     */
    void setup_system(const bool initial_step) override;

    void assemble_system(AssemblyOptions opts) override;

    /** Takes care of setting the boundary values on the solution vector.
     *
     *  Ensures that the solution vector's entries respect the
     *  boundary values for our problem.  Having refined the mesh (or just
     *  started computations), there might be new nodal points on the
     *  boundary. These have values that are simply interpolated from the
     *  previous mesh in `refine_mesh()`, instead of the correct boundary
     *  values. This is fixed up by setting all boundary nodes of the current
     *  solution vector explicit to the right value.
     */
    void set_boundary_values();

    /** Computes the norm of the nonlinear (discrete) residual.
     *
     * We use this function to
     * monitor convergence of the Newton iteration. The function takes a step
     * length $\alpha^n$ as argument to compute the residual of $u^n + \alpha^n
     * \; \delta u^n$. This is something one typically needs for step length
     * control, although we will not use this feature here.
     */
    double compute_residual(const double alpha) const;

    /** Computes the step length $\alpha^n$ in each Newton iteration.
     *
     * We here use a fixed step length.
     */
    double determine_step_length() const;

    /** Solves the linear system in one Newton step and updates the solution vector.
     */
    void solve() override;

    /** Adaptively refines the mesh based on the Kelley estimator.
     *
     * Not currently used.
     */
    void refine_mesh();

    void output_results(int cycle) const;

    using DiscretePDE<dim>::case_;
    using DiscretePDE<dim>::params_;
    using DiscretePDE<dim>::solver_params_;

    using DiscretePDE<dim>::tria_;
    using DiscretePDE<dim>::dof_handler_;
    using DiscretePDE<dim>::sparsity_pattern_;
    using DiscretePDE<dim>::system_matrix_;
    using DiscretePDE<dim>::solution_;
    using DiscretePDE<dim>::update_;
    using DiscretePDE<dim>::rhs_;

    FE_Q<dim> fe_;
    AffineConstraints<double> hanging_node_constraints;
};


}
}

#endif
