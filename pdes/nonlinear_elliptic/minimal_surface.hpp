#ifndef PARAMSIM_MINIMAL_SURFACE_HPP_
#define PARAMSIM_MINIMAL_SURFACE_HPP_

#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>

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
class MinimalSurface : public DiscretePDE<dim, FE_Q<dim>>
{
public:
    using fe_type = FE_Q<dim>;
    using vector_type = typename DiscretePDE<dim,fe_type>::vector_type;

    MinimalSurface(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params);

    bool is_symm_positive_definite() const override {
        return true;
    }

    void assemble_system(AssemblyOptions opts, const vector_type& state,
                         dealii::SparseMatrix<double>& mat, vector_type& rhs) const override;

    void output_results(int mesh_number, const vector_type& solution) const override;

protected:

    /** Computes the norm of the nonlinear (discrete) residual.
     *
     * We use this function to
     * monitor convergence of the Newton iteration. The function takes a step
     * length $\alpha^n$ as argument to compute the residual of $u^n + \alpha^n
     * \; \delta u^n$. This is something one typically needs for step length
     * control, although we will not use this feature here.
     */
    double compute_residual(const double alpha, const vector_type& solution,
                            const vector_type& update) const;

    using DiscretePDE<dim,fe_type>::case_;
    using DiscretePDE<dim,fe_type>::params_;

    using DiscretePDE<dim,fe_type>::tria_;
    using DiscretePDE<dim,fe_type>::dof_handler_;
    using DiscretePDE<dim,fe_type>::fe_;
    using DiscretePDEBase::affine_constraints_;
};


}
}

#endif
