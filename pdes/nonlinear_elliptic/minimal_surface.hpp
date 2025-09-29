#ifndef PARAMSIM_MINIMAL_SURFACE_HPP_
#define PARAMSIM_MINIMAL_SURFACE_HPP_

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
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

    void evaluate_residual(const vector_type& state, vector_type& rhs) const;

    void output_results(int mesh_number, const vector_type& solution) const override;

protected:

    /* Computes the norm of the nonlinear (discrete) residual.
     *
     * We use this function to
     * monitor convergence of the Newton iteration. The function takes a step
     * length $\alpha^n$ as argument to compute the residual of $u^n + \alpha^n
     * \; \delta u^n$. This is something one typically needs for step length
     * control, although we will not use this feature here.
     */
    [[deprecated("Outdated; incorrect residual")]]
    double compute_residual(const double alpha, const vector_type& solution,
                            const vector_type& update) const;

    using DiscretePDE<dim,fe_type>::case_;
    using DiscretePDE<dim,fe_type>::params_;

    using DiscretePDE<dim,fe_type>::tria_;
    using DiscretePDE<dim,fe_type>::dof_handler_;
    using DiscretePDE<dim,fe_type>::fe_;
    using DiscretePDEBase::affine_constraints_;

private:
    /// Evaluates the PDE residual at one quadrature point and adds it to an output vector.
    void evaluate_point_residual(const FEValues<dim>& fe_values,
                                 const std::vector<Tensor<1,dim>>& solution_gradients,
                                 const unsigned q,
                                 Vector<double>& elem_rhs) const
    {
        const unsigned dofs_per_cell = fe_.n_dofs_per_cell();
        const double coeff =
          1.0 / std::sqrt(1 + solution_gradients[q] *
                                solution_gradients[q]);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            // residual of operator
            elem_rhs(i) -= (fe_values.shape_grad(i, q)  // \nabla \phi_i
                            * coeff                     // * a_n
                            * solution_gradients[q] // * \nabla u_n
                            * fe_values.JxW(q));        // * dx
            // source term
            const auto &x_q = fe_values.quadrature_point(q);
            elem_rhs(i) += (fe_values.shape_value(i, q) *          // phi_i(x_q)
                            case_->get_right_hand_side()->value(x_q) *   // f(x_q)
                            fe_values.JxW(q));                      // dx
        }
    }
};


}
}

#endif
