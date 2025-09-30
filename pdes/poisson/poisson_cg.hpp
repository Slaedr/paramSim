#ifndef PARAMSIM_POISSON_CG_HPP_
#define PARAMSIM_POISSON_CG_HPP_

#include <memory>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include "../pdebase.hpp"
#include "../../cases/case.hpp"

namespace paramsim {
namespace pde {

using namespace dealii;

/**
 * Solves the Poisson equation with non-homogeneous Dirichlet BCs.
 */
template <int dim>
class PoissonCG : public DiscretePDE<dim, FE_Q<dim>>
{
public:
    using fe_type = FE_Q<dim>;
    using vector_type = typename DiscretePDE<dim,fe_type>::vector_type;

    PoissonCG(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params);

    bool is_symm_positive_definite() const override {
        return true;
    }

    void assemble_system(AssemblyOptions opts, const vector_type& state,
                         dealii::SparseMatrix<double>& mat, vector_type& rhs) const override;

    /// Evaluates the nonlinear residual at a given state.
    void evaluate_residual(const vector_type& state, vector_type& rhs) const override;

    void output_results(int mesh_number, const vector_type& solution) const override;

private:

    using DiscretePDE<dim,fe_type>::case_;
    using DiscretePDE<dim,fe_type>::params_;

    using DiscretePDE<dim,fe_type>::tria_;
    using DiscretePDE<dim,fe_type>::dof_handler_;
    using DiscretePDE<dim,fe_type>::fe_;
    using DiscretePDE<dim,fe_type>::affine_constraints_;

    /// Evaluates the PDE residual at one quadrature point and adds it to an output vector.
    void evaluate_point_residual(const FEValues<dim>& fe_values,
                                 const std::vector<Tensor<1,dim>>& solution_gradients,
                                 const unsigned q,
                                 Vector<double>& elem_rhs) const
    {
        const unsigned dofs_per_cell = fe_.n_dofs_per_cell();

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            // residual of operator
            elem_rhs(i) -= (fe_values.shape_grad(i, q)  // \nabla \phi_i
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
