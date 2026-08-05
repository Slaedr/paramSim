#ifndef PARAMSIM_GELFAND_HPP_
#define PARAMSIM_GELFAND_HPP_

#include <memory>

#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>

#include "../../cases/case.hpp"
#include "../pdebase.hpp"

namespace paramsim {
namespace pde {

/**
 * Discretizes the dimension-independent Gelfand equation
 * \f$-\Delta u = \exp(u) + f\f$ with continuous finite elements.
 *
 * @tparam dim Spatial dimension.
 */
template <int dim>
class Gelfand : public DiscretePDE<dim, dealii::FE_Q<dim>> {
public:
    using fe_type = dealii::FE_Q<dim>;
    using vector_type = typename DiscretePDE<dim, fe_type>::vector_type;

    /**
     * Constructs a Gelfand discretization for a case and PDE parameters.
     *
     * @param test_case Case providing the geometry, forcing, and boundary data.
     * @param params Finite-element and mesh parameters.
     */
    Gelfand(std::shared_ptr<const Case<dim>> test_case,
            const PDEParams& params);

    /**
     * Reports whether the Jacobian is guaranteed symmetric positive definite.
     *
     * @return False because the negative exponential mass term can make the
     * Jacobian indefinite.
     */
    bool is_symm_positive_definite() const override
    {
        return false;
    }

    /**
     * Assembles the Newton Jacobian and negative residual.
     *
     * @param opts Assembly options shared by all PDE discretizations.
     * @param state Current nonlinear state.
     * @param mat Global Jacobian matrix to overwrite.
     * @param rhs Global negative-residual vector to overwrite.
     */
    void assemble_system(AssemblyOptions opts, const vector_type& state,
                         dealii::SparseMatrix<double>& mat,
                         vector_type& rhs) const override;

    /**
     * Evaluates the negative residual at a nonlinear state.
     *
     * @param state Nonlinear state at which to evaluate the residual.
     * @param rhs Global negative-residual vector to overwrite.
     */
    void evaluate_residual(const vector_type& state,
                           vector_type& rhs) const override;

protected:
    using DiscretePDE<dim, fe_type>::case_;
    using DiscretePDE<dim, fe_type>::dof_handler_;
    using DiscretePDE<dim, fe_type>::fe_;
    using DiscretePDEBase::affine_constraints_;
};

} // namespace pde
} // namespace paramsim

#endif
