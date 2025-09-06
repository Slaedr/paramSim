#ifndef PARAMSIM_POISSON_CG_HPP_
#define PARAMSIM_POISSON_CG_HPP_

#include <memory>

#include <deal.II/fe/fe_q.h>

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

    PoissonCG(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params,
              const SolverParams& solver_params);

    bool is_symm_positive_definite() const override {
        return true;
    }

    void assemble_system(AssemblyOptions opts, const vector_type& state,
                         dealii::SparseMatrix<double>& mat, vector_type& rhs) const override;

    void output_results(int mesh_number, const vector_type& solution) const override;

private:

    using DiscretePDE<dim,fe_type>::case_;
    using DiscretePDE<dim,fe_type>::params_;

    using DiscretePDE<dim,fe_type>::tria_;
    using DiscretePDE<dim,fe_type>::dof_handler_;
    using DiscretePDE<dim,fe_type>::fe_;
    using DiscretePDE<dim,fe_type>::affine_constraints_;
};

}
}

#endif
