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
class PoissonCG : public DiscretePDE<dim>
{
public:
    PoissonCG(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params,
              const SolverParams& solver_params);

    void run() override;
    
    std::shared_ptr<Vector<double>> create_solution_vector() const;

private:
    void setup_system(bool initial_step) override;
    void assemble_system(AssemblyOptions opts) override;
    void solve() override;
    void output_results(int cycle) const;

    using DiscretePDE<dim>::case_;
    using DiscretePDE<dim>::params_;
    using DiscretePDE<dim>::solver_params_;

    using DiscretePDE<dim>::tria_;
    using DiscretePDE<dim>::dof_handler_;
    using DiscretePDE<dim>::sparsity_pattern_;
    using DiscretePDE<dim>::system_matrix_;
    using DiscretePDE<dim>::solution_;
    using DiscretePDE<dim>::rhs_;
      
    FE_Q<dim> fe_;
};

}
}

#endif
