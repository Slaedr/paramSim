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
class PoissonCG : public PDESolver<dim>
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

    using PDESolver<dim>::case_;
    using PDESolver<dim>::params_;
    using PDESolver<dim>::solver_params_;

    using PDESolver<dim>::tria_;
    using PDESolver<dim>::dof_handler_;
    using PDESolver<dim>::sparsity_pattern_;
    using PDESolver<dim>::system_matrix_;
    using PDESolver<dim>::solution_;
    using PDESolver<dim>::rhs_;
      
    FE_Q<dim> fe_;
};

}
}

#endif
