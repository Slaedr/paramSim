#ifndef PARAMSIM_POISSON_CG_HPP_
#define PARAMSIM_POISSON_CG_HPP_

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

/**
 * Solves the Poisson equation with non-homogeneous Dirichlet BCs.
 */
template <int dim>
class PoissonCG : public PDESolver<dim>
{
public:
  PoissonCG(const PDEParams<dim>& params, const SolverParams& solver_params);

  void run() override;
  
  std::shared_ptr<Vector<double>> create_solution_vector() const;

private:
  void make_grid(unsigned resolution);
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(int cycle) const;

  using PDESolver<dim>::params_;
  using PDESolver<dim>::solver_params_;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};

}
}

#endif
