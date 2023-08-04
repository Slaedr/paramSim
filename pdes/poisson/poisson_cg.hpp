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

// @sect3{The <code>PoissonCG</code> class template}

// This is again the same <code>PoissonCG</code> class as in the previous
// example. The only difference is that we have now declared it as a class
// with a template parameter, and the template parameter is of course the
// spatial dimension in which we would like to solve the Laplace equation. Of
// course, several of the member variables depend on this dimension as well,
// in particular the Triangulation class, which has to represent
// quadrilaterals or hexahedra, respectively. Apart from this, everything is
// as before.
template <int dim>
class PoissonCG : public PDESolver<dim>
{
public:
  PoissonCG(std::shared_ptr<const Case<dim>> tcase, int degree, unsigned int initial_resolution,
      int refine_levels, const std::string& output_path);

  void run() override;
  
  std::shared_ptr<Vector<double>> create_solution_vector() const;

private:
  void make_grid(unsigned resolution);
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(int cycle) const;

  using PDESolver<dim>::tcase_;
  using PDESolver<dim>::fe_degree_;
  using PDESolver<dim>::init_res_;

  int num_cycles_;

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
