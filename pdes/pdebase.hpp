#ifndef PARAMSIM_PDEBASE_HPP_
#define PARAMSIM_PDEBASE_HPP_

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>


#include "../geometrybase.hpp"
#include "../cases/case.hpp"

namespace paramsim {

using gl_int_t = dealii::types::global_dof_index;
 
//enum class MeshRefineMode
//{
//  global_refinement,
//  adaptive_refinement
//};

struct PDEParams {
    std::string pde_solver;
    int fe_degree;
    unsigned initial_resolution;
    int refine_levels;
    bool is_adaptive;
    std::string output_path;
};

struct SolverParams {
    double tolerance;
    int max_its;
};

/// Any needed options for the system assembly operation.
struct AssemblyOptions {
    /** For hybridized discretizations, whether to assemble the 'local' domain system (true) or
     * the 'global' skeleton solution (false).
     */
    bool reconstruct_from_trace;
};

template <int dim>
class PDESolver
{
public:
    PDESolver(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params,
              const SolverParams& solver_params);

    virtual ~PDESolver() { }
    
    void make_grid(unsigned n_cell_dir);

    virtual void setup_system(bool initial_step) = 0;
    virtual void assemble_system(AssemblyOptions opts) = 0;
    virtual void solve() = 0;

    virtual void run() = 0;

    const dealii::Triangulation<dim>& get_triangulation() const { return tria_; }
    dealii::Triangulation<dim>& get_triangulation() { return tria_; }
    const dealii::DoFHandler<dim>& get_dof_handler() const { return dof_handler_; }
    const dealii::Vector<double> get_solution() const { return solution_; }

protected:
    std::shared_ptr<const Case<dim>> case_;
    const PDEParams params_;
    const SolverParams solver_params_;

    dealii::Triangulation<dim> tria_;

    dealii::DoFHandler<dim> dof_handler_;

    dealii::SparsityPattern sparsity_pattern_;
    dealii::SparseMatrix<double> system_matrix_;

    dealii::Vector<double> solution_;
    dealii::Vector<double> update_;
    dealii::Vector<double> rhs_;
};

template <int dim>
std::unique_ptr<PDESolver<dim>> create_pde_solver(std::shared_ptr<const Case<dim>> test_case,
                                                  const PDEParams& params,
                                                  const SolverParams& solver_params);

}

#endif
