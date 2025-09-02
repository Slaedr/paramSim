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

/// Spatial-dimension-agnostic abstract PDE discretization interface.
class DiscretePDEBase
{
public:
    DiscretePDEBase(const PDEParams& params, const SolverParams& solver_params);
    virtual ~DiscretePDEBase() { }

    virtual void make_grid(unsigned n_cell_dir) = 0;
    virtual void setup_system(bool initial_step) = 0;
    virtual void assemble_system(AssemblyOptions opts) = 0;
    virtual void solve() = 0;
    virtual void run() = 0;

protected:
    const PDEParams params_;
    const SolverParams solver_params_;

    dealii::SparsityPattern sparsity_pattern_;
    dealii::SparseMatrix<double> system_matrix_;

    dealii::Vector<double> solution_;
    dealii::Vector<double> update_;
    dealii::Vector<double> rhs_;
};

/// Parts of the abstract PDE discretization interface that depend on the spatial dimension.
template <int dim>
class DiscretePDE : public DiscretePDEBase
{
public:
    DiscretePDE(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params,
              const SolverParams& solver_params);

    virtual ~DiscretePDE() { }
    
    void make_grid(unsigned n_cell_dir) override;

    const dealii::Triangulation<dim>& get_triangulation() const { return tria_; }
    dealii::Triangulation<dim>& get_triangulation() { return tria_; }
    const dealii::DoFHandler<dim>& get_dof_handler() const { return dof_handler_; }
    const dealii::Vector<double> get_solution() const { return solution_; }

protected:
    std::shared_ptr<const Case<dim>> case_;

    dealii::Triangulation<dim> tria_;
    dealii::DoFHandler<dim> dof_handler_;
};

template <int dim>
std::unique_ptr<DiscretePDE<dim>> create_discrete_pde(std::shared_ptr<const Case<dim>> test_case,
                                                      const PDEParams& params,
                                                      const SolverParams& solver_params);

}

#endif
