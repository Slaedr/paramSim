#ifndef PARAMSIM_PDEBASE_HPP_
#define PARAMSIM_PDEBASE_HPP_

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/function.h>

#include "../geometrybase.hpp"
#include "../cases/case.hpp"

namespace paramsim {
 
enum class MeshRefineMode
{
  global_refinement,
  adaptive_refinement
};

/// Common parameters needed for solving most scalar PDEs
template <int dim>
struct PDEParams1 {
    /// Geometry
    std::shared_ptr<const DomainGeometry<dim>> geom;
    /// Source term
    std::shared_ptr<const dealii::Function<dim>> rhs_function;
    /// Dirichlet boundary condition
    std::shared_ptr<const dealii::Function<dim>> dirichlet_bc_function;
    /// Neumann boundary condition
    std::shared_ptr<const FaceFunction<dim>> neumann_bc_function;
    /// Exact solution (if available)
    std::shared_ptr<const dealii::Function<dim>> solution_function;

    unsigned int dirichlet_marker;
    unsigned int neumann_marker;
};

template <int dim>
struct PDEParams {
    const std::string pde_solver;
    std::shared_ptr<const Case<dim>> test_case;
    int fe_degree;
    unsigned initial_resolution;
    int refine_levels;
};

template <int dim>
class PDESolver
{
public:
    PDESolver(std::shared_ptr<const Case<dim>> test_case, int fe_degree,
            unsigned initial_cell_resolution)
        : tcase_{test_case}, fe_degree_{fe_degree}, init_res_{initial_cell_resolution}
    { }

    virtual ~PDESolver() { }

    virtual void run() = 0;

protected:
    std::shared_ptr<const Case<dim>> tcase_;
    int fe_degree_;
    unsigned int init_res_;
};

template <int dim>
std::unique_ptr<PDESolver<dim>> create_pde_solver(const PDEParams<dim>& params);

}

#endif
