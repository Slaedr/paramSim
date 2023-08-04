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

template <int dim>
struct PDEParams {
    const std::string pde_solver;
    std::shared_ptr<const Case<dim>> test_case;
    int fe_degree;
    unsigned initial_resolution;
    int refine_levels;
    std::string output_path;
};

template <int dim>
class PDESolver
{
public:
    PDESolver(std::shared_ptr<const Case<dim>> test_case, int fe_degree,
            unsigned initial_cell_resolution, const std::string& output_path)
        : tcase_{test_case}, fe_degree_{fe_degree}, init_res_{initial_cell_resolution},
        output_path_{output_path}
    { }

    virtual ~PDESolver() { }

    virtual void run() = 0;

protected:
    std::shared_ptr<const Case<dim>> tcase_;
    int fe_degree_;
    unsigned int init_res_;
    std::string output_path_;
};

template <int dim>
std::unique_ptr<PDESolver<dim>> create_pde_solver(const PDEParams<dim>& params);

}

#endif
