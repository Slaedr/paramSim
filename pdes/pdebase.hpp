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
    bool is_adaptive;
    std::string output_path;
};

struct SolverParams {
    double tolerance;
    int max_its;
};

template <int dim>
class PDESolver
{
public:
    PDESolver(const PDEParams<dim>& params, const SolverParams& solver_params)
        : params_{params}, solver_params_{solver_params}
    { }

    virtual ~PDESolver() { }

    virtual void run() = 0;

protected:
    const PDEParams<dim> params_;
    const SolverParams solver_params_;
};

template <int dim>
std::unique_ptr<PDESolver<dim>> create_pde_solver(const PDEParams<dim>& params,
                                                  const SolverParams& solver_params);

}

#endif
