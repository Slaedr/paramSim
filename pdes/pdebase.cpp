
#include "pdebase.hpp"

#include <stdexcept>
#include <memory>

#include "poisson/poisson_cg.hpp"
#include "convdiff/convdiff_hdg.hpp"
#include "nonlinear_elliptic/minimal_surface.hpp"

namespace paramsim {

template <int dim>
std::unique_ptr<PDESolver<dim>> create_pde_solver(const PDEParams<dim>& params,
                                                  const SolverParams& solver_params)
{
    if(params.pde_solver == "poisson_cg") {
        return std::make_unique<pde::PoissonCG<dim>>(params, solver_params);
    } else if(params.pde_solver == "convdiff_hdg") {
        auto ccase = std::dynamic_pointer_cast<const convdiffcase_verification<dim>>(
                params.test_case);
        if(!ccase) {
            throw std::runtime_error("Invalid case for HDG convdiff!");
        }
        return std::make_unique<pde::ConvdiffHDG<dim>>(params, solver_params);
    } else if(params.pde_solver == "minimal_surface") {
        return std::make_unique<pde::MinimalSurface<dim>>(params, solver_params);
    } else {
        throw std::runtime_error("Unsupported PDE solver!");
    }
}

template std::unique_ptr<PDESolver<2>> create_pde_solver(const PDEParams<2>& params,
                                                         const SolverParams& sparams);

}
