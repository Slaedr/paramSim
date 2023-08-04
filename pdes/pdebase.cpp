
#include "pdebase.hpp"

#include <stdexcept>
#include <memory>

#include "poisson/poisson_cg.hpp"
#include "convdiff/convdiff_hdg.hpp"

namespace paramsim {

template <int dim>
std::unique_ptr<PDESolver<dim>> create_pde_solver(const PDEParams<dim>& params)
{
    if(params.pde_solver == "poisson_cg") {
        return std::make_unique<pde::PoissonCG<dim>>(params.test_case, params.fe_degree,
                params.initial_resolution, params.refine_levels, params.output_path);
    } else if(params.pde_solver == "convdiff_hdg") {
        auto ccase = std::dynamic_pointer_cast<const convdiffcase_verification<dim>>(
                params.test_case);
        if(!ccase) {
            throw std::runtime_error("Invalid case for HDG convdiff!");
        }
        return std::make_unique<pde::ConvdiffHDG<dim>>(ccase, params.fe_degree,
                params.initial_resolution, MeshRefineMode::global_refinement,
                params.refine_levels, params.output_path);
    } else {
        throw std::runtime_error("Unsupported PDE solver!");
    }
}

template std::unique_ptr<PDESolver<2>> create_pde_solver(const PDEParams<2>& params);

}
