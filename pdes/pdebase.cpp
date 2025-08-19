
#include "pdebase.hpp"

#include <stdexcept>
#include <memory>
#include <iostream>

#include "../utils/error_handling.hpp"
#include "poisson/poisson_cg.hpp"
#include "convdiff/convdiff_hdg.hpp"
#include "nonlinear_elliptic/minimal_surface.hpp"

namespace paramsim {

template <int dim>
PDESolver<dim>::PDESolver(std::shared_ptr<const Case<dim>> test_case,
                          const PDEParams& params, const SolverParams& solver_params)
    : case_{test_case}, params_{params}, solver_params_{solver_params}, dof_handler_(tria_)
{ 
}

template <int dim>
void PDESolver<dim>::make_grid(const unsigned n_cell_dir)
{
    tria_.clear();
    auto geom = case_->get_geometry();
    geom->generate_grid(tria_, n_cell_dir);
    //triangulation.refine_global(5);
    geom->set_boundary_ids(tria_);

    std::cout << "   Number of active cells: " << tria_.n_active_cells()
              << std::endl
              << "   Total number of cells: " << tria_.n_cells()
              << std::endl;
}

template class PDESolver<2>;

template <int dim>
std::unique_ptr<PDESolver<dim>> create_pde_solver(std::shared_ptr<const Case<dim>> test_case,
                                                  const PDEParams& params,
                                                  const SolverParams& solver_params)
{
    if(params.pde_solver == "poisson_cg") {
        return std::make_unique<pde::PoissonCG<dim>>(test_case, params, solver_params);
    } else if(params.pde_solver == "convdiff_hdg") {
        auto ccase = std::dynamic_pointer_cast<const ConvDiffCase<dim>>(test_case);
        if(!ccase) {
            throw std::runtime_error("Invalid case for HDG convdiff!");
        }
        return std::make_unique<pde::ConvdiffHDG<dim>>(test_case, params, solver_params);
    } else if(params.pde_solver == "minimal_surface") {
        return std::make_unique<pde::MinimalSurface<dim>>(test_case, params, solver_params);
    } else {
        throw std::runtime_error("Unsupported PDE solver!");
    }
}

template std::unique_ptr<PDESolver<2>> create_pde_solver(std::shared_ptr<const Case<2>>,
                                                         const PDEParams&,
                                                         const SolverParams&);

}
