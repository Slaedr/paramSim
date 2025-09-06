#include "grid_convergence.hpp"

#include <vector>
#include <cmath>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>

#include "../utils/error_handling.hpp"
#include "../solvers/newton.hpp"

namespace paramsim {
namespace testutils {

template <int dim>
double test_grid_convergence(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params,
                             const SolverParams sparams)
{
    if(params.refine_levels < 2) {
        throw std::runtime_error("Not enough refinement levels to test grid convergence!");
    }
    assert(!params.is_adaptive);
    std::shared_ptr<DiscretePDEBase> pdeb = create_discrete_pde<dim>(test_case, params, sparams);
    auto pde = std::dynamic_pointer_cast<DiscretePDE<dim,dealii::FE_Q<dim>>>(pdeb);
    auto scase = std::dynamic_pointer_cast<const HasExactSolution<dim>>(test_case);
    if(!scase) {
        throw TypeNotSupportedError("Case must have exact solution for grid convergence!");
    }

    solver::NewtonSolver solver(pde, sparams);
    using vector_type = typename DiscretePDEBase::vector_type;
    vector_type u;
    pde->allocate_solution_vector(u);

    dealii::ConvergenceTable convergence_table;
    std::vector<double> l2_errors;
    std::vector<double> h;
    h.reserve(params.refine_levels);
    std::vector<double> slopes(params.refine_levels-1);
    unsigned resolution = params.initial_resolution;

    for(int imesh = 0; imesh < params.refine_levels; imesh++, resolution*=2)
    {
        solver.reinit();
        solver.solve(u);

        const dealii::DoFHandler<dim>& dof_handler = pde->get_dof_handler();
        const dealii::Triangulation<dim>& triangulation = pde->get_triangulation();
        const vector_type& solution = u;

        convergence_table.add_value("cells", pde->get_triangulation().n_active_cells());
        convergence_table.add_value("dofs", dof_handler.n_dofs());
        //h.push_back(std::pow(static_cast<double>(pde->get_triangulation().n_active_cells()),
        //                     1.0/dim));
        h.push_back(1.0/static_cast<double>(resolution));

        dealii::Vector<float> difference_per_cell(triangulation.n_active_cells());

        dealii::VectorTools::integrate_difference(dof_handler,
                                                  solution,
                                                  *scase->get_exact_solution(),
                                                  difference_per_cell,
                                                  dealii::QGauss<dim>(params.fe_degree + 2),
                                                  dealii::VectorTools::L2_norm);
        const double post_error =
            dealii::VectorTools::compute_global_error(triangulation, difference_per_cell,
                                                      dealii::VectorTools::L2_norm);

        convergence_table.add_value("val L2", post_error);
        convergence_table.set_scientific("val L2", true);
        convergence_table.set_precision("val L2", 3);
        l2_errors.push_back(post_error);
        std::cout << "Mesh size h = " << h[imesh] << std::endl;
        std::cout << "Error = " << l2_errors[imesh] << std::endl;
        if(imesh > 0) {
            slopes[imesh-1] = (std::log10(l2_errors[imesh]) - std::log10(l2_errors[imesh-1])) /
                (std::log10(h[imesh]) - std::log10(h[imesh-1]));
        }

        pde->refine_mesh_and_interpolate_solution(u);
    }

    if(scase && scase->get_exact_solution()) {
        convergence_table.evaluate_convergence_rates(
                "val L2", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.write_text(std::cout);
    }
    return slopes.back();
}

template double test_grid_convergence(std::shared_ptr<const Case<2>>, const PDEParams&,
                                      SolverParams);

}
}
