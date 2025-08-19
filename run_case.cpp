/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Martin Kronbichler, Technische Universität München,
 *         Scott T. Miller, The Pennsylvania State University, 2013
 */

// @sect3{Include files}
//
// Most of the deal.II include files have already been covered in previous
// examples and are not commented on.

#include <iostream>
#include <boost/program_options/options_description.hpp>

#include <deal.II/base/multithread_info.h>

#include "utils/cmdparser.hpp"
#include "cases/case.hpp"
#include "pdes/pdebase.hpp"

namespace bpo = boost::program_options;
using namespace paramsim;

int main(int argc, char *argv[])
{
    // Reads DEAL_II_NUM_THREADS env var
    dealii::MultithreadInfo::set_thread_limit();

    constexpr unsigned int dim = 2;

    // Common options description//
    bpo::options_description common_desc
        ("Solves one problem given one set of parameters.");
    add_common_options(common_desc, "");

    // complete all options addition before calling the following line
    const bpo::variables_map common_cmdmap = get_cmd_args(argc, argv, common_desc);
    if(common_cmdmap.count("help")) {
        std::cout << common_desc << std::endl;
        return 0;
    }

    const auto common_params = get_common_params(common_cmdmap);

    std::shared_ptr<const Case<dim>> tcase = create_case<dim>(common_params, argc, argv);

    PDEParams pdeparams{common_params.solver_str, common_params.fe_degree,
                        common_params.initial_resolution, common_params.refine_levels,
                        common_params.is_adaptive, common_params.outpath};
    SolverParams solver_params{common_params.tolerance, common_params.max_outer_its};

    auto pdesolver = create_pde_solver(tcase, pdeparams, solver_params);

    try
    {
        std::cout << "Solving" << std::endl
                  << "=======" << std::endl
                  << std::endl;

        pdesolver->run();

        std::cout << std::endl;
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
