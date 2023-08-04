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

#include "cmdparser.hpp"
#include "cases/case.hpp"
#include "pdes/pdebase.hpp"

namespace bpo = boost::program_options;
using namespace paramsim;

int main(int argc, char *argv[])
{
  // Reads DEAL_II_NUM_THREADS env var
  dealii::MultithreadInfo::set_thread_limit();
  
  constexpr unsigned int dim = 2;

  bpo::options_description common_desc
      ("Solves one problem given one set of parameters.");
  add_common_options(common_desc, "");

  // complete all options addition before calling the following line
  const bpo::variables_map common_cmdmap = get_cmd_args(argc, argv, common_desc);
  if(common_cmdmap.count("help")) {
      std::cout << common_desc << std::endl;
      return 0;
  }

  const auto case_str = common_cmdmap["case"].as<std::string>();
  const auto solver_str = common_cmdmap["solver"].as<std::string>();
  const auto refine_levels = common_cmdmap["refine_levels"].as<int>();
  const auto initial_resolution = common_cmdmap["initial_resolution"].as<unsigned>();
  const auto fe_degree = common_cmdmap["fe_degree"].as<int>();
  const auto outpath = common_cmdmap["output_prefix"].as<std::string>();

  std::shared_ptr<Case<dim>> tcase = create_case<dim>(case_str);
    
  bpo::options_description case_desc
      (std::string("Solves the case ") + case_str + " given one set of parameters.");
  tcase->add_case_cmd_args(case_desc);
  
  const bpo::variables_map case_cmdmap = get_cmd_args(argc, argv, case_desc);
  tcase->initialize(case_cmdmap);

  PDEParams<dim> pdeparams{solver_str, tcase, fe_degree, initial_resolution, refine_levels,
      outpath};

  auto pdesolver = create_pde_solver(pdeparams);

  try
  {
      std::cout << "Solving with Q1 elements, global refinement" << std::endl
                << "===========================================" << std::endl
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

