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

#include <deal.II/base/multithread_info.h>

#include "convdiff_hdg.hpp"
#include "cases/step51.hpp"

int main()
{
  // Reads DEAL_II_NUM_THREADS env var
  dealii::MultithreadInfo::set_thread_limit();
  
  const unsigned int dim = 2;

  std::vector<convdiff_hdg::DomainGeometry<dim>::bc_mark_desc> bcmarks;
  bcmarks.push_back(std::make_pair(1, 
      [](const dealii::Point<dim>& p) {
          if ((std::fabs(p(0) - (-1)) < 1e-12) ||
              (std::fabs(p(1) - (-1)) < 1e-12))
              return true;
          else
              return false;
      }));
  
  auto geom = std::make_shared<cases::step51::Cube<dim>>(bcmarks);
  
  const int refine_levels = 8;
  const unsigned int initial_resolution = 2;

  auto exact_soln = std::make_shared<cases::step51::Solution<dim>>();
  auto exact_soln_grad = std::make_shared<cases::step51::SolutionAndGradient<dim>>();
  auto rhs = std::make_shared<cases::step51::RightHandSide<dim>>();
  auto dirichlet_bc = std::make_shared<cases::step51::Solution<dim>>();
  auto neumann_bc = std::make_shared<cases::step51::Neumann<dim>>();
  auto conv_vel = std::make_shared<cases::step51::ConvectionVelocity<dim>>();

  convdiff_hdg::CDParams<dim> cdparams {
      geom, conv_vel, rhs, dirichlet_bc, neumann_bc, exact_soln, exact_soln_grad, 0, 1 };

  try
    {
      // Now for the three calls to the main class in complete analogy to
      // step-7.
      {
        std::cout << "Solving with Q1 elements, adaptive refinement"
                  << std::endl
                  << "============================================="
                  << std::endl
                  << std::endl;

        convdiff_hdg::HDG<dim> hdg_problem(1, convdiff_hdg::adaptive_refinement, cdparams);
        hdg_problem.run(refine_levels, initial_resolution);

        std::cout << std::endl;
      }

      {
        std::cout << "Solving with Q1 elements, global refinement" << std::endl
                  << "===========================================" << std::endl
                  << std::endl;

        convdiff_hdg::HDG<dim> hdg_problem(1, convdiff_hdg::global_refinement, cdparams);
        hdg_problem.run(refine_levels, initial_resolution);

        std::cout << std::endl;
      }

      //{
      //  std::cout << "Solving with Q3 elements, global refinement" << std::endl
      //            << "===========================================" << std::endl
      //            << std::endl;

      //  convdiff_hdg::HDG<dim> hdg_problem(3, convdiff_hdg::global_refinement);
      //  hdg_problem.run();

      //  std::cout << std::endl;
      //}
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
