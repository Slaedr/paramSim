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

#include "cmdparser.hpp"
#include "convdiff_hdg.hpp"
#include "cases/step51.hpp"

int main(int argc, char *argv[])
{
  // Reads DEAL_II_NUM_THREADS env var
  dealii::MultithreadInfo::set_thread_limit();

  const auto params = convdiff_hdg::parse_cmd_options_ensemble(argc, argv);
  
  const unsigned int dim = 2;

  std::vector<convdiff_hdg::DomainGeometry<dim>::bc_mark_desc> bcmarks;
  bcmarks.push_back(std::make_pair(1, 
      [](const dealii::Point<dim>& p) {
          //if ((std::fabs(p(0) - (-1)) < 1e-12) ||
          //    (std::fabs(p(1) - (-1)) < 1e-12))
          //    return true;
          //else
          //    return false;
          (void)p;
          return false;
      }));
  
  auto geom = std::make_shared<cases::step51::Cube<dim>>(bcmarks);

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
      {
        std::cout << "Solving with Q1 elements, global refinement" << std::endl
                  << "===========================================" << std::endl
                  << std::endl;

        convdiff_hdg::HDG<dim> hdg_problem(1, convdiff_hdg::global_refinement, cdparams);
        hdg_problem.run(params.refine_levels, params.initial_resolution);

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

namespace unused
{
  using namespace dealii;

  template <int dim>
  class Rectangle : public convdiff_hdg::DomainGeometry<dim>
  {
  public:
      virtual void generate_grid(dealii::Triangulation<dim>& tria,
              const unsigned int initial_resolution) const override
      {
          const dealii::Tensor<1,dim> dims = upper_right - lower_left;
          const auto num_cells_per_dim = get_cells_per_dim(dims, initial_resolution);
          // The 'true' below "colorizes" the mesh; boundary IDs are set
          dealii::GridGenerator::subdivided_hyper_rectangle(tria, num_cells_per_dim,
                  lower_left, upper_right);
      }
  
  protected:
      dealii::Point<dim> lower_left{-1.0, -1.0};
      dealii::Point<dim> upper_right{2.0, 1.0};
  
      std::vector<unsigned int> get_cells_per_dim(const dealii::Tensor<1,dim>& dims,
              const unsigned int init_res) const
      {
          static_assert(dim == 2, "Not implemented for anything but 2D!");
          const double min_dim = std::min(dims[0], dims[1]);
          const double max_dim = std::max(dims[0], dims[1]);
          const int min_idx = dims[0] < dims[1] ? 0 : 1;
          std::vector<unsigned int> num_cells_per_dim(dim);
          num_cells_per_dim[min_idx] = init_res;
          const auto other_res = static_cast<unsigned int>(std::round(max_dim / min_dim * init_res));
          const auto other_idx = (min_idx + 1) % dim;
          num_cells_per_dim[other_idx] = other_res;
          std::cout << "Num cells per dim = " << num_cells_per_dim[0] << ", " 
              << num_cells_per_dim[1] << std::endl;
          return num_cells_per_dim;
      }
  };
}
