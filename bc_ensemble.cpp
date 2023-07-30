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

// We start by putting all of our classes into their own namespace.
namespace cd_ensemble
{
  using namespace dealii;

  class Rectangle : public convdiff_hdg::DomainGeometry
  {
  public:
      using DomainGeometry::dim;
  
      virtual void generate_grid(dealii::Triangulation<dim>& tria,
              const unsigned int initial_resolution) const override
      {
          const dealii::Tensor<1,dim> dims = upper_right - lower_left;
          const auto num_cells_per_dim = get_cells_per_dim(dims, initial_resolution);
          // The 'true' below "colorizes" the mesh; boundary IDs are set
          dealii::GridGenerator::subdivided_hyper_rectangle(tria, num_cells_per_dim,
                  lower_left, upper_right);
          tria.refine_global(3 - dim);
      }
  
  protected:
      dealii::Point<dim> lower_left{-1.0, -1.0};
      dealii::Point<dim> upper_right{2.0, 1.0};
  
      std::vector<unsigned int> get_cells_per_dim(const dealii::Tensor<1,dim>& dims,
              const unsigned int init_res) const
      {
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

  // @sect3{Equation data}
  //
  // The structure of the analytic solution is the same as in step-7. There are
  // two exceptions. Firstly, we also create a solution for the 3d case, and
  // secondly, we scale the solution so its norm is of order unity for all
  // values of the solution width.
  template <int dim>
  class SolutionBase
  {
  protected:
    static const unsigned int n_source_centers = 3;
    static const Point<dim>   source_centers[n_source_centers];
    static const double       width;
  };


  template <>
  const Point<1>
    SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] =
      {Point<1>(-1.0 / 3.0), Point<1>(0.0), Point<1>(+1.0 / 3.0)};


  template <>
  const Point<2>
    SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] =
      {Point<2>(-0.5, +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.5, -0.5)};

  template <>
  const Point<3>
    SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] = {
      Point<3>(-0.5, +0.5, 0.25),
      Point<3>(-0.6, -0.5, -0.125),
      Point<3>(+0.5, -0.5, 0.5)};

  template <int dim>
  const double SolutionBase<dim>::width = 1. / 5.;


  template <int dim>
  class Solution : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      double sum = 0;
      for (unsigned int i = 0; i < this->n_source_centers; ++i)
        {
          const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
          sum +=
            std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
        }

      return sum /
             std::pow(2. * numbers::PI * this->width * this->width, dim / 2.);
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> sum;
      for (unsigned int i = 0; i < this->n_source_centers; ++i)
        {
          const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

          sum +=
            (-2 / (this->width * this->width) *
             std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) *
             x_minus_xi);
        }

      return sum /
             std::pow(2. * numbers::PI * this->width * this->width, dim / 2.);
    }
  };



  // This class implements a function where the scalar solution and its negative
  // gradient are collected together. This function is used when computing the
  // error of the HDG approximation and its implementation is to simply call
  // value and gradient function of the Solution class.
  template <int dim>
  class SolutionAndGradient : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    SolutionAndGradient()
      : Function<dim>(dim + 1)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  v) const override
    {
      AssertDimension(v.size(), dim + 1);
      Solution<dim>  solution;
      Tensor<1, dim> grad = solution.gradient(p);
      for (unsigned int d = 0; d < dim; ++d)
        v[d] = -grad[d];
      v[dim] = solution.value(p);
    }
  };



  // Next comes the implementation of the convection velocity. As described in
  // the introduction, we choose a velocity field that is $(y, -x)$ in 2d and
  // $(y, -x, 1)$ in 3d. This gives a divergence-free velocity field.
  template <int dim>
  class ConvectionVelocity : public TensorFunction<1, dim>
  {
  public:
    ConvectionVelocity()
      : TensorFunction<1, dim>()
    {}

    virtual Tensor<1, dim> value(const Point<dim> &p) const override
    {
      Tensor<1, dim> convection;
      switch (dim)
        {
          case 1:
            convection[0] = 1;
            break;
          case 2:
            convection[0] = p[1];
            convection[1] = -p[0];
            break;
          case 3:
            convection[0] = p[1];
            convection[1] = -p[0];
            convection[2] = 1;
            break;
          default:
            Assert(false, ExcNotImplemented());
        }
      return convection;
    }
  };



  // The last function we implement is the right hand side for the
  // manufactured solution. It is very similar to step-7, with the exception
  // that we now have a convection term instead of the reaction term. Since
  // the velocity field is incompressible, i.e., $\nabla \cdot \mathbf{c} =
  // 0$, the advection term simply reads $\mathbf{c} \nabla u$.
  template <int dim>
  class RightHandSide : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      ConvectionVelocity<dim> convection_velocity;
      Tensor<1, dim>          convection = convection_velocity.value(p);
      double                  sum        = 0;
      for (unsigned int i = 0; i < this->n_source_centers; ++i)
        {
          const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

          sum +=
            ((2 * dim - 2 * convection * x_minus_xi -
              4 * x_minus_xi.norm_square() / (this->width * this->width)) /
             (this->width * this->width) *
             std::exp(-x_minus_xi.norm_square() / (this->width * this->width)));
        }

      return sum /
             std::pow(2. * numbers::PI * this->width * this->width, dim / 2.);
    }
  };

  template <int dim>
  class ZeroNeumann : public convdiff_hdg::FaceFunction<dim>
  {
  public:
    virtual double value_normal(const Point<dim>& /*p*/ = 0,
            const Tensor<1,dim>& /*normal*/ = 0,
            const unsigned int = 0) const override
    {
        return 0;
    }
  };

  template <int dim>
  class Neumann : public convdiff_hdg::FaceFunction<dim>
  {
  public:
    virtual double value_normal(const Point<dim>& p, const Tensor<1,dim>& normal,
            const unsigned int = 0) const override
    {
        return -exact_solution.gradient(p) *
                            normal +
                          convection.value(p) * normal *
                            exact_solution.value(p);
    }
  private:
    Solution<dim> exact_solution;
    ConvectionVelocity<dim> convection;
  };

} // end of namespace cd_ensemble



int main(int argc, char *argv[])
{
  // Reads DEAL_II_NUM_THREADS env var
  dealii::MultithreadInfo::set_thread_limit();

  const auto runparams = convdiff_hdg::parse_cmd_options_ensemble(argc, argv);

  constexpr unsigned int dim = 2;

  auto geom = std::make_shared<cd_ensemble::Rectangle>();

  const int poly_degree = 1;

  auto exact_soln = std::make_shared<cd_ensemble::Solution<dim>>();
  auto exact_soln_grad = std::make_shared<cd_ensemble::SolutionAndGradient<dim>>();
  auto rhs = std::make_shared<cd_ensemble::RightHandSide<dim>>();
  auto dirichlet_bc = std::make_shared<cd_ensemble::Solution<dim>>();
  auto neumann_bc = std::make_shared<cd_ensemble::Neumann<dim>>();
  auto conv_vel = std::make_shared<cd_ensemble::ConvectionVelocity<dim>>();
  
  convdiff_hdg::CDParams<dim> cdparams {
      geom, conv_vel, rhs, dirichlet_bc, neumann_bc, exact_soln, exact_soln_grad };

  try
    {
      {
        std::cout << "Solving with Q1 elements, global refinement" << std::endl
                  << "===========================================" << std::endl
                  << std::endl;

        convdiff_hdg::HDG<dim> hdg_problem(poly_degree, convdiff_hdg::global_refinement,
                cdparams);
        hdg_problem.run(runparams.refine_levels, runparams.initial_resolution);

        std::cout << std::endl;
      }

      // This might be a good use-case for mesh-independent learning later.
      //{
      //  std::cout << "Solving with Q1 elements, adaptive refinement"
      //            << std::endl
      //            << "============================================="
      //            << std::endl
      //            << std::endl;

      //  convdiff_hdg::HDG<dim> hdg_problem(1, convdiff_hdg::adaptive_refinement,
      //          conv_vel, rhs, dirichlet_bc, neumann_bc, exact_soln, exact_soln_grad);
      //  hdg_problem.run(5);
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
