/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2012 - 2023 by the deal.II authors
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
 * Author: Sven Wetterauer, University of Heidelberg, 2012
 */

#include "minimal_surface.hpp"


#include <fstream>
#include <iostream>


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>


#include "../pdebase.hpp"
#include "../../utils/function_norms.hpp"
#include "../../cases/case.hpp"


namespace paramsim {
namespace pde {

using namespace dealii;

template <int dim>
MinimalSurface<dim>::MinimalSurface(std::shared_ptr<const Case<dim>> tcase, const PDEParams& params,
                                    const SolverParams& s_params)
: DiscretePDE<dim,fe_type>(tcase, params, s_params)
{
}

// @sect4{MinimalSurface::assemble_system}

// The matrix and right hand side functions depend on the
// previous iteration's solution. As discussed in the introduction, we need
// to use zero boundary values for the Newton updates; we compute them at
// the end of this function.
//
// The top of the function contains the usual boilerplate code, setting up
// the objects that allow us to evaluate shape functions at quadrature
// points and temporary storage locations for the local matrices and
// vectors, as well as for the gradients of the previous solution at the
// quadrature points. We then start the loop over all cells:
template <int dim>
void MinimalSurface<dim>::assemble_system(AssemblyOptions, const vector_type& state,
                                          dealii::SparseMatrix<double>& mat, vector_type& rhs) const
{
  const QGauss<dim> quadrature_formula(fe_.degree + 1);

  mat = 0;
  rhs = 0;

  FEValues<dim> fe_values(fe_, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points
                          | update_JxW_values);

  const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler_.active_cell_iterators())
  {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);

      // For the assembly of the linear system, we have to obtain the values
      // of the previous solution's gradients at the quadrature
      // points. There is a standard way of doing this: the
      // FEValues::get_function_gradients function takes a vector that
      // represents a finite element field defined on a DoFHandler, and
      // evaluates the gradients of this field at the quadrature points of the
      // cell with which the FEValues object has last been reinitialized.
      // The values of the gradients at all quadrature points are then written
      // into the second argument:
      fe_values.get_function_gradients(state, old_solution_gradients);

      // With this, we can then do the integration loop over all quadrature
      // points and shape functions.  Having just computed the gradients of
      // the old solution in the quadrature points, we are able to compute
      // the coefficients $a_{n}$ in these points.  The assembly of the
      // system itself then looks similar to what we always do with the
      // exception of the nonlinear terms, as does copying the results from
      // the local objects into the global ones:
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
          const double coeff =
            1.0 / std::sqrt(1 + old_solution_gradients[q] *
                                  old_solution_gradients[q]);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  (((fe_values.shape_grad(i, q)      // ((\nabla \phi_i
                     * coeff                         //   * a_n
                     * fe_values.shape_grad(j, q))   //   * \nabla \phi_j)
                    -                                //  -
                    (fe_values.shape_grad(i, q)      //  (\nabla \phi_i
                     * coeff * coeff * coeff         //   * a_n^3
                     * (fe_values.shape_grad(j, q)   //   * (\nabla \phi_j
                        * old_solution_gradients[q]) //      * \nabla u_n)
                     * old_solution_gradients[q]))   //   * \nabla u_n)))
                   * fe_values.JxW(q));              // * dx

              // residual of operator
              cell_rhs(i) -= (fe_values.shape_grad(i, q)  // \nabla \phi_i
                              * coeff                     // * a_n
                              * old_solution_gradients[q] // * \nabla u_n
                              * fe_values.JxW(q));        // * dx
              // source term
              const auto &x_q = fe_values.quadrature_point(q);
              cell_rhs(i) += (fe_values.shape_value(i, q) *          // phi_i(x_q)
                              case_->get_right_hand_side()->value(x_q) *   // f(x_q)
                              fe_values.JxW(q));                      // dx
          }
      }

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            mat.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));

          rhs(local_dof_indices[i]) += cell_rhs(i);
      }
  }

  // we remove hanging nodes from the system
  affine_constraints_.condense(mat);
  affine_constraints_.condense(rhs);

  // apply zero boundary values to the linear system that defines the Newton updates
  // $\delta u^n$:
  ////////// MOVED TO PDEBASE
  //for(auto bc : case_->get_dirichlet_bcs()) {
  //  std::map<types::global_dof_index, double> boundary_values;
  //  VectorTools::interpolate_boundary_values(dof_handler_,
  //                                           bc.bc_id, Functions::ZeroFunction<dim>(),
  //                                           boundary_values);
  //  MatrixTools::apply_boundary_values(boundary_values,
  //                                     mat,
  //                                     update_,
  //                                     rhs);
  //}
}


// @sect4{MinimalSurface::compute_residual}

// In order to monitor convergence, we need a way to compute the norm of the
// (discrete) residual, i.e., the norm of the vector
// $\left<F(u^n),\varphi_i\right>$ with $F(u)=-\nabla \cdot \left(
// \frac{1}{\sqrt{1+|\nabla u|^{2}}}\nabla u \right)$ as discussed in the
// introduction. It turns out that (although we don't use this feature in
// the current version of the program) one needs to compute the residual
// $\left<F(u^n+\alpha^n\;\delta u^n),\varphi_i\right>$ when determining
// optimal step lengths, and so this is what we implement here: the function
// takes the step length $\alpha^n$ as an argument. The original
// functionality is of course obtained by passing a zero as argument.
//
// In the function below, we first set up a vector for the residual, and
// then a vector for the evaluation point $u^n+\alpha^n\;\delta u^n$. This
// is followed by the same boilerplate code we use for all integration
// operations:
template <int dim>
double MinimalSurface<dim>::compute_residual(const double alpha, const vector_type& solution,
                                             const vector_type& update) const
{
    Vector<double> residual(dof_handler_.n_dofs());

    Vector<double> evaluation_point(dof_handler_.n_dofs());
    evaluation_point = solution;
    evaluation_point.add(alpha, update);

    const QGauss<dim> quadrature_formula(fe_.degree + 1);
    FEValues<dim>     fe_values(fe_,
                            quadrature_formula,
                            update_gradients | update_quadrature_points |
                              update_JxW_values);

    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double>              cell_residual(dofs_per_cell);
    std::vector<Tensor<1, dim>> gradients(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler_.active_cell_iterators())
    {
        cell_residual = 0;
        fe_values.reinit(cell);

        // The actual computation is much as in
        // <code>assemble_system()</code>. We first evaluate the gradients of
        // $u^n+\alpha^n\,\delta u^n$ at the quadrature points, then compute
        // the coefficient $a_n$, and then plug it all into the formula for
        // the residual:
        fe_values.get_function_gradients(evaluation_point, gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double coeff =
                1. / std::sqrt(1 + gradients[q] * gradients[q]);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                cell_residual(i) -= (fe_values.shape_grad(i, q) // \nabla \phi_i
                                     * coeff                    // * a_n
                                     * gradients[q]             // * \nabla u_n
                                     * fe_values.JxW(q));       // * dx
            }
        }
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            residual(local_dof_indices[i]) += cell_residual(i);
        }
    }

    // At the end of this function we also have to deal with the hanging node
    // constraints and with the issue of boundary values. With regard to the
    // latter, we have to set to zero the elements of the residual vector for
    // all entries that correspond to degrees of freedom that sit at the
    // boundary. The reason is that because the value of the solution there is
    // fixed, they are of course no "real" degrees of freedom and so, strictly
    // speaking, we shouldn't have assembled entries in the residual vector
    // for them. However, as we always do, we want to do exactly the same
    // thing on every cell and so we didn't want to deal with the question
    // of whether a particular degree of freedom sits at the boundary in the
    // integration above. Rather, we will simply set to zero these entries
    // after the fact. To this end, we need to determine which degrees
    // of freedom do in fact belong to the boundary and then loop over all of
    // those and set the residual entry to zero. This happens in the following
    // lines which we have already seen used in step-11, using the appropriate
    // function from namespace DoFTools:
    affine_constraints_.condense(residual);

    for (const types::global_dof_index i : DoFTools::extract_boundary_dofs(dof_handler_)) {
        residual(i) = 0;
    }

    // At the end of the function, we return the norm of the residual:
    return residual.l2_norm();
}


// @sect4{MinimalSurface::output_results}

// This last function to be called from `run()` outputs the current solution
// (and the Newton update) in graphical form as a VTU file. It is entirely the
// same as what has been used in previous tutorials.
template <int dim>
void MinimalSurface<dim>::output_results(const int refinement_cycle,
                                         const vector_type& solution) const
{
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler_);
    data_out.add_data_vector(solution, "solution");
    //data_out.add_data_vector(update_, "update");
    data_out.build_patches();

    const std::string file_prefix = params_.output_path + "-" +
        Utilities::int_to_string(refinement_cycle, 2);
    const std::string filename = file_prefix + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    std::ofstream b_output(file_prefix + "-boundary.vtk");
    DataOutFaces<dim> data_out_boundary(true);
    std::vector<std::string> face_name(1, "solution");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        face_component_type(1, DataComponentInterpretation::component_is_scalar);
    data_out_boundary.add_data_vector(dof_handler_,
                                      solution,
                                      face_name,
                                      face_component_type);
    data_out_boundary.build_patches(fe_.degree);
    data_out_boundary.write_vtk(b_output);
    b_output.close();
}

#if 0
// @sect4{MinimalSurface::run}

// In the run function, we build the first grid and then have the top-level
// logic for the Newton iteration.
//
// As described in the introduction, the domain is the unit disk around
// the origin, created in the same way as shown in step-6. The mesh is
// globally refined twice followed later on by several adaptive cycles.
//
// Before starting the Newton loop, we also need to do a bit of
// setup work: We need to create the basic data structures and
// ensure that the first Newton iterate already has the correct
// boundary values, as discussed in the introduction.
template <int dim>
void MinimalSurface<dim>::run()
{
    this->make_grid(params_.initial_resolution);
    //setup_system(/*first time=*/true);
    //set_boundary_values();

    if(params_.is_adaptive) {
        // The Newton iteration starts next. We iterate until the (norm of the)
        // residual computed at the end of the previous iteration is less than
        // $10^{-3}$, as checked at the end of the `do { ... } while` loop that
        // starts here. Because we don't have a reasonable value to initialize
        // the variable, we just use the largest value that can be represented
        // as a `double`.
        double last_residual_norm = std::numeric_limits<double>::max();
        int refinement_cycle = 0;
        do
        {
            std::cout << "Adaptive mesh refinement step " << refinement_cycle << std::endl;

            if (refinement_cycle != 0) {
              //refine_mesh();
            }

            // set up FEValues for residual norm computation
            const QGauss<dim> quadrature_formula(fe_.degree + 1);
            FEValues<dim> fe_values(fe_,
                                    quadrature_formula,
                                    update_quadrature_points | update_JxW_values
                                    | update_values);

            // On every mesh we do exactly five Newton steps. We print the initial
            // residual here and then start the iterations on this mesh.
            //
            // In every Newton step the system matrix and the right hand side have
            // to be computed first, after which we store the norm of the right
            // hand side as the residual to check against when deciding whether to
            // stop the iterations. We then solve the linear system (the function
            // also updates $u^{n+1}=u^n+\alpha^n\;\delta u^n$) and output the
            // norm of the residual at the end of this Newton step.
            //
            // After the end of this loop, we then also output the solution on the
            // current mesh in graphical form and increment the counter for the
            // mesh refinement cycle.
            std::cout << "  Initial residual: " << compute_residual(0) << std::endl;

            for(int inner_it = 0; inner_it < solver_params_.max_its; ++inner_it)
            {
                //assemble_system(AssemblyOptions{false});
                last_residual_norm = utils::compute_Lp_norm(fe_values, dof_handler_,
                                                            rhs_, 2);
                solve();
                std::cout << "  Residual norm: " << last_residual_norm << std::endl;
            }

            output_results(refinement_cycle);

            ++refinement_cycle;
            std::cout << std::endl;
        }
        while (last_residual_norm > solver_params_.tolerance &&
               refinement_cycle < params_.refine_levels);
    } else {
        std::cout << "Running globally-refined meshes.\n";
        double init_res = compute_residual(0);
        for(int imesh = 0; imesh < params_.refine_levels; imesh++) {
            std::cout << "  Grid " << imesh << ": Initial residual norm: " << init_res << std::endl;
            double last_residual_norm = std::numeric_limits<double>::max();
            const int max_its = (imesh == params_.refine_levels - 1) ?
              solver_params_.max_its : 10;
            const double tolerance = (imesh == params_.refine_levels - 1) ?
              solver_params_.tolerance : 1e-1;

            // set up FEValues for residual norm computation
            const QGauss<dim> quadrature_formula(fe_.degree + 1);
            FEValues<dim> fe_values(fe_,
                                    quadrature_formula,
                                    update_quadrature_points | update_JxW_values
                                    | update_values);

            for(int inner_it = 0; inner_it < max_its; ++inner_it) {
                // compute RHS, Jacobian matrix
                //assemble_system(AssemblyOptions{false});
                // Maybe use function L2 norm for determining convergence
                last_residual_norm = utils::compute_Lp_norm(fe_values, dof_handler_,
                                                            rhs_, 2);
                // Set reference norm
                // Sometimes, if the boundary function is aliased on a very coarse grid,
                //  the initial residual can be zero. If so, update it on a finer grid.
                if((imesh == 0 && inner_it == 0) || (inner_it == 0 && init_res < 1e-14)) {
                    init_res = last_residual_norm;
                }
                solve();
                std::cout << "    Iter. " << inner_it
                          << " Abs. residual L2 norm: " << last_residual_norm << std::endl;
                if(last_residual_norm / init_res < tolerance) {
                    std::cout << "Converged in " << inner_it + 1 << " iterations." << std::endl;
                    break;
                }
            }
            std::cout << "Relative residual = " << last_residual_norm / init_res << std::endl;
            output_results(imesh);
            //refine_mesh();
        }
    }
}
#endif

template class MinimalSurface<2>;

} // namespace
}

