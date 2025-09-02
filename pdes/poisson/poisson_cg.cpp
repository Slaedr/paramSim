/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2023 by the deal.II authors
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
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */
#include "poisson_cg.hpp"

#include <fstream>
#include <iostream>

// @sect3{Include files}

// The first few (many?) include files have already been used in the previous
// example, so we will not explain their meaning here again.
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

//#include <deal.II/base/logstream.h>

// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:

namespace paramsim {
namespace pde {

template <int dim>
PoissonCG<dim>::PoissonCG(std::shared_ptr<const Case<dim>> tcase, const PDEParams& params,
                          const SolverParams& sparams)
  : DiscretePDE<dim>(tcase, params, sparams), fe_(params.fe_degree)
{ }

template <int dim>
std::shared_ptr<Vector<double>> PoissonCG<dim>::create_solution_vector() const
{
    auto vec = std::make_shared<Vector<double>>();
    vec->reinit(dof_handler_.n_dofs());
    return vec;
}

template <int dim>
void PoissonCG<dim>::setup_system(bool)
{
    dof_handler_.distribute_dofs(fe_);

    std::cout << "   Number of degrees of freedom: " << dof_handler_.n_dofs()
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler_.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_, dsp);
    sparsity_pattern_.copy_from(dsp);

    system_matrix_.reinit(sparsity_pattern_);

    solution_.reinit(dof_handler_.n_dofs());
    rhs_.reinit(dof_handler_.n_dofs());
}


template <int dim>
void PoissonCG<dim>::assemble_system(AssemblyOptions)
{
  QGauss<dim> quadrature_formula(fe_.degree + 1);

  // In order to evaluate the non-constant
  // right hand side function we now also need the quadrature points on the
  // cell we are presently on in addition to values and
  // gradients of the shape function from the FEValues object, as well as the
  // quadrature weights, FEValues::JxW(). We can tell the FEValues object to
  // do for us by also giving it the #update_quadrature_points flag:
  FEValues<dim> fe_values(fe_,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // We then again define the same abbreviation as in the previous program.
  // The value of this variable of course depends on the dimension which we
  // are presently using, but the FiniteElement class does all the necessary
  // work for you and you don't have to care about the dimension dependent
  // parts:
  const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Next, we again have to loop over all cells and assemble local
  // contributions.  Note, that a cell is a quadrilateral in two space
  // dimensions, but a hexahedron in 3d. In fact, the
  // <code>active_cell_iterator</code> data type is something different,
  // depending on the dimension we are in, but to the outside world they look
  // alike and you will probably never see a difference. In any case, the real
  // type is hidden by using `auto`:
  for (const auto &cell : dof_handler_.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      // Now we have to assemble the local matrix and right hand side. This is
      // done exactly like in the previous example, but now we revert the
      // order of the loops (which we can safely do since they are independent
      // of each other) and merge the loops for the local matrix and the local
      // vector as far as possible to make things a bit faster.
      //
      // Assembling the right hand side presents the only significant
      // difference to how we did things in step-3: Instead of using a
      // constant right hand side with value 1, we use the object representing
      // the right hand side and evaluate it at the quadrature points:
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

            const auto &x_q = fe_values.quadrature_point(q_index);
            cell_rhs(i) += (fe_values.shape_value(i, q_index) *          // phi_i(x_q)
                            case_->get_right_hand_side()->value(x_q) *   // f(x_q)
                            fe_values.JxW(q_index));                      // dx
          }
      // As a final remark to these loops: when we assemble the local
      // contributions into <code>cell_matrix(i,j)</code>, we have to multiply
      // the gradients of shape functions $i$ and $j$ at point number
      // q_index and
      // multiply it with the scalar weights JxW. This is what actually
      // happens: <code>fe_values.shape_grad(i,q_index)</code> returns a
      // <code>dim</code> dimensional vector, represented by a
      // <code>Tensor@<1,dim@></code> object, and the operator* that
      // multiplies it with the result of
      // <code>fe_values.shape_grad(j,q_index)</code> makes sure that the
      // <code>dim</code> components of the two vectors are properly
      // contracted, and the result is a scalar floating point number that
      // then is multiplied with the weights. Internally, this operator* makes
      // sure that this happens correctly for all <code>dim</code> components
      // of the vectors, whether <code>dim</code> be 2, 3, or any other space
      // dimension; from a user's perspective, this is not something worth
      // bothering with, however, making things a lot simpler if one wants to
      // write code dimension independently.

      // With the local systems assembled, the transfer into the global matrix
      // and right hand side is done exactly as before, but here we have again
      // merged some loops for efficiency:
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            system_matrix_.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          rhs_(local_dof_indices[i]) += cell_rhs(i);
        }
    }

  // As the final step in this function, we wanted to have non-homogeneous
  // boundary values in this example, unlike the one before. This is a simple
  // task, we only have to replace the Functions::ZeroFunction used there by an
  // object of the class which describes the boundary values we would like to
  // use (i.e. the <code>BoundaryValues</code> class declared above):
  //
  // The function VectorTools::interpolate_boundary_values() will only work
  // on faces that have been marked with boundary indicator 0 (because that's
  // what we say the function should work on with the second argument below).
  // If there are faces with boundary id other than 0, then the function
  // interpolate_boundary_values will do nothing on these faces. For
  // the Laplace equation doing nothing is equivalent to assuming that
  // on those parts of the boundary a zero Neumann boundary condition holds.
  for(auto bc : case_->get_dirichlet_bcs()) {
      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(dof_handler_,
                                               bc.bc_id, *bc.bc_fn,
                                               boundary_values);
      MatrixTools::apply_boundary_values(boundary_values,
                                         system_matrix_,
                                         solution_,
                                         rhs_);
  }
}

template <int dim>
void PoissonCG<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  PreconditionSSOR<SparseMatrix<double> > precondition;
  precondition.initialize(
    system_matrix_, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(.8));
  solver.solve(system_matrix_, solution_, rhs_, precondition);

  // We have made one addition, though: since we suppress output from the
  // linear solvers, we have to print the number of iterations by hand.
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}

template <int dim>
void PoissonCG<dim>::output_results(const int cycle) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler_);
  data_out.add_data_vector(solution_, "solution");

  data_out.build_patches();

  const std::string file_prefix = params_.output_path + "-" + std::to_string(cycle);
  std::ofstream output(file_prefix + ".vtk");
  data_out.write_vtk(output);
   
  std::ofstream b_output(file_prefix + "-boundary.vtk");
  DataOutFaces<dim> data_out_boundary(true);
  std::vector<std::string> face_name(1, "solution");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      face_component_type(1, DataComponentInterpretation::component_is_scalar);
  data_out_boundary.add_data_vector(dof_handler_,
                                    solution_,
                                    face_name,
                                    face_component_type);
  data_out_boundary.build_patches(fe_.degree);
  data_out_boundary.write_vtk(b_output);
  b_output.close();
}

template <int dim>
void PoissonCG<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;
  
  auto scase = std::dynamic_pointer_cast<const HasExactSolution<dim>>(this->case_);
  ConvergenceTable convergence_table;

  for(int imesh = 0; imesh < params_.refine_levels; imesh++)
  {
    const unsigned resolution = this->params_.initial_resolution * (imesh+1);
    this->make_grid(resolution);
    setup_system(false);
    assemble_system(AssemblyOptions{false});
    solve();
    output_results(imesh);

    convergence_table.add_value("cells", tria_.n_active_cells());
    convergence_table.add_value("dofs", dof_handler_.n_dofs());

    if(scase && scase->get_exact_solution()) {
      Vector<float> difference_per_cell(tria_.n_active_cells());

      VectorTools::integrate_difference(dof_handler_,
                                        solution_,
                                        *scase->get_exact_solution(),
                                        difference_per_cell,
                                        QGauss<dim>(params_.fe_degree + 2),
                                        VectorTools::L2_norm);
      const double post_error =
        VectorTools::compute_global_error(tria_, difference_per_cell,
                                          VectorTools::L2_norm);

      convergence_table.add_value("val L2", post_error);
      convergence_table.set_scientific("val L2", true);
      convergence_table.set_precision("val L2", 3);
    }
  }
  
  if(scase && scase->get_exact_solution()) {
    convergence_table.evaluate_convergence_rates(
            "val L2", "cells", ConvergenceTable::reduction_rate_log2, dim);
    convergence_table.write_text(std::cout);
  }
}

template class PoissonCG<2>;

}
}
