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
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

//#include <deal.II/base/logstream.h>

// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:

namespace paramsim {
namespace pde {

template <int dim>
PoissonCG<dim>::PoissonCG(std::shared_ptr<const Case<dim>> tcase, const PDEParams& params)
  : DiscretePDE<dim,fe_type>(tcase, params)
{ }

template <int dim>
void PoissonCG<dim>::evaluate_residual(const vector_type& state, vector_type& rhs) const
{
    rhs = 0;

    QGauss<dim> quadrature_formula(fe_.degree + 1);
    const unsigned int n_q_points    = quadrature_formula.size();

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

    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<Tensor<1, dim>> solution_gradients(n_q_points);

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
        cell_rhs    = 0;

        fe_values.get_function_gradients(state, solution_gradients);

        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            evaluate_point_residual(fe_values, solution_gradients, q_index, cell_rhs);
        }

        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
        {
            rhs(local_dof_indices[i]) += cell_rhs(i);
        }

        affine_constraints_.condense(rhs);
    }
}

// The assembly is in error-correction form, so even though this is a linear problem,
// it's supposed to be solved by a Newton-like approach.
template <int dim>
void PoissonCG<dim>::assemble_system(AssemblyOptions, const vector_type& state,
                                     dealii::SparseMatrix<double>& mat, vector_type& rhs) const
{
    mat = 0;
    rhs = 0;

    QGauss<dim> quadrature_formula(fe_.degree + 1);
    const unsigned int n_q_points    = quadrature_formula.size();

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
    std::vector<Tensor<1, dim>> solution_gradients(n_q_points);

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

        fe_values.get_function_gradients(state, solution_gradients);

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
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                     fe_values.JxW(q_index));           // dx
            }
            evaluate_point_residual(fe_values, solution_gradients, q_index, cell_rhs);
        }

        affine_constraints_.condense(mat);
        affine_constraints_.condense(rhs);

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
                mat.add(local_dof_indices[i],
                                local_dof_indices[j],
                                cell_matrix(i, j));

            rhs(local_dof_indices[i]) += cell_rhs(i);
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
    //// MOVED TO PDEBASE
    //for(auto bc : case_->get_dirichlet_bcs()) {
    //    std::map<types::global_dof_index, double> boundary_values;
    //    VectorTools::interpolate_boundary_values(dof_handler_,
    //                                             bc.bc_id, *bc.bc_fn,
    //                                             boundary_values);
    //    MatrixTools::apply_boundary_values(boundary_values, mat, state, rhs);
    //}
}

template <int dim>
void PoissonCG<dim>::output_results(const int cycle, const vector_type& solution) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler_);
  data_out.add_data_vector(solution, "solution");

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
                                    solution,
                                    face_name,
                                    face_component_type);
  data_out_boundary.build_patches(fe_.degree);
  data_out_boundary.write_vtk(b_output);
  b_output.close();
}

template class PoissonCG<2>;

}
}
